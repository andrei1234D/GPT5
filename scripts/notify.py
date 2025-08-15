# scripts/notify.py
import os, re, sys, time, requests, pytz
from datetime import datetime

from pathlib import Path
from universe import load_universe
from features import build_features
from filters import is_garbage, daily_index_filter
from proxies import get_spy_ctx, derive_proxies, fund_proxies_from_feats, catalyst_severity_from_feats
from fundamentals import fetch_next_earnings_days  # still used for EARNINGS_SOON sev
from data_fetcher import fetch_pe_for_top          # your PE fetcher
from prompt_blocks import BASELINE_HINTS, build_prompt_block
from gpt_client import call_gpt5 
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE
from time_utils import seconds_until_target_hour
from debugger import post_debug_inputs_to_discord

from trash_ranker import RobustRanker
from quick_scorer import rank_stage1

TZ = pytz.timezone("Europe/Bucharest")

def log(m): print(m, flush=True)
def need_env(name):
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Missing env: {name}", flush=True); sys.exit(1)
    return v

OPENAI_API_KEY = need_env("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = need_env("DISCORD_WEBHOOK_URL")

# ——— Strengthen system prompt ———
SYSTEM_PROMPT_TOP20_EXT = SYSTEM_PROMPT_TOP20 + """
INPUT EXTRAS:
- Each candidate block may include:
  • DATA_AVAILABILITY: which of {FUNDAMENTALS, VALUATION, RISKS, CATALYSTS} are MISSING or PARTIAL.
  • BASELINE_HINTS: exact baselines for each category.
  • PROXIES: price/volume proxies with severities (−5..+5) and directions (+/−).
  • PROXIES_FUNDAMENTALS: {GROWTH_TECH, MARGIN_TREND_TECH, FCF_TREND_TECH, OP_EFF_TREND_TECH} as signed severities (−5..+5).
  • PROXIES_CATALYSTS: {TECH_BREAKOUT, TECH_BREAKDOWN, DIP_REVERSAL, EARNINGS_SOON} with signed severities.
  • CATALYST_TIMING_HINTS: e.g., TECH_BREAKOUT=Today/None.
  • EXPECTED_VOLATILITY_PCT: derived from ATR%.
  • FVA_HINT: technical fair-value anchor seed, derived only from indicators.
  • PE_HINT: optional numeric P/E (trailing preferred; forward if trailing unavailable).

MANDATORY HANDLING:
- If DATA_AVAILABILITY says a category is MISSING ⇒ set that category exactly to its BASELINE_HINTS value (do NOT set 0).
- If PARTIAL ⇒ use only the provided PROXIES; keep unaddressed sub-factors at baseline.
- Never assume unknown = bad; unknown = baseline.
- PROXIES map 1–5 severity to the factor ranges you already have.
- CATALYST PROXY MAPPING (apply to 'Near-Term Catalysts' base 100):
    • TECH_BREAKOUT (+1..+5)  ⇒ +10, +20, +30, +45, +60
    • DIP_REVERSAL (+1..+5)   ⇒ +8, +12, +18, +24, +30
    • TECH_BREAKDOWN (−1..−5) ⇒ −10, −20, −30, −45, −60
    • EARNINGS_SOON (+1..+5)  ⇒ +5, +8, +10, +12, +15
  Timing multiplier applies for TECH_BREAKOUT timing: Today × 1.50.
- P/E OVERLAY: apply only as specified in Technical Valuation & Risks sections; never use P/E to modify Market & Sector,
  Quality (Tech Proxies), or Near-Term Catalysts.
"""


force = os.getenv("FORCE_RUN", "").lower() in {"1","true","yes"}

def fail(msg: str):
    log(f"[ERROR] {msg}")
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username":"Daily Stock Alert","content":f"⚠️ {msg}"}, timeout=60)
    except Exception:
        pass

def dump_blocks_pre_gpt(
    blocks: list[str],
    user_prompt: str | None,
    tz,
    out_dir: str = "data/logs",
    preview_lines: int = 12,
    echo_stdout: bool = False,
) -> str:
    """
    Writes a full copy of the 10 candidate blocks (exact strings sent to GPT)
    to data/logs/blocks_to_gpt_<timestamp>.txt. Optionally echoes a short
    preview to stdout. Returns the file path.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"blocks_to_gpt_{ts}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== 10 BLOCKS SENT TO GPT ===\n")
        for i, b in enumerate(blocks, 1):
            f.write(f"\n--- BLOCK {i}/{len(blocks)} ---\n")
            f.write(b.rstrip() + "\n")
        if user_prompt is not None:
            f.write("\n=== FULL USER PROMPT (for reproducibility) ===\n")
            f.write(user_prompt.rstrip() + "\n")

    if echo_stdout:
        print(f"[DEBUG] Wrote GPT input blocks to {path}")
        for i, b in enumerate(blocks, 1):
            head = "\n".join(b.splitlines()[:preview_lines])
            print(f"\n[DEBUG] BLOCK {i} preview:\n{head}\n... (truncated)")

    return path


def main():
    now = datetime.now(TZ)
    log(f"[INFO] Start {now.isoformat()} Europe/Bucharest. FORCE_RUN={force}")

    # 1) Load universe
    universe = load_universe()  # list[(ticker, company)]
    log(f"[INFO] Universe size: {len(universe)}")

    # 2) Features for all
    feats_map = build_features(universe, batch_size=int(os.getenv("YF_CHUNK_SIZE", "80")))
    if not feats_map:
        return fail("No features computed (network/data)")

    # 3) Trash filter (legacy fast filter)
    kept = []
    for t, name in universe:
        row = feats_map.get(t)
        if not row:
            continue
        feats = row["features"]
        if not is_garbage(feats):   # your existing coarse filter
            kept.append((t, name, feats))
    log(f"[INFO] After trash filter: {len(kept)} remain")
    if not kept:
        return fail("All filtered in trash stage")

    # 4) Daily index filter (placeholder)
    today_context = {"bench_trend": "up", "sector_trend": "up", "breadth50": 55}
    kept2 = [(t,n,f) for (t,n,f) in kept if daily_index_filter(f, today_context)]
    log(f"[INFO] After daily index filter: {len(kept2)} remain")
    if not kept2:
        return fail("All filtered by daily context")

    # 5) Stage-1 — QUICK pass (resilient) -> pre_top200
    pre_top200, quick_scored, removed = rank_stage1(
        kept2,
        keep=int(os.getenv("STAGE1_KEEP", "200")),
        mode=os.getenv("STAGE1_MODE", "loose"),
        rescue_frac=float(os.getenv("STAGE1_RESCUE_FRAC", "0.15")),
        log_dir="data"
    )
    log(f"[INFO] Stage-1 survivors for Stage-2: {len(pre_top200)}")

    # 6) Stage-2 — THOROUGH RobustRanker on those survivors -> resort -> Top-10
    ranker = RobustRanker()
    ranker.fit_cross_section([f for (t, n, f, s, meta) in pre_top200])

    thorough_ranked = []
    for (t, n, f, _score, _meta) in pre_top200:
        if ranker.should_drop(f):   # uses env-tunable HardFilter
            continue
        score, parts = ranker.composite_score(f)
        thorough_ranked.append((t, n, f, score))

    thorough_ranked.sort(key=lambda x: x[3], reverse=True)
    top200 = thorough_ranked[:200]
    if not top200:
        return fail("No candidates after Stage-2 robust scorer")
    log(f"[INFO] Stage-2 leader: {top200[0][0]}")

    # 7) Prepare TOP 10 blocks + debug payloads (use the thorough Top-10)
    top10 = top200[:10]
    tickers_top10 = [t for t, _, _, _ in top10]

    spy_ctx = get_spy_ctx()
    pe_map = fetch_pe_for_top(tickers_top10)
    earn_days_map = fetch_next_earnings_days(tickers_top10)

    blocks = []
    debug_inputs = {}
    baseline_str = "; ".join([f"{k}={v}" for k, v in BASELINE_HINTS.items()])

    for t, name, feats, _ in top10:
        proxies = derive_proxies(feats, spy_ctx)
        fund_proxy = fund_proxies_from_feats(feats)
        cat = catalyst_severity_from_feats(feats)

        earn_days = earn_days_map.get(t)
        if earn_days is None: earn_sev = 0
        elif earn_days <= 0:  earn_sev = 5
        elif earn_days <= 3:  earn_sev = 4
        elif earn_days <= 7:  earn_sev = 3
        elif earn_days <= 14: earn_sev = 2
        else:                 earn_sev = 0

        pe_hint = pe_map.get(t)

        block_text, debug_dict = build_prompt_block(
            t=t, name=name, feats=feats, proxies=proxies, fund_proxy=fund_proxy,
            cat=cat, earn_sev=earn_sev, fm={},  # we no longer pass valuation/fundamentals
            baseline_hints=BASELINE_HINTS, baseline_str=baseline_str,
            pe_hint=pe_hint  # <-- ensure your build_prompt_block supports this arg
        )
        blocks.append(block_text)
        debug_inputs[t] = debug_dict

    blocks_text = "\n\n".join(blocks)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(today=now.strftime("%b %d"), blocks=blocks_text)

    # --- NEW: log the exact inputs we’re about to send ---
    if os.getenv("LOG_GPT_INPUT", "1").lower() in {"1", "true", "yes"}:
        echo = os.getenv("LOG_GPT_INPUT_STDOUT", "0").lower() in {"1", "true", "yes"}
        dump_path = dump_blocks_pre_gpt(blocks, user_prompt, TZ, echo_stdout=echo)
        log(f"[INFO] Dumped pre-GPT blocks to {dump_path}")

    # 8) GPT-5 adjudication
    
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20, user_prompt, max_tokens=13000, timeout=float(os.getenv("OPENAI_TIMEOUT","360")))
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")

    # 9) Parse selected tickers from GPT output and post debug
    RE_PICK_TICKER = re.compile(r"(?im)^\s*(?:\d+\)\s*)?(?:\*\*)?([A-Z][A-Z0-9.\-]{1,10})\s+[–-]")
    RE_FORECAST_TICK = re.compile(r"(?im)Forecast\s+image\s+URL:\s*https?://[^/]+/stocks/([A-Z0-9.\-]+)/forecast\b")
    picked = list({*RE_PICK_TICKER.findall(final_text), *RE_FORECAST_TICK.findall(final_text)})

    if picked:
        post_debug_inputs_to_discord(picked, debug_inputs, DISCORD_WEBHOOK_URL)
    else:
        log("[WARN] Could not parse any selected ticker from GPT output; skipping debug post.")

    # 10) Save & (optionally) wait until 08:00
    with open("daily_pick.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    log("[INFO] Draft saved to daily_pick.txt")

    if not force:
        wait_s = max(0, seconds_until_target_hour(8, 0, TZ))
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # 11) Send to Discord
    embed = {"title": f"Daily Stock Pick — {datetime.now(TZ).strftime('%Y-%m-%d')}", "description": final_text}
    payload = {"username": "Daily Stock Alert", "embeds": [embed]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=60).raise_for_status()
        log("[INFO] Posted alert to Discord ✅")
    except Exception as e:
        log(f"[ERROR] Discord webhook error: {repr(e)}")

if __name__ == "__main__":
    main()
