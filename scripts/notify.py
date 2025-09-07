# scripts/notify.py
import os, re, sys, time, requests, pytz
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from universe import load_universe
from features import build_features
from filters import is_garbage, daily_index_filter
from proxies import (
    get_spy_ctx,
    derive_proxies,
    fund_proxies_from_feats,
    catalyst_severity_from_feats,
)
from fundamentals import fetch_next_earnings_days
from data_fetcher import fetch_valuations_for_top
from prompt_blocks import BASELINE_HINTS, build_prompt_block
from gpt_client import call_gpt5
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE
from time_utils import seconds_until_target_hour
from debugger import post_debug_inputs_to_discord

from trash_ranker import RobustRanker, pick_top_stratified
from quick_scorer import rank_stage1, quick_score  # we rescore after adding PE
from alphavantage_jit import get_news_sentiment_bulk


TZ = pytz.timezone("Europe/Bucharest")

def log(m): print(m, flush=True)

def need_env(name):
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Missing env: {name}", flush=True); sys.exit(1)
    return v

OPENAI_API_KEY = need_env("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = need_env("DISCORD_WEBHOOK_URL")
DISCORD_DEBUG_WEBHOOK_URL = os.getenv("DISCORD_DEBUG_WEBHOOK_URL") or DISCORD_WEBHOOK_URL

# ——— Strengthened system prompt (adds strict PLAN rules) ———

ALLOW_BLOWOFF_PROBE = os.getenv("ALLOW_BLOWOFF_PROBE", "0") == "1"
SYSTEM_PROMPT_TOP20_EXT = SYSTEM_PROMPT_TOP20 + """
INPUT EXTRAS:
- Each candidate block may include:
  • BASELINE_HINTS: exact baselines for each category.
  • PROXIES: price/volume proxies with severities (−5..+5) and directions (+/−).
  • PROXIES_FUNDAMENTALS: {{GROWTH_TECH, MARGIN_TREND_TECH, FCF_TREND_TECH, OP_EFF_TREND_TECH}} as signed severities (−5..+5).
  • PROXIES_CATALYSTS: {{TECH_BREAKOUT, TECH_BREAKDOWN, DIP_REVERSAL, EARNINGS_SOON}} with signed severities.
  • CATALYST_TIMING_HINTS: e.g., TECH_BREAKOUT=Today/None.
  • EXPECTED_VOLATILITY_PCT: derived from ATR% (clip to 1..6).
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
- P/E OVERLAY: apply only inside Technical Valuation & Risks; never touch Market & Sector, Quality (Tech Proxies), or Near-Term Catalysts.

PLAN BUILDER (MANDATORY; DO NOT SKIP):
- FAIR-VALUE ANCHOR & PLAN (tech first with tiny PE tilt):
  1) Compute a single $FVA for TODAY.
     Start from FVA_HINT and apply a trend-aware catch-up:
       FVA := max(FVA_HINT, 0.98×SMA50, 0.96×EMA50).
     Then allow a tiny PE bias (only if PE_HINT present; keep within the global clamp below; VALUATION_HISTORY is −5..+5):
       • If PE_HINT ≤ 10 and VALUATION_HISTORY ≥ +2 → tilt FVA up by +1%..+3%.
       • If PE_HINT ≥ 50 and VALUATION_HISTORY ≤ −2 → tilt FVA down by −1%..−3%.
  2) Global clamp: |FVA − PRICE| ≤ 25% unless very strong technical evidence.
     Treat as “very strong” only if TECH_BREAKOUT ≥ +4 AND RSI14 ≥ 75 AND vsSMA50 ≥ +20% AND Vol_vs_20d% ≥ +150%.
  3) Let EV = EXPECTED_VOLATILITY_PCT, with EV := min(max(EV, 1), 6).
     Early-trend EV floor: if 52≤RSI14≤66 AND 0≤vsSMA50≤8 AND vsSMA200≥10 → EV := max(EV, 4).
  4) Standard plan:
     Buy = FVA × (1 − 0.8×EV/100) … FVA × (1 + 0.8×EV/100);
     Stop = FVA × (1 − 2.0×EV/100);
     Target = FVA × (1 + 3.0×EV/100).
     Sanity: if stop ≥ buy_low → push to min(buy_low×0.99, FVA×(1 − 2.2×EV/100));
             if target ≤ buy_high → push to max(buy_high×1.05, FVA×(1 + 3.2×EV/100)).
 5) Overheat/BLOWOFF guard (any of the following):
      A) RSI14 ≥ 75 AND vsSMA50 ≥ 30% AND Vol_vs_20d% ≥ +80%
      B) RSI14 ≥ 80 AND vsSMA50 ≥ 40%
      C) RSI14 ≥ 78 AND vsSMA50 ≥ 60%
   → Output: "No trade — momentum blowoff; wait for cooling. (Anchor: $FVA)"

     → Normally output: "No trade — momentum blowoff; wait for cooling. (Anchor: $FVA)"
     → If CONFIG.ALLOW_BLOWOFF_PROBE=1 AND Vol_vs_20d% < 150:
         Use PROBE PLAN instead (tight risk):
           EVp := max(EV, 4);
           Anchor := max(min(FVA, PRICE×1.02), PRICE×0.97);
           Buy = PRICE × (1 − 0.5×EVp/100) … PRICE × (1 + 0.2×EVp/100);
           Stop = max(PRICE × (1 − 2.2×EVp/100), EMA20×0.995 if EMA20<PRICE);
           Target = PRICE × (1 + 3.8×EVp/100);
           Output exactly: "Probe buy X–Y; Stop Z; Target T; Max hold time: ≤ 6 months (Anchor: $Anchor) (Parabolic risk)"
  6) Rounding: round all $ to 2 decimals.
  7) Guards:
      • If after sanity Target ≤ CURRENT_PRICE → "No trade — extended; wait for pullback. (Anchor: $FVA)".
      • If CURRENT_PRICE > buy_high but CURRENT_PRICE < Target → append " (Wait for pullback into range.)".
      • If CURRENT_PRICE < buy_low by more than ~2×EV% → append " (Accumulation zone)".
- Keep plans terse; only the single-line plan plus the required summary fields.
"""
force = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def fail(msg: str):
    log(f"[ERROR] {msg}")
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"username": "Daily Stock Alert", "content": f"⚠️ {msg}"},
            timeout=60,
        )
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
    Writes a full copy of the candidate blocks (exact strings sent to GPT)
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
    universe = load_universe()
    log(f"[INFO] Universe size: {len(universe)}")

    # 2) Features for all
    feats_map = build_features(universe, batch_size=int(os.getenv("YF_CHUNK_SIZE", "80")))
    if not feats_map:
        return fail("No features computed (network/data)")

    # 3) Trash filter
    kept = []
    for t, name in universe:
        row = feats_map.get(t)
        if not row:
            continue
        feats = row["features"]
        if not is_garbage(feats):
            kept.append((t, name, feats))
    log(f"[INFO] After trash filter: {len(kept)} remain")
    if not kept:
        return fail("All filtered in trash stage")

    # 4) Daily index filter (placeholder)
    today_context = {"bench_trend": "up", "sector_trend": "up", "breadth50": 55}
    kept2 = [(t, n, f) for (t, n, f) in kept if daily_index_filter(f, today_context)]
    log(f"[INFO] After daily index filter: {len(kept2)} remain")
    if not kept2:
        return fail("All filtered by daily context")

    # 5) Stage-1 — QUICK pass
    pre_top200, quick_scored, removed = rank_stage1(
        kept2,
        keep=int(os.getenv("STAGE1_KEEP", "200")),
        mode=os.getenv("STAGE1_MODE", "loose"),
        rescue_frac=float(os.getenv("STAGE1_RESCUE_FRAC", "0.15")),
        log_dir="data"
    )
    log(f"[INFO] Stage-1 survivors for Stage-2: {len(pre_top200)}")

    # --- Optional: P/E refinement on the Stage-1 pool (fixed & simplified) ---
    if os.getenv("STAGE1_PE_RESCORE", "1").lower() in {"1", "true", "yes"} and pre_top200:
        pool_n = int(os.getenv("STAGE1_PE_POOL", "2000"))
        pool_tickers = [t for (t, _n, _f, _s, _m) in quick_scored[:pool_n]]
        pe_map = {}
        try:
            from data_fetcher import fetch_pe_for_top
            pe_map = fetch_pe_for_top(pool_tickers) or {}
        except Exception as e:
            log(f"[WARN] Stage-1 P/E refine skipped (fetch error): {e!r}")

        # Inject P/E and rescore once, preserving meta
        rescored = []
        mode = os.getenv("STAGE1_MODE", "loose")
        for (t, n, f, _s, m) in pre_top200:
            pe = pe_map.get(t)
            if pe is not None and pe > 0:
                f["val_PE"] = pe
            s2, parts2 = quick_score(f, mode=mode)
            # Preserve important meta fields
            meta2 = {
                "parts": parts2,
                "tags": m.get("tags", []),
                "tier": m.get("tier"),
                "avg_dollar_vol_20d": m.get("avg_dollar_vol_20d"),
            }
            rescored.append((t, n, f, s2, meta2))

        rescored.sort(key=lambda x: x[3], reverse=True)
        pre_top200 = rescored[:int(os.getenv("STAGE1_KEEP", "200"))]
        log("[INFO] Stage-1 P/E refine applied.")

     # 6) Load Stage-2 merged results instead of re-scoring
    path = "data/stage2_merged.csv"
    if not os.path.exists(path):
        return fail(f"{path} not found")

    df = pd.read_csv(path)
    if df.empty:
        return fail("stage2_merged.csv is empty")

    # preserve the CSV's order
    df = df.sort_values("merged_score", ascending=False).reset_index(drop=True)

    # reload full features for all tickers
    universe = [(t, "") for t in df["ticker"]]
    feats_map = build_features(universe, batch_size=int(os.getenv("YF_CHUNK_SIZE", "80")))

    ranked = []
    for _, row in df.iterrows():
        t = row["ticker"]
        n = row.get("name", t)

        # fresh features
        f = feats_map.get(t, {}).get("features", {})
        if f is None:
            f = {}

        # overlay scores/meta from stage2_merged.csv
        f.update(row.to_dict())

        s = row.get("merged_score", 0.0)

        # maintain the merged CSV order
        ranked.append((t, n, f, s, {}))




# 7) Stratified Top-10 from merged CSV
    top10 = pick_top_stratified(
        ranked,
        total=10,
        min_small=int(os.getenv("STAGE2_MIN_SMALL", "5")),
        min_large=int(os.getenv("STAGE2_MIN_LARGE", "5")),
        pe_min=int(os.getenv("STAGE2_MIN_PE", "5")),
    )
    tickers_top10 = [t for (t, _, _, _, _) in top10]

    log(f"[INFO] Using stratified Top-10 from stage2_merged.csv: {', '.join(tickers_top10)}")

    spy_ctx = get_spy_ctx()
    earn_days_map = fetch_next_earnings_days(tickers_top10)
    valuations_map = fetch_valuations_for_top(tickers_top10)

    # --- Fetch bulk news for all tickers ---
    try:
        news_map = get_news_sentiment_bulk(tickers_top10, days=7, limit=50)
        log(f"[INFO] Pulled news for {len(news_map)} tickers via AlphaVantage bulk API")
    except Exception as e:
        log(f"[WARN] Failed to fetch bulk news: {e}")
        news_map = {t: [] for t in tickers_top10}

    blocks = []
    debug_inputs = {}
    baseline_str = "; ".join([f"{k}={v}" for k, v in BASELINE_HINTS.items()])

    # cutoff for 1 week
    cutoff = datetime.utcnow() - timedelta(days=7)

    for (t, name, feats, _score, _meta) in top10:
        proxies = derive_proxies(feats, spy_ctx)
        fund_proxy = fund_proxies_from_feats(feats)
        cat = catalyst_severity_from_feats(feats)

        earn_days = earn_days_map.get(t)
        if earn_days is None:
            earn_sev = 0
        elif earn_days <= 0:
            earn_sev = 5
        elif earn_days <= 3:
            earn_sev = 4
        elif earn_days <= 7:
            earn_sev = 3
        elif earn_days <= 14:
            earn_sev = 2
        else:
            earn_sev = 0

        vals = (valuations_map.get(t) or {})
        pe_hint = vals.get("PE")
        fm = {
            "PE":        vals.get("PE"),
            "PS":        vals.get("PS"),
            "EV_EBITDA": vals.get("EV_EBITDA"),
            "EV_REV":    vals.get("EV_REV"),
            "PEG":       vals.get("PEG"),
            "FCF_YIELD": vals.get("FCF_YIELD"),
        }

        # --- Filter to last 7 days, take up to 2 news ---
        news_items = [
            n for n in news_map.get(t, [])
            if n.get("published_at") and datetime.strptime(n["published_at"], "%Y%m%dT%H%M%S") >= cutoff
        ]
        if news_items:
            news_lines = [f"- {n['title']} ({n['sentiment']}, {n['source']})" for n in news_items[:2]]
            news_text = "\n".join(news_lines)
        else:
            news_text = "N/A"

        block_text, debug_dict = build_prompt_block(
            t=t, name=name, feats=feats, proxies=proxies, fund_proxy=fund_proxy,
            cat=cat, earn_sev=earn_sev, fm=fm,
            baseline_hints=BASELINE_HINTS, baseline_str=baseline_str,
            pe_hint=pe_hint
        )
        block_text += f"\n\nNews:\n{news_text}"
        blocks.append(block_text)
        debug_inputs[t] = debug_dict

    blocks_text = "\n\n".join(blocks)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(
        today=now.strftime("%b %d"),
        blocks=blocks_text,
)




    # 8) GPT adjudication
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20_EXT, user_prompt, max_tokens=13000)
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")

    # 9) Parse selected tickers from GPT output and post debug
    RE_PICK_TICKER = re.compile(
        r"(?m)^\s*\*\*([A-Z]{1,10}(?:[.\-][A-Z0-9]{1,5})?)\s+[–—-]\s"
    )
    RE_FORECAST_TICK = re.compile(
        r"(?im)Forecast\s+image\s+URL:\s*https?://[^/]+/stocks/([A-Z0-9.\-]+)/forecast\b"
    )

    valid_set = set(debug_inputs.keys())
    a = RE_PICK_TICKER.findall(final_text)
    b = RE_FORECAST_TICK.findall(final_text)
    raw = list(dict.fromkeys(a + b))
    picked = [t for t in raw if t in valid_set]
    junk = [t for t in raw if t not in valid_set]
    if junk:
        log(f"[INFO] Ignored non-ticker matches: {junk}")

    force_debug = os.getenv("DEBUGGER_FORCE_POST", "1").lower() in {"1", "true", "yes"}
    if not picked and force_debug:
        picked = tickers_top10[:]
        log("[INFO] No valid tickers parsed; forcing debug post for Stage-2 Top-10.")
    elif picked:
        log(f"[INFO] Parsed tickers for debug: {', '.join(picked)}")
    else:
        log("[WARN] No tickers parsed and DEBUGGER_FORCE_POST=0; will skip debug post.")

    try:
        if picked:
            post_debug_inputs_to_discord(picked, debug_inputs, DISCORD_DEBUG_WEBHOOK_URL)
            log(f"[INFO] Posted debug embeds for: {', '.join(picked)}")
    except Exception as e:
        log(f"[WARN] Debug post failed: {e!r}")

    for t in picked:
        dbg = debug_inputs.get(t)
        if not dbg:
            continue
        is_probe = dbg.get("RISK_GUARDS", {}).get("PROBE_ELIGIBLE") == "Yes"
        plan_re = re.search(rf"\*\*{t}\s+–\s.*?^Plan:\s+(Buy|No trade).*?\(Anchor: \$([0-9.]+)\)", final_text, re.MULTILINE | re.DOTALL)
        certainty_re = re.search(rf"\*\*{t}\s+–\s.*?Certainty: (\d+)%", final_text, re.IGNORECASE)
        if plan_re and certainty_re:
            action = plan_re.group(1)
            certainty = int(certainty_re.group(1))
            if action == "No trade" and certainty >= 80 and is_probe:
                fva = float(plan_re.group(2))
                price = _to_float(dbg["CURRENT_PRICE"])
                ev = max(_to_float(dbg["PROXIES_BLOCK"]["EXPECTED_VOLATILITY_PCT"]), 4)
                anchor = max(min(fva, price * 1.02), price * 0.97)
                buy_lo = round(price * (1 - 0.005 * ev), 2)
                buy_hi = round(price * (1 + 0.002 * ev), 2)
                stop = round(max(price * (1 - 0.022 * ev),
                                 dbg["INDICATORS_BLOCK"]["EMA20"] * 0.995 if dbg["INDICATORS_BLOCK"]["EMA20"] < price else price), 2)
                target = round(price * (1 + 0.038 * ev), 2)
                replacement = f"Probe buy {buy_lo}–{buy_hi}; Stop {stop}; Target {target}; Max hold time: ≤ 6 months (Anchor: ${anchor:.2f}) (Parabolic risk)"
                final_text = re.sub(
                    rf"(?<=\*\*{t}\s+–\s).*?No trade.*?\(Anchor: \$[0-9.]+\)",
                    replacement,
                    final_text,
                    flags=re.DOTALL
                )
                log(f"[INFO] Overrode no-trade for {t} → PROBE BUY ⚡")

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
    embed = {
        "title": f"Daily Stock Pick — {datetime.now(TZ).strftime('%Y-%m-%d')}",
        "description": final_text,
    }
    payload = {"username": "Daily Stock Alert", "embeds": [embed]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=60).raise_for_status()
        log("[INFO] Posted alert to Discord ✅")
    except Exception as e:
        log(f"[ERROR] Discord webhook error: {repr(e)}")

if __name__ == "__main__":
    main()
