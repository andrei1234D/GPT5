# scripts/notify.py
import os, re, sys, time, requests, pytz
from datetime import datetime
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
SYSTEM_PROMPT_TOP20_EXT = SYSTEM_PROMPT_TOP20 + """
INPUT EXTRAS:
- Each candidate block may include:
  • BASELINE_HINTS: exact baselines for each category.
  • PROXIES: price/volume proxies with severities (−5..+5) and directions (+/−).
  • PROXIES_FUNDAMENTALS: {GROWTH_TECH, MARGIN_TREND_TECH, FCF_TREND_TECH, OP_EFF_TREND_TECH} as signed severities (−5..+5).
  • PROXIES_CATALYSTS: {TECH_BREAKOUT, TECH_BREAKDOWN, DIP_REVERSAL, EARNINGS_SOON} with signed severities.
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
  1) Compute a single $FVA for TODAY. Start from FVA_HINT and adjust modestly only if indicators clearly justify it.
  2) Tiny PE bias (only if PE_HINT present; always stay within the global clamp below; VALUATION_HISTORY is the signed proxy −5..+5):
     • If PE_HINT ≤ 10 and VALUATION_HISTORY ≥ +2 → tilt FVA up by +1%..+3%.
     • If PE_HINT ≥ 50 and VALUATION_HISTORY ≤ −2 → tilt FVA down by −1%..−3%.
  3) Global clamp: |FVA − PRICE| ≤ 20% unless very strong technical evidence. Treat as “very strong” only if
     TECH_BREAKOUT ≥ +4 AND RSI14 ≥ 75 AND vsSMA50 ≥ +20% AND Vol_vs_20d% ≥ +150%. Otherwise honor the 20% clamp.
  4) Let EV = EXPECTED_VOLATILITY_PCT, with EV := min(max(EV, 1), 6).
  5) Buy range = FVA × (1 − 0.8×EV/100) … FVA × (1 + 0.8×EV/100)
  6) Stop loss = FVA × (1 − 2.0×EV/100)
  7) Profit target = FVA × (1 + 3.0×EV/100)
  8) Sanity: if stop ≥ buy_low, push stop to min(buy_low×0.99, FVA×(1 − 2.2×EV/100));
             if target ≤ buy_high, push target to max(buy_high×1.05, FVA×(1 + 3.2×EV/100)).
  9) Rounding: round all $ to 2 decimals.
 10) Output exactly: "Buy X–Y; Stop Z; Target T; Max hold time: ≤ 1 year (Anchor: $FVA)".

GUARDS (avoid nonsensical plans):
  • If after sanity Target ≤ CURRENT_PRICE → do NOT invent higher targets. Prefer: "No trade — extended; wait for pullback. (Anchor: $FVA)".
  • If CURRENT_PRICE > buy_high but CURRENT_PRICE < Target → keep the standard plan text and append " (Wait for pullback into range.)".
  • If CURRENT_PRICE < buy_low by more than ~2×EV% → append " (Accumulation zone)".
- Keep plans terse; do not explain math in prose; only the single-line plan plus the required summary fields.
"""
force = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}

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

def _z(mu, sd, x):
    if x is None: return None
    try:
        sd = sd if (sd and sd > 1e-12) else 1.0
        z = (float(x) - float(mu)) / float(sd)
        if math.isnan(z) or math.isinf(z): return None
        return max(-6.0, min(6.0, z))
    except Exception:
        return None

def compute_val_z_stats(vals_map):
    # vals_map: {ticker: {"PE":..,"PS":..,"EV_REV":..,"EV_EBITDA":..,"PEG":..,"FCF_YIELD":..}}
    keys = ["PE","PEG","EV_EBITDA","EV_REV","PS","FCF_YIELD"]
    cols = {k: [] for k in keys}
    for v in vals_map.values():
        for k in keys:
            x = v.get(k)
            try:
                if x is not None:
                    fx = float(x)
                    if not math.isnan(fx) and not math.isinf(fx):
                        cols[k].append(fx)
            except Exception:
                pass
    stats = {}
    for k, arr in cols.items():
        if not arr:
            stats[k] = (0.0, 1.0)
        else:
            mu = float(np.median(arr))
            mad = float(np.median(np.abs(np.array(arr) - mu)))
            sd = mad * 1.4826 if mad > 1e-12 else (float(np.std(arr)) or 1.0)
            stats[k] = (mu, sd if sd > 1e-9 else 1.0)
    return stats

def trend_quality_from_feats(f):
    # 0..1 score: banded RSI, EMA alignment, non-manic participation, and low extension
    rsi   = safe(f.get("RSI14"), None)
    vsem50= safe(f.get("vsEMA50"), None)
    vsem200=safe(f.get("vsEMA200"), None)
    px    = safe(f.get("price"), None)
    e50   = safe(f.get("EMA50"), None)
    e200  = safe(f.get("EMA200"), None)
    vol20 = safe(f.get("vol_vs20"), None)

    # RSI band 45..70 good
    if rsi is None: rsi_band = 0.5
    elif rsi <= 35 or rsi >= 80: rsi_band = 0.1
    elif 45 <= rsi <= 70: rsi_band = 1.0
    else: rsi_band = 0.6

    # EMA alignment
    ema_align = 1.0 if (px and e50 and e200 and px>e50>e200) else (0.6 if (e50 and e200 and e50>e200) else 0.2)

    # Volume participation (not blow-off)
    if vol20 is None: vol_part = 0.6
    elif vol20 < 40: vol_part = 0.4
    elif 60 <= vol20 <= 180: vol_part = 1.0
    elif 180 < vol20 <= 260: vol_part = 0.7
    else: vol_part = 0.3

    # Extension penalty from vsEMA50
    if vsem50 is None: ext_pen = 0.0
    elif vsem50 <= 15: ext_pen = 0.0
    elif vsem50 <= 30: ext_pen = 0.15
    elif vsem50 <= 45: ext_pen = 0.30
    else: ext_pen = 0.45

    # Combine (subtract extension)
    score = max(0.0, min(1.0, 0.35*rsi_band + 0.35*ema_align + 0.30*vol_part - ext_pen))
    return {
        "score": round(score, 2),
        "rsi_band": round(rsi_band, 2),
        "ema_align": round(ema_align, 2),
        "vol_part": round(vol_part, 2),
        "ext_pen": round(ext_pen, 2),
    }



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

    # Optional: P/E refinement on a small pool
    if os.getenv("STAGE1_PE_RESCORE", "1").lower() in {"1", "true", "yes"}:
        pool_n = int(os.getenv("STAGE1_PE_POOL", "2000"))
        pool_tickers = [t for (t, _n, _f, _s, _m) in quick_scored[:pool_n]]
        try:
            from data_fetcher import fetch_pe_for_top
            pe_map = fetch_pe_for_top(pool_tickers)
        except Exception as e:
            log(f"[WARN] Stage-1 P/E refine skipped (fetch error): {e!r}")
            pe_map = {}
        pre_top200_pe = []
        for (t, n, f, s, meta) in pre_top200:
            pe = pe_map.get(t)
            if pe is not None and pe > 0:
                f["val_PE"] = pe
            pre_top200_pe.append((t, n, f, s, meta))
            rescored = []
            for (t, n, f, _s, _m) in pre_top200_pe:
                s2, parts2 = quick_score(f, mode=os.getenv("STAGE1_MODE", "loose"))
                rescored.append((t, n, f, s2, {"parts": parts2, "tags": _m.get("tags", [])}))
                # preserve original meta fields (tier, ADV) to keep later logs/filters coherent
                rescored.append((
                    t, n, f, s2,
                    {
                        "parts": parts2, "tags": _m.get("tags", []),
                        "tier": _m.get("tier"), "avg_dollar_vol_20d": _m.get("avg_dollar_vol_20d")
                    }
                ))
            rescored.sort(key=lambda x: x[3], reverse=True)
            pre_top200 = rescored[:int(os.getenv("STAGE1_KEEP", "200"))]
            log("[INFO] Stage-1 P/E refine applied.")

    # 6) Stage-2 — RobustRanker
    ranker = RobustRanker()
    tickers_pre = [t for (t, n, f, _score, _meta) in pre_top200]
    vals_pre = fetch_valuations_for_top(tickers_pre)

    for (t, n, f, _score, _meta) in pre_top200:
        v = (vals_pre.get(t) or {})
        f["val_PE"]        = v.get("PE")
        f["val_PS"]        = v.get("PS")
        f["val_EV_REV"]    = v.get("EV_REV")
        f["val_EV_EBITDA"] = v.get("EV_EBITDA")
        f["val_PEG"]       = v.get("PEG")
        f["val_FCF_YIELD"] = v.get("FCF_YIELD")

    ranker.fit_cross_section([f for (t, n, f, _score, _meta) in pre_top200])

    thorough_ranked = []
    for (t, n, f, _score, _meta) in pre_top200:
        if ranker.should_drop(f):
            continue
        score, parts = ranker.composite_score(f)
        thorough_ranked.append((t, n, f, score))
    thorough_ranked.sort(key=lambda x: x[3], reverse=True)
    top200 = thorough_ranked[:200]
    if not top200:
        return fail("No candidates after Stage-2 robust scorer")
    log(f"[INFO] Stage-2 leader: {top200[0][0]}")

    # 7) Stratified Top-10
    top10 = pick_top_stratified(
        top200,
        total=10,
        min_small=int(os.getenv("STAGE2_MIN_SMALL", "5")),
        min_large=int(os.getenv("STAGE2_MIN_LARGE", "5")),
        pe_min=int(os.getenv("STAGE2_MIN_PE", "5")),
    )
    tickers_top10 = [t for (t, _, _, _) in top10]

    spy_ctx = get_spy_ctx()
    earn_days_map = fetch_next_earnings_days(tickers_top10)
    valuations_map = fetch_valuations_for_top(tickers_top10)

    blocks = []
    debug_inputs = {}
    baseline_str = "; ".join([f"{k}={v}" for k, v in BASELINE_HINTS.items()])

    for (t, name, feats, _score) in top10:
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

        block_text, debug_dict = build_prompt_block(
            t=t, name=name, feats=feats, proxies=proxies, fund_proxy=fund_proxy,
            cat=cat, earn_sev=earn_sev, fm=fm,
            baseline_hints=BASELINE_HINTS, baseline_str=baseline_str,
            pe_hint=pe_hint
        )
        blocks.append(block_text)
        debug_inputs[t] = debug_dict

    blocks_text = "\n\n".join(blocks)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(
        today=now.strftime("%b %d"),
        blocks=blocks_text,
    )

    # Log the exact inputs we’re about to send to GPT
    if os.getenv("LOG_GPT_INPUT", "1").lower() in {"1", "true", "yes"}:
        echo = os.getenv("LOG_GPT_INPUT_STDOUT", "0").lower() in {"1", "true", "yes"}
        dump_path = dump_blocks_pre_gpt(blocks, user_prompt, TZ, echo_stdout=echo)
        log(f"[INFO] Dumped pre-GPT blocks to {dump_path}")

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
