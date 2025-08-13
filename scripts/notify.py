# scripts/notify_main.py
import os
import re
import sys
import time
from datetime import datetime

import requests
import pytz

from universe import load_universe
from features import build_features
from filters import is_garbage, daily_index_filter
from scoring import base_importance_score
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE

from proxies import get_spy_ctx, derive_proxies, fund_proxies_from_feats, catalyst_severity_from_feats
from fundamentals import fetch_next_earnings_days, fetch_funda_valuation_for_top
from prompt_blocks import BASELINE_HINTS, build_prompt_block
from gpt_client import call_gpt5, apply_personal_bonuses_to_text
from debugger import post_debug_inputs_to_discord
from time_utils import seconds_until_target_hour

TZ = pytz.timezone("Europe/Bucharest")

def log(m): print(m, flush=True)
def need_env(name):
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Missing env: {name}", flush=True); sys.exit(1)
    return v

OPENAI_API_KEY = need_env("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = need_env("DISCORD_WEBHOOK_URL")
force = os.getenv("FORCE_RUN", "").lower() in {"1","true","yes"}

# ——— Strengthen system prompt (same logic you had) ———
SYSTEM_PROMPT_TOP20_EXT = SYSTEM_PROMPT_TOP20 + """
INPUT EXTRAS:
- Each candidate block may include:
  • DATA_AVAILABILITY: which of {FUNDAMENTALS, VALUATION, RISKS, CATALYSTS} are MISSING or PARTIAL.
  • BASELINE_HINTS: exact baselines for each category.
  • VALUATION FIELDS (if present): PE, PE_SECTOR, EV_EBITDA, PS, FCF_YIELD, PEG.
  • PROXIES: simple, price/volume-only signals with 1–5 severity and a direction (+/-).
  • PROXIES_FUNDAMENTALS: {GROWTH_TECH, MARGIN_TREND_TECH, FCF_TREND_TECH, OP_EFF_TREND_TECH} as signed severities (−5..+5).
  • PROXIES_CATALYSTS: {TECH_BREAKOUT, TECH_BREAKDOWN, DIP_REVERSAL, EARNINGS_SOON} with signed severities like +1..+5 or -1..-5.
  • CATALYST_TIMING_HINTS: TECH_BREAKOUT=Today/None.
  • EXPECTED_VOLATILITY_PCT: derived from ATR%, use as 'Expected volatility' in the Certainty rule.
  • FVA_HINT: a technical fair-value anchor seed, derived only from supplied indicators.

MANDATORY HANDLING:
- If DATA_AVAILABILITY says a category is MISSING ⇒ set that category exactly to its BASELINE_HINTS value (do NOT set 0).
- If PARTIAL ⇒ use only the provided FIELDS and PROXIES for that category; keep unaddressed sub-factors at baseline.
- Never assume unknown = bad; unknown = baseline.
- PROXIES map 1–5 severity to the factor ranges you already have.
- CATALYST PROXY MAPPING (apply to 'Near-Term Catalysts' base 100):
    • TECH_BREAKOUT (+1..+5)  ⇒ +10, +20, +30, +45, +60
    • DIP_REVERSAL (+1..+5)   ⇒ +8, +12, +18, +24, +30
    • TECH_BREAKDOWN (-1..-5) ⇒ −10, −20, −30, −45, −60
    • EARNINGS_SOON (+1..+5)  ⇒ +5, +8, +10, +12, +15
  Timing multiplier applies for TECH_BREAKOUT timing: Today×1.50.
- FUNDAMENTALS PROXY MAPPING (start at 125 baseline; clamp total from these tech proxies to ±35 overall).
"""

def fail(msg: str):
    log(f"[ERROR] {msg}")
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username":"Daily Stock Alert","content":f"⚠️ {msg}"}, timeout=60)
    except Exception:
        pass

def _fallback_score(feats: dict) -> float:
    # Robust, cheap fallback if scoring.base_importance_score is unavailable/throws
    try:
        vs200 = feats.get("vsSMA200") or 0.0
        vs50  = feats.get("vsSMA50") or 0.0
        r60   = feats.get("r60") or 0.0
        atr   = feats.get("ATRpct") or 4.0
        dd    = feats.get("drawdown_pct") or 0.0
        return (0.5*vs200 + 0.3*vs50 + 0.3*r60) - 0.4*max(0, atr-6) + 0.15*min(0, dd+15)
    except Exception:
        return 0.0

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
    kept2 = [(t,n,f) for (t,n,f) in kept if daily_index_filter(f, today_context)]
    log(f"[INFO] After daily index filter: {len(kept2)} remain")
    if not kept2:
        return fail("All filtered by daily context")

    # 5) Score & top-200
# 5) Rank to 200 using a simple composite score you already import
    ranked = []
    for (t, n, f) in kept2:
            try:
             score = base_importance_score(f)  # from scoring.py
            except Exception:
                score = 0.0
    ranked.append((t, n, f, score))

    ranked.sort(key=lambda x: x[3], reverse=True)
    top200 = ranked[:200]
    if not top200:
        return fail("No candidates after ranking")
    log(f"[INFO] Reduced to top 200. Example leader: {top200[0][0]}")
    # 6) Prepare TOP 10 blocks + debug payloads
    top20 = top200[:10]
    tickers_top20 = [t for t, _, _, _ in top20]

    spy_ctx = get_spy_ctx()
    earn_days_map = fetch_next_earnings_days(tickers_top20)
    funda_map = fetch_funda_valuation_for_top(tickers_top20)

    blocks = []
    debug_inputs = {}
    baseline_str = "; ".join([f"{k}={v}" for k, v in BASELINE_HINTS.items()])

    for t, name, feats, _ in top20:
        # proxies & catalysts
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

        fm = funda_map.get(t, {}) or {}

        # exact prompt block + full debug dict
        block_text, debug_dict = build_prompt_block(
            t=t, name=name, feats=feats, proxies=proxies, fund_proxy=fund_proxy,
            cat=cat, earn_sev=earn_sev, fm=fm, baseline_hints=BASELINE_HINTS, baseline_str=baseline_str
        )

        blocks.append(block_text)
        debug_inputs[t] = debug_dict

    blocks_text = "\n\n".join(blocks)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(today=now.strftime("%b %d"), blocks=blocks_text)

    # 7) GPT-5 adjudication
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20_EXT, user_prompt, max_tokens=13000)
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")

    # 8) Personal bonuses
    final_text = apply_personal_bonuses_to_text(final_text)

    # 8.1) Pull selected tickers from GPT output and post debug inputs
    RE_PICK_TICKER = re.compile(r"(?im)^\s*(?:\d+\)\s*)?(?:\*\*)?([A-Z][A-Z0-9.\-]{1,10})\s+[–-]")
    RE_FORECAST_TICK = re.compile(r"(?im)Forecast\s+image\s+URL:\s*https?://[^/]+/stocks/([A-Z0-9.\-]+)/forecast\b")

    picked = RE_PICK_TICKER.findall(final_text)
    picked += RE_FORECAST_TICK.findall(final_text)

    seen = set(); picked_unique = []
    for x in picked:
        if x not in seen:
            seen.add(x); picked_unique.append(x)

    if not picked_unique:
        if debug_inputs:
            picked_unique = [next(iter(debug_inputs.keys()))]

    if picked_unique:
        post_debug_inputs_to_discord(picked_unique, debug_inputs, DISCORD_WEBHOOK_URL)
    else:
        log("[WARN] Could not parse any selected ticker from GPT output; skipping debug post.")

    # 9) Save & (optionally) wait until 08:00
    with open("daily_pick.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    log("[INFO] Draft saved to daily_pick.txt")

    if not force:
        wait_s = seconds_until_target_hour(8, 0, TZ)
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # 10) Send to Discord
    embed = {
        "title": f"Daily Stock Pick — {datetime.now(TZ).strftime('%Y-%m-%d')}",
        "description": final_text
    }
    payload = {"username": "Daily Stock Alert", "embeds": [embed]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=60).raise_for_status()
        log("[INFO] Posted alert to Discord ✅")
    except Exception as e:
        log(f"[ERROR] Discord webhook error: {repr(e)}")

if __name__ == "__main__":
    main()
