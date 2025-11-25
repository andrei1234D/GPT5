# scripts/notify.py
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import pytz
import requests

from gpt_client import call_gpt5
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE
from time_utils import seconds_until_target_hour

from llm_data_builder import build_llm_today_data
from brain_ranker import rank_with_brain

# --- Timezone
TZ = pytz.timezone("Europe/Bucharest")

# --- Helpers
def log(m: str) -> None:
    print(m, flush=True)

def need_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Missing env: {name}", flush=True)
        sys.exit(1)
    return v

OPENAI_API_KEY = need_env("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = need_env("DISCORD_WEBHOOK_URL")
DISCORD_DEBUG_WEBHOOK_URL = os.getenv("DISCORD_DEBUG_WEBHOOK_URL") or DISCORD_WEBHOOK_URL

ALLOW_BLOWOFF_PROBE = os.getenv("ALLOW_BLOWOFF_PROBE", "0") == "1"
# leave SYSTEM prompt unchanged; we feed a minimal CSV as the user blocks
SYSTEM_PROMPT_TOP20_EXT = SYSTEM_PROMPT_TOP20

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
    sys.exit(1)

# ---------- LLM jsonl loader (for minimal 10 indices) ----------
def _load_llm_records(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    recs: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                t = str(rec.get("Ticker"))
                if t:
                    recs[t] = rec
            except Exception:
                continue
    return recs

def _num(v):
    try:
        return float(v)
    except Exception:
        return None

def _fmt(v):
    # write numeric as plain (no trailing zeros explosion); keep None as empty
    if v is None:
        return ""
    try:
        fv = float(v)
        # compact formatting (up to 6 decimals if needed)
        s = f"{fv:.6f}".rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(v)

def main():
    now = datetime.now(TZ)
    log(f"[INFO] Start {now.isoformat()} Europe/Bucharest. FORCE_RUN={force}")

    # === Ensure stage2 is present ===
    stage2_path = "data/stage2_merged.csv"
    if not os.path.exists(stage2_path):
        return fail(f"{stage2_path} not found")

    df = pd.read_csv(stage2_path)
    if df.empty:
        return fail("stage2_merged.csv is empty")

    # === Build LLM_today_data for Brain (top-30 tickers) ===
    try:
        tickers_llm = build_llm_today_data(
            stage2_path=stage2_path,
            out_path="data/LLM_today_data.jsonl",
            top_n=30,
        )
    except Exception as e:
        return fail(f"LLM_today_data build failed: {repr(e)}")

    log(f"[INFO] Built LLM_today_data.jsonl for {len(tickers_llm)} tickers.")

    # === Run Brain: get top-10 tickers + scores ===
    try:
        tickers_top10, brain_scores = rank_with_brain(
            llm_data_path="data/LLM_today_data.jsonl",
            top_k=10,
        )
    except Exception as e:
        return fail(f"Brain ranking failed: {repr(e)}")

    if not tickers_top10:
        return fail("Brain returned no top tickers.")

    log(f"[INFO] Brain Top-10 tickers: {', '.join(tickers_top10)}")

    # === Load LLM records to extract the EXACT 10 indices we want ===
    llm_map = _load_llm_records("data/LLM_today_data.jsonl")

    # Columns: exactly 10 indices + BrainScore (ticker is first col)
    header = [
        "Ticker",
        "BrainScore",
        "current_price",
        "RSI14",
        "MACD_hist",
        "Momentum",
        "ATRpct",
        "volatility_30",
        "pos_30d",
        "EMA50",
        "EMA200",
        "MarketTrend",
    ]

    rows = []
    for t in tickers_top10:
        rec = llm_map.get(t, {})
        row = [
            t,
            _fmt(brain_scores.get(t)),
            _fmt(_num(rec.get("current_price"))),
            _fmt(_num(rec.get("RSI14"))),
            _fmt(_num(rec.get("MACD_hist"))),
            _fmt(_num(rec.get("Momentum"))),
            _fmt(_num(rec.get("ATR%"))),            # stored as "ATR%" in jsonl
            _fmt(_num(rec.get("volatility_30"))),
            _fmt(_num(rec.get("pos_30d"))),
            _fmt(_num(rec.get("EMA50"))),
            _fmt(_num(rec.get("EMA200"))),
            str(rec.get("MarketTrend", "")),
        ]
        rows.append(row)

    # === Save minimal payload for traceability (ONLY 10 indices + BrainScore) ===
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = logs_dir / "blocks_to_gpt_latest.txt"

    # Write as CSV (comma-separated, header + 10 rows)
    with blocks_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            # escape commas only in MarketTrend if needed (unlikely)
            safe_row = [r.replace(",", " ") if isinstance(r, str) else r for r in row]
            f.write(",".join(safe_row) + "\n")

    log(f"[INFO] Wrote minimal GPT payload to {blocks_path}")

    # === Feed GPT the SAME minimal CSV as blocks ===
    # Put the CSV under the "CANDIDATES:" section
    csv_content = blocks_path.read_text(encoding="utf-8").strip()
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(
        today=now.strftime("%b %d"),
        blocks=csv_content,
    )

    # Also save the exact user + system prompts for full reproducibility
    with (logs_dir / "gpt_user_prompt.txt").open("w", encoding="utf-8") as f:
        f.write(user_prompt)
    with (logs_dir / "gpt_system_prompt.txt").open("w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT_TOP20_EXT)

    # === GPT call ===
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20_EXT, user_prompt, max_tokens=4500)
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")

    # === Save output ===
    with open("daily_pick.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    log("[INFO] Draft saved to daily_pick.txt")

    if not force:
        wait_s = max(0, seconds_until_target_hour(8, 0, TZ))
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # === Send to Discord ===
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
