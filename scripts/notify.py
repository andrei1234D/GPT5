import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import requests

from gpt_client import call_gpt5
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE
from time_utils import seconds_until_target_hour

from llm_data_builder import build_llm_today_data
from brain_ranker import rank_with_brain
from gpt_block_builder import build_gpt_blocks

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
SYSTEM_PROMPT_TOP20_EXT = SYSTEM_PROMPT_TOP20 + """
... (your strict PLAN rules here, unchanged) ...
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
    sys.exit(1)


def load_news_summary(path: str = "data/news_summary_top10.txt") -> dict:
    """Load the GPT-3.5 condensed news summaries for the top-10 tickers."""
    if not os.path.exists(path):
        log(f"[WARN] News summary file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    log(f"[INFO] Loaded news summaries from {path}")
    blocks = {}
    current_ticker = None
    current_lines = []
    for line in content.splitlines():
        if line.startswith("### "):
            if current_ticker and current_lines:
                blocks[current_ticker] = "\n".join(current_lines).strip()
            current_ticker = line.replace("### ", "").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_ticker and current_lines:
        blocks[current_ticker] = "\n".join(current_lines).strip()
    return blocks


def main():
    now = datetime.now(TZ)
    log(f"[INFO] Start {now.isoformat()} Europe/Bucharest. FORCE_RUN={force}")

    # === 6) Load Stage-2 merged results (needed just to ensure it's there) ===
    path = "data/stage2_merged.csv"
    if not os.path.exists(path):
        return fail(f"{path} not found")

    df = pd.read_csv(path)
    if df.empty:
        return fail("stage2_merged.csv is empty")

    # === 7) Build LLM_today_data for Brain (top-30 tickers) ===
    try:
        tickers_llm = build_llm_today_data(
            stage2_path=path,
            out_path="data/LLM_today_data.jsonl",
            top_n=30,
        )
    except Exception as e:
        return fail(f"LLM_today_data build failed: {repr(e)}")

    log(f"[INFO] Built LLM_today_data.jsonl for {len(tickers_llm)} tickers.")

    # === 8) Run Brain to get top-10 tickers and Brain scores ===
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

    # === 9) Load precomputed news summaries (still optional) ===
    news_blocks = load_news_summary()

    # === 10) Build light GPT-5 blocks (BrainScore + key indices) ===
    blocks_text = build_gpt_blocks(
        top_tickers=tickers_top10,
        brain_scores=brain_scores,
        llm_data_path="data/LLM_today_data.jsonl",
        stage2_path=path,
        news_blocks=news_blocks,
    )

    now = datetime.now(TZ)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(
        today=now.strftime("%b %d"),
        blocks=blocks_text,
    )

    # === 10b) Save blocks for debugging ===
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = logs_dir / "blocks_to_gpt_latest.txt"
    try:
        with blocks_path.open("w", encoding="utf-8") as f:
            f.write(blocks_text)
        log(f"[INFO] Wrote prompt blocks to {blocks_path}")
    except Exception as e:
        log(f"[WARN] Failed to save prompt blocks: {e}")

    # === 11) GPT adjudication (LIGHTER) ===
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20_EXT, user_prompt, max_tokens=4500)
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")

    # === 12) Save output ===
    with open("daily_pick.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    log("[INFO] Draft saved to daily_pick.txt")

    if not force:
        wait_s = max(0, seconds_until_target_hour(8, 0, TZ))
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # === 13) Send to Discord ===
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
