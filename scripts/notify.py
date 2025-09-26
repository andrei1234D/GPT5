# scripts/notify.py
import os, sys, time, requests, pytz
import pandas as pd
from datetime import datetime, timedelta, UTC
from pathlib import Path

from features import build_features
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

from trash_ranker import pick_top_stratified

# --- Timezone
TZ = pytz.timezone("Europe/Bucharest")

# --- Helpers
def log(m): 
    print(m, flush=True)

def need_env(name):
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


def load_news_summary(path="data/news_summary_top10.txt"):
    """Load the GPT-3.5 condensed news summaries for the top-10 tickers."""
    if not os.path.exists(path):
        log(f"[WARN] News summary file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    log(f"[INFO] Loaded news summaries from {path}")
    # Break into blocks by ### ticker
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

    # === 6) Load Stage-2 merged results ===
    path = "data/stage2_merged.csv"
    if not os.path.exists(path):
        return fail(f"{path} not found")

    df = pd.read_csv(path)
    df = df.where(pd.notnull(df), None)  # keep None for NaN

    if df.empty:
        return fail("stage2_merged.csv is empty")

    if "merged_final_score" in df.columns:
        df = df.sort_values("merged_final_score", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("merged_score", ascending=False).reset_index(drop=True)

    # Build ranked list
    universe = [(t, "") for t in df["ticker"]]
    feats_map = build_features(universe, batch_size=int(os.getenv("YF_CHUNK_SIZE", "80")))
    ranked = []
    for _, row in df.iterrows():
        t = row["ticker"]
        n = (row.get("company") or t)
        f = feats_map.get(t, {}).get("features", {}) or {}
        f.update(row.to_dict())
        s = row.get("merged_score", 0.0)
        ranked.append((t, n, f, s, {}))

    # === 7) Pick stratified Top-10 ===
    top10 = pick_top_stratified(
        ranked,
        total=10,
        min_small=int(os.getenv("STAGE2_MIN_SMALL", "5")),
        min_large=int(os.getenv("STAGE2_MIN_LARGE", "5")),
        pe_min=int(os.getenv("STAGE2_MIN_PE", "5")),
    )
    tickers_top10 = [t for (t, _, _, _, _) in top10]
    log(f"[INFO] Using stratified Top-10 from stage2_merged.csv: {', '.join(tickers_top10)}")

    # === 8) Fetch fundamentals ===
    spy_ctx = get_spy_ctx()
    earn_days_map = fetch_next_earnings_days(tickers_top10)
    valuations_map = fetch_valuations_for_top(tickers_top10)

    # === 9) Load precomputed news summaries ===
    news_blocks = load_news_summary()

    blocks = []
    debug_inputs = {}
    baseline_str = "; ".join([f"{k}={v}" for k, v in BASELINE_HINTS.items()])

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

        vals = valuations_map.get(t) or {}
        pe_hint = vals.get("PE")
        fm = {
            "PEG": feats.get("val_PEG"),
            "YoY_Growth": feats.get("val_YoY"),
            "PE": feats.get("val_PE"),
            "PS": vals.get("PS"),
            "EV_EBITDA": vals.get("EV_EBITDA"),
            "EV_REV": vals.get("EV_REV"),
            "FCF_YIELD": vals.get("FCF_YIELD"),
        }

        block_text, debug_dict = build_prompt_block(
            t=t, name=name, feats=feats, proxies=proxies, fund_proxy=fund_proxy,
            cat=cat, earn_sev=earn_sev, fm=fm,
            baseline_hints=BASELINE_HINTS, baseline_str=baseline_str,
            pe_hint=pe_hint,
        )

        # Attach GPT-3.5 condensed news if available
        if t in news_blocks:
            block_text += f"\n\n### {t} News\n{news_blocks[t]}"
        else:
            block_text += f"\n\n### {t} News\n- N/A\nImpact: 0"

        blocks.append(block_text)
        debug_inputs[t] = debug_dict

    blocks_text = "\n\n".join(blocks)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(
        today=now.strftime("%b %d"),
        blocks=blocks_text,
    )

    # === 9b) Save blocks for debugging ===
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = logs_dir / "blocks_to_gpt_latest.txt"
    try:
        with blocks_path.open("w", encoding="utf-8") as f:
            f.write(blocks_text)
        log(f"[INFO] Wrote prompt blocks to {blocks_path}")
    except Exception as e:
        log(f"[WARN] Failed to save prompt blocks: {e}")

    # === 10) GPT adjudication ===
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20_EXT, user_prompt, max_tokens=13000)
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")

    # === 11) Save output ===
    with open("daily_pick.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    log("[INFO] Draft saved to daily_pick.txt")

    if not force:
        wait_s = max(0, seconds_until_target_hour(8, 0, TZ))
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # === 12) Send to Discord ===
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
