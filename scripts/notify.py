# scripts/notify.py
import os
import sys
import time
import json
import csv
import re
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import requests

from gpt_client import call_gpt5
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE
from time_utils import seconds_until_target_hour

TZ = pytz.timezone("Europe/Bucharest")


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


# ---------- helpers ----------
def _load_llm_records(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = str(rec.get("Ticker") or "")
            if t:
                out[t] = rec
    return out


def load_news_summary(path: str = "data/news_summary_top10.txt") -> dict[str, str]:
    """
    Parses file shaped like:
      ### TICKER
      - bullet
      - bullet
    Returns {ticker: "bullet1 | bullet2"} (single-line, compact).
    """
    p = Path(path)
    if not p.exists():
        log(f"[WARN] News summary file not found: {path}")
        return {}

    txt = p.read_text(encoding="utf-8").strip()
    blocks: dict[str, str] = {}
    ticker = None
    lines_accum = []
    for line in txt.splitlines():
        if line.startswith("### "):
            if ticker is not None:
                bullets = [l for l in lines_accum if l.strip() and not l.startswith("###")]
                compact = " | ".join(x.strip("- ").strip() for x in bullets if x.strip())
                blocks[ticker] = compact or "N/A"
            ticker = line.replace("### ", "").strip()
            lines_accum = []
        else:
            lines_accum.append(line)

    if ticker is not None:
        bullets = [l for l in lines_accum if l.strip() and not l.startswith("###")]
        compact = " | ".join(x.strip("- ").strip() for x in bullets if x.strip())
        blocks[ticker] = compact or "N/A"

    return blocks


def _post_discord(webhook_url: str, title: str, description: str) -> None:
    """
    Posts an embed to Discord. Truncates description to stay within Discord limits.
    """
    max_desc = 3800
    if description and len(description) > max_desc:
        description = description[: max_desc - 50].rstrip() + "\n\n[truncated]"

    embed = {"title": title, "description": description}
    payload = {"username": "Daily Stock Alert", "embeds": [embed]}
    requests.post(webhook_url, json=payload, timeout=60).raise_for_status()


def _score_pct_per_point() -> float:
    try:
        data = json.loads(Path("knobs/score_calibration.json").read_text(encoding="utf-8"))
        mu_perfect = float(data.get("mu_perfect", 1.672281))
        return (mu_perfect * 100.0) / 1000.0
    except Exception:
        return 0.1672281


def _load_score_calibration() -> dict | None:
    try:
        data = json.loads(Path("knobs/score_calibration.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    bins = int(data.get("bins", 0))
    mu_adj = data.get("mu_adj")
    if not bins or not isinstance(mu_adj, list) or len(mu_adj) != bins:
        return None
    return {
        "bins": bins,
        "pred_min": float(data.get("pred_min", 0.0)),
        "pred_max": float(data.get("pred_max", 1.0)),
        "mu_adj": mu_adj,
    }


def _expected_return_from_pred(pred: float, calib: dict | None) -> float | None:
    if calib is None or pred is None or not math.isfinite(pred):
        return None
    pred_min = float(calib["pred_min"])
    pred_max = float(calib["pred_max"])
    bins = int(calib["bins"])
    if not math.isfinite(pred_min) or not math.isfinite(pred_max) or pred_max <= pred_min:
        return None
    idx = int((pred - pred_min) / (pred_max - pred_min) * bins)
    if idx < 0:
        idx = 0
    elif idx >= bins:
        idx = bins - 1
    try:
        mu = float(calib["mu_adj"][idx])
        if not math.isfinite(mu):
            return None
        return mu * 100.0
    except Exception:
        return None


def _extract_ticker(text: str) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Expect "TICKER – Full Name" or "TICKER - Full Name"
        if " - " in s or " – " in s:
            parts = re.split(r"\s[–-]\s", s, maxsplit=1)
            if parts:
                t = parts[0].strip()
                if re.fullmatch(r"[A-Z0-9.\-]+", t):
                    return t
        # Fallback: line starts with ticker
        m = re.match(r"^([A-Z0-9.\-]{1,15})\b", s)
        if m:
            return m.group(1)
    return None


def _extract_score(text: str) -> int | None:
    if not text:
        return None
    for line in text.splitlines():
        if line.lower().startswith("score"):
            m = re.search(r"(\d{1,4})", line)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 <= val <= 1000:
                        return val
                except Exception:
                    pass
    return None


def _replace_score_line(text: str, score: int) -> str:
    if not text:
        return text
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.lower().startswith("score"):
            lines[i] = f"Score (0-1000): {score}"
            replaced = True
            break
    if not replaced:
        lines.append(f"Score (0-1000): {score}")
    return "\n".join(lines)


def _advice_from_score(score: int) -> str:
    if score > 800:
        return "Ultra Strong Buy"
    if score > 700:
        return "Strong Buy"
    if score > 600:
        return "Buy"
    return "Ignore"


def _replace_advice_line(text: str, advice: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.lower().startswith("advice"):
            lines[i] = f"ADVICE: {advice}"
            replaced = True
            break
    if not replaced:
        lines.append(f"ADVICE: {advice}")
    return "\n".join(lines)


def _append_score_legend(text: str) -> str:
    score = _extract_score(text)
    pct_per_point = _score_pct_per_point()
    expected_line = ""
    # Prefer calibrated expected return from model prediction (top10_ml.csv)
    expected = None
    gpt_score = None
    try:
        ticker = _extract_ticker(text)
        if ticker:
            top_df = pd.read_csv("data/top10_ml.csv")
            top_df["ticker"] = top_df["ticker"].astype(str).str.strip()
            row = top_df[top_df["ticker"] == ticker].head(1)
            if not row.empty:
                if "pred_score" in row.columns:
                    pred = float(row.iloc[0]["pred_score"])
                    expected = _expected_return_from_pred(
                        pred, _load_score_calibration()
                    )
                if "gpt_score" in row.columns:
                    gpt_score = int(row.iloc[0]["gpt_score"])
    except Exception:
        expected = None
        gpt_score = None
    if gpt_score is not None:
        text = _replace_score_line(text, gpt_score)
        text = _replace_advice_line(text, _advice_from_score(gpt_score))
        score = gpt_score
    if expected is None and score is not None:
        expected = score * pct_per_point
    if expected is not None and math.isfinite(expected):
        expected_line = f"Expected return (approx): {expected:.1f}%\n"
    legend = (
        "\n\n---\n"
        "**Legend**\n"
        ">800 Ultra Strong Buy\n"
        ">700 Strong Buy\n"
        ">600 Buy\n"
        "<599 Ignore\n"
        f"{expected_line}"
        "MAE~25%\n"
        "\n"
    )
    return (text or "").rstrip() + legend

def main():
    now = datetime.now(TZ)
    log(f"[INFO] Start {now.isoformat()} Europe/Bucharest. FORCE_RUN={force}")

    top_path = "data/top10_ml.csv"
    if not os.path.exists(top_path):
        return fail(f"{top_path} not found")

    top_df = pd.read_csv(top_path)
    if top_df.empty or "ticker" not in top_df.columns:
        return fail("top10_ml.csv is empty or missing 'ticker'")

    # Build ticker -> company name map if present
    name_map: dict[str, str] = {}
    if "company" in top_df.columns:
        for _, row in top_df.iterrows():
            t = str(row.get("ticker") or "")
            if not t:
                continue
            n = row.get("company")
            if isinstance(n, str):
                name_map[t] = n

    # Preserve rank order from file if available
    if "rank" in top_df.columns:
        top_df = top_df.sort_values("rank")
    tickers_to_gpt = top_df["ticker"].astype(str).str.strip().tolist()
    if not tickers_to_gpt:
        return fail("top10_ml.csv contains no tickers.")

    log(f"[INFO] ML Top-10 tickers (ordered): {', '.join(tickers_to_gpt)}")

    # Load pipeline news summary (optional hint; GPT will still browse for up-to-date data)
    news_map = load_news_summary("data/news_summary_top10.txt")

    # Build MINIMAL CSV for GPT (ranked top10)
    header = ["Rank", "TickerName", "Ticker", "PipelineNews"]

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = logs_dir / "blocks_to_gpt_latest.txt"

    with blocks_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, t in enumerate(tickers_to_gpt, start=1):
            name = name_map.get(t, "")
            ticker_name = f"{t} - {name}" if name else t
            row = [
                idx,
                ticker_name,
                t,
                news_map.get(t, "N/A"),
            ]
            writer.writerow(row)

    log(f"[INFO] Wrote ranked GPT payload to {blocks_path}")

    # Build user prompt using the SAME CSV
    csv_content = blocks_path.read_text(encoding="utf-8").strip()
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(
        today=now.strftime("%b %d"),
        blocks=csv_content,
    )

    # Save prompts for reproducibility
    (logs_dir / "gpt_user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    (logs_dir / "gpt_system_prompt.txt").write_text(SYSTEM_PROMPT_TOP20_EXT, encoding="utf-8")

    # GPT call (generate final alert text)
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20_EXT, user_prompt, max_tokens=2500)
    except Exception as e:
        return fail(f"GPT failed: {repr(e)}")

    # Append legend for Discord readability
    final_text = _append_score_legend(final_text)

    # Save output
    Path("daily_pick.txt").write_text(final_text, encoding="utf-8")
    log("[INFO] Draft saved to daily_pick.txt")

    # Optional wait until 08:00 Europe/Bucharest
    if not force:
        wait_s = max(0, seconds_until_target_hour(8, 0, TZ))
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # Send to Discord (main + optional debug)
    title = f"Daily Stock Pick — {datetime.now(TZ).strftime('%Y-%m-%d')}"
    try:
        _post_discord(DISCORD_WEBHOOK_URL, title=title, description=final_text)
        log("[INFO] Posted alert to Discord ✅")
    except Exception as e:
        log(f"[ERROR] Discord webhook error: {repr(e)}")

    try:
        if DISCORD_DEBUG_WEBHOOK_URL and DISCORD_DEBUG_WEBHOOK_URL != DISCORD_WEBHOOK_URL:
            _post_discord(DISCORD_DEBUG_WEBHOOK_URL, title=title + " [debug]", description=final_text)
            log("[INFO] Posted alert to Discord (debug) ✅")
    except Exception as e:
        log(f"[WARN] Discord debug webhook error: {repr(e)}")


if __name__ == "__main__":
    main()
