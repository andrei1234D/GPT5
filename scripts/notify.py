# scripts/notify.py
import os
import sys
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import requests

from gpt_client import call_gpt5
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE
from time_utils import seconds_until_target_hour
from brain_ranker import rank_with_brain

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


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


def _fmt(v):
    if v is None:
        return ""
    try:
        fv = float(v)
        s = f"{fv:.6f}".rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(v)


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


def main():
    now = datetime.now(TZ)
    log(f"[INFO] Start {now.isoformat()} Europe/Bucharest. FORCE_RUN={force}")

    # 1) Ensure stage2 exists (sanity check) + build ticker -> name map
    stage2_path = "data/stage2_merged.csv"
    if not os.path.exists(stage2_path):
        return fail(f"{stage2_path} not found")
    df = pd.read_csv(stage2_path)
    if df.empty:
        return fail("stage2_merged.csv is empty")

    # Build ticker -> company name map
    name_map: dict[str, str] = {}
    if "ticker" in df.columns:
        for _, row in df.iterrows():
            t = str(row.get("ticker") or "")
            if not t:
                continue
            n = row.get("name")
            if isinstance(n, str):
                name_map[t] = n

    # 2) Ensure feature JSONL exists (built in prepare-data job)
    llm_today_path = "data/LLM_today_data.jsonl"
    if not os.path.exists(llm_today_path):
        return fail(f"{llm_today_path} not found. It should be built in the prepare-data job.")
    log(f"[INFO] Found {llm_today_path}")

    # 3) Brain rank — top 10 + pred_scores (ScoreBotSlim)
    try:
        tickers_top10, pred_scores = rank_with_brain(
            stage2_path=stage2_path,
            llm_data_path=llm_today_path,
            top_k=10,
        )
    except Exception as e:
        return fail(f"Brain ranking failed: {repr(e)}")
    if not tickers_top10:
        return fail("Brain returned no top tickers.")
    log(f"[INFO] Brain Top-10 tickers: {', '.join(tickers_top10)}")

    # 4) Load feature records + News
    llm_map = _load_llm_records(llm_today_path)
    news_map = load_news_summary("data/news_summary_top10.txt")

    # 4.1) Parse and integrate News Impact into pred_scores
    import re
    news_text_path = "data/news_summary_top10.txt"
    impact_pattern = re.compile(r"###\s*([A-Z0-9._-]+).*?Impact:\s*([+-]?\d+)", re.DOTALL)
    news_impact: dict[str, int] = {}

    if os.path.exists(news_text_path):
        txt = Path(news_text_path).read_text(encoding="utf-8")
        for match in impact_pattern.finditer(txt):
            ticker, impact_str = match.groups()
            try:
                news_impact[ticker] = int(impact_str)
            except Exception:
                continue
    else:
        log(f"[WARN] Missing {news_text_path}, skipping news impact integration")

    if not news_impact:
        log("[WARN] No Impact lines found in news summary; all news impacts = 0")

    adjusted_scores = {}
    for t in tickers_top10:
        brain = float(pred_scores.get(t, 0.0))
        impact = int(news_impact.get(t, 0))
        adjusted = brain + impact
        adjusted_scores[t] = adjusted
        impact_str = f"+{impact}" if impact > 0 else str(impact) if impact != 0 else "NA"
        log(f"[ADJUST] {t}: pred_score={brain:.2f}, news impact={impact_str}, adjusted={adjusted:.2f}")

    pred_scores = adjusted_scores
    log("[INFO] pred_scores updated with news impact adjustments.")

    # 4.5) Filter: select only top 1 unless multiple have pred_score >= 720
    score_threshold = 720.0
    candidates = [(t, pred_scores.get(t, 0.0)) for t in tickers_top10]
    candidates.sort(key=lambda x: float(x[1]), reverse=True)

    tickers_to_gpt = []
    if candidates:
        tickers_to_gpt = [candidates[0][0]]
        for t, sc in candidates[1:]:
            if _num(sc) is not None and float(sc) >= score_threshold:
                tickers_to_gpt.append(t)

    log(f"[INFO] Filtered candidates: {len(tickers_to_gpt)} ticker(s) selected (threshold={score_threshold})")
    log(f"[INFO] Tickers for GPT: {', '.join(tickers_to_gpt)}")

    # 5) Build MINIMAL CSV for GPT + traceability (UPDATED)
    # NOTE: llm_today_data.jsonl now contains your engineered features.
    header = [
        "TickerName",
        "Ticker",
        "pred_score",          # renamed from BrainScore
        "current_price",
        "Volatility_30D",
        "Volatility_252D",
        "Momentum_63D",
        "Momentum_126D",
        "Momentum_252D",
        "RSE",
        "CBP",
        "SMC",
        "TSS",
        "pos_52w",
        "pos_30d",
        "RSI_14",
        "Month",
        "News",
    ]

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = logs_dir / "blocks_to_gpt_latest.txt"

    with blocks_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in tickers_to_gpt:
            rec = llm_map.get(t, {})
            name = name_map.get(t, "")
            ticker_name = f"{t} - {name}" if name else t

            row = [
                ticker_name,
                t,
                _fmt(pred_scores.get(t)),
                _fmt(_num(rec.get("current_price"))),
                _fmt(_num(rec.get("Volatility_30D"))),
                _fmt(_num(rec.get("Volatility_252D"))),
                _fmt(_num(rec.get("Momentum_63D"))),
                _fmt(_num(rec.get("Momentum_126D"))),
                _fmt(_num(rec.get("Momentum_252D"))),
                _fmt(_num(rec.get("RSE"))),
                _fmt(_num(rec.get("CBP"))),
                _fmt(_num(rec.get("SMC"))),
                _fmt(_num(rec.get("TSS"))),
                _fmt(_num(rec.get("pos_52w"))),
                _fmt(_num(rec.get("pos_30d"))),
                _fmt(_num(rec.get("RSI_14"))),
                str(rec.get("Month", "")),
                news_map.get(t, "N/A"),
            ]
            writer.writerow(row)

    log(f"[INFO] Wrote minimal GPT payload (incl. News) to {blocks_path}")


if __name__ == "__main__":
    main()
