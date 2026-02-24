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
    p20 = data.get("p20")
    p50 = data.get("p50")
    p80 = data.get("p80")
    if not isinstance(p20, list) or len(p20) != bins:
        p20 = None
    if not isinstance(p50, list) or len(p50) != bins:
        p50 = None
    if not isinstance(p80, list) or len(p80) != bins:
        p80 = None
    return {
        "bins": bins,
        "pred_min": float(data.get("pred_min", 0.0)),
        "pred_max": float(data.get("pred_max", 1.0)),
        "mu_adj": mu_adj,
        "p20": p20,
        "p50": p50,
        "p80": p80,
    }


def _load_score_return_calibration() -> dict | None:
    try:
        data = json.loads(Path("knobs/score_calibration.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    bins = int(data.get("score_bins", 0))
    p20 = data.get("score_p20")
    p50 = data.get("score_p50")
    p80 = data.get("score_p80")
    mean = data.get("score_mean")
    if not bins:
        return None
    if not isinstance(p50, list) or len(p50) != bins:
        return None
    if not isinstance(p20, list) or len(p20) != bins:
        p20 = None
    if not isinstance(p80, list) or len(p80) != bins:
        p80 = None
    if not isinstance(mean, list) or len(mean) != bins:
        mean = None
    return {
        "bins": bins,
        "score_min": int(data.get("score_min", 0)),
        "score_max": int(data.get("score_max", 1000)),
        "p20": p20,
        "p50": p50,
        "p80": p80,
        "mean": mean,
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


def _expected_band_from_pred(
    pred: float, calib: dict | None
) -> tuple[float, float] | None:
    if calib is None or pred is None or not math.isfinite(pred):
        return None
    p20 = calib.get("p20")
    p80 = calib.get("p80")
    if not isinstance(p20, list) or not isinstance(p80, list):
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
        lo = float(p20[idx])
        hi = float(p80[idx])
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        return lo * 100.0, hi * 100.0
    except Exception:
        return None


def _expected_median_from_pred(pred: float, calib: dict | None) -> float | None:
    if calib is None or pred is None or not math.isfinite(pred):
        return None
    p50 = calib.get("p50")
    if not isinstance(p50, list):
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
        med = float(p50[idx])
        if not math.isfinite(med):
            return None
        return med * 100.0
    except Exception:
        return None


def _score_bin_idx(score: int, calib: dict) -> int | None:
    if score is None or not math.isfinite(score):
        return None
    bins = int(calib.get("bins", 0))
    score_min = int(calib.get("score_min", 0))
    score_max = int(calib.get("score_max", 1000))
    if bins <= 0 or score_max <= score_min:
        return None
    idx = int((score - score_min) / (score_max - score_min) * bins)
    if idx < 0:
        idx = 0
    elif idx >= bins:
        idx = bins - 1
    return idx


def _expected_median_from_score(score: int, calib: dict | None) -> float | None:
    if calib is None:
        return None
    idx = _score_bin_idx(score, calib)
    if idx is None:
        return None
    try:
        med = float(calib["p50"][idx])
        if not math.isfinite(med):
            return None
        return med * 100.0
    except Exception:
        return None


def _expected_mean_from_score(score: int, calib: dict | None) -> float | None:
    if calib is None:
        return None
    mean = calib.get("mean")
    if not isinstance(mean, list):
        return None
    idx = _score_bin_idx(score, calib)
    if idx is None:
        return None
    try:
        mu = float(mean[idx])
        if not math.isfinite(mu):
            return None
        return mu * 100.0
    except Exception:
        return None


def _expected_band_from_score(score: int, calib: dict | None) -> tuple[float, float] | None:
    if calib is None:
        return None
    p20 = calib.get("p20")
    p80 = calib.get("p80")
    if not isinstance(p20, list) or not isinstance(p80, list):
        return None
    idx = _score_bin_idx(score, calib)
    if idx is None:
        return None
    try:
        lo = float(p20[idx])
        hi = float(p80[idx])
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        return lo * 100.0, hi * 100.0
    except Exception:
        return None


def _score_thresholds_from_calib(calib: dict | None) -> dict | None:
    if calib is None:
        return None
    p50 = calib.get("p50")
    bins = int(calib.get("bins", 0))
    score_min = int(calib.get("score_min", 0))
    score_max = int(calib.get("score_max", 1000))
    if not isinstance(p50, list) or bins <= 0 or score_max <= score_min:
        return None
    targets = {20.0: None, 30.0: None, 45.0: None}
    for i, v in enumerate(p50):
        if v is None or not math.isfinite(v):
            continue
        pct = float(v) * 100.0
        for t in list(targets.keys()):
            if targets[t] is None and pct >= t:
                score = score_min + (score_max - score_min) * (i / bins)
                targets[t] = int(round(score))
    return {
        "score_20": targets[20.0],
        "score_30": targets[30.0],
        "score_45": targets[45.0],
    }


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
        return "**🟢🟢🟢 Ultra Strong Buy**"
    if score > 700:
        return "**🟢🟢 Strong Buy**"
    if score > 600:
        return "**🟢 Buy**"
    return "**🔴 Ignore**"


def _advice_from_expected_median(median_pct: float) -> str:
    if median_pct >= 45.0:
        return "**🟢🟢🟢 Ultra Strong Buy**"
    if median_pct >= 30.0:
        return "**🟢🟢 Strong Buy**"
    if median_pct >= 20.0:
        return "**🟢 Buy**"
    return "**🔴 Ignore**"


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
    median_line = ""
    mean_line = ""
    band_line = ""
    legend_score_lines = ""
    # Prefer calibrated expected return from model prediction (top10_ml.csv)
    expected_median = None
    expected_mean = None
    gpt_score = None
    score_calib_by_score = _load_score_return_calibration()
    score_thresholds = _score_thresholds_from_calib(score_calib_by_score)
    try:
        ticker = _extract_ticker(text)
        if ticker:
            top_df = pd.read_csv("data/top10_ml.csv")
            top_df["ticker"] = top_df["ticker"].astype(str).str.strip()
            row = top_df[top_df["ticker"] == ticker].head(1)
            if not row.empty:
                calib = _load_score_calibration()
                if "pred_score" in row.columns:
                    pred = float(row.iloc[0]["pred_score"])
                    expected_median = _expected_median_from_pred(pred, calib)
                    expected_mean = _expected_return_from_pred(pred, calib)
                    band = _expected_band_from_pred(pred, calib)
                    if band is not None:
                        band_line = (
                            f"Expected range (P20–P80): {band[0]:.1f}%–{band[1]:.1f}%\n"
                        )
                if "gpt_score" in row.columns:
                    gpt_score = int(row.iloc[0]["gpt_score"])
    except Exception:
        expected_median = None
        expected_mean = None
        gpt_score = None
    if gpt_score is not None:
        if score_calib_by_score is not None:
            median_by_score = _expected_median_from_score(gpt_score, score_calib_by_score)
            mean_by_score = _expected_mean_from_score(gpt_score, score_calib_by_score)
            band_by_score = _expected_band_from_score(gpt_score, score_calib_by_score)
            if median_by_score is not None:
                expected_median = median_by_score
            if mean_by_score is not None:
                expected_mean = mean_by_score
            if band_by_score is not None:
                band_line = (
                    f"Expected range (P20–P80): {band_by_score[0]:.1f}%–{band_by_score[1]:.1f}%\n"
                )
        text = _replace_score_line(text, gpt_score)
        if expected_median is not None and math.isfinite(expected_median):
            advice = _advice_from_expected_median(expected_median)
        else:
            advice = _advice_from_score(gpt_score)
        text = _replace_advice_line(text, advice)
        score = gpt_score
    if expected_median is not None and math.isfinite(expected_median):
        median_line = f"Expected return (typical): {expected_median:.1f}%\n"
    if expected_mean is not None and math.isfinite(expected_mean):
        mean_line = f"Expected return (average): {expected_mean:.1f}%\n"
    if not median_line and not mean_line and score is not None:
        expected_mean = score * pct_per_point
        mean_line = f"Expected return (approx, score-based): {expected_mean:.1f}%\n"
    if score_thresholds and all(v is not None for v in score_thresholds.values()):
        s20 = score_thresholds["score_20"]
        s30 = score_thresholds["score_30"]
        s45 = score_thresholds["score_45"]
        legend_score_lines = (
            f"**🟢🟢🟢 Ultra Strong Buy** (Score ≥ {s45})\n"
            f"**🟢🟢 Strong Buy** (Score {s30}–{s45-1})\n"
            f"**🟢 Buy** (Score {s20}–{s30-1})\n"
            f"**🔴 Ignore** (Score < {s20})\n"
        )
    legend = (
        "\n\n---\n"
        "**Legend (score tiers, derived from typical return)**\n"
        f"{legend_score_lines if legend_score_lines else '**🟢🟢🟢 Ultra Strong Buy**\\n**🟢🟢 Strong Buy**\\n**🟢 Buy**\\n**🔴 Ignore**\\n'}"
        f"{median_line}"
        f"{mean_line}"
        f"{band_line}"
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
    header = ["Rank", "TickerName", "Ticker", "Score", "PipelineNews"]

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = logs_dir / "blocks_to_gpt_latest.txt"

    with blocks_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, t in enumerate(tickers_to_gpt, start=1):
            name = name_map.get(t, "")
            ticker_name = f"{t} - {name}" if name else t
            score = ""
            if "gpt_score" in top_df.columns:
                try:
                    score = int(
                        top_df.loc[top_df["ticker"] == t, "gpt_score"]
                        .astype("Int64")
                        .iloc[0]
                    )
                except Exception:
                    score = ""
            row = [
                idx,
                ticker_name,
                t,
                score,
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
