# scripts/quick_scorer.py
from __future__ import annotations
from typing import Dict, List, Tuple
import os, math, time, csv, logging

# -------- logging --------
def _cfg_log():
    lvl = (os.getenv("QUICK_LOG_LEVEL") or os.getenv("RANKER_LOG_LEVEL") or "INFO").upper()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=getattr(logging, lvl, logging.INFO),
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_cfg_log()
log = logging.getLogger("quick_scorer")

# -------- helpers --------
def safe(x, d=0.0):
    try:
        if x is None: return d
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf): return d
        return xf
    except: return d

def clamp(v, lo, hi): return max(lo, min(hi, v))

def _tags(feats: Dict) -> List[str]:
    """Lightweight protections to avoid cutting winners too soon."""
    t: List[str] = []
    rsi = safe(feats.get("RSI14"), None)
    vs200 = safe(feats.get("vsSMA200"), 0.0)
    vs50  = safe(feats.get("vsSMA50"), 0.0)
    r60   = safe(feats.get("r60"), 0.0)
    d20   = safe(feats.get("d20"), 0.0)
    v20   = safe(feats.get("vol_vs20"), 0.0)
    is20h = bool(feats.get("is_20d_high"))

    if r60 >= 20 and vs200 >= 0:
        t.append("RS_trend_protect")
    if is20h and (rsi is not None and 55 <= rsi <= 85):
        t.append("breakout_protect")
    if (0 <= vs50 <= 12) and (5 <= r60 <= 40) and (rsi is not None and 45 <= rsi <= 70):
        t.append("pullback_protect")
    if v20 >= 180 and rsi is not None and rsi >= 78 and d20 >= 12:
        t.append("pump_risk")
    return t

def quick_score(feats: Dict, mode: str = "loose") -> Tuple[float, Dict]:
    """
    Fast, resilient first-pass score ~[-100..+200] (then we just sort).
    Tuned to NOT over-penalize strong trends; mild risk penalties only.
    """
    # weights by mode
    if mode == "strict":
        w_trend, w_momo, w_struct, w_risk = 0.45, 0.30, 0.15, 0.10
        atr_soft, vol_soft, dd_soft = 5.5, 220, -40
    elif mode == "normal":
        w_trend, w_momo, w_struct, w_risk = 0.48, 0.32, 0.13, 0.07
        atr_soft, vol_soft, dd_soft = 6.0, 250, -45
    else:  # loose
        w_trend, w_momo, w_struct, w_risk = 0.50, 0.34, 0.11, 0.05
        atr_soft, vol_soft, dd_soft = 6.5, 300, -50

    vs50  = safe(feats.get("vsSMA50"), 0.0)
    vs200 = safe(feats.get("vsSMA200"), 0.0)
    r60   = safe(feats.get("r60"), 0.0)
    r120  = safe(feats.get("r120"), 0.0)
    rsi   = safe(feats.get("RSI14"), None)
    macd  = safe(feats.get("MACD_hist"), 0.0)
    atr   = safe(feats.get("ATRpct"), 0.0)
    dd    = safe(feats.get("drawdown_pct"), 0.0)  # negative is worse
    v20   = safe(feats.get("vol_vs20"), 0.0)
    px    = safe(feats.get("price"), 0.0)
    sma50 = safe(feats.get("SMA50"), None)
    avwap = safe(feats.get("AVWAP252"), None)

    # trend: favor alignment + healthy RSI band
    rsi_band = 0.0
    if rsi is not None:
        if 47 <= rsi <= 68: rsi_band = 1.0
        elif 40 <= rsi < 47: rsi_band = (rsi - 40) / 7.0
        elif 68 < rsi <= 75: rsi_band = (75 - rsi) / 7.0

    trend = (
        0.55 * clamp(vs200, -40, 80) / 80.0 +
        0.35 * clamp(vs50,  -40, 80) / 80.0 +
        0.10 * clamp(macd,  -1.5, 1.5) / 1.5
    ) * 100.0 + 8.0 * rsi_band

    # momentum: 20–120 day persistence
    momo = (
        0.45 * clamp(r60,  -40, 80) / 80.0 +
        0.30 * clamp(r120, -40, 80) / 80.0 +
        0.25 * clamp(feats.get("d20") or 0.0, -25, 40) / 40.0
    ) * 100.0

    # structure: simple alignment
    struct = 0.0
    if sma50 and avwap:
        if px > sma50 > avwap:
            struct = 12.0
        elif px < sma50 < avwap:
            struct = -8.0
        else:
            struct = 3.0 if px > avwap else -2.0

    # risk (soft)
    risk_pen = 0.0
    if atr > atr_soft:
        # soft: scale gently
        risk_pen -= min(12.0, (atr - atr_soft) * 1.5)
    if v20 > vol_soft:
        risk_pen -= min(8.0, (v20 - vol_soft) / 40.0)
    if dd < dd_soft:
        risk_pen -= min(8.0, (abs(dd) - abs(dd_soft)) / 4.0)
    if rsi is not None and rsi >= 85:
        risk_pen -= 8.0
    if rsi is not None and rsi >= 80 and v20 >= 200:
        risk_pen -= 6.0

    score = (
        w_trend  * trend +
        w_momo   * momo +
        w_struct * struct +
        w_risk   * risk_pen
    )

    return score, {
        "trend": trend, "momo": momo, "struct": struct, "risk_pen": risk_pen
    }

def rank_stage1(
    universe: List[Tuple[str, str, Dict]],
    keep: int = 200,
    mode: str = None,
    rescue_frac: float = None,
    log_dir: str = "data"
) -> Tuple[List[Tuple[str,str,Dict,float,Dict]], List[Tuple[str,str,Dict,float,Dict]], List[Tuple]]:
    """
    Returns:
      pre_topK:  list of (t, name, feats, score, parts) kept (sorted desc)
      quick_scored_all: same for ALL scored (sorted desc)
      removed_rows: diagnostics rows for CSV (ticker, reason,...)
    """
    mode = (mode or os.getenv("STAGE1_MODE", "loose")).lower()
    keep = int(os.getenv("STAGE1_KEEP", str(keep)))
    rescue_frac = float(os.getenv("STAGE1_RESCUE_FRAC", str(rescue_frac if rescue_frac is not None else 0.15)))
    os.makedirs(log_dir, exist_ok=True)

    scored = []
    removed = []
    protected_bucket = []

    for (t, n, f) in universe:
        s, parts = quick_score(f, mode=mode)
        tags = _tags(f)
        row = (t, n, f, s, {"parts": parts, "tags": tags})
        scored.append(row)

    scored.sort(key=lambda x: x[3], reverse=True)

    # protection: keep a slice of protected names even if not in top-K
    top_main = scored[:keep]
    borderline = scored[keep:]

    # collect protected from borderline
    for r in borderline:
        if any(tag.endswith("_protect") for tag in r[4]["tags"]):
            protected_bucket.append(r)

    # limit rescued count
    max_rescue = max(0, int(keep * rescue_frac))
    rescued = protected_bucket[:max_rescue]

    pre_topK = (top_main + rescued)
    # re-sort after adding rescued, and trim to keep
    pre_topK.sort(key=lambda x: x[3], reverse=True)
    pre_topK = pre_topK[:keep]

    # removed diagnostics (everything not in pre_topK)
    kept_set = {t for (t, *_rest) in [(t, n) for (t, n, *_x) in pre_topK]}
    for (t, n, f, s, meta) in scored:
        if t not in kept_set:
            reason = "below_cutoff"
            if "pump_risk" in meta["tags"]:
                reason = "below_cutoff_pump_risk"
            removed.append((t, n, s,
                            f.get("price"), f.get("vsSMA50"), f.get("vsSMA200"),
                            f.get("RSI14"), f.get("ATRpct"), f.get("r60"),
                            f.get("vol_vs20"), f.get("drawdown_pct"),
                            ";".join(meta["tags"]), reason))

    # write CSVs
    try:
        # kept
        path_kept = os.path.join(log_dir, "stage1_kept.csv")
        with open(path_kept, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker","company","score","price","vsSMA50","vsSMA200","RSI14","ATR%","r60","vol_vs20","drawdown%","tags"])
            for (t, n, feats, s, meta) in pre_topK:
                w.writerow([t, n, f"{s:.2f}", feats.get("price"), feats.get("vsSMA50"), feats.get("vsSMA200"),
                            feats.get("RSI14"), feats.get("ATRpct"), feats.get("r60"),
                            feats.get("vol_vs20"), feats.get("drawdown_pct"), ";".join(meta["tags"])])
        log.info(f"[Stage1] kept={len(pre_topK)} -> {path_kept}")

        # removed
        path_removed = os.path.join(log_dir, "stage1_removed.csv")
        with open(path_removed, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker","company","score","price","vsSMA50","vsSMA200","RSI14","ATR%","r60","vol_vs20","drawdown%","tags","reason"])
            for row in removed:
                w.writerow(row)
        log.info(f"[Stage1] removed={len(removed)} -> {path_removed}")
    except Exception as e:
        log.warning(f"[Stage1] CSV logging failed: {e!r}")

    return pre_topK, scored, removed
