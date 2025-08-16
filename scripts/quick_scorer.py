# scripts/quick_scorer.py
from __future__ import annotations
from typing import Dict, List, Tuple
import os, math, csv, logging
import numpy as np

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

def _xs_stats(feats_list: List[Dict], keys: List[str]) -> Dict[str, Tuple[float, float]]:
    """Cross-sectional robust stats (median, MAD*1.4826; fallback to std)."""
    stats: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals: List[float] = []
        for f in feats_list:
            v = safe(f.get(k), None)  # keep None for invalid/NaN/inf
            if v is not None:
                vals.append(v)
        if not vals:
            stats[k] = (0.0, 1.0)
            continue
        arr = np.array(vals, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = mad * 1.4826 if mad > 1e-12 else float(np.std(arr) or 1.0)
        if not (scale > 1e-9): scale = 1.0
        stats[k] = (med, scale)
    return stats

def _z(xs: Dict[str, Tuple[float,float]], key: str, x: float|None, lo=-4.0, hi=4.0) -> float:
    if x is None: return 0.0
    med, scale = xs.get(key, (0.0, 1.0))
    try: z = (float(x) - med) / (scale if scale != 0 else 1.0)
    except: z = 0.0
    return float(clamp(z, lo, hi))

def _rsi_band_val(rsi: float|None) -> float:
    """Sweet-spot band shaping for RSI; returns 0..1."""
    if rsi is None: return 0.0
    r = float(rsi)
    if 47 <= r <= 68: return 1.0
    if 40 <= r < 47:  return (r - 40) / 7.0
    if 68 < r <= 75:  return (75 - r) / 7.0
    return 0.0

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
    if v20 <= -60:
        t.append("low_liquidity_today")
    return t

# ---- P/E tilt (optional; uses feats["val_PE"] or feats["PE"] if present) ----
def _pe_tilt_points(pe: float|None, rsi: float|None, vol_vs20: float|None) -> float:
    """
    Returns a small tilt in [-20 .. +20] points (will be weighted later).
    - Low P/E gets a bonus that saturates below ~10.
    - Very high P/E gets a mild penalty, stronger only if overbought/frothy.
    - Missing/None => 0 (neutral).
    """
    if pe is None or pe <= 0:  # treat invalid as neutral
        return 0.0

    # Base shape: map P/E to a bounded score
    # <=10: +20, 10..15: +12..+8, 15..25: +6..+0, 25..40: 0..-6, >40: -8
    if pe <= 10:
        base = 20.0
    elif pe <= 15:
        base = 12.0 - (pe - 10) * (4.0 / 5.0)    # 12 -> 8
    elif pe <= 25:
        base = 6.0 - (pe - 15) * (0.6)           # 6 -> 0
    elif pe <= 40:
        base = 0.0 - (pe - 25) * (0.4)           # 0 -> -6
    else:
        base = -8.0

    # Froth guard: only let strong negatives apply when momentum is hot
    rsi = None if rsi is None else float(rsi)
    vol_vs20 = None if vol_vs20 is None else float(vol_vs20)
    if base < 0:
        if not ( (rsi is not None and rsi >= 75) or (vol_vs20 is not None and vol_vs20 >= 200) ):
            base *= 0.5  # soften penalty if not overbought/frothy

    return float(clamp(base, -20.0, 20.0))

def quick_score(
    feats: Dict,
    mode: str = "loose",
    xs: Dict[str, Tuple[float, float]] | None = None
) -> Tuple[float, Dict]:
    """
    Fast, resilient first-pass score ~[-100..+200] (then we sort).
    Uses cross-sectional z-scores when provided (cheap & adaptive).
    Includes a small P/E tilt iff P/E exists in feats.
    """
    # weights by mode (sum to 1.0). We carve out a small bucket for P/E.
    pe_weight_env = float(os.getenv("QS_PE_WEIGHT", "0.06"))  # default 6%
    pe_weight = float(clamp(pe_weight_env, 0.0, 0.12))

    if mode == "strict":
        w_trend, w_momo, w_struct, w_risk = 0.44, 0.29, 0.14, 0.07
    elif mode == "normal":
        w_trend, w_momo, w_struct, w_risk = 0.47, 0.31, 0.12, 0.04
    else:  # loose
        w_trend, w_momo, w_struct, w_risk = 0.49, 0.33, 0.10, 0.02

    # Normalize to leave room for pe_weight
    rem = max(1e-9, (w_trend + w_momo + w_struct + w_risk))
    scale = max(0.0, 1.0 - pe_weight) / rem
    w_trend  *= scale
    w_momo   *= scale
    w_struct *= scale
    w_risk   *= scale
    w_pe = pe_weight

    # regime thresholds
    if mode == "strict":
        atr_soft, vol_soft, dd_soft = 5.5, 220, -40
    elif mode == "normal":
        atr_soft, vol_soft, dd_soft = 6.0, 250, -45
    else:
        atr_soft, vol_soft, dd_soft = 6.5, 300, -50

    vs50  = safe(feats.get("vsSMA50"), 0.0)
    vs200 = safe(feats.get("vsSMA200"), 0.0)
    r60   = safe(feats.get("r60"), 0.0)
    r120  = safe(feats.get("r120"), 0.0)
    d20   = safe(feats.get("d20"), 0.0)
    rsi   = safe(feats.get("RSI14"), None)
    macd  = safe(feats.get("MACD_hist"), 0.0)
    atr   = safe(feats.get("ATRpct"), 0.0)
    dd    = safe(feats.get("drawdown_pct"), 0.0)  # negative is worse
    v20   = safe(feats.get("vol_vs20"), 0.0)
    px    = safe(feats.get("price"), 0.0)
    sma50 = safe(feats.get("SMA50"), None)
    avwap = safe(feats.get("AVWAP252"), None)

    # Pull P/E if pre-attached (no network here)
    pe_val = None
    for k in ("val_PE", "PE", "pe", "pe_hint"):
        if feats.get(k) is not None:
            try:
                pe_val = float(feats.get(k))
            except Exception:
                pass
            break

    # --- cross-sectional z where available ---
    use_xs = xs is not None and os.getenv("QS_USE_XS", "1").lower() in {"1", "true", "yes"}
    if use_xs:
        z_vs200 = _z(xs, "vsSMA200", vs200)
        z_vs50  = _z(xs, "vsSMA50",  vs50)
        z_r60   = _z(xs, "r60",      r60)
        z_r120  = _z(xs, "r120",     r120)
        z_d20   = _z(xs, "d20",      d20)
        z_atr   = _z(xs, "ATRpct",   atr)
        z_v20   = _z(xs, "vol_vs20", v20)
    else:
        z_vs200 = clamp(vs200, -40, 80) / 20.0
        z_vs50  = clamp(vs50,  -40, 80) / 20.0
        z_r60   = clamp(r60,   -40, 80) / 20.0
        z_r120  = clamp(r120,  -40, 80) / 20.0
        z_d20   = clamp(d20,   -25, 40) / 10.0
        z_atr   = (atr - 6.0) / 2.0
        z_v20   = (v20 - 200.0) / 80.0

    # trend: favor alignment + healthy RSI band (adaptive)
    rsi_band = _rsi_band_val(rsi)
    trend = (0.5 * clamp(z_vs200/4.0, -1, 1) +
             0.35 * clamp(z_vs50/4.0,  -1, 1) +
             0.15 * clamp(macd/1.5,     -1, 1)) * 100.0 + 8.0 * rsi_band

    # momentum: 20–120 day persistence (adaptive)
    momo = (0.42 * clamp(z_r60/4.0,  -1, 1) +
            0.30 * clamp(z_r120/4.0, -1, 1) +
            0.28 * clamp(z_d20/4.0,  -1, 1)) * 100.0

    # structure: simple alignment + anchor premium (price vs AVWAP252)
    struct = 0.0
    if sma50 and avwap:
        if px > sma50 > avwap:
            struct += 12.0
        elif px < sma50 < avwap:
            struct -= 8.0
        else:
            struct += 3.0 if px > avwap else -2.0
    if avwap:
        prem = (px / avwap) - 1.0
        if prem < 0:
            struct += clamp(abs(prem) * 20.0, 0.0, 6.0)
        else:
            struct -= clamp(prem * 25.0, 0.0, 8.0)

    # risk (soft, adaptive)
    risk_pen = 0.0
    if use_xs:
        risk_pen -= clamp(max(0.0, z_atr), 0.0, 3.0) * 3.5
        risk_pen -= clamp(max(0.0, z_v20), 0.0, 3.0) * 2.0
    else:
        if atr > atr_soft:
            risk_pen -= min(12.0, (atr - atr_soft) * 1.5)
        if v20 > vol_soft:
            risk_pen -= min(8.0, (v20 - vol_soft) / 40.0)
    if dd < dd_soft:
        risk_pen -= min(8.0, (abs(dd) - abs(dd_soft)) / 4.0)
    if rsi is not None and rsi >= 85:
        risk_pen -= 8.0
    if rsi is not None and rsi >= 88 and d20 >= 150:
        risk_pen -= 10.0
    if rsi is not None and rsi >= 80 and v20 >= 200:
        risk_pen -= 6.0

    # P/E tilt (neutral if missing)
    pe_points_raw = _pe_tilt_points(pe_val, rsi, v20)  # [-20..+20]
    pe_score = clamp(pe_points_raw / 20.0, -1.0, 1.0) * 100.0  # normalize

    score = (w_trend  * trend +
             w_momo   * momo +
             w_struct * struct +
             w_risk   * risk_pen +
             w_pe     * pe_score)

    return score, {
        "trend": trend, "momo": momo, "struct": struct,
        "risk_pen": risk_pen, "pe_tilt_pts": pe_points_raw, "pe_weight": w_pe
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
      pre_topK:  list of (t, name, feats, score, meta) kept (sorted desc)
      quick_scored_all: same for ALL scored (sorted desc)
      removed_rows: diagnostics rows for CSV (ticker, reason,...)
    """
    mode = (mode or os.getenv("STAGE1_MODE", "loose")).lower()
    keep = int(os.getenv("STAGE1_KEEP", str(keep)))
    rescue_frac = float(os.getenv("STAGE1_RESCUE_FRAC", str(rescue_frac if rescue_frac is not None else 0.15)))
    write_csv = os.getenv("STAGE1_WRITE_CSV", "1").lower() in {"1","true","yes"}
    os.makedirs(log_dir, exist_ok=True)

    # --- compute cross-sectional stats once (cheap) ---
    feat_list = [f for (_t, _n, f) in universe]
    xs_keys = ["vsSMA50","vsSMA200","d20","r60","r120","ATRpct","vol_vs20"]
    xs = _xs_stats(feat_list, xs_keys)

    scored: List[Tuple[str,str,Dict,float,Dict]] = []
    removed: List[Tuple] = []
    protected_bucket: List[Tuple[str,str,Dict,float,Dict]] = []

    for (t, n, f) in universe:
        s, parts = quick_score(f, mode=mode, xs=xs)
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
    pre_topK.sort(key=lambda x: x[3], reverse=True)
    pre_topK = pre_topK[:keep]

    # removed diagnostics (everything not in pre_topK)
    kept_set = {t for (t, _n, _f, _s, _m) in pre_topK}
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

    # write CSVs (optional for speed)
    if write_csv:
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
