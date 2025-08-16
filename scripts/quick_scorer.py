# scripts/quick_scorer.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, math, csv, logging
import numpy as np

_TIER_THR_LARGE_USD = None
_TIER_THR_MID_USD   = None

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
            v = f.get(k)
            try:
                if v is not None:
                    v = float(v)
                    if not (math.isnan(v) or math.isinf(v)):
                        vals.append(v)
            except Exception:
                pass
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

def _auto_liq_thresholds(universe, p_large=80, p_mid=40):
    """
    Infer LARGE/MID cutoffs from cross-section percentiles of avg_dollar_vol_20d.
    Falls back to 50M / 10M if too few values.
    """
    advs = []
    for (_t, _n, f) in universe:
        a = f.get("avg_dollar_vol_20d")
        try:
            if a is None: 
                continue
            a = float(a)
            if a > 0 and not math.isinf(a) and not math.isnan(a):
                advs.append(a)
        except Exception:
            pass

    if len(advs) >= 30:
        arr = np.array(advs, dtype=float)
        thr_large = float(np.percentile(arr, p_large))
        thr_mid   = float(np.percentile(arr, p_mid))
    else:
        thr_large = 50_000_000.0
        thr_mid   = 10_000_000.0
    return thr_large, thr_mid

def _compute_and_set_tier_thresholds(universe):
    """
    Populate module-level thresholds using env overrides if provided,
    otherwise auto-derive from the current universe.
    """
    global _TIER_THR_LARGE_USD, _TIER_THR_MID_USD
    env_large = os.getenv("TIER_LARGE_USD", "").strip()
    env_mid   = os.getenv("TIER_MEDIUM_USD", "").strip()
    if env_large and env_mid:
        try:
            _TIER_THR_LARGE_USD = float(env_large)
            _TIER_THR_MID_USD   = float(env_mid)
        except Exception:
            _TIER_THR_LARGE_USD, _TIER_THR_MID_USD = _auto_liq_thresholds(universe)
    else:
        _TIER_THR_LARGE_USD, _TIER_THR_MID_USD = _auto_liq_thresholds(universe)

# ---- TIERING POLICY ------------------------------------------------
def _tier_from_thresholds(feats: Dict) -> str:
    """Map ADV to 'large'/'mid'/'small' using global thresholds; missing -> 'small'."""
    adv = feats.get("avg_dollar_vol_20d")
    try:
        a = float(adv) if adv is not None else None
    except Exception:
        a = None
    if a is None or _TIER_THR_LARGE_USD is None or _TIER_THR_MID_USD is None:
        return "small"
    if a >= _TIER_THR_LARGE_USD:
        return "large"
    if a >= _TIER_THR_MID_USD:
        return "mid"
    return "small"

def _tier_fallback_small_vs_large(feats: Dict) -> str:
    """
    Legacy fallback: classify small vs large by cap/liquidity/price when no policy applies.
    """
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None
    cap = _safe_float(feats.get("market_cap"))
    liq = _safe_float(feats.get("avg_dollar_vol_20d"))
    price = _safe_float(feats.get("price"))

    cap_cut = float(os.getenv("QS_SMALL_CAP_MAX", "2000000000"))   # $2B default
    liq_cut = float(os.getenv("QS_SMALL_LIQ_MAX", "10000000"))     # $10M avg $vol
    px_cut  = float(os.getenv("QS_SMALL_PRICE_MAX", "12"))

    if cap is not None:
        return "small" if cap <= cap_cut else "large"
    if liq is not None:
        return "small" if liq <= liq_cut else "large"
    if price is not None:
        return "small" if price <= px_cut else "large"
    return "large"

def _tier_map_for_universe(universe: List[Tuple[str,str,Dict]]) -> Dict[str, str]:
    """
    Returns {ticker: 'small'|'large'} according to TIER_POLICY.
    - TOPK_ADV: Top-K by ADV -> 'large'; others -> 'small'. Missing ADV uses TIER_BACKFILL_UNKNOWN_AS.
    - THRESH:   Use feats['liq_tier'] from features.py if present (LARGE/MID/SMALL),
                else derive thresholds cross-sectionally. 'mid' maps to 'large' weights.
    """
    policy = (os.getenv("TIER_POLICY", "THRESH") or "THRESH").upper()
    backfill_unknown = (os.getenv("TIER_BACKFILL_UNKNOWN_AS", "small") or "small").lower()
    if backfill_unknown not in {"small","large"}:
        backfill_unknown = "small"

    tier_map: Dict[str, str] = {}

    if policy == "TOPK_ADV":
        # Build ADV list
        pairs = []
        for (t, _n, f) in universe:
            adv = f.get("avg_dollar_vol_20d")
            try:
                adv = float(adv) if adv is not None else None
            except Exception:
                adv = None
            pairs.append((t, adv))
        # sort by adv desc, ignore None
        valid = [(t, a) for (t, a) in pairs if a is not None]
        valid.sort(key=lambda x: x[1], reverse=True)

        K = int(os.getenv("TIER_TOPK_LARGE", "500"))
        large_set = set([t for (t, _a) in valid[:K]])

        for (t, a) in pairs:
            if t in large_set:
                tier_map[t] = "large"
            else:
                if a is None:
                    tier_map[t] = backfill_unknown
                else:
                    tier_map[t] = "small"

        log.info(f"[Stage1] TIER_POLICY=TOPK_ADV: large={len(large_set)} (K={K}), backfill_unknown_as={backfill_unknown}")
        return tier_map

    # THRESH or anything else -> use thresholds / features.py precomputed tag
    _compute_and_set_tier_thresholds(universe)
    if _TIER_THR_LARGE_USD is not None and _TIER_THR_MID_USD is not None:
        log.info(f"[Stage1] TIER_POLICY=THRESH thresholds: LARGE≥{_TIER_THR_LARGE_USD:,.0f} USD, "
                 f"MID≥{_TIER_THR_MID_USD:,.0f} USD")

    for (t, _n, f) in universe:
        pre = (f.get("liq_tier") or "").upper()
        if pre in {"LARGE","MID","SMALL"}:
            tier_map[t] = "large" if pre in {"LARGE","MID"} else "small"
        else:
            tri = _tier_from_thresholds(f)  # large/mid/small
            tier_map[t] = "large" if tri in {"large","mid"} else "small"
    return tier_map

# ---- scoring pieces ------------------------------------------------
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

def _tier_params(tier: str, mode: str, pe_weight_base: float):
    """
    Return (weights, pe_weight, atr_soft, vol_soft, dd_soft).
    Small caps: slightly harsher risk + a bit more structure attention.
    Large caps: slightly more trend/momo + stronger valuation tilt.
    """
    if tier == "small":
        pe_w = float(os.getenv("QS_PE_WEIGHT_SMALL", str(pe_weight_base)))  # default same as base
        if mode == "strict":
            atr_soft, vol_soft, dd_soft = 5.0, 200, -42
        elif mode == "normal":
            atr_soft, vol_soft, dd_soft = 5.5, 220, -45
        else:
            atr_soft, vol_soft, dd_soft = 6.0, 240, -48
        weights = dict(trend=0.44, momo=0.30, struct=0.14, risk=0.12)
    else:
        pe_w = float(os.getenv("QS_PE_WEIGHT_LARGE", str(pe_weight_base + 0.02)))  # +2% nudge
        if mode == "strict":
            atr_soft, vol_soft, dd_soft = 5.5, 240, -42
        elif mode == "normal":
            atr_soft, vol_soft, dd_soft = 6.0, 280, -45
        else:
            atr_soft, vol_soft, dd_soft = 6.5, 300, -50
        weights = dict(trend=0.50, momo=0.33, struct=0.11, risk=0.04)
    return weights, pe_w, atr_soft, vol_soft, dd_soft

# ---- P/E tilt (optional; uses feats["val_PE"] or feats["PE"] if present) ----
def _pe_tilt_points(pe: float|None, rsi: float|None, vol_vs20: float|None) -> float:
    if pe is None or pe <= 0:
        return 0.0
    if pe <= 10: base = 20.0
    elif pe <= 15: base = 12.0 - (pe - 10) * (4.0 / 5.0)
    elif pe <= 25: base = 6.0 - (pe - 15) * 0.6
    elif pe <= 40: base = 0.0 - (pe - 25) * 0.4
    else: base = -8.0
    rsi = None if rsi is None else float(rsi)
    vol_vs20 = None if vol_vs20 is None else float(vol_vs20)
    if base < 0 and not ((rsi is not None and rsi >= 75) or (vol_vs20 is not None and vol_vs20 >= 200)):
        base *= 0.5
    return float(clamp(base, -20.0, 20.0))

def quick_score(
    feats: Dict,
    mode: str = "loose",
    xs: Dict[str, Tuple[float, float]] | None = None,
    tier: Optional[str] = None
) -> Tuple[float, Dict]:
    """
    Fast, resilient first-pass score ~[-100..+200] (then we sort).
    Cross-sectional z-scores optional. Tier-aware weights & thresholds.
    Includes a small P/E tilt iff P/E exists in feats.
    """
    # Tier fallback if not supplied (kept for safety)
    tier = (tier or _tier_fallback_small_vs_large(feats))

    # base P/E weight (can be overridden per-tier)
    pe_weight_base = float(os.getenv("QS_PE_WEIGHT", "0.06"))
    weights, pe_weight, atr_soft, vol_soft, dd_soft = _tier_params(tier, mode, pe_weight_base)

    w_trend, w_momo, w_struct, w_risk = weights["trend"], weights["momo"], weights["struct"], weights["risk"]
    rem = max(1e-9, (w_trend + w_momo + w_struct + w_risk))
    scale = max(0.0, 1.0 - pe_weight) / rem
    w_trend  *= scale
    w_momo   *= scale
    w_struct *= scale
    w_risk   *= scale
    w_pe = pe_weight

    vs50  = safe(feats.get("vsSMA50"), 0.0)
    vs200 = safe(feats.get("vsSMA200"), 0.0)
    r60   = safe(feats.get("r60"), 0.0)
    r120  = safe(feats.get("r120"), 0.0)
    d20   = safe(feats.get("d20"), 0.0)
    rsi   = safe(feats.get("RSI14"), None)
    macd  = safe(feats.get("MACD_hist"), 0.0)
    atr   = safe(feats.get("ATRpct"), 0.0)
    dd    = safe(feats.get("drawdown_pct"), 0.0)
    v20   = safe(feats.get("vol_vs20"), 0.0)
    px    = safe(feats.get("price"), 0.0)
    sma50 = safe(feats.get("SMA50"), None)
    avwap = safe(feats.get("AVWAP252"), None)

    # EMA features (if present)
    vsE50  = safe(feats.get("vsEMA50"), None)
    vsE200 = safe(feats.get("vsEMA200"), None)
    e50s   = safe(feats.get("EMA50_slope_5d"), None)

    # Pull P/E if pre-attached (no network here)
    pe_val = None
    for k in ("val_PE", "PE", "pe", "pe_hint"):
        if feats.get(k) is not None:
            try:
                pe_val = float(feats.get(k))
            except Exception:
                pass
            break

    # --- cross-sectional z where available (include EMA keys) ---
    use_xs = xs is not None and os.getenv("QS_USE_XS", "1").lower() in {"1", "true", "yes"}
    if use_xs:
        z_vs200 = _z(xs, "vsSMA200", vs200)
        z_vs50  = _z(xs, "vsSMA50",  vs50)
        z_r60   = _z(xs, "r60",      r60)
        z_r120  = _z(xs, "r120",     r120)
        z_d20   = _z(xs, "d20",      d20)
        z_atr   = _z(xs, "ATRpct",   atr)
        z_v20   = _z(xs, "vol_vs20", v20)
        z_vsE50 = _z(xs, "vsEMA50",  vsE50)
        z_vsE200= _z(xs, "vsEMA200", vsE200)
        z_e50s  = _z(xs, "EMA50_slope_5d", e50s)
    else:
        z_vs200 = clamp(vs200, -40, 80) / 20.0
        z_vs50  = clamp(vs50,  -40, 80) / 20.0
        z_r60   = clamp(r60,   -40, 80) / 20.0
        z_r120  = clamp(r120,  -40, 80) / 20.0
        z_d20   = clamp(d20,   -25, 40) / 10.0
        z_atr   = (atr - 6.0) / 2.0
        z_v20   = (v20 - 200.0) / 80.0
        z_vsE50 = (0.0 if vsE50  is None else clamp(vsE50,  -40, 80) / 20.0)
        z_vsE200= (0.0 if vsE200 is None else clamp(vsE200, -40, 80) / 20.0)
        z_e50s  = (0.0 if e50s   is None else clamp(e50s,   -20, 30) / 10.0)

    # trend: add EMA alignment modestly
    rsi_band = _rsi_band_val(rsi)
    trend = (
        0.40 * clamp(z_vs200/4.0, -1, 1) +
        0.25 * clamp(z_vs50/4.0,  -1, 1) +
        0.15 * clamp(z_vsE200/4.0,-1, 1) +
        0.10 * clamp(z_vsE50/4.0, -1, 1) +
        0.10 * clamp(macd/1.5,    -1, 1)
    ) * 100.0 + 8.0 * rsi_band

    # momentum: include EMA50 slope
    momo = (
        0.36 * clamp(z_r60/4.0,   -1, 1) +
        0.27 * clamp(z_r120/4.0,  -1, 1) +
        0.22 * clamp(z_d20/4.0,   -1, 1) +
        0.15 * clamp(z_e50s/4.0,  -1, 1)
    ) * 100.0

    # structure: SMA/AVWAP + EMA alignment bonus
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
    # EMA stack nudge
    if (vsE50 is not None and vsE200 is not None):
        if vsE50 > 0 and vsE200 > 0:
            struct += 3.0
        elif vsE50 < 0 and vsE200 < 0:
            struct -= 2.0

    # risk (soft, tier-aware)
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
    # froth/overbought
    if rsi is not None and rsi >= 85:
        risk_pen -= 8.0
    if rsi is not None and rsi >= 88 and d20 >= 150:
        risk_pen -= 10.0
    if rsi is not None and rsi >= 80 and v20 >= 200:
        risk_pen -= 6.0
    # small-cap extras: thin tape and liquidity drought
    if tier == "small":
        liq_min = float(os.getenv("QS_MIN_DOLLAR_VOL_20D", "2000000"))  # $2M
        liq = safe(feats.get("avg_dollar_vol_20d"), None)
        if liq is not None and liq < liq_min:
            # up to -10 for illiquidity; smooth scaling
            risk_pen -= clamp((liq_min - liq) / max(liq_min, 1.0), 0.0, 1.0) * 10.0
        if v20 <= -40:  # today is unusually quiet vs avg
            risk_pen -= clamp(abs(v20 + 40.0) / 40.0, 0.0, 1.0) * 6.0

    # P/E tilt (neutral if missing)
    pe_points_raw = _pe_tilt_points(pe_val, rsi, v20)  # [-20..+20]
    pe_score = clamp(pe_points_raw / 20.0, -1.0, 1.0) * 100.0

    score = (w_trend  * trend +
             w_momo   * momo +
             w_struct * struct +
             w_risk   * risk_pen +
             w_pe     * pe_score)

    return score, {
        "trend": trend, "momo": momo, "struct": struct, "risk_pen": risk_pen,
        "pe_tilt_pts": pe_points_raw, "pe_weight": w_pe, "tier": tier
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
    Tiering respects TIER_POLICY (TOPK_ADV or THRESH) and quotas.
    """
    mode = (mode or os.getenv("STAGE1_MODE", "loose")).lower()
    keep = int(os.getenv("STAGE1_KEEP", str(keep)))
    rescue_frac = float(os.getenv("STAGE1_RESCUE_FRAC", str(rescue_frac if rescue_frac is not None else 0.15)))
    write_csv = os.getenv("STAGE1_WRITE_CSV", "1").lower() in {"1","true","yes"}
    os.makedirs(log_dir, exist_ok=True)

    # cross-sectional stats (include EMA keys)
    feat_list = [f for (_t, _n, f) in universe]
    xs_keys = ["vsSMA50","vsSMA200","d20","r60","r120","ATRpct","vol_vs20","vsEMA50","vsEMA200","EMA50_slope_5d"]
    xs = _xs_stats(feat_list, xs_keys)

    # Build tier mapping for the whole universe according to policy
    tier_map = _tier_map_for_universe(universe)

    scored: List[Tuple[str,str,Dict,float,Dict]] = []
    removed: List[Tuple] = []
    protected_bucket_small: List[Tuple[str,str,Dict,float,Dict]] = []
    protected_bucket_large: List[Tuple[str,str,Dict,float,Dict]] = []

    for (t, n, f) in universe:
        tier = tier_map.get(t) or _tier_fallback_small_vs_large(f)
        # store tier on feats so downstream code can see it too
        f["liq_tier"] = tier
        s, parts = quick_score(f, mode=mode, xs=xs, tier=tier)
        tags = _tags(f)
        meta = {"parts": parts, "tags": tags, "tier": tier, "avg_dollar_vol_20d": f.get("avg_dollar_vol_20d")}
        row = (t, n, f, s, meta)
        scored.append(row)

    scored.sort(key=lambda x: x[3], reverse=True)

    # protection: keep a slice of protected names (per-tier)
    top_main = scored[:keep]
    borderline = scored[keep:]

    for r in borderline:
        tier = r[4].get("tier")
        if any(tag.endswith("_protect") for tag in r[4]["tags"]):
            if tier == "small":
                protected_bucket_small.append(r)
            else:
                protected_bucket_large.append(r)

    max_rescue = max(0, int(keep * rescue_frac))
    # split rescue capacity roughly across tiers (by proportion present)
    small_ct = sum(1 for (_t,_n,_f,_s,m) in scored if m["tier"]=="small")
    large_ct = len(scored) - small_ct
    small_ct = max(1, small_ct)
    large_ct = max(1, large_ct)
    small_quota = int(max_rescue * (small_ct / (small_ct + large_ct)))
    large_quota = max_rescue - small_quota

    rescued = protected_bucket_small[:small_quota] + protected_bucket_large[:large_quota]

    pre = (top_main + rescued)
    pre.sort(key=lambda x: x[3], reverse=True)
    pre = pre[:keep]

    # ---- tier quotas merge (final stratification) ----
    min_small = int(os.getenv("STAGE1_MIN_SMALL", "0"))
    min_large = int(os.getenv("STAGE1_MIN_LARGE", "0"))
    # clamp impossible combination
    if min_small + min_large > keep:
        over = min_small + min_large - keep
        if min_small >= min_large:
            min_small = max(0, min_small - over)
        else:
            min_large = max(0, min_large - over)

    # If quotas are zero, skip stratification
    if (min_small + min_large) > 0:
        small_sorted = [r for r in pre if r[4]["tier"] == "small"]
        large_sorted = [r for r in pre if r[4]["tier"] == "large"]
        other = [r for r in pre if r[4]["tier"] not in {"small","large"}]  # kept for completeness

        pick_small = small_sorted[:min_small]
        pick_large = large_sorted[:min_large]

        # fill remaining slots by best available regardless of tier
        remainder = keep - len(pick_small) - len(pick_large)
        # FIX: avoid set() on tuples containing dicts (unhashable). Compare by ticker.
        taken_tickers = {r[0] for r in (pick_small + pick_large)}
        pool = [r for r in pre if r[0] not in taken_tickers]
        tail = pool[:remainder]
        pre_topK = (pick_small + pick_large + tail)
        pre_topK.sort(key=lambda x: x[3], reverse=True)
    else:
        pre_topK = pre

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
                            meta.get("tier"), f.get("avg_dollar_vol_20d"),
                            ";".join(meta["tags"]), reason))

    # write CSVs (optional for speed)
    if write_csv:
        try:
            path_kept = os.path.join(log_dir, "stage1_kept.csv")
            with open(path_kept, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ticker","company","score","tier","avg_dollar_vol_20d",
                            "price","vsSMA50","vsSMA200","RSI14","ATR%","r60",
                            "vol_vs20","drawdown%","tags"])
                for (t, n, feats, s, meta) in pre_topK:
                    w.writerow([t, n, f"{s:.2f}", meta.get("tier"), feats.get("avg_dollar_vol_20d"),
                                feats.get("price"), feats.get("vsSMA50"), feats.get("vsSMA200"),
                                feats.get("RSI14"), feats.get("ATRpct"), feats.get("r60"),
                                feats.get("vol_vs20"), feats.get("drawdown_pct"),
                                ";".join(meta["tags"])] )
            log.info(f"[Stage1] kept={len(pre_topK)} -> {path_kept}")

            path_removed = os.path.join(log_dir, "stage1_removed.csv")
            with open(path_removed, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ticker","company","score","tier","avg_dollar_vol_20d",
                            "price","vsSMA50","vsSMA200","RSI14","ATR%","r60",
                            "vol_vs20","drawdown%","tags","reason"])
                for row in removed:
                    w.writerow(row)
            log.info(f"[Stage1] removed={len(removed)} -> {path_removed}")
        except Exception as e:
            log.warning(f"[Stage1] CSV logging failed: {e!r}")

    return pre_topK, scored, removed
