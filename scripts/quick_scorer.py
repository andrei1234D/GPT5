# scripts/quick_scorer.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, math, csv, logging, time
import numpy as np
from features import build_features

"""
ENV knobs this scorer honors (new ones marked ★):

# Weighting (tier-aware)
QS_W_TREND_SMALL,  QS_W_MOMO_SMALL,  QS_W_STRUCT_SMALL,  QS_W_RISK_SMALL
QS_W_TREND_LARGE,  QS_W_MOMO_LARGE,  QS_W_STRUCT_LARGE,  QS_W_RISK_LARGE
QS_PE_WEIGHT, QS_PE_WEIGHT_SMALL, QS_PE_WEIGHT_LARGE


# Anchor & structure shaping (tier-aware via *_SMALL / *_LARGE)
QS_USE_FVA=1|0
QS_FVA_PEN_MAX / QS_FVA_PEN_MAX_SMALL / QS_FVA_PEN_MAX_LARGE
QS_FVA_BONUS_MAX / QS_FVA_BONUS_MAX_SMALL / QS_FVA_BONUS_MAX_LARGE
QS_FVA_KO_PCT / QS_FVA_KO_PCT_SMALL / QS_FVA_KO_PCT_LARGE
QS_STRUCT_PREM_CAP / QS_STRUCT_PREM_CAP_SMALL / QS_STRUCT_PREM_CAP_LARGE

# Anticipation (lead setup) bias
QS_SETUP_ATR_MAX (6.0), QS_SETUP_VS50_MIN (-4), QS_SETUP_VS50_MAX (10),
QS_SETUP_VS200_MIN (-2), QS_SETUP_VOL_ABS_MAX (60),
QS_SETUP_RSI_LO (45), QS_SETUP_RSI_HI (62),
QS_SETUP_E50S_MIN (0.5), QS_SETUP_E50S_MAX (8),
QS_SETUP_CAP_SMALL (12), QS_SETUP_CAP_LARGE (14)

# Momentum “chase” penalty
QS_MOMO_CHASE_KNEE (35), QS_MOMO_CHASE_SLOPE (0.6), QS_MOMO_CHASE_PEN_MAX (15)

# Valuation overlay into 'struct'
QS_VAL_OVERLAY=1|0
QS_VAL_OVERLAY_MAX (12)

# ★ Overheat-but-allow-probe path
ALLOW_BLOWOFF_PROBE=1|0          (default 1)
PROBE_MIN_EV=4                   (ATR% threshold to allow probes)
PROBE_MAX_VOL_SPIKE=150
OVERHEAT_A_RSI=78  OVERHEAT_B_RSI=83  OVERHEAT_C_RSI=87
OVERHEAT_A_VS50=20 OVERHEAT_B_VS50=35 OVERHEAT_C_VS50=50
PROBE_ADD_BACK_A=4  PROBE_ADD_BACK_B=7  PROBE_ADD_BACK_C=10
PROBE_STRUCT_SHARE=0.45

# ★ Momentum-carry discount on FVA-KO penalty
KO_MOMO_DISCOUNT=0.50
KO_MOMO_DISCOUNT_STRONG=0.35

# Stage-1 CSV dumps
STAGE1_WRITE_TOPN_CSV=1|0  STAGE1_TOPN_CSV (2000)  STAGE1_TOPN_PATH
STAGE1_WRITE_CSV=1|0
"""

_TIER_THR_LARGE_USD = None
_TIER_THR_MID_USD   = None
YF_CHUNK_SIZE   = int(os.getenv("YF_CHUNK_SIZE", "80"))
YF_MAX_RETRIES  = int(os.getenv("YF_MAX_RETRIES", "5"))
YF_RETRY_SLEEP  = float(os.getenv("YF_RETRY_SLEEP", "2.5"))


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

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return float(default)

def _env_tiered(base: str, tier: str, default: float) -> float:
    """Tier-aware env lookup: prefer BASE_SMALL/BASE_LARGE, fall back to BASE, else default."""
    tkey = f"{base}_{'SMALL' if tier == 'small' else 'LARGE'}"
    try:
        if os.getenv(tkey) is not None:
            return float(os.getenv(tkey))
        if os.getenv(base) is not None:
            return float(os.getenv(base))
    except Exception:
        pass
    return float(default)

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
    """Infer LARGE/MID cutoffs from cross-section percentiles of avg_dollar_vol_20d."""
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
    """Populate module-level thresholds using env overrides if provided, else auto-derive."""
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
    if a >= _TIER_THR_LARGE_USD: return "large"
    if a >= _TIER_THR_MID_USD:   return "mid"
    return "small"

def _tier_fallback_small_vs_large(feats: Dict) -> str:
    """Legacy fallback: classify small vs large by cap/liquidity/price when no policy applies."""
    def _safe_float(x):
        try: return float(x)
        except Exception: return None
    cap = _safe_float(feats.get("market_cap"))
    liq = _safe_float(feats.get("avg_dollar_vol_20d"))
    price = _safe_float(feats.get("price"))
    cap_cut = float(os.getenv("QS_SMALL_CAP_MAX", "2000000000"))   # $2B default
    liq_cut = float(os.getenv("QS_SMALL_LIQ_MAX", "10000000"))     # $10M avg $vol
    px_cut  = float(os.getenv("QS_SMALL_PRICE_MAX", "12"))
    if cap is not None:   return "small" if cap <= cap_cut else "large"
    if liq is not None:   return "small" if liq <= liq_cut else "large"
    if price is not None: return "small" if price <= px_cut else "large"
    return "large"

def _tier_map_for_universe(universe: List[Tuple[str,str,Dict]]) -> Dict[str, str]:
    """Return {ticker: 'small'|'large'} by policy."""
    policy = (os.getenv("TIER_POLICY", "THRESH") or "THRESH").upper()
    backfill_unknown = (os.getenv("TIER_BACKFILL_UNKNOWN_AS", "small") or "small").lower()
    if backfill_unknown not in {"small","large"}: backfill_unknown = "small"
    tier_map: Dict[str, str] = {}
    if policy == "TOPK_ADV":
        pairs = []
        for (t, _n, f) in universe:
            adv = f.get("avg_dollar_vol_20d")
            try: adv = float(adv) if adv is not None else None
            except Exception: adv = None
            pairs.append((t, adv))
        valid = [(t, a) for (t, a) in pairs if a is not None]
        valid.sort(key=lambda x: x[1], reverse=True)
        K = int(os.getenv("TIER_TOPK_LARGE", "500"))
        large_set = set([t for (t, _a) in valid[:K]])
        for (t, a) in pairs:
            if t in large_set: tier_map[t] = "large"
            else: tier_map[t] = backfill_unknown if a is None else "small"
        log.info(f"[Stage1] TIER_POLICY=TOPK_ADV: large={len(large_set)} (K={K}), backfill_unknown_as={backfill_unknown}")
        return tier_map
    _compute_and_set_tier_thresholds(universe)
    if _TIER_THR_LARGE_USD is not None and _TIER_THR_MID_USD is not None:
        log.info(f"[Stage1] TIER_POLICY=THRESH thresholds: LARGE≥{_TIER_THR_LARGE_USD:,.0f} USD, MID≥{_TIER_THR_MID_USD:,.0f} USD")
    for (t, _n, f) in universe:
        pre = (f.get("liq_tier") or "").upper()
        if pre in {"LARGE","MID","SMALL"}:
            tier_map[t] = "large" if pre in {"LARGE","MID"} else "small"
        else:
            tri = _tier_from_thresholds(f)
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

# --- lead/anticipation setup points (favor coils likely to break) ---
def _lead_setup_points(vs50, vs200, r60, rsi, atr, v20, vsE50, vsE200, e50s, tier: str) -> float:
    """
    Reward tight, aligned bases likely to break soon (not already extended).
    Returns capped points (±cap).
    """
    def g(name, default):
        return _env_float(name, default)

    atr_max   = g("QS_SETUP_ATR_MAX",   6.0)
    vs50_max  = g("QS_SETUP_VS50_MAX", 10.0)
    vs50_min  = g("QS_SETUP_VS50_MIN", -4.0)
    vs200_min = g("QS_SETUP_VS200_MIN",-2.0)
    vabs_max  = g("QS_SETUP_VOL_ABS_MAX", 60.0)
    rsi_lo    = g("QS_SETUP_RSI_LO",   45.0)
    rsi_hi    = g("QS_SETUP_RSI_HI",   62.0)
    e50s_lo   = g("QS_SETUP_E50S_MIN", 0.5)
    e50s_hi   = g("QS_SETUP_E50S_MAX", 8.0)

    pts = 0.0
    # coil / tightness near trend
    if (vs200 is not None and vs200 >= vs200_min) and (vs50_min <= vs50 <= vs50_max):
        if (atr is not None and atr <= atr_max) and (v20 is not None and abs(v20) <= vabs_max):
            pts += 10.0
    # alignment + early RSI
    if (vsE50 is not None and vsE200 is not None):
        if vsE50 > -5 and vsE200 >= -3 and (rsi is not None and rsi_lo <= rsi <= rsi_hi):
            pts += 6.0
    # early slope
    if e50s is not None and e50s_lo <= e50s <= e50s_hi:
        pts += 4.0
    # haircut if already chase-y
    if (vs50 is not None and vs50 > 20) or (rsi is not None and rsi > 75):
        pts -= 8.0

    cap = _env_float(
        "QS_SETUP_CAP_SMALL" if tier == "small" else "QS_SETUP_CAP_LARGE",
        _env_float("QS_SETUP_CAP", 12.0 if tier == "small" else 14.0)
    )
    return clamp(pts, -cap * 0.5, cap)

# --- overheat / blowoff probe --------------------------------------
def _overheat_level(rsi: Optional[float], vs50: Optional[float]) -> int:
    """0 = none, 1/2/3 = A/B/C levels of overheat."""
    if rsi is None or vs50 is None: return 0
    A_RSI = _env_float("OVERHEAT_A_RSI", 78.0)
    B_RSI = _env_float("OVERHEAT_B_RSI", 83.0)
    C_RSI = _env_float("OVERHEAT_C_RSI", 87.0)
    A_V50 = _env_float("OVERHEAT_A_VS50", 20.0)
    B_V50 = _env_float("OVERHEAT_B_VS50", 35.0)
    C_V50 = _env_float("OVERHEAT_C_VS50", 50.0)
    if rsi >= C_RSI and vs50 >= C_V50: return 3
    if rsi >= B_RSI and vs50 >= B_V50: return 2
    if rsi >= A_RSI and vs50 >= A_V50: return 1
    return 0

def _apply_probe_path(rsi, vs50, v20, atr, struct, risk_pen) -> Tuple[float, float, bool, int]:
    """
    If ALLOW_BLOWOFF_PROBE and conditions OK (ATR high enough, vol spike not crazy),
    give back part of the penalties to keep a controlled 'probe' alive.
    Returns (struct, risk_pen, probe_ok, level)
    """
    allow = (os.getenv("ALLOW_BLOWOFF_PROBE", "1").lower() in {"1","true","yes"})
    lvl = _overheat_level(rsi, vs50)
    if not allow or lvl == 0:
        return struct, risk_pen, False, 0

    ev_ok   = (atr is not None) and (atr >= _env_float("PROBE_MIN_EV", 4.0))
    vol_ok  = (v20 is None) or (v20 <= _env_float("PROBE_MAX_VOL_SPIKE", 150.0))
    if not (ev_ok and vol_ok):
        return struct, risk_pen, False, lvl

    # distribute add-back between struct and risk
    add_A = _env_float("PROBE_ADD_BACK_A", 4.0)
    add_B = _env_float("PROBE_ADD_BACK_B", 7.0)
    add_C = _env_float("PROBE_ADD_BACK_C", 10.0)
    add = add_A if lvl == 1 else add_B if lvl == 2 else add_C
    share = _env_float("PROBE_STRUCT_SHARE", 0.45)   # remainder goes to risk
    struct += add * share
    risk_pen += add * (1.0 - share)  # risk_pen is negative; adding back makes it less punitive
    return struct, risk_pen, True, lvl

def _tags(feats: Dict) -> List[str]:
    """Lightweight protections & risk flags."""
    t: List[str] = []
    rsi = safe(feats.get("RSI14"), None)
    vs200 = safe(feats.get("vsSMA200"), 0.0)
    vs50  = safe(feats.get("vsSMA50"), 0.0)
    r60   = safe(feats.get("r60"), 0.0)
    d20   = safe(feats.get("d20"), 0.0)
    v20   = safe(feats.get("vol_vs20"), 0.0)
    atr   = safe(feats.get("ATRpct"), 0.0)
    is20h = bool(feats.get("is_20d_high"))

    if r60 >= 20 and vs200 >= 0: t.append("RS_trend_protect")
    if is20h and (rsi is not None and 55 <= rsi <= 85): t.append("breakout_protect")
    if (0 <= vs50 <= 12) and (5 <= r60 <= 40) and (rsi is not None and 45 <= rsi <= 70): t.append("pullback_protect")
    if v20 >= 180 and rsi is not None and rsi >= 78 and d20 >= 12: t.append("pump_risk")
    if v20 <= -60: t.append("low_liquidity_today")

    # distribution / stall hints
    try:
        macd = float(feats.get("MACD_hist")) if feats.get("MACD_hist") is not None else None
    except Exception:
        macd = None
    if (rsi is not None and 55 <= rsi <= 70) and (vs50 > 0) and (((v20 or 0) <= -20) or (macd is not None and macd < 0)):
        t.append("stall_risk")

    # coil/setup protector
    try: setup_atr = _env_float("QS_SETUP_ATR_MAX", 6.0)
    except Exception: setup_atr = 6.0
    if (-4 <= vs50 <= 10) and (vs200 >= -2) and (rsi is not None and 45 <= rsi <= 62) and (atr <= setup_atr) and (-60 <= v20 <= 40):
        t.append("setup_protect")

    # overheat flag (for visibility)
    lvl = _overheat_level(rsi, vs50)
    if lvl >= 1:
        t.append(f"overheat_L{lvl}")
    return t

def _tier_params(tier: str, mode: str, pe_weight_base: float):
    """Return (weights, pe_weight, atr_soft, vol_soft, dd_soft)."""
    def _w(env_name: str, fallback: float) -> float:
        try: return float(os.getenv(env_name, str(fallback)))
        except Exception: return fallback
    if tier == "small":
        wt_trend  = _w("QS_W_TREND_SMALL", 0.44)
        wt_momo   = _w("QS_W_MOMO_SMALL",  0.30)
        wt_struct = _w("QS_W_STRUCT_SMALL",0.14)
        wt_risk   = _w("QS_W_RISK_SMALL",  0.12)
        weights = dict(trend=wt_trend, momo=wt_momo, struct=wt_struct, risk=wt_risk)
        pe_w = _env_float("QS_PE_WEIGHT_SMALL", pe_weight_base)
        if mode == "strict":   atr_soft, vol_soft, dd_soft = 5.0, 200, -42
        elif mode == "normal": atr_soft, vol_soft, dd_soft = 5.5, 220, -45
        else:                  atr_soft, vol_soft, dd_soft = 6.0, 240, -48
    else:
        wt_trend  = _w("QS_W_TREND_LARGE", 0.50)
        wt_momo   = _w("QS_W_MOMO_LARGE",  0.33)
        wt_struct = _w("QS_W_STRUCT_LARGE",0.11)
        wt_risk   = _w("QS_W_RISK_LARGE",  0.04)
        weights = dict(trend=wt_trend, momo=wt_momo, struct=wt_struct, risk=wt_risk)
        pe_w = _env_float("QS_PE_WEIGHT_LARGE", pe_weight_base + 0.02)
        if mode == "strict":   atr_soft, vol_soft, dd_soft = 5.5, 240, -42
        elif mode == "normal": atr_soft, vol_soft, dd_soft = 6.0, 280, -45
        else:                  atr_soft, vol_soft, dd_soft = 6.5, 300, -50
    return weights, pe_w, atr_soft, vol_soft, dd_soft

# ---- P/E tilt ----
def _pe_tilt_points(pe: float|None, rsi: float|None, vol_vs20: float|None) -> float:
    if pe is None or pe <= 0: return 0.0
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
    """Fast, resilient first-pass score with anticipation bias, overheat probe, and tier-aware anchors."""
    tier = (tier or _tier_fallback_small_vs_large(feats))

    pe_weight_base = _env_float("QS_PE_WEIGHT", 0.06)
    weights, pe_weight, atr_soft, vol_soft, dd_soft = _tier_params(tier, mode, pe_weight_base)

    w_trend, w_momo, w_struct, w_risk = weights["trend"], weights["momo"], weights["struct"], weights["risk"]
    rem = max(1e-9, (w_trend + w_momo + w_struct + w_risk))
    scale = max(0.0, 1.0 - pe_weight) / rem
    w_trend  *= scale; w_momo *= scale; w_struct *= scale; w_risk *= scale
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
    vsE50  = safe(feats.get("vsEMA50"), None)
    vsE200 = safe(feats.get("vsEMA200"), None)
    e50s   = safe(feats.get("EMA50_slope_5d"), None)

    pe_val = None
    for k in ("val_PE", "PE", "pe", "pe_hint"):
        if feats.get(k) is not None:
            try: pe_val = float(feats.get(k))
            except Exception: pass

    # --- In-motion bonus: Reward near-breakout momentum with strong structure ---
    in_motion_bonus = 0
    if vs50 >= 10 and e50s is not None and e50s >= 2.0 and dd >= -20:
        if vs50 >= 14 and e50s >= 3:
            in_motion_bonus = 12
        elif vs50 >= 12 and e50s >= 2.5:
            in_motion_bonus = 9
        else:
            in_motion_bonus = 7


    # cross-sectional z
    use_xs = xs is not None and os.getenv("QS_USE_XS", "1").lower() in {"1", "true", "yes"}
    if use_xs:
        z_vs200 = _z(xs, "vsSMA200", vs200); z_vs50  = _z(xs, "vsSMA50",  vs50)
        z_r60   = _z(xs, "r60",      r60);   z_r120  = _z(xs, "r120",     r120)
        z_d20   = _z(xs, "d20",      d20);   z_atr   = _z(xs, "ATRpct",   atr)
        z_v20   = _z(xs, "vol_vs20", v20);   z_vsE50 = _z(xs, "vsEMA50",  vsE50)
        z_vsE200= _z(xs, "vsEMA200", vsE200);z_e50s  = _z(xs, "EMA50_slope_5d", e50s)
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

           # trend (EMA alignment + RSI band)
    rsi_band = _rsi_band_val(rsi)

    trend_raw = (
        0.40 * clamp(z_vs200/4.0, -1, 1) +
        0.25 * clamp(z_vs50/4.0,  -1, 1) +
        0.15 * clamp(z_vsE200/4.0,-1, 1) +
        0.10 * clamp(z_vsE50/4.0, -1, 1) +
        0.10 * clamp(macd/1.5,    -1, 1)
    ) * 100.0 + 8.0 * rsi_band

    trend = trend_raw

    # 🔥 Dynamic dampener: reduce weight of trend if stock is already extended
    if rsi is not None and vs50 is not None:
        if rsi > 70 and vs50 > 20:
            if rsi < 75 and vs50 < 35:
                damp = 0.75   # mildly extended
            elif rsi < 80 and vs50 < 45:
                damp = 0.65   # clearly extended
            else:
                damp = 0.50   # very extended / blowoff
            trend *= damp


    early_turn = 0.0
    if rsi is not None and 45 <= rsi <= 60 and e50s is not None and e50s > 1.5 and vs200 > -10 and vs200 < 10:
        early_turn = 6.0
    
    # momentum (EMA50 slope) + chase penalty
    momo_raw = (
        0.36 * clamp(z_r60/4.0,   -1, 1) +
        0.27 * clamp(z_r120/4.0,  -1, 1) +
        0.22 * clamp(z_d20/4.0,   -1, 1) +
        0.15 * clamp(z_e50s/4.0,  -1, 1)
    ) * 100.0

    momo = momo_raw
    momo += early_turn
    try:
        knee  = _env_float("QS_MOMO_CHASE_KNEE", 35.0)
        slope = _env_float("QS_MOMO_CHASE_SLOPE", 0.6)
        cap   = _env_float("QS_MOMO_CHASE_PEN_MAX", 15.0)
    except Exception:
        knee, slope, cap = 35.0, 0.6, 15.0




    # classic chase penalty
    if r60 is not None and r60 > knee:
        penalty = min(cap, (r60 - knee) * slope)
        momo -= penalty

    # 🔥 Dynamic dampener for extended momo
    if rsi is not None and rsi > 72 and vs50 > 25:
        damp = 0.7 if rsi < 78 else 0.55  # moderate vs. very extended
        momo *= damp



    # acceleration & continuation
    try:
        if use_xs:
            accel_z = clamp(((z_r60 - z_r120) / 2.0), -1.0, 1.0)
        else:
            accel_z = clamp(((r60 - r120) / 20.0), -1.0, 1.0)
    except Exception:
        accel_z = 0.0
    momo += 8.0 * accel_z  # reward acceleration a bit

    continuation = 0.0
    if (vsE50 is not None and 3.0 <= vsE50 <= 18.0) and (vsE200 is not None and vsE200 >= -2.0) and \
       (rsi is not None and 52.0 <= rsi <= 68.0) and (e50s is not None and e50s > 0.0):
        if (v20 is None) or (-20.0 <= v20 <= 180.0):
            continuation = 8.0
    momo += continuation

    # structure (AVWAP/MAs, FVA, valuation overlay)
    struct = 0.0
    if sma50 and avwap:
        if px > sma50 > avwap: struct += 12.0
        elif px < sma50 < avwap: struct -= 8.0
        else: struct += 3.0 if px > avwap else -2.0
    if avwap:
        prem = (px / avwap) - 1.0
        if prem < 0: struct += clamp(abs(prem) * 20.0, 0.0, 6.0)
        else:
            prem_cap = _env_tiered("QS_STRUCT_PREM_CAP", tier, 8.0)
            struct -= clamp(prem * 25.0, 0.0, prem_cap)
    if (vsE50 is not None and vsE200 is not None):
        if vsE50 > 0 and vsE200 > 0: struct += 3.0
        elif vsE50 < 0 and vsE200 < 0: struct -= 2.0

    # FVA discounts/penalties with momentum-carry discount on KO
    fva = None
    if os.getenv("QS_USE_FVA", "1").lower() in {"1","true","yes"}:
        pen   = _env_tiered("QS_FVA_PEN_MAX",   tier, 12.0)
        bonus = _env_tiered("QS_FVA_BONUS_MAX", tier, 6.0)
        ko_pct= _env_tiered("QS_FVA_KO_PCT",    tier, 35.0)

        fva = safe(
            feats.get("fva_hint") if feats.get("fva_hint") is not None else feats.get("FVA_HINT"),
            None
        )

        if (fva is not None) and (px is not None):
            disc = (fva - px) / max(abs(px), 1e-9) * 100.0

            # ✅ Undervalued → bonus
            if disc > 0:
                struct += clamp(disc/20.0, 0.0, 1.0) * bonus

            # 🚨 Overvalued zone → handle in tiers
            else:
                gap = (px - fva) / max(abs(fva), 1e-9) * 100.0

                if gap <= 15:
                    # Slight premium → treat as early breakout setup
                    struct += clamp((15 - gap)/15.0, 0.0, 1.0) * (bonus * 0.5)

                elif (r60 is not None and r60 >= 30):
                    # Breakout in progress with strong trend support
                    struct += bonus * 0.3

                elif gap >= ko_pct:
                    # Heavy KO penalty for runaway price with no momentum support
                    ko_pen = min(40.0, (gap - ko_pct) * 0.6)

                    strong_trend = (r60 is not None and r60 >= 30) and (vs50 is not None and vs50 >= 15)
                    discount = 1.0
                    if strong_trend and accel_z > 0.10:
                        discount = _env_float("KO_MOMO_DISCOUNT", 0.50)
                        if accel_z >= 0.35:
                            discount = _env_float("KO_MOMO_DISCOUNT_STRONG", 0.35)

                    struct -= ko_pen * discount

                else:
                    # Mild overvaluation → soft penalty
                    struct -= clamp(abs(disc)/20.0, 0.0, 1.0) * pen


    # Valuation overlay (optional)
    if os.getenv("QS_VAL_OVERLAY", "1").lower() in {"1","true","yes"}:
        cap_overlay = _env_float("QS_VAL_OVERLAY_MAX", 12.0)
        pe     = safe(feats.get("val_PE")        if feats.get("val_PE")        is not None else feats.get("PE"), None)
        peg    = safe(feats.get("val_PEG")       if feats.get("val_PEG")       is not None else feats.get("PEG"), None)
        fcfy   = safe(feats.get("val_FCF_YIELD") if feats.get("val_FCF_YIELD") is not None else feats.get("FCF_YIELD"), None)
        ev_eb  = safe(feats.get("val_EV_EBITDA") if feats.get("val_EV_EBITDA") is not None else feats.get("EV_EBITDA"), None)
        ev_rev = safe(feats.get("val_EV_REV")    if feats.get("val_EV_REV")    is not None else feats.get("EV_REV"), None)
        ps     = safe(feats.get("val_PS")        if feats.get("val_PS")        is not None else feats.get("PS"), None)
        pts = 0.0
        if pe and pe > 0:
            if pe <= 10: pts += 18
            elif pe <= 12: pts += 14
            elif pe <= 18 and (safe(feats.get("REL_STRENGTH"),0) >= 1 or safe(feats.get("VALUATION_HISTORY"),0) >= 0): pts += 8
            elif 30 <= pe <= 40 and ((safe(feats.get("RSI14"),0) >= 75) or (safe(feats.get("vsSMA50"),0) >= 20)): pts -= 4
            elif pe >= 50:
                hot = (safe(feats.get("RSI14"),0) >= 80 and safe(feats.get("vol_vs20"),0) >= 200)
                pts -= (15 if hot else 10)
            pts = clamp(pts, -15, 20)
        if peg is not None:   pts += (6 if peg <= 1 else 4 if peg <= 1.5 else 1 if peg <= 2.5 else -4)
        if fcfy is not None:  pts += (10 if fcfy >= 6 else 6 if fcfy >= 3 else 2 if fcfy >= 1 else 0 if fcfy >= 0 else -6)
        if ev_eb is not None: pts += (8 if ev_eb <= 10 else 4 if ev_eb <= 15 else 0 if ev_eb <= 25 else -3 if ev_eb <= 30 else -6)
        if ev_rev is not None:pts += (8 if ev_rev <= 2 else 5 if ev_rev <= 5 else 0 if ev_rev <= 10 else -4 if ev_rev <= 20 else -8)
        if ps is not None:    pts += (6 if ps <= 2 else 3 if ps <= 5 else 0 if ps <= 10 else -3 if ps <= 15 else -5)
        struct += clamp(pts, -cap_overlay, cap_overlay)

    # anticipation boost
    struct += _lead_setup_points(vs50, vs200, r60, rsi, atr, v20, vsE50, vsE200, e50s, tier)

    # risk (soft, tier-aware)
    risk_pen = 0.0
    if use_xs:
        risk_pen -= clamp(max(0.0, z_atr), 0.0, 3.0) * 3.5
        risk_pen -= clamp(max(0.0, z_v20), 0.0, 3.0) * 2.0
    else:
        if atr > atr_soft:  risk_pen -= min(12.0, (atr - atr_soft) * 1.5)
        if v20 > vol_soft:  risk_pen -= min(8.0, (v20 - vol_soft) / 40.0)
    if dd < dd_soft: risk_pen -= min(8.0, (abs(dd) - abs(dd_soft)) / 4.0)
    if rsi is not None and rsi >= 85: risk_pen -= 8.0
    if rsi is not None and rsi >= 88 and d20 >= 150: risk_pen -= 10.0
    if rsi is not None and rsi >= 80 and v20 >= 200: risk_pen -= 6.0
    if tier == "small":
        liq_min = _env_float("QS_MIN_DOLLAR_VOL_20D", 2_000_000.0)
        liq = safe(feats.get("avg_dollar_vol_20d"), None)
        if liq is not None and liq < liq_min:
            risk_pen -= clamp((liq_min - liq) / max(liq_min, 1.0), 0.0, 1.0) * 10.0
        if v20 <= -40:
            risk_pen -= clamp(abs(v20 + 40.0) / 40.0, 0.0, 1.0) * 6.0

    # stall / deceleration penalties
    try:
        if 'accel_z' in locals() and accel_z < -0.25:
            risk_pen -= min(8.0, (abs(accel_z) - 0.25) * 20.0)
    except Exception:
        pass
    try:
        if e50s is not None and e50s <= 0.0:
            risk_pen -= 5.0
    except Exception:
        pass
    try:
        macd_loc = float(macd) if macd is not None else None
        if (macd_loc is not None and macd_loc < 0.0) and (rsi is not None and 55.0 <= rsi <= 75.0) and (vs50 is not None and vs50 > 0.0):
            risk_pen -= min(10.0, abs(macd_loc) * 2.0)
    except Exception:
        pass
    try:
        if (rsi is not None and 55.0 <= rsi <= 70.0) and (vs50 is not None and vs50 > 0.0) and (v20 is not None and v20 <= -20.0):
            risk_pen -= 4.0
    except Exception:
        pass

    # ★ Overheat probe: give back some points to keep “probe-sized” trades alive
    struct, risk_pen, probe_ok, probe_lvl = _apply_probe_path(rsi, vs50, v20, atr, struct, risk_pen)

    # P/E overlay
    pe_points_raw = _pe_tilt_points(pe_val, rsi, v20)  # [-20..+20]
    pe_score = clamp(pe_points_raw / 20.0, -1.0, 1.0) * 100.0

    score = (
    w_trend  * trend +
    w_momo   * momo +
    w_struct * struct +
    w_risk   * risk_pen +
    in_motion_bonus +
    w_pe     * pe_score
)

    return score, {
        "trend_raw": trend_raw, "trend": trend,
        "momo_raw": momo_raw,   "momo": momo,
        "struct": struct, "risk_pen": risk_pen,
        "pe_tilt_pts": pe_points_raw, "pe_weight": w_pe, "tier": tier,
        "fva_hint": fva, "price": px,
        "accel_z": accel_z,
        "probe_ok": bool(probe_ok),
        "probe_lvl": int(probe_lvl),
    }


def rank_stage1(
    universe: List[Tuple[str, str, Dict]],
    keep: int = 200,
    mode: str = None,
    rescue_frac: float = None,
    log_dir: str = "data"
) -> Tuple[List[Tuple[str,str,Dict,float,Dict]], List[Tuple[str,str,Dict,float,Dict]], List[Tuple]]:
    """Run Stage-1 quick pass with tiering, probe-aware protection rescue and CSV logs."""
    mode = (mode or os.getenv("STAGE1_MODE", "loose")).lower()
    keep = int(os.getenv("STAGE1_KEEP", str(keep)))
    rescue_frac = float(os.getenv("STAGE1_RESCUE_FRAC", str(rescue_frac if rescue_frac is not None else 0.15)))
    write_csv = os.getenv("STAGE1_WRITE_CSV", "1").lower() in {"1","true","yes"}
    os.makedirs(log_dir, exist_ok=True)

    feat_list = [f for (_t, _n, f) in universe]
    xs_keys = ["vsSMA50","vsSMA200","d20","r60","r120","ATRpct","vol_vs20","vsEMA50","vsEMA200","EMA50_slope_5d"]
    xs = _xs_stats(feat_list, xs_keys)

    tier_map = _tier_map_for_universe(universe)

    scored: List[Tuple[str,str,Dict,float,Dict]] = []
    removed: List[Tuple] = []
    protected_bucket_small: List[Tuple[str,str,Dict,float,Dict]] = []
    protected_bucket_large: List[Tuple[str,str,Dict,float,Dict]] = []

    for (t, n, f) in universe:
        tier = tier_map.get(t) or _tier_fallback_small_vs_large(f)
        f["liq_tier"] = tier

        # --- base score ---
        base_score, parts = quick_score(f, mode=mode, xs=xs, tier=tier)

        # --- enhancer applied ---
        final_score, bonus, reason_log, phase = enhance_score_for_strong_buy(f, base_score, parts)


        tags = _tags(f)
        if parts.get("probe_ok"):
            tags.append(f"probe_ok_L{parts.get('probe_lvl', 0)}")

        meta = {
            "parts": parts, "tags": tags, "tier": tier,
            "avg_dollar_vol_20d": f.get("avg_dollar_vol_20d"),
            "enhancer_bonus": bonus,
            "base_score": base_score,
            "final_score": final_score,
            "enhancer_reason": reason_log,
            "phase": phase
        }

        row = (t, n, f, final_score, meta)
        scored.append(row)

    scored.sort(key=lambda x: x[3], reverse=True)

    # rescue logic unchanged...
    top_main = scored[:keep]
    borderline = scored[keep:]
    for r in borderline:
        tier = r[4].get("tier")
        if any(tag.endswith("_protect") or tag.startswith("probe_ok") for tag in r[4]["tags"]):
            (protected_bucket_small if tier == "small" else protected_bucket_large).append(r)

    max_rescue = max(0, int(keep * rescue_frac))
    small_ct = sum(1 for (_t,_n,_f,_s,m) in scored if m["tier"]=="small")
    large_ct = len(scored) - small_ct
    small_ct = max(1, small_ct); large_ct = max(1, large_ct)
    small_quota = int(max_rescue * (small_ct / (small_ct + large_ct)))
    large_quota = max_rescue - small_quota
    rescued = protected_bucket_small[:small_quota] + protected_bucket_large[:large_quota]

    pre = (top_main + rescued)
    pre.sort(key=lambda x: x[3], reverse=True)
    pre = pre[:keep]

    # --- CSV logs ---
    if write_csv:
        try:
            path_kept = os.path.join(log_dir, "stage1_kept.csv")
            with open(path_kept, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ticker","company","final_score","enhancer_bonus","tier","avg_dollar_vol_20d",
                    "price","vsSMA50","vsSMA200","RSI14","ATR%","r60",
                    "vol_vs20","drawdown%","tags"
                ])
                for (t, n, feats, s, meta) in pre:
                    w.writerow([
                        t, n, f"{s:.2f}",
                        f"{meta.get('enhancer_bonus', 0.0):.2f}",
                        meta.get("tier"), feats.get("avg_dollar_vol_20d"),
                        feats.get("price"), feats.get("vsSMA50"), feats.get("vsSMA200"),
                        feats.get("RSI14"), feats.get("ATRpct"), feats.get("r60"),
                        feats.get("vol_vs20"), feats.get("drawdown_pct"),
                        ";".join(meta["tags"])
                    ])
            log.info(f"[Stage1] kept={len(pre)} -> {path_kept}")

            path_bonus = os.path.join(log_dir, "stage1_enhancer_bonus.csv")
            with open(path_bonus, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ticker","company","base_score","trend","struct","risk_pen",
                    "enhancer_bonus","final_score","tags"
                ])
                for (t, n, feats, s, meta) in pre:
                    parts = meta.get("parts", {})
                    w.writerow([
                        t, n, f"{meta.get('base_score', 0.0):.2f}",
                        parts.get("trend"), parts.get("struct"), parts.get("risk_pen"),
                        f"{meta.get('enhancer_bonus', 0.0):.2f}", f"{s:.2f}",
                        ";".join(meta["tags"])
                    ])
            log.info(f"[Stage1] enhancer bonuses written -> {path_bonus}")
        except Exception as e:
            log.warning(f"[Stage1] CSV logging failed: {e!r}")

    return pre, scored, removed

def fetch_features_with_retries(universe):
    """Fetch features with retry + sleep guard (to avoid rate-limits)."""
    last_exc = None
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            feats = build_features(universe, batch_size=YF_CHUNK_SIZE)
            if feats:   # success
                return feats
        except Exception as e:
            last_exc = e
            print(f"[WARN] build_features failed (attempt {attempt}/{YF_MAX_RETRIES}): {e}", flush=True)
        if attempt < YF_MAX_RETRIES:
            time.sleep(YF_RETRY_SLEEP * attempt)  # exponential backoff
    raise RuntimeError(f"Failed to fetch features after {YF_MAX_RETRIES} attempts: {last_exc}")


# === CUSTOM MODIFICATION START ===
# Injecting strong buy prioritization into quick score logic

def enhance_score_for_strong_buy(
    feats: Dict[str, float],
    base_score: float,
    parts: Dict
) -> Tuple[float, float, str, str]:
    """
    Returns (enhanced_score, bonus_points, reason_log, phase).
    Enhances early-stage strong buy setups, penalizes mid/late breakouts aggressively.
    """
    bonus = 0.0
    reason_log = []
    phase = "Neutral"  # Default

    # === Align keys with quick_score ===
    vsSMA20        = safe(feats.get("vsSMA20"))
    vsSMA50        = safe(feats.get("vsSMA50"))
    vsSMA200       = safe(feats.get("vsSMA200"))
    vsEMA50        = safe(feats.get("vsEMA50"))
    EMA50_slope_5d = safe(feats.get("EMA50_slope_5d"))
    RSI14          = safe(feats.get("RSI14"))
    r20            = safe(feats.get("r20"))
    r60            = safe(feats.get("r60"))
    MACD_hist      = safe(feats.get("MACD_hist"))
    vol_vs20       = safe(feats.get("vol_vs20"))
    drawdown_pct   = safe(feats.get("drawdown_pct"))
    REL_STRENGTH   = safe(feats.get("REL_STRENGTH"))
    CATALYST_HINT  = str(feats.get("CATALYST_TIMING_HINTS") or "")

    # Internals from quick_score
    trend    = parts.get("trend", 0)
    struct   = parts.get("struct", 0)
    risk_pen = parts.get("risk_pen", 0)

    # === Phase 1: Early setups (reward) ===
    if trend > 60 and struct > 0 and risk_pen > -8:
        phase = "Early"
        if 2 < vsEMA50 < 18 and 52 <= RSI14 <= 68 and EMA50_slope_5d > 1.5:
            bonus += 15; reason_log.append("Early breakout setup")
        if -4 <= vsSMA50 <= 10 and r20 > 5 and vol_vs20 < 40:
            bonus += 10; reason_log.append("Healthy pullback entry")
        if 10 < vsSMA50 < 25 and 55 <= RSI14 <= 70 and EMA50_slope_5d > 1.5:
            bonus += 10; reason_log.append("Continuation (not extended)")
        if r60 < 90 and EMA50_slope_5d > 2 and REL_STRENGTH >= 1:
            bonus += 8; reason_log.append("Early momentum")
        if "EARNINGS_SOON" in CATALYST_HINT or "TECH_BREAKOUT" in CATALYST_HINT:
            bonus += 10; reason_log.append("Catalyst hint")
        if MACD_hist > 0 and drawdown_pct > -10:
            bonus += 5; reason_log.append("Positive MACD with shallow DD")
    else:
        if trend <= 60: reason_log.append("Blocked: weak trend")
        if struct <= 0: reason_log.append("Blocked: no structure")
        if risk_pen <= -8: reason_log.append("Blocked: extended risk_pen")

    # === Phase 2: Mid extension (warning zone) ===
    if RSI14 >= 68 and vsSMA50 >= 15:
        phase = "Mid"
        penalty = 10 + (vsSMA50 // 10) * 3
        bonus -= penalty
        reason_log.append(f"Mid extension penalty -{penalty}")

    # === Phase 3: Late extension (harsh penalties) ===
    if trend >= 85 and RSI14 >= 70 and vsSMA50 >= 25:
        phase = "Late"
        penalty = 20 + ((vsSMA50 // 10) * 5)
        penalty = min(penalty, 45)  # cap
        bonus -= penalty
        reason_log.append(f"Late-stage breakout penalty -{penalty}")

        # soften if structure is very strong
        if struct > 12:
            soft_back = penalty * 0.5
            bonus += soft_back
            reason_log.append(f"Support softened +{soft_back}")

    # === Blowoff / overheat guards ===
    if RSI14 > 72 and vsSMA50 > 30:
        bonus -= 15; reason_log.append("Medium blowoff")
    if RSI14 > 74 and vsSMA50 > 40:
        bonus -= 25; reason_log.append("Heavy blowoff")
    if RSI14 >= 78 and vsSMA50 > 50:
        bonus -= 35; reason_log.append("Parabolic")

    # === Extra guards ===
    if r60 > 150 or r20 > 70:
        bonus -= 15; reason_log.append("Excessive r20/r60")
    if vol_vs20 > 150 and RSI14 > 70:
        bonus -= 10; reason_log.append("Volume + RSI overheat")
    if RSI14 > 65 and MACD_hist < 0:
        bonus -= 8; reason_log.append("Bearish divergence")

    # === Weak base penalty ===
    if EMA50_slope_5d < 1 and vol_vs20 > 60 and MACD_hist < 0:
        bonus -= 8; reason_log.append("Weak base penalty")

    # Clamp to range
    bonus = max(min(bonus, 40), -50)

    return base_score + bonus, bonus, "; ".join(reason_log), phase

# === main entrypoint for Stage-1 ===
if __name__ == "__main__":
    import sys
    import pandas as pd
    from features import build_features
    from universe import load_universe
    from filters import is_garbage, daily_index_filter

    try:
        input_path = sys.argv[sys.argv.index("--input") + 1] if "--input" in sys.argv else "data/universe_clean.csv"
        output_path = sys.argv[sys.argv.index("--output") + 1] if "--output" in sys.argv else "data/stage1_kept.csv"
    except Exception:
        input_path = "data/universe_clean.csv"
        output_path = "data/stage1_kept.csv"

    log.info(f"[Stage1] Input universe: {input_path} → Output: {output_path}")

    # 1) Load universe
    universe = load_universe()
    if not universe:
        log.error("[Stage1] Universe is empty!")
        sys.exit(1)

    # 2) Features
    feats_map = fetch_features_with_retries(universe)
    if not feats_map:
        log.error("[Stage1] Failed to build features")
        sys.exit(1)

    # 3) Trash filter
    kept = []
    for t, name in universe:
        row = feats_map.get(t)
        if not row:
            continue
        feats = row["features"]
        if not is_garbage(feats):
            kept.append((t, name, feats))

    if not kept:
        log.error("[Stage1] All filtered out in trash stage")
        sys.exit(1)
    log.info(f"[Stage1] After trash filter: {len(kept)} remain")

    # 4) Daily index filter
    today_ctx = {"bench_trend": "up", "sector_trend": "up", "breadth50": 55}
    kept2 = [(t, n, f) for (t, n, f) in kept if daily_index_filter(f, today_ctx)]
    if not kept2:
        log.error("[Stage1] All filtered by daily context")
        sys.exit(1)
    log.info(f"[Stage1] After daily index filter: {len(kept2)} remain")

    # 5) Rank
    pre, scored, removed = rank_stage1(
        kept2,
        keep=int(os.getenv("STAGE1_KEEP", "200")),
        mode=os.getenv("STAGE1_MODE", "loose"),
        rescue_frac=float(os.getenv("STAGE1_RESCUE_FRAC", "0.15")),
        log_dir="data"
    )

    # 6) Write kept set
    pd.DataFrame(
        [(t, n, s, m.get("tier")) for (t, n, _f, s, m) in pre],
        columns=["ticker", "company", "score", "tier"]
    ).to_csv(output_path, index=False)
    log.info(f"[Stage1] Final survivors={len(pre)} → {output_path}")
