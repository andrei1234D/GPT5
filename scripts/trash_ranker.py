# scripts/trash_ranker.py
from __future__ import annotations
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import time, random, functools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import os
import logging
import numpy as np

from data_fetcher import fetch_valuations_for_top


"""
New/updated ENV knobs (★ = new here, aligned with quick_scorer):

# Hard filter controls
HARD_DROP_MODE=off|loose|normal|strict
HARD_GRACE_ATR=2.0
HARD_EMA_ROLLOVER=1
TR_EARN_BLACKOUT_DAYS=0

# Early turn detector bands
EARLY_TURN_RSI_LO=47.0  EARLY_TURN_RSI_HI=63.0
EARLY_TURN_MIN_E50_SLOPE=2.0
EARLY_TURN_VS200_LO=-12.0 EARLY_TURN_VS200_HI=8.0
EARLY_TURN_VOL_LO=110.0 EARLY_TURN_VOL_HI=220.0
EARLY_TURN_VS200_PAD=5.0

# Composite weights/stability caps profile via RANKER_PROFILE=A|B|C (or SELECTION_MODE)
# (Base weights in RankerParams; profile tweaks in _apply_profile)

# FVA KO threshold (0 disables)
QS_FVA_KO_PCT=35

# ★ Overheat / probe path (keep hot leaders with controlled haircut)
ALLOW_BLOWOFF_PROBE=1
PROBE_MIN_EV=4                 # ATR% min to allow probe
PROBE_MAX_VOL_SPIKE=150        # vol_vs20 max to allow probe
OVERHEAT_A_RSI=78 OVERHEAT_B_RSI=83 OVERHEAT_C_RSI=87
OVERHEAT_A_VS50=20 OVERHEAT_B_VS50=35 OVERHEAT_C_VS50=50
PROBE_ADD_BACK_A=4 PROBE_ADD_BACK_B=7 PROBE_ADD_BACK_C=10  # points (of 100) given back
PROBE_STRUCT_SHARE=0.45        # fraction to struct (rest to stability)

# ★ Momentum-carry discount for FVA KO penalty
KO_MOMO_DISCOUNT=0.50
KO_MOMO_DISCOUNT_STRONG=0.35

# ATH guard (keep existing)
ATH_GUARD=1
ATH_NEAR_PCT=1.0
ATH_MIN_RSI=80
ATH_MIN_VS50=25
ATH_VOL_RELIEF=60
ATH_SCORE_HAIRCUT=22

RANKER_VERBOSE=1|0
RANKER_LOG_EVERY=500
RANKER_LOG_LEVEL=INFO|DEBUG|...
"""

# ---------- logging setup ----------
def _maybe_configure_logging():
    level_name = os.getenv("RANKER_LOG_LEVEL", "").upper().strip()
    if not level_name:
        return
    level = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_maybe_configure_logging()
logger = logging.getLogger("trash_ranker")


# ---------- utilities ----------
def safe(x, default=None):
    try:
        if x is None:
            return default
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return default
        return xf
    except Exception:
        return default

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def tri_sweetspot(x: Optional[float], lo: float, mid_lo: float, mid_hi: float, hi: float) -> float:
    """
    0..1 score peaking in [mid_lo, mid_hi], linearly rising/falling, 0 outside [lo,hi].
    """
    if x is None:
        return 0.0
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x <= lo or x >= hi:
        return 0.0
    if mid_lo <= x <= mid_hi:
        return 1.0
    if x < mid_lo:
        return (x - lo) / max((mid_lo - lo), 1e-9)
    return (hi - x) / max((hi - mid_hi), 1e-9)

def bool01(x) -> Optional[float]:
    if x is True:
        return 1.0
    if x is False:
        return 0.0
    return None

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return float(default)

def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, None) or "").strip().lower()
    if not v:
        return default
    return v in {"1","true","yes","y","on"}

# ---------- overheat / probe helpers ----------
def _overheat_level(rsi: Optional[float], vs50: Optional[float]) -> int:
    """
    0 = none, 1/2/3 = A/B/C levels of overheat based on RSI and vsSMA50.
    """
    if rsi is None or vs50 is None:
        return 0
    A_RSI = _env_float("OVERHEAT_A_RSI", 78.0)
    B_RSI = _env_float("OVERHEAT_B_RSI", 83.0)
    C_RSI = _env_float("OVERHEAT_C_RSI", 87.0)
    A_V50 = _env_float("OVERHEAT_A_VS50", 20.0)
    B_V50 = _env_float("OVERHEAT_B_VS50", 35.0)
    C_V50 = _env_float("OVERHEAT_C_VS50", 50.0)
    r, v = float(rsi), float(vs50)
    if r >= C_RSI and v >= C_V50: return 3
    if r >= B_RSI and v >= B_V50: return 2
    if r >= A_RSI and v >= A_V50: return 1
    return 0

def _probe_gate(feats: Dict) -> Tuple[bool, int]:
    """
    Check whether the "probe" path should be allowed for this name now.
    Profile-aware:
      - Aggressive (A): permissive on probes
      - Balanced (B): default behavior
      - Conservative (C): mostly blocks probes
    """
    prof = (os.getenv("RANKER_PROFILE", os.getenv("SELECTION_MODE", "B")) or "B").upper()

    if prof == "C":
        # Conservative: no probes at all
        return False, 0

    if not _env_bool("ALLOW_BLOWOFF_PROBE", True):
        return False, 0

    rsi  = safe(feats.get("RSI14"), None)
    vs50 = safe(feats.get("vsSMA50"), None)
    atr  = safe(feats.get("ATRpct"), None)
    v20  = safe(feats.get("vol_vs20"), None)

    lvl = _overheat_level(rsi, vs50)
    if lvl == 0:
        return False, 0

    ev_ok  = (atr is not None) and (atr >= _env_float("PROBE_MIN_EV", 4.0))
    vol_ok = (v20 is None) or (v20 <= _env_float("PROBE_MAX_VOL_SPIKE", 150.0))

    # Aggressive: looser probe rules
    if prof == "A":
        ev_ok = True   # ignore ATR floor
        vol_ok = True  # ignore volume spike cap

    return (ev_ok and vol_ok), lvl


# ---------- hard filters ----------
@dataclass
class HardFilter:
    # base thresholds
    min_price: float = 3.0
    max_atr_pct: float = 12.0
    max_vssma200_neg: float = -45.0
    max_drawdown_neg: float = -70.0
    pump_rsi: float = 83.0
    pump_vol_vs20: float = 400.0
    pump_d5: float = 18.0

    # EMA-specific soft hardening
    ema_rollover_drop: int = field(default_factory=lambda: int(os.getenv("HARD_EMA_ROLLOVER", "1")))

    # strictness mode
    mode: str = field(default_factory=lambda: os.getenv("HARD_DROP_MODE", "loose").lower())
    grace_atr: float = field(default_factory=lambda: float(os.getenv("HARD_GRACE_ATR", "2.0")))

    # Optional: ignore tickers right on earnings day to reduce whipsaw risk
    earnings_blackout_days: int = field(default_factory=lambda: int(os.getenv("TR_EARN_BLACKOUT_DAYS", "0")))

    # --- early-turn detector (keeps candidates that are just starting to trend up) ---
    def _is_early_turn(self, feats: Dict) -> bool:
        rsi      = safe(feats.get("RSI14"), None)
        e50s     = safe(feats.get("EMA50_slope_5d"), None)
        v20      = safe(feats.get("vol_vs20"), None)
        px       = safe(feats.get("price"), None)
        e20      = safe(feats.get("EMA20"), None)
        vsem200  = safe(feats.get("vsEMA200"), None)

        if (rsi is None) or (e50s is None) or (px is None) or (e20 is None) or (vsem200 is None):
            return False

        rsi_lo   = _env_float("EARLY_TURN_RSI_LO", 47.0)
        rsi_hi   = _env_float("EARLY_TURN_RSI_HI", 63.0)
        slope_lo = _env_float("EARLY_TURN_MIN_E50_SLOPE", 2.0)
        vs200_lo = _env_float("EARLY_TURN_VS200_LO", -12.0)
        vs200_hi = _env_float("EARLY_TURN_VS200_HI", 8.0)
        vol_lo   = _env_float("EARLY_TURN_VOL_LO", 110.0)
        vol_hi   = _env_float("EARLY_TURN_VOL_HI", 220.0)

        rsi_ok   = rsi_lo <= rsi <= rsi_hi
        slope_ok = e50s >= slope_lo
        vol_ok   = (v20 is None) or (vol_lo <= v20 <= vol_hi)
        px_ok    = px >= e20
        near200  = vs200_lo <= vsem200 <= vs200_hi

        return rsi_ok and slope_ok and vol_ok and px_ok and near200

    def why_garbage(self, feats: Dict) -> Tuple[bool, str]:
        # Earnings blackout
        try:
            dse = feats.get("days_since_earnings")
            if (dse is not None) and (int(dse) <= self.earnings_blackout_days):
                return True, "earnings_blackout"
        except Exception:
            pass

        t = self._scaled()
        price = safe(feats.get("price"), None)
        atrp  = safe(feats.get("ATRpct"), None)
        vs200 = safe(feats.get("vsSMA200"), None)
        dd    = safe(feats.get("drawdown_pct"), None)
        rsi   = safe(feats.get("RSI14"), None)
        v20   = safe(feats.get("vol_vs20"), None)
        d5    = safe(feats.get("d5"), None)
        r60   = safe(feats.get("r60"), 0.0)
        vs50  = safe(feats.get("vsSMA50"), None)

        # Trend leaders get a little ATR grace
        trend_leader = (r60 is not None and r60 >= 20) and (vs200 is not None and vs200 >= 0)
        early_turn = self._is_early_turn(feats)
        probe_ok, probe_lvl = _probe_gate(feats)

        if (price is None) or (price < t["min_price"]):
            return True, f"price<{t['min_price']} (price={price})"

        # ATR cap with relief for leaders and early turns
        if atrp is not None:
            lim = t["max_atr_pct"]
            if trend_leader:
                lim += self.grace_atr
            if early_turn:
                lim += self.grace_atr
            if atrp > lim:
                return True, f"ATRpct>{lim} (atr={atrp}, leader={trend_leader}, early={early_turn})"

        # vs200 drop test (allow a bit more weakness if it's an early turn)
        if vs200 is not None:
            lim_200 = t["max_vssma200_neg"]
            if early_turn:
                lim_200 -= _env_float("EARLY_TURN_VS200_PAD", 5.0)
            if vs200 < lim_200:
                return True, f"vsSMA200<{lim_200} (vs200={vs200}, early={early_turn})"

        if dd is not None and dd < t["max_drawdown_neg"]:
            return True, f"drawdown<{t['max_drawdown_neg']} (dd={dd})"

        # Pump-and-dump pattern
        if (rsi is not None and rsi >= t["pump_rsi"]) and \
           (v20 is not None and v20 >= t["pump_vol_vs20"]) and \
           (d5  is not None and d5  >= t["pump_d5"]):
            # If probe path is allowed, keep with tag; otherwise apply old rule
            if probe_ok:
                feats["probe_ok"] = True
                feats["probe_lvl"] = probe_lvl
                return False, f"probe_keep_L{probe_lvl}"

            if self.mode in {"off", "loose", "normal"} and (trend_leader or early_turn):

                return False, "pump_pattern_soft_keep"
            return True, "pump pattern (RSI/vol/d5)"

        # Valuation blow-off only if mania momentum present
        pe  = safe(feats.get("val_PE"), None)
        ps  = safe(feats.get("val_PS"), None)
        peg = safe(feats.get("val_PEG"), None)
        if ((pe is not None and pe >= 80) or (ps is not None and ps >= 40) or (peg is not None and peg >= 5.0)):
            hot = ((rsi is not None and rsi >= t["pump_rsi"]) and
                   (v20 is not None and v20 >= t["pump_vol_vs20"]) and
                   (d5  is not None and d5  >= t["pump_d5"]))
            if hot:
                if probe_ok:
                    return False, f"valuation_probe_keep_L{probe_lvl}"
                if self.mode in {"off", "loose"} and trend_leader :
                    return False, "valuation-blowoff-soft-keep"
                return True, "valuation-blowoff (PE/PS/PEG extreme + RSI/vol spike)"

        # Optional EMA rollover hard filter — skip if it's an early turn
        if self.ema_rollover_drop:
            e50  = safe(feats.get("EMA50"), None)
            e200 = safe(feats.get("EMA200"), None)
            px   = safe(feats.get("price"), None)
            if e50 and e200 and px:
                if (e50 < e200) and (px < e50) and self.mode in {"normal", "strict"}:
                    if not early_turn:
                        return True, "ema_rollover (px<EMA50<EMA200)"
        return False, ""

    def is_garbage(self, feats: Dict) -> bool:
        dropped, _ = self.why_garbage(feats)
        return dropped

    def _scaled(self):
        """
        Scale thresholds dynamically by profile (A/B/C).
        A = Aggressive (looser filters)
        B = Balanced (moderate filters)
        C = Conservative (stricter filters)
        """
        prof = (os.getenv("RANKER_PROFILE", os.getenv("SELECTION_MODE", "B")) or "B").upper()

        if prof == "A":
            # Looser filters → allow more volatile, extended names
            return dict(
                min_price=self.min_price,
                max_atr_pct=self.max_atr_pct + 4.0,   # tolerate higher ATR
                max_vssma200_neg=self.max_vssma200_neg - 10.0,  # allow weaker base
                max_drawdown_neg=self.max_drawdown_neg - 10.0,  # tolerate deeper DD
                pump_rsi=self.pump_rsi + 4.0,
                pump_vol_vs20=self.pump_vol_vs20 + 100.0,
                pump_d5=self.pump_d5 + 4.0,
            )

        if prof == "C":
            # Conservative → harsh risk filters
            return dict(
                min_price=max(self.min_price, 5.0),  # no penny junk
                max_atr_pct=self.max_atr_pct - 3.0,  # low ATR tolerance
                max_vssma200_neg=self.max_vssma200_neg + 10.0,  # base must be solid
                max_drawdown_neg=self.max_drawdown_neg + 10.0,  # shallow DD only
                pump_rsi=self.pump_rsi - 3.0,        # earlier pump detection
                pump_vol_vs20=self.pump_vol_vs20 - 100.0,
                pump_d5=self.pump_d5 - 3.0,
            )

        # Default = Balanced (B)
        return dict(
            min_price=self.min_price,
            max_atr_pct=self.max_atr_pct,          # baseline
            max_vssma200_neg=self.max_vssma200_neg,
            max_drawdown_neg=self.max_drawdown_neg,
            pump_rsi=self.pump_rsi,
            pump_vol_vs20=self.pump_vol_vs20,
            pump_d5=self.pump_d5,
        )


# ---------- composite score ----------
@dataclass
class RankerParams:
    # weights (keep valuation relatively small to avoid cheap/expensive bias)
    w_trend: float = 0.30
    w_momo: float = 0.32
    w_struct: float = 0.14
    w_stab: float = 0.14
    w_blowoff: float = 0.02
    w_value: float = 0.08  # reduced valuation influence

    # soft caps used in stability penalties
    atr_soft_cap: float = 6.0
    vol20_soft_cap: float = 250.0
    dd_soft_cap: float = -35.0


@dataclass
class RobustRanker:
    params: RankerParams = field(default_factory=RankerParams)
    hard: HardFilter = field(default_factory=HardFilter)
    stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    verbose: bool = field(default_factory=lambda: os.getenv("RANKER_VERBOSE", "").strip().lower() in {"1","true","yes"})
    log_every: int = field(default_factory=lambda: int(os.getenv("RANKER_LOG_EVERY", "500")))

    def set_verbose(self, enabled: bool, level: int = logging.INFO):
        self.verbose = bool(enabled)
        logger.setLevel(level)

    def _robust_z(self, arr: List[float]) -> Tuple[float, float]:
        a = np.array(
            [x for x in arr if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))],
            dtype=float,
        )
        if a.size == 0:
            return (0.0, 1.0)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = mad * 1.4826 if mad > 1e-12 else (float(np.std(a)) or 1.0)
        if not (scale > 1e-9):
            scale = 1.0
        return (med, scale)

    def fit_cross_section(self, feats_list: List[Dict]) -> None:
        keys = [
            "vsSMA50","vsSMA200","d20","RSI14","MACD_hist","ATRpct","drawdown_pct","vol_vs20","r60","r120",
            "vsEMA50","vsEMA200","EMA50_slope_5d"
        ]
        self.stats.clear()
        for k in keys:
            vals = [safe(f.get(k), None) for f in feats_list]
            med, scale = self._robust_z(vals)
            self.stats[k] = (med, scale)
        if self.verbose:
            logger.info(f"[Ranker] fit_cross_section on {len(feats_list)} names")

    def _zs(self, k: str, x: Optional[float]) -> float:
        if x is None:
            return 0.0
        med, scale = self.stats.get(k, (0.0, 1.0))
        try:
            z = (float(x) - med) / (scale if scale != 0 else 1.0)
        except Exception:
            z = 0.0
        return float(clamp(z, -4.0, 4.0))

    def _value_tilt(self, feats: Dict) -> float:
        """
        -1..+1 valuation score from optional fields (missing ~ neutral).
        """
        pe     = safe(feats.get("val_PE"), None)
        ps     = safe(feats.get("val_PS"), None)
        evrev  = safe(feats.get("val_EV_REV"), None)
        ebitda = safe(feats.get("val_EV_EBITDA"), None)
        peg    = safe(feats.get("val_PEG"), None)
        fcfy   = safe(feats.get("val_FCF_YIELD"), None)  # percent

        def to_pm01(u: float) -> float:
            return clamp(u * 2.0 - 1.0, -1.0, 1.0)

        s = 0.0
        s += 0.25 * to_pm01(tri_sweetspot(pe,     lo=5.0,  mid_lo=10.0, mid_hi=25.0, hi=60.0))
        s += 0.15 * to_pm01(tri_sweetspot(ps,     lo=0.5,  mid_lo=2.0,  mid_hi=10.0, hi=30.0))
        s += 0.15 * to_pm01(tri_sweetspot(evrev,  lo=0.5,  mid_lo=2.0,  mid_hi=10.0, hi=30.0))
        s += 0.15 * to_pm01(tri_sweetspot(ebitda, lo=5.0,  mid_lo=8.0,  mid_hi=20.0, hi=40.0))
        s += 0.15 * to_pm01(tri_sweetspot(peg,    lo=0.3,  mid_lo=0.8,  mid_hi=1.8,  hi=4.0))
        s += 0.15 * to_pm01(tri_sweetspot(fcfy,   lo=-5.0, mid_lo=1.0,  mid_hi=8.0,  hi=20.0))
        return clamp(s, -1.0, 1.0)

    def composite_score(self, feats: Dict, context: Optional[Dict]=None) -> Tuple[float, Dict[str, float]]:
        P = self.params
        prof = (os.getenv("RANKER_PROFILE", os.getenv("SELECTION_MODE", "B")) or "B").upper()

        try:
            # -------- Inputs --------
            vs50   = safe(feats.get("vsSMA50"), None)
            vs200  = safe(feats.get("vsSMA200"), None)
            macd   = safe(feats.get("MACD_hist"), None)
            rsi    = safe(feats.get("RSI14"), None)
            px     = safe(feats.get("price"), None)
            vsem50 = safe(feats.get("vsEMA50"), None)
            vsem200= safe(feats.get("vsEMA200"), None)
            sma50  = safe(feats.get("SMA50"), None)
            sma200 = safe(feats.get("SMA200"), None)
            avwap  = safe(feats.get("AVWAP252"), None)
            e50    = safe(feats.get("EMA50"), None)
            e200   = safe(feats.get("EMA200"), None)

            v20    = safe(feats.get("vol_vs20"), None)
            atrp   = safe(feats.get("ATRpct"), None)
            dd     = safe(feats.get("drawdown_pct"), None)

            pe     = safe(feats.get("val_PE"), None)
            ps     = safe(feats.get("val_PS"), None)
            rev    = safe(feats.get("val_EV_REV"), None)

            # -------- Trend (z-scores, not divided) --------
            rsi_band = tri_sweetspot(rsi, lo=40, mid_lo=52, mid_hi=68, hi=75)

            # Trend: give z-scores more bite
            trend = (
                0.6*self._zs("vsSMA200", vs200) +
                0.5*self._zs("vsSMA50",  vs50) +
                0.3*(clamp(macd or 0.0, -2.0, 2.0)) +
                0.4*rsi_band*4
            )

            # -------- Structure --------
            struct = 0.0
            if sma50 and sma200: struct += 2 if sma50 > sma200 else -1
            if e50 and e200:     struct += 2 if e50 > e200 else -1
            if px and e50:       struct += 1 if px > e50 else -0.5
            if px and avwap:     struct += 2 if px > avwap else -1
            if px and e50 and e200:
                if px > e50 > e200: struct += 1
                elif px < e50 < e200: struct -= 1

            # -------- Stability --------
            stab = 0.0
            if atrp: stab -= (atrp - P.atr_soft_cap)
            if dd:   stab -= (abs(min(dd,0.0)) - abs(P.dd_soft_cap)) / 5
            if v20:  stab -= (v20 - P.vol20_soft_cap) / 50
            if feats.get("liq_tier") == "small" and v20 and v20 >= 300:
                stab -= 2

            # -------- Blowoff --------
            blow = 0.0
            if rsi and rsi >= 80 and v20 and v20 >= 200: blow -= 3
            if rsi and rsi >= 85: blow -= 5
            if vsem50 and vsem50 >= 25: blow -= 1
            if prof == "A": blow *= 0.5
            if prof == "C": blow *= 1.5

            # -------- Valuation --------
            val = self._value_tilt(feats) * 5  # expand effect
            if pe and pe >= 80: val -= 3
            if ps and ps >= 40: val -= 3
            if rev and rev >= 20: val -= 3

            # -------- Combine --------
            parts = {"trend": trend, "struct": struct, "stability": stab, "blowoff": blow, "value": val}
            weights = {"trend": 0.25, "struct": 0.30, "stability": 0.25, "blowoff": 0.10, "value": 0.10}

            score_unit = sum(weights[k] * parts[k] for k in parts)

            # Final scale into -100..150
            val_boost = self._valuation_boost(feats)
            scr = clamp(score_unit * 35, -100, 150)
            parts["valuation_boost"] = val_boost
            return scr, parts

        except Exception as e:
            if self.verbose:
                logger.exception(f"[Ranker] composite_score error: {e}")
            return (-999.0, {"error": 1.0})

    def should_drop(self, feats: Dict) -> bool:
        drop, reason = self.hard.why_garbage(feats)
        if drop and self.verbose:
            logger.info(f"[Ranker] drop: {reason}")
        return drop

    def _apply_profile(self) -> None:
        """
        A = Aggressive (more forgiving of volatility/valuations)
        B = Balanced (default, moderate risk filter)
        C = Conservative (strict risk & valuation focus)
        """
        prof = (os.getenv("RANKER_PROFILE", os.getenv("SELECTION_MODE", "B")) or "B").upper()
        P = self.params

        if prof == "A":
            # Aggressive: more tolerance to volatility, lighter blowoff penalties
            P.w_trend, P.w_struct, P.w_stab, P.w_blowoff, P.w_value = \
                0.25, 0.28, 0.18, 0.09, 0.10
            P.atr_soft_cap, P.vol20_soft_cap, P.dd_soft_cap = \
                8.0, 350.0, -45.0   # wide tolerance

        elif prof == "C":
            # Conservative: harsher risk & valuation penalties, durability focus
            P.w_trend, P.w_struct, P.w_stab, P.w_blowoff, P.w_value = \
                0.15, 0.35, 0.30, 0.15, 0.05
            P.atr_soft_cap, P.vol20_soft_cap, P.dd_soft_cap = \
                5.0, 180.0, -25.0   # stricter thresholds

        else:  # B = Balanced (default)
            P.w_trend, P.w_struct, P.w_stab, P.w_blowoff, P.w_value = \
                0.20, 0.30, 0.25, 0.15, 0.10
            P.atr_soft_cap, P.vol20_soft_cap, P.dd_soft_cap = \
                6.0, 250.0, -35.0   # middle ground

    def score_universe(self, universe: List[Tuple[str, str, Dict]], context: Optional[Dict]=None):
        """
        Input: universe as iterable of (ticker, name, feats)
        Output: ranked list of (ticker, name, feats, score, parts)
        """
        self._apply_profile()
        feat_list = [f for _,_,f in universe]
        self.fit_cross_section(feat_list)
        ranked = []
        for i, (t, n, f) in enumerate(universe, 1):
            if self.should_drop(f):
                continue
            tr_score, tr_parts = self.composite_score(f, context)
            qs_score = f.get("final_score", f.get("score", 0.0)) # QS score from stage1_kept.csv
            merged_score, merged_parts = merge_tr_qs(tr_score, tr_parts, qs_score)

            # Add qs_score and tr_score explicitly into feats for export
            f["_qs_score"] = qs_score
            f["_tr_score"] = tr_score

            ranked.append((t, n, f, merged_score, merged_parts))


            if self.verbose and self.log_every and (i % self.log_every == 0):
                logger.info(f"[Ranker] progress: {i} processed, {len(ranked)} kept")
        ranked.sort(key=lambda x: x[3], reverse=True)
        if self.verbose and ranked:
            logger.info(f"[Ranker] done: kept {len(ranked)} / {len(universe)} ; leader={ranked[0][0]} score={ranked[0][3]:.2f}")
        return ranked
    def _valuation_boost(self, feats: Dict) -> float:
        """
        Enhanced valuation scoring using PE, PEG, and growth rate (YoY or forward).
        Prioritizes forward-looking PEG ratios when available and penalizes unstable backward metrics.
        """

        pe   = safe(feats.get("val_PE"), None)
        peg  = safe(feats.get("val_PEG"), None)
        yoy  = safe(feats.get("val_YoY"), None)
        gr   = safe(feats.get("val_growth_fwd"), None) or yoy  # allow external forward growth injection

        if pe is None or pe <= 0:
            return 0.0

        boost = 0.0

        # --- PE scoring ---
        if pe < 8:
            boost += 10
        elif 8 <= pe <= 15:
            boost += 6
        elif 15 < pe <= 25:
            boost += 2
        elif 25 < pe <= 35:
            boost -= 3
        elif pe > 35:
            boost -= 10

        # --- Growth scoring (YoY or forward) ---
        if gr is not None:
            # Clamp extreme growth values for safety
            gr = float(gr)
            if gr < -1: gr = -1
            if gr > 3: gr = 3

            if gr <= 0:
                boost -= 10  # contraction
            elif 0 < gr < 0.05:
                boost -= 3   # flat
            elif 0.05 <= gr <= 0.3:
                boost += 5   # healthy
            elif 0.3 < gr <= 0.8:
                boost += 10  # strong
            elif 0.8 < gr <= 1.5:
                boost += 15  # hypergrowth
            elif gr > 1.5:
                boost += 8   # very high growth (limited cap)

        # --- PEG scoring (prefer forward-looking) ---
        if peg and peg > 0:
            peg_val = float(peg)
            # Filter out unrealistic PEGs (>10)
            if 0 < peg_val <= 10:
                if peg_val < 0.8:
                    boost += 15  # undervalued vs. growth
                elif 0.8 <= peg_val <= 1.5:
                    boost += 8   # fair value
                elif 1.5 < peg_val <= 3.0:
                    boost -= 5   # overvalued
                else:
                    boost -= 10  # extremely overvalued
            else:
                boost -= 5
        else:
            # If no PEG but we have growth → small fallback bonus
            if gr and gr > 0.2 and pe < 20:
                boost += 5

        # --- Penalize mismatch cases ---
        if yoy and yoy < 0 and pe and pe > 30:
            boost -= 10  # expensive with negative growth

        # --- Bound final score ---
        boost = max(-27, min(27, boost))
        return boost



# --- Stratified top-N selector (5 small + 5 large with >=5 P/E present) ---
def _tier_of_row(row):
    """
    Determine tier of a ranked row. Expects row = (t, n, f, score[, parts]).
    Uses feats['liq_tier'] if present. Accepts 'small'/'SMALL' and 'large'/'LARGE'.
    """
    f = row[2]
    tier_raw = f.get("liq_tier")
    if not tier_raw:
        return "unknown"
    tier = str(tier_raw).lower()
    if tier in {"small", "large"}:
        return tier
    if tier in {"sm", "s"}:
        return "small"
    if tier in {"lg", "l", "big"}:
        return "large"
    return "unknown"

def _has_pe(row):
    """
    True if row has a positive, finite val_PE or PE proxy.
    """
    f = row[2]
    pe = (f.get("val_PE") if f.get("val_PE") is not None else f.get("PE"))
    try:
        if pe is None:
            return False
        pef = float(pe)
        return (pef > 0) and (not math.isinf(pef)) and (not math.isnan(pef))
    except Exception:
        return False

def pick_top_stratified(
    ranked: List[Tuple],
    total: int = 10,
    min_small: int = 5,
    min_large: int = 5,
    pe_min: int = 5,
) -> List[Tuple]:
    """
    Select 'total' rows from a ranked list, aiming for:
      - at least 'min_small' small-cap names
      - at least 'min_large' large-cap names
      - at least 'pe_min' names with val_PE present (>0)
    Preserves score order as much as possible; falls back gracefully.
    """
    if total <= 0:
        return []

    # Clamp impossible quotas
    if min_small + min_large > total:
        over = (min_small + min_large) - total
        if min_small >= min_large:
            min_small = max(0, min_small - over)
        else:
            min_large = max(0, min_large - over)

    # Partition the ranked pool
    small = [r for r in ranked if _tier_of_row(r) == "small"]
    large = [r for r in ranked if _tier_of_row(r) == "large"]

    # Start with quotas (truncate if not enough in a bucket)
    pick_small = small[:min_small]
    pick_large = large[:min_large]
    picked = pick_small + pick_large
    picked_ids = {r[0] for r in picked}  # ticker set

    def pool_not_picked():
        return [r for r in ranked if r[0] not in picked_ids]

    def count_pe(rows):
        return sum(1 for r in rows if _has_pe(r))

    # Fill the rest
    while len(picked) < total:
        remaining = pool_not_picked()
        if not remaining:
            break
        need_pe = max(0, pe_min - count_pe(picked))
        if need_pe > 0:
            next_with_pe = next((r for r in remaining if _has_pe(r)), None)
            if next_with_pe is not None:
                picked.append(next_with_pe)
                picked_ids.add(next_with_pe[0])
                continue
        picked.append(remaining[0])
        picked_ids.add(remaining[0][0])

    # Improve PE coverage
    pe_have = count_pe(picked)
    if pe_have < pe_min:
        insertables = [r for r in ranked if (r[0] not in picked_ids) and _has_pe(r)]

        def tier_counts(rows):
            s = sum(1 for r in rows if _tier_of_row(r) == "small")
            l = sum(1 for r in rows if _tier_of_row(r) == "large")
            return s, l

        picks_sorted_lo = sorted(picked, key=lambda x: x[3])  # lowest score first
        ins_idx = 0
        while (pe_have < pe_min) and (ins_idx < len(insertables)):
            incoming = insertables[ins_idx]
            ins_idx += 1
            removed_any = False
            for cand_out in list(picks_sorted_lo):
                if _has_pe(cand_out):
                    continue
                tmp = [r for r in picked if r[0] != cand_out[0]]
                s_cnt, l_cnt = tier_counts(tmp)
                if (s_cnt >= min_small) and (l_cnt >= min_large):
                    picked = tmp + [incoming]
                    picked_ids = {r[0] for r in picked}
                    picks_sorted_lo = sorted(picked, key=lambda x: x[3])
                    pe_have = count_pe(picked)
                    removed_any = True
                    break
            if not removed_any:
                continue

    # ✅ Always rank by merged_final_score if available, else fallback
    picked.sort(
        key=lambda x: (
            x[2].get("merged_final_score")  # new preferred field
            or x[2].get("_merged_score")    # fallback to TR merge
            or x[3]                         # fallback to score
        ),
        reverse=True,
    )
    return picked[:total]

import pandas as pd

def merge_tr_qs(tr_score: float, tr_parts: Dict[str, float], qs_score: Optional[float]) -> Tuple[float, Dict[str, float]]:
    try:
        qs_score = float(qs_score) if qs_score is not None else 0.0
    except Exception:
        qs_score = 0.0
    tr_scaled = tr_score
    # --- Weighted blend (Conservative TR bias) ---
    w_tr = 0.75
    w_qs = 0.25
    merged_score = (w_tr * tr_scaled) + (w_qs * qs_score)


    # --- Conservative penalties ---
    if tr_score < -50:
        merged_score -= 20
    if tr_score < -75:
        merged_score -= 40

    # --- Synergy bonuses ---
    if tr_score > 50 and qs_score > 75:
        merged_score += 15
    if tr_score > 70 and qs_score > 85:
        merged_score += 25

    merged_parts = dict(tr_parts)
    merged_parts["final_score"] = merged_score

    return merged_score, merged_parts

def normalize_qs(qs_score: float) -> float:
    if qs_score is None:
        return 0.0
    try:
        qs = float(qs_score)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, (qs - 50.0) / 50.0))



def merge_stage1_with_tr(stage1_path: str, out_path: str = "data/stage2_merged.csv"):
    df = pd.read_csv(stage1_path)
    ranker = RobustRanker()

    universe = []
    for _, row in df.iterrows():
        feats = row.to_dict()
        ticker = feats.get("ticker", "")
        name   = feats.get("company", feats.get("name", ticker))

        # --- Step 1: Base computation from your own logic
        pe, yoy, peg, trailing_pe, forward_pe = compute_pe_yoy_peg(ticker)
        feats["val_PE"] = pe
        feats["val_YoY"] = yoy
        feats["val_PEG"] = peg
        feats["val_PE_trailing"] = trailing_pe
        feats["val_PE_forward"] = forward_pe

        print(f"[PE/PEG] {ticker}: PE={pe}, YoY={yoy}, PEG={peg}")

        # --- Step 2: Try to refresh with live Yahoo data if available
        try:
            vals = fetch_valuations_for_top([ticker]).get(ticker, {})
            # Overwrite ONLY if Yahoo returns valid non-null data
            if vals:
                if vals.get("PE") is not None:
                    feats["val_PE"] = vals["PE"]
                if vals.get("PEG") is not None:
                    feats["val_PEG"] = vals["PEG"]
                # For YoY growth: you could use derived growth if available
                if vals.get("YoY") is not None:
                    feats["val_YoY"] = vals["YoY"]

                print(f"[YF OVERRIDE] {ticker}: Overwrote with Yahoo → PE={vals.get('PE')}, PEG={vals.get('PEG')}, YoY={vals.get('YoY')}")
        except Exception as e:
            logger.warning(f"[YF OVERRIDE] {ticker}: fetch_valuations_for_top failed → {e}")

        universe.append((ticker, name, feats))

    ranked = ranker.score_universe(universe)

    rows = []
    for t, n, f, tr_score, parts in ranked:
        qs_score = f.get("final_score", 0.0)

        merged_score, merged_parts = merge_tr_qs(tr_score, parts, qs_score)

        row = {
            "ticker": t,
            "name": n,
            "qs_score": qs_score,
            "tr_score": tr_score,
            "merged_score": merged_score,
            "merged_qs_score": qs_score,
            "merged_tr_score": tr_score,
            "probe_ok": f.get("probe_ok", False),
            "probe_lvl": f.get("probe_lvl", 0),
            "valuation_boost": parts.get("valuation_boost", 0.0),
            "val_PE": f.get("val_PE"),
            "val_YoY": f.get("val_YoY"),
            "val_PEG": f.get("val_PEG"),
            "val_PE_trailing": f.get("val_PE_trailing"),
            "val_PE_forward": f.get("val_PE_forward"),
        }
        row.update({f"tr_{k}": v for k, v in parts.items()})
        row.update({f"merged_{k}": v for k, v in merged_parts.items()})
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df["merged_final_score"] = (
        out_df["merged_score"] + out_df.get("valuation_boost", 0)
    )

    out_df = out_df.sort_values("merged_final_score", ascending=False)

    out_df.to_csv(out_path, index=False)
    print(f"[merge_stage1_with_tr] Saved merged scores to {out_path}")
    return out_df

@functools.lru_cache(maxsize=256)
def get_financials_cached(ticker: str):
    """Cached fetch for financial statements to reduce Yahoo rate hits."""
    tk = yf.Ticker(ticker)
    fin = tk.financials
    if fin is None or fin.empty:
        fin = tk.annual_financials
    return fin


def compute_yoy_growth(ticker: str, retries: int = 3):
    """
    Compute historical YoY growth using Net Income or EPS as fallback.
    Includes rate-limit handling, backoff, and caching.
    """
    for attempt in range(retries):
        try:
            logger.debug(f"[YoY] {ticker}: Fetching financials (attempt {attempt + 1})…")

            fin = get_financials_cached(ticker)

            if fin is None or fin.empty:
                logger.debug(f"[YoY] {ticker}: No financial data available.")
                return None

            # ✅ Normalize index names for consistency
            if "Net Income" not in fin.index and "Net Income Applicable To Common Shares" in fin.index:
                fin.rename(index={"Net Income Applicable To Common Shares": "Net Income"}, inplace=True)
            if "Basic EPS" not in fin.index and "Diluted EPS" in fin.index:
                fin.rename(index={"Diluted EPS": "Basic EPS"}, inplace=True)

            # --- Try Net Income YoY first
            if "Net Income" in fin.index:
                net_income = fin.loc["Net Income"].dropna().values[::-1]
                if len(net_income) >= 2 and net_income[-2] != 0:
                    yoy = (net_income[-1] - net_income[-2]) / abs(net_income[-2])
                    logger.info(f"[YoY] {ticker}: Historical YoY net income growth = {yoy:.4f}")
                    return float(yoy)

            # --- Fallback: EPS YoY
            if "Basic EPS" in fin.index:
                eps = fin.loc["Basic EPS"].dropna().values[::-1]
                if len(eps) >= 2 and eps[-2] != 0:
                    eps_growth = (eps[-1] - eps[-2]) / abs(eps[-2])
                    logger.info(f"[YoY] {ticker}: Fallback EPS YoY growth = {eps_growth:.4f}")
                    return float(eps_growth)

            logger.debug(f"[YoY] {ticker}: No valid YoY growth data found.")
            return None

        except YFRateLimitError:
            # Intelligent exponential backoff with jitter
            wait = min(1800, 60 * (attempt + 1) + random.uniform(3, 8))
            logger.warning(f"[YoY] {ticker}: Rate limited by Yahoo. Sleeping {wait:.1f}s before retry…")
            time.sleep(wait)
            continue

        except Exception as e:
            logger.warning(f"[YoY] {ticker} unexpected error: {e}", exc_info=True)
            wait = 15 * (attempt + 1) + random.uniform(2, 6)
            logger.info(f"[YoY] {ticker}: Retrying after {wait:.1f}s due to error…")
            time.sleep(wait)
            continue

    logger.info(f"[YoY] {ticker}: No valid historical YoY found after {retries} retries.")
    return None
def compute_pe_yoy_peg(ticker: str):
    """
    Fetch trailing & forward PE, compute forward (or fallback YoY) growth, and PEG.
    Prefers Yahoo's built-in forward PEG (pegRatio) when available and valid.
    PEG = PE / Growth if pegRatio missing.
    Returns: (pe, growth, peg, trailing_pe, forward_pe)
    """
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            # --- PE ---
            trailing_pe = info.get("trailingPE")
            forward_pe  = info.get("forwardPE")

            try:
                trailing_pe = float(trailing_pe) if trailing_pe and trailing_pe > 0 else None
            except Exception:
                trailing_pe = None
            try:
                forward_pe = float(forward_pe) if forward_pe and forward_pe > 0 else None
            except Exception:
                forward_pe = None

            # pick main PE (prefer trailing)
            pe = trailing_pe or forward_pe

            # --- Growth ---
            growth = None
            growth_fields = [
                info.get("earningsGrowth"),
                info.get("revenueGrowth"),
                info.get("earningsQuarterlyGrowth")
            ]

            for g in growth_fields:
                try:
                    g = float(g)
                    if g and g > 0:
                        growth = g
                        break  # ✅ Use first valid Yahoo growth, don't overwrite
                except:
                    continue

            # fallback to historical YoY ONLY if no valid growth from Yahoo
            if growth is None:
                growth = compute_yoy_growth(ticker)
                if growth is not None:
                    logger.info(f"[PE/PEG] {ticker}: Using fallback historical YoY growth={growth:.4f}")

            # --- PEG ---
            peg = None
            peg_yf = info.get("pegRatio")

            # ✅ Use Yahoo PEG if valid (0 < peg < 10)
            if peg_yf is not None:
                try:
                    peg_yf = float(peg_yf)
                    if 0 < peg_yf < 10:
                        peg = peg_yf
                        logger.info(f"[PE/PEG] {ticker}: Using Yahoo pegRatio={peg:.2f}")
                except Exception:
                    pass

            # fallback PEG only if Yahoo PEG missing/invalid
            if peg is None and pe and growth and growth > 0:
                try:
                    peg = pe / growth
                    if not (0 < peg < 10):
                        peg = None
                    else:
                        logger.info(f"[PE/PEG] {ticker}: Calculated PEG from PE/growth={peg:.2f}")
                except Exception:
                    peg = None


            logger.info(
                f"[PE/PEG] {ticker}: trailing={trailing_pe}, forward={forward_pe}, "
                f"growth={growth}, peg={peg}"
            )

            return pe, growth, peg, trailing_pe, forward_pe

        except Exception as e:
            wait = random.choice([3, 5, 7])
            logger.warning(f"[PE/PEG] {ticker} error: {e}. Retrying in {wait}s…", exc_info=True)
            time.sleep(wait)

    logger.error(f"[PE/PEG] {ticker}: failed after retries")
    return None, None, None, None, None



__all__ = [
    "HardFilter",
    "RankerParams",
    "RobustRanker",
    "pick_top_stratified",
]
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to stage1_kept.csv")
    parser.add_argument("--output", required=True, help="Path to save merged results (stage2_merged.csv)")
    args = parser.parse_args()

    merge_stage1_with_tr(args.input, args.output)
