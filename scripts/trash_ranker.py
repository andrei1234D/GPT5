from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math, os, logging, numpy as np

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
        if x is None: return default
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf): return default
        return xf
    except Exception:
        return default

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def tri_sweetspot(x: Optional[float], lo: float, mid_lo: float, mid_hi: float, hi: float) -> float:
    """
    Returns a 0..1 score that peaks in the [mid_lo, mid_hi] band,
    linearly rises from lo..mid_lo and falls from mid_hi..hi, zero outside.
    """
    if x is None: return 0.5
    try: x = float(x)
    except: return 0.5
    if x <= lo or x >= hi: return 0.0
    if mid_lo <= x <= mid_hi: return 1.0
    if x < mid_lo: return (x - lo) / max((mid_lo - lo), 1e-9)
    return (hi - x) / max((hi - mid_hi), 1e-9)

def bool01(x) -> Optional[float]:
    if x is True: return 1.0
    if x is False: return 0.0
    return None

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

    # strictness mode
    mode: str = field(default_factory=lambda: os.getenv("HARD_DROP_MODE", "loose").lower())
    grace_atr: float = field(default_factory=lambda: float(os.getenv("HARD_GRACE_ATR", "2.0")))

    def _scaled(self):
        m = self.mode
        if m == "off":
            return dict(
                min_price=self.min_price,
                max_atr_pct=self.max_atr_pct + 999,
                max_vssma200_neg=self.max_vssma200_neg - 999,
                max_drawdown_neg=self.max_drawdown_neg - 999,
                pump_rsi=self.pump_rsi + 999,
                pump_vol_vs20=self.pump_vol_vs20 + 999,
                pump_d5=self.pump_d5 + 999
            )
        if m == "strict":
            return dict(
                min_price=self.min_price,
                max_atr_pct=self.max_atr_pct - 2.0,
                max_vssma200_neg=self.max_vssma200_neg + 5.0,
                max_drawdown_neg=self.max_drawdown_neg + 5.0,
                pump_rsi=self.pump_rsi - 2.0,
                pump_vol_vs20=self.pump_vol_vs20 - 50.0,
                pump_d5=self.pump_d5 - 2.0
            )
        if m == "normal":
            return dict(
                min_price=self.min_price,
                max_atr_pct=self.max_atr_pct - 1.0,
                max_vssma200_neg=self.max_vssma200_neg + 2.0,
                max_drawdown_neg=self.max_drawdown_neg + 2.0,
                pump_rsi=self.pump_rsi - 1.0,
                pump_vol_vs20=self.pump_vol_vs20 - 25.0,
                pump_d5=self.pump_d5 - 1.0
            )
        # loose default
        return dict(
            min_price=self.min_price,
            max_atr_pct=self.max_atr_pct + 1.5,
            max_vssma200_neg=self.max_vssma200_neg - 2.0,
            max_drawdown_neg=self.max_drawdown_neg - 2.0,
            pump_rsi=self.pump_rsi + 1.5,
            pump_vol_vs20=self.pump_vol_vs20 + 50.0,
            pump_d5=self.pump_d5 + 2.0
        )

    def why_garbage(self, feats: Dict) -> Tuple[bool, str]:
        t = self._scaled()
        price = safe(feats.get("price"), None)
        atrp  = safe(feats.get("ATRpct"), None)
        vs200 = safe(feats.get("vsSMA200"), None)
        dd    = safe(feats.get("drawdown_pct"), None)
        rsi   = safe(feats.get("RSI14"), None)
        v20   = safe(feats.get("vol_vs20"), None)
        d5    = safe(feats.get("d5"), None)
        r60   = safe(feats.get("r60"), 0.0)

        # Never drop trend leaders on small ATR overruns (grace)
        trend_leader = (r60 is not None and r60 >= 20) and (vs200 is not None and vs200 >= 0)

        if (price is None) or (price < t["min_price"]):
            return True, f"price<{t['min_price']} (price={price})"

        if atrp is not None:
            lim = t["max_atr_pct"]
            if trend_leader:
                lim += self.grace_atr
            if atrp > lim:
                return True, f"ATRpct>{lim} (atr={atrp}, leader={trend_leader})"

        if vs200 is not None and vs200 < t["max_vssma200_neg"]:
            return True, f"vsSMA200<{t['max_vssma200_neg']} (vs200={vs200})"
        if dd is not None and dd < t["max_drawdown_neg"]:
            return True, f"drawdown<{t['max_drawdown_neg']} (dd={dd})"

        # Pump-and-dump pattern
        if (rsi is not None and rsi >= t["pump_rsi"]) and \
           (v20 is not None and v20 >= t["pump_vol_vs20"]) and \
           (d5  is not None and d5  >= t["pump_d5"]):
            # allow soft mode to keep but flag
            if self.mode in {"off","loose"} and trend_leader:
                return False, "pump_pattern_soft_keep"
            return True, "pump pattern (RSI/vol/d5)"

        # Conservative valuation blow-off: only triggers with mania momentum
        pe  = safe(feats.get("val_PE"), None)
        ps  = safe(feats.get("val_PS"), None)
        peg = safe(feats.get("val_PEG"), None)
        if ((pe is not None and pe >= 80) or (ps is not None and ps >= 40) or (peg is not None and peg >= 5.0)):
            if (rsi is not None and rsi >= t["pump_rsi"]) and (v20 is not None and v20 >= t["pump_vol_vs20"]) and (d5 is not None and d5 >= t["pump_d5"]):
                if self.mode in {"off", "loose"} and trend_leader:
                    return False, "valuation-blowoff-soft-keep"
                return True, "valuation-blowoff (PE/PS/PEG extreme + RSI/vol spike)"

        return False, ""

    def is_garbage(self, feats: Dict) -> bool:
        dropped, _ = self.why_garbage(feats)
        return dropped

# ---------- composite score ----------
@dataclass
class RankerParams:
    w_trend: float = 0.32
    w_momo:  float = 0.28
    w_struct: float = 0.14
    w_stab:  float = 0.14
    w_blowoff: float = 0.02
    w_value: float = 0.10   # valuation tilt
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
        a = np.array([x for x in arr if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))], dtype=float)
        if a.size == 0:
            return (0.0, 1.0)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = mad * 1.4826 if mad > 1e-12 else (float(np.std(a)) or 1.0)
        if not (scale > 1e-9): scale = 1.0
        return (med, scale)

    def fit_cross_section(self, feats_list: List[Dict]) -> None:
        keys = ["vsSMA50","vsSMA200","d20","RSI14","MACD_hist","ATRpct","drawdown_pct","vol_vs20","r60","r120"]
        self.stats.clear()
        for k in keys:
            vals = [safe(f.get(k), None) for f in feats_list]
            med, scale = self._robust_z(vals)
            self.stats[k] = (med, scale)
        if self.verbose:
            logger.info(f"[Ranker] fit_cross_section on {len(feats_list)} names")

    def _zs(self, k: str, x: Optional[float]) -> float:
        if x is None: return 0.0
        med, scale = self.stats.get(k, (0.0, 1.0))
        try: z = (float(x) - med) / (scale if scale != 0 else 1.0)
        except: z = 0.0
        return float(clamp(z, -4.0, 4.0))

    def _value_tilt(self, feats: Dict) -> float:
        """
        Returns a -1..+1 valuation score from (optionally present) fields:
        val_PE, val_PS, val_EV_REV, val_EV_EBITDA, val_PEG, val_FCF_YIELD (percent).
        Missing values contribute near-neutral.
        """
        pe     = safe(feats.get("val_PE"), None)
        ps     = safe(feats.get("val_PS"), None)
        evrev  = safe(feats.get("val_EV_REV"), None)
        ebitda = safe(feats.get("val_EV_EBITDA"), None)
        peg    = safe(feats.get("val_PEG"), None)
        fcfy   = safe(feats.get("val_FCF_YIELD"), None)  # percent

        def to_pm01(u: float) -> float:
            # map 0..1 sweetspot score → -1..+1
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
        try:
            vs50  = safe(feats.get("vsSMA50"), 0.0)
            vs200 = safe(feats.get("vsSMA200"), 0.0)
            macd  = safe(feats.get("MACD_hist"), 0.0)
            rsi   = safe(feats.get("RSI14"),   None)
            rsi_band = tri_sweetspot(rsi, lo=35, mid_lo=47, mid_hi=68, hi=75)
            trend = 0.35*(self._zs("vsSMA200", vs200)/4.0) + 0.25*(self._zs("vsSMA50", vs50)/4.0) + 0.25*(clamp(macd,-1.5,1.5)/1.5) + 0.15*rsi_band

            d20  = safe(feats.get("d20"), 0.0)
            r60  = safe(feats.get("r60"), 0.0)
            r120 = safe(feats.get("r120"), 0.0)
            is20h = feats.get("is_20d_high")
            near20 = 1.0 if (is20h or (d20 is not None and d20 >= 8 and safe(feats.get("RSI14"), 50) <= 72)) else 0.0
            momo = 0.38*(self._zs("d20", d20)/4.0) + 0.28*(self._zs("r60", r60)/4.0) + 0.24*(self._zs("r120", r120)/4.0) + 0.10*near20

            sma50 = safe(feats.get("SMA50"), None)
            sma200= safe(feats.get("SMA200"), None)
            px    = safe(feats.get("price"), None)
            avwap = safe(feats.get("AVWAP252"), None)
            align = 0.0
            if sma50 is not None and sma200 is not None:
                align += 0.6 if (sma50 > sma200) else -0.2
            if px is not None and avwap is not None:
                align += 0.4 if (px > avwap) else -0.1
            if px is not None and sma50 is not None and sma200 is not None:
                if px > sma50 > sma200: align += 0.25
                elif px < sma50 < sma200: align -= 0.20
                if avwap is not None:
                    bullish_stack = (sma50 > sma200)
                    if bullish_stack ^ (px > avwap):
                        align -= 0.10
            struct = clamp(align, -1.0, 1.0)

            atrp = safe(feats.get("ATRpct"), None)
            dd   = safe(feats.get("drawdown_pct"), None)
            v20  = safe(feats.get("vol_vs20"), None)
            atr_pen = 0.0 if atrp is None else (-clamp((atrp - P.atr_soft_cap)/P.atr_soft_cap, 0.0, 1.0))
            dd_pen  = 0.0 if dd   is None else (-clamp((abs(min(dd,0.0)) - abs(P.dd_soft_cap))/25.0, 0.0, 1.0))
            vol_pen = 0.0 if v20  is None else (-clamp((v20 - P.vol20_soft_cap)/P.vol20_soft_cap, 0.0, 1.0))
            stability = clamp(0.6*atr_pen + 0.3*dd_pen + 0.1*vol_pen, -1.0, 0.0)

            # Blowoff detector
            blow = 0.0
            if (rsi is not None and rsi >= 80) and (v20 is not None and v20 >= 200) and (d20 is not None and d20 >= 15):
                blow = -1.0
            elif (rsi is not None and rsi >= 85):
                blow = -0.7

            # Make blowoffs harsher under extreme valuations
            pe  = safe(feats.get("val_PE"), None)
            ps  = safe(feats.get("val_PS"), None)
            peg = safe(feats.get("val_PEG"), None)
            if ((pe is not None and pe >= 60) or (ps is not None and ps >= 30) or (peg is not None and peg >= 4.0)):
                if (rsi is not None and rsi >= 80) and (v20 is not None and v20 >= 200) and (d20 is not None and d20 >= 15):
                    blow = -1.0

            # Valuation tilt (−1..+1 → part)
            value = self._value_tilt(feats)

            parts = {"trend": trend, "momo": momo, "struct": struct, "stability": stability, "blowoff": blow, "value": value}
            w = {"trend": P.w_trend, "momo": P.w_momo, "struct": P.w_struct, "stability": P.w_stab, "blowoff": P.w_blowoff, "value": P.w_value}
            active_w = sum(w.values()) or 1.0
            score_unit = sum(w[k]*v for k,v in parts.items())
            scr = clamp((score_unit / active_w) * 100.0, -100.0, 100.0)
            if self.verbose and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Ranker] parts trend=%.3f momo=%.3f struct=%.3f stab=%.3f blow=%.3f value=%.3f -> score=%.2f",
                             parts["trend"], parts["momo"], parts["struct"], parts["stability"], parts["blowoff"], parts["value"], scr)
            return (scr, parts)
        except Exception as e:
            if self.verbose: logger.exception(f"[Ranker] composite_score error: {e}")
            return (-999.0, {"error": 1.0})

    def should_drop(self, feats: Dict) -> bool:
        drop, reason = self.hard.why_garbage(feats)
        if drop and self.verbose:
            logger.info(f"[Ranker] drop: {reason}")
        return drop

    def score_universe(self, universe: List[Tuple[str, str, Dict]], context: Optional[Dict]=None):
        feat_list = [f for _,_,f in universe]
        self.fit_cross_section(feat_list)
        ranked = []
        for i, (t, n, f) in enumerate(universe, 1):
            if self.should_drop(f):
                continue
            score, parts = self.composite_score(f, context)
            ranked.append((t, n, f, score, parts))
            if self.verbose and self.log_every and (i % self.log_every == 0):
                logger.info(f"[Ranker] progress: {i} processed, {len(ranked)} kept")
        ranked.sort(key=lambda x: x[3], reverse=True)
        if self.verbose and ranked:
            logger.info(f"[Ranker] done: kept {len(ranked)} / {len(universe)} ; leader={ranked[0][0]} score={ranked[0][3]:.2f}")
        return ranked
