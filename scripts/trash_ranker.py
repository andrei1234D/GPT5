# scripts/trash_ranker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

# ---------- utilities ----------
def safe(x, default=None):
    try:
        if x is None: return default
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return default
        return float(x)
    except Exception:
        return default

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def tri_sweetspot(x: Optional[float], lo: float, mid_lo: float, mid_hi: float, hi: float) -> float:
    """
    Triangular utility: 0 at lo/hi, 1 at [mid_lo, mid_hi], linear in between.
    Returns in [0,1]. Missing -> 0.5 neutral.
    """
    if x is None: return 0.5
    x = float(x)
    if x <= lo or x >= hi: return 0.0
    if mid_lo <= x <= mid_hi: return 1.0
    if x < mid_lo:  # rising edge
        return (x - lo) / (mid_lo - lo)
    # falling edge
    return (hi - x) / (hi - mid_hi)

def bool01(x) -> Optional[float]:
    if x is True: return 1.0
    if x is False: return 0.0
    return None

# ---------- hard filters (small-cap friendly) ----------
@dataclass
class HardFilter:
    min_price: float = 3.0          # allow legit small caps above $3
    max_atr_pct: float = 12.0       # extreme daily noise
    max_vssma200_neg: float = -45.0 # deep downtrend
    max_drawdown_neg: float = -70.0 # catastrophic 52w drawdown
    pump_rsi: float = 83.0          # blow-off risk when paired with vol spike
    pump_vol_vs20: float = 400.0
    pump_d5: float = 18.0

    def is_garbage(self, feats: Dict) -> bool:
        price = safe(feats.get("price"), default=None)
        atrp  = safe(feats.get("ATRpct"), default=None)
        vs200 = safe(feats.get("vsSMA200"), default=None)
        dd    = safe(feats.get("drawdown_pct"), default=None)
        rsi   = safe(feats.get("RSI14"), default=None)
        v20   = safe(feats.get("vol_vs20"), default=None)
        d5    = safe(feats.get("d5"), default=None)

        # Low price micro-pennies (but keep small caps >= $3)
        if (price is None) or (price < self.min_price):
            return True

        if atrp is not None and atrp > self.max_atr_pct:
            return True
        if vs200 is not None and vs200 < self.max_vssma200_neg:
            return True
        if dd is not None and dd < self.max_drawdown_neg:
            return True

        # Pump-and-dump profile: parabolic, hyper-volume, very overbought in days
        if (rsi is not None and rsi >= self.pump_rsi) and \
           (v20 is not None and v20 >= self.pump_vol_vs20) and \
           (d5 is not None and d5 >= self.pump_d5):
            return True

        return False

# ---------- composite score ----------
@dataclass
class RankerParams:
    w_trend: float = 0.35
    w_momo:  float = 0.30
    w_struct: float = 0.15
    w_stab:  float = 0.15
    w_blowoff: float = 0.05
    # risk caps
    atr_soft_cap: float = 6.0
    vol20_soft_cap: float = 250.0
    dd_soft_cap: float = -35.0

@dataclass
class RobustRanker:
    params: RankerParams = field(default_factory=RankerParams)
    hard: HardFilter = field(default_factory=HardFilter)

    # cross-sectional medians/MADs per run (built at call time; no persistence needed)
    stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def _robust_z(self, arr: List[float]) -> Tuple[float, float]:
        a = np.array([x for x in arr if x is not None and not math.isnan(x)], dtype=float)
        if a.size == 0:
            return (0.0, 1.0)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = mad * 1.4826 if mad > 1e-12 else (np.std(a) or 1.0)
        return (med, float(scale if scale > 1e-9 else 1.0))

    def fit_cross_section(self, feats_list: List[Dict]) -> None:
        """
        Build robust median/MAD for continuous features used in scoring.
        Neutralizes universe-to-universe scaling and is missing-friendly.
        """
        keys = [
            "vsSMA50","vsSMA200","d20","RSI14","MACD_hist","ATRpct",
            "drawdown_pct","vol_vs20","r60","r120"
        ]
        for k in keys:
            vals = [safe(f.get(k), None) for f in feats_list]
            med, scale = self._robust_z(vals)
            self.stats[k] = (med, scale)

    def _zs(self, k: str, x: Optional[float]) -> float:
        """robust z-score (winsorized to +/-4). Missing -> 0."""
        if x is None: return 0.0
        med, scale = self.stats.get(k, (0.0, 1.0))
        z = (float(x) - med) / (scale if scale != 0 else 1.0)
        return float(clamp(z, -4.0, 4.0))

    def composite_score(self, feats: Dict, context: Optional[Dict]=None) -> Tuple[float, Dict[str, float]]:
        P = self.params
        # --- Trend quality (like “are you above the big currents?”)
        vs50  = safe(feats.get("vsSMA50"), 0.0)
        vs200 = safe(feats.get("vsSMA200"), 0.0)
        macd  = safe(feats.get("MACD_hist"), 0.0)
        rsi   = safe(feats.get("RSI14"),   None)

        rsi_band = tri_sweetspot(rsi, lo=35, mid_lo=47, mid_hi=68, hi=75)  # favor 47–68
        trend = (
            0.35 * (self._zs("vsSMA200", vs200) / 4.0) +   # stronger weight
            0.25 * (self._zs("vsSMA50",  vs50)  / 4.0) +
            0.25 * (clamp(macd, -1.5, 1.5) / 1.5) +
            0.15 * rsi_band
        )  # in approx [-1,1]

        # --- Momentum persistence (avoid 1-3 day pops)
        d20  = safe(feats.get("d20"), 0.0)
        r60  = safe(feats.get("r60"), 0.0)
        r120 = safe(feats.get("r120"), 0.0)
        is20h = feats.get("is_20d_high")
        momo = (
            0.40 * (self._zs("d20", d20)   / 4.0) +
            0.30 * (self._zs("r60", r60)   / 4.0) +
            0.20 * (self._zs("r120", r120) / 4.0) +
            0.10 * (bool01(is20h) if bool01(is20h) is not None else 0.0)
        )

        # --- Structure (alignment & anchored refs)
        sma50 = safe(feats.get("SMA50"), None)
        sma200= safe(feats.get("SMA200"), None)
        px    = safe(feats.get("price"), None)
        avwap = safe(feats.get("AVWAP252"), None)
        align = 0.0
        if sma50 is not None and sma200 is not None:
            align += 0.6 if (sma50 > sma200) else -0.2
        if px is not None and avwap is not None:
            align += 0.4 if (px > avwap) else -0.1
        struct = clamp(align, -1.0, 1.0)

        # --- Stability / risk penalties (soft caps)
        atrp = safe(feats.get("ATRpct"), None)
        dd   = safe(feats.get("drawdown_pct"), None)  # negative is bad
        v20  = safe(feats.get("vol_vs20"), None)

        # penalties in [-1,0]
        atr_pen  = 0.0 if atrp is None else (-clamp((atrp - P.atr_soft_cap)/P.atr_soft_cap, 0.0, 1.0))
        dd_pen   = 0.0 if dd   is None else (-clamp((abs(min(dd,0.0)) - abs(P.dd_soft_cap))/abs(P.dd_soft_cap), 0.0, 1.0))
        vol_pen  = 0.0 if v20  is None else (-clamp((v20 - P.vol20_soft_cap)/P.vol20_soft_cap, 0.0, 1.0))
        stability = clamp(0.6*atr_pen + 0.3*dd_pen + 0.1*vol_pen, -1.0, 0.0)

        # --- Blow-off/exhaustion penalty (don’t auto-kill legit breakouts)
        blow = 0.0
        if (rsi is not None and rsi >= 80) and (v20 is not None and v20 >= 200) and (d20 is not None and d20 >= 15):
            blow = -1.0  # classic exhaustion/pump pattern
        elif (rsi is not None and rsi >= 85):
            blow = -0.7

        # combine (only count features we can actually observe)
        parts = {
            "trend": trend,
            "momo": momo,
            "struct": struct,
            "stability": stability,
            "blowoff": blow
        }
        # missing-aware weighting: only weight parts that are not neutral-missing
        w = {
            "trend": P.w_trend,
            "momo": P.w_momo,
            "struct": P.w_struct,
            "stability": P.w_stab,
            "blowoff": P.w_blowoff
        }
        active_w = 0.0
        score = 0.0
        for k, v in parts.items():
            # treat "None" pieces as neutral 0 (but they never are None here), still count weight
            score += w[k] * v
            active_w += w[k]
        # normalize to [-100,100]
        scr = (score / max(active_w, 1e-9)) * 100.0
        return (scr, parts)

    # public API
    def should_drop(self, feats: Dict) -> bool:
        return self.hard.is_garbage(feats)

    def score_universe(
        self,
        universe: List[Tuple[str, str, Dict]],
        context: Optional[Dict]=None
    ) -> List[Tuple[str, str, Dict, float, Dict[str,float]]]:
        """
        universe: list of (ticker, company, features)
        returns: sorted descending by composite score
        """
        feat_list = [f for _,_,f in universe]
        self.fit_cross_section(feat_list)

        ranked = []
        for t, n, f in universe:
            if self.should_drop(f):
                continue
            score, parts = self.composite_score(f, context)
            ranked.append((t, n, f, score, parts))
        ranked.sort(key=lambda x: x[3], reverse=True)
        return ranked
