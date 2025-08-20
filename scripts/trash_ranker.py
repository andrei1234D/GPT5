# scripts/trash_ranker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os, math, logging
import numpy as np

logger = logging.getLogger("trash_ranker")
if not logger.handlers:
    logging.basicConfig(level=getattr(logging, (os.getenv("RANKER_LOG_LEVEL") or "INFO").upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ----------------- helpers -----------------
def safe(x, d=0.0):
    try:
        if x is None: return d
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf): return d
        return xf
    except Exception:
        return d

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def bool01(x) -> int:
    return 1 if bool(x) else 0

def tri_sweetspot(x: Optional[float], lo: float, mid_lo: float, mid_hi: float, hi: float) -> float:
    """0..1 ramp in [lo..mid_lo], 1 in [mid_lo..mid_hi], down to 0 at hi."""
    if x is None: return 0.0
    x = float(x)
    if x <= lo: return 0.0
    if x >= hi: return 0.0
    if x < mid_lo:
        return (x - lo) / max(1e-9, (mid_lo - lo))
    if x <= mid_hi:
        return 1.0
    return (hi - x) / max(1e-9, (hi - mid_hi))

# --------------- dataclasses ---------------
@dataclass
class RankerParams:
    total: int = 30
    min_small: int = 15
    min_large: int = 15
    require_pe_min: int = 8  # at least this many with PE present
    chase_pen_r60_knee: float = float(os.getenv("TR_CHASE_KNEE", "40"))
    chase_pen_slope: float = float(os.getenv("TR_CHASE_SLOPE", "0.7"))
    chase_pen_cap: float = float(os.getenv("TR_CHASE_MAX", "18"))
    blowoff_rsi: float = 80.0
    blowoff_vol: float = 220.0
    earnings_blackout_days: int = int(os.getenv("TR_EARN_BLACKOUT_DAYS", "1"))  # skip same-day prints if desired
    verbose: bool = os.getenv("RANKER_VERBOSE", "").strip().lower() in {"1","true","yes"}

@dataclass
class HardFilter:
    earnings_blackout_days: int = 0
    min_price: float = 1.0
    min_adv20: float = 0.0
    allow_otc: bool = False

    def why_garbage(self, row) -> Optional[str]:
        t, n, feats, s, meta = row
        px = safe(feats.get("price"), None)
        if px is not None and px < self.min_price:
            return "too_cheap_price"
        adv = safe(feats.get("avg_dollar_vol_20d"), None)
        if adv is not None and adv < self.min_adv20:
            return "too_illiquid"
        if not self.allow_otc and str(feats.get("exchange","")).upper() in {"OTC","PINK"}:
            return "otc_blocked"
        dse = feats.get("days_since_earnings")
        try:
            if dse is not None and int(dse) <= self.earnings_blackout_days:
                return "earnings_blackout"
        except Exception:
            pass
        return None

    def is_garbage(self, row) -> bool:
        return self.why_garbage(row) is not None

# ---------------- Ranker -------------------
class RobustRanker:
    def __init__(self, params: Optional[RankerParams] = None):
        self.params = params or RankerParams()
        self.stats: Dict[str, Tuple[float,float]] = {}
        self.verbose = self.params.verbose

    # cross-sectional robust stats
    def _robust(self, arr: List[float]) -> Tuple[float,float]:
        a = np.array([x for x in arr if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))], dtype=float)
        if a.size == 0: return (0.0, 1.0)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = mad * 1.4826 if mad > 1e-12 else float(np.std(a) or 1.0)
        if not (scale > 1e-9): scale = 1.0
        return (med, scale)

    def fit_cross_section(self, ranked: List[Tuple[str,str,Dict,float,Dict]]):
        keys = ["vsSMA50","vsSMA200","r60","r120","d20","RSI14","ATRpct","vol_vs20","EMA50_slope_5d","vsEMA50","vsEMA200","drawdown_pct"]
        data: Dict[str, List[float]] = {k: [] for k in keys}
        for (_t,_n,f,_s,_m) in ranked:
            for k in keys:
                x = f.get(k)
                try:
                    if x is not None:
                        xv = float(x)
                        if not (math.isnan(xv) or math.isinf(xv)):
                            data[k].append(xv)
                except Exception:
                    pass
        self.stats = {k: self._robust(v) for k,v in data.items()}

    def _z(self, key: str, x: float|None, lo=-4.0, hi=4.0) -> float:
        if x is None: return 0.0
        med, scale = self.stats.get(key, (0.0, 1.0))
        try:
            z = (float(x) - med) / (scale if scale != 0 else 1.0)
        except Exception:
            z = 0.0
        return float(clamp(z, lo, hi))

    def composite_score(self, feats: Dict) -> Tuple[float, Dict[str,float]]:
        P = self.params
        vs50  = safe(feats.get("vsSMA50"), 0.0)
        vs200 = safe(feats.get("vsSMA200"), 0.0)
        r60   = safe(feats.get("r60"), 0.0)
        r120  = safe(feats.get("r120"), 0.0)
        d20   = safe(feats.get("d20"), 0.0)
        rsi   = safe(feats.get("RSI14"), None)
        macd  = safe(feats.get("MACD_hist"), 0.0)
        atr   = safe(feats.get("ATRpct"), 0.0)
        v20   = safe(feats.get("vol_vs20"), 0.0)
        dd    = safe(feats.get("drawdown_pct"), 0.0)
        px    = safe(feats.get("price"), 0.0)
        avwap = safe(feats.get("AVWAP252"), None)
        vsem50= safe(feats.get("vsEMA50"), None)
        vsem200=safe(feats.get("vsEMA200"), None)
        e50s  = safe(feats.get("EMA50_slope_5d"), None)
        pe    = None
        for k in ("val_PE","PE","pe","pe_hint"):
            if feats.get(k) is not None:
                try: pe = float(feats.get(k)); break
                except Exception: pass

        # z-scores
        z_vs50  = self._z("vsSMA50", vs50);     z_vs200 = self._z("vsSMA200", vs200)
        z_r60   = self._z("r60", r60);          z_r120  = self._z("r120", r120)
        z_d20   = self._z("d20", d20);          z_atr   = self._z("ATRpct", atr)
        z_v20   = self._z("vol_vs20", v20);     z_e50s  = self._z("EMA50_slope_5d", e50s)

        # parts
        rsi_band = tri_sweetspot(rsi, lo=40, mid_lo=52, mid_hi=68, hi=75)
        trend = (
            0.32 * clamp(z_vs200/4.0, -1, 1) +
            0.20 * clamp(z_vs50/4.0,  -1, 1) +
            0.10 * clamp((vsem200 or 0)/20.0, -1, 1) +
            0.10 * clamp((vsem50 or 0)/20.0,  -1, 1) +
            0.08 * clamp(macd/1.5,           -1, 1)
        ) * 100.0 + 8.0 * rsi_band

        momo = (
            0.36 * clamp(z_r60/4.0,  -1, 1) +
            0.26 * clamp(z_r120/4.0, -1, 1) +
            0.22 * clamp(z_d20/4.0,  -1, 1) +
            0.16 * clamp(z_e50s/4.0, -1, 1)
        ) * 100.0

        # chase penalty
        if r60 is not None and r60 > P.chase_pen_r60_knee:
            momo -= min(P.chase_pen_cap, (r60 - P.chase_pen_r60_knee) * P.chase_pen_slope)

        # acceleration & continuation bonuses (prefer beginning/continuation, not exhaustion)
        try:
            accel = clamp(((z_r60 - z_r120) / 2.0), -1.0, 1.0)
        except Exception:
            accel = 0.0
        momo += 8.0 * accel
        continuation = 0.0
        if (vsem50 is not None and 3.0 <= vsem50 <= 18.0) and (vsem200 is not None and vsem200 >= -2.0) and \
           (rsi is not None and 52.0 <= rsi <= 68.0) and (e50s is not None and e50s > 0.0):
            if (v20 is None) or (-20.0 <= v20 <= 180.0):
                continuation = 8.0
        momo += continuation

        # structure
        struct = 0.0
        if avwap:
            prem = (px/avwap) - 1.0
            if prem < 0: struct += clamp(abs(prem)*20.0, 0.0, 6.0)
            else:        struct -= clamp(prem*25.0, 0.0, 8.0)
        if vsem50 and vsem200:
            if vsem50 > 0 and vsem200 > 0: struct += 3.0
            elif vsem50 < 0 and vsem200 < 0: struct -= 2.0

        # stability (inverse risk)
        stability = 0.0
        if atr is not None: stability += clamp((6.0 - atr)/6.0, -1.0, 1.0) * 10.0
        if dd is not None and dd > -50: stability += clamp((50.0 + dd)/50.0, 0.0, 1.0) * 4.0

        # blowoff/late-stage penalty
        blow = 0.0
        if (rsi is not None and rsi >= P.blowoff_rsi) and (v20 is not None and v20 >= P.blowoff_vol):
            blow -= 12.0
        if rsi is not None and rsi >= 88 and v20 >= 250:
            blow -= 8.0
        # stall penalties
        stall = 0.0
        if accel < -0.25:    stall -= min(8.0, (abs(accel) - 0.25) * 20.0)
        if e50s is not None and e50s <= 0.0: stall -= 5.0
        if (rsi is not None and 55 <= rsi <= 75) and (macd < 0.0) and (vs50 > 0.0): stall -= 8.0
        if (rsi is not None and 55 <= rsi <= 70) and (vs50 > 0.0) and (v20 is not None and v20 <= -20): stall -= 4.0

        # tiny valuation tilt
        val = 0.0
        if pe is not None and pe > 0:
            if pe <= 10: val += 8.0
            elif 10 < pe <= 18: val += 4.0
            elif 30 <= pe <= 40: val -= 2.0
            elif pe >= 50: val -= 6.0

        parts = {
            "trend": trend, "momo": momo, "struct": struct, "stability": stability,
            "blowoff": blow, "stall": stall, "value": val, "accel": accel, "continuation": continuation
        }
        # weights
        wt_trend, wt_momo, wt_struct, wt_stab, wt_blow, wt_stall, wt_val = 0.34, 0.34, 0.12, 0.06, 0.06, 0.06, 0.02
        score = (wt_trend*trend + wt_momo*momo + wt_struct*struct + wt_stab*stability +
                 wt_blow*blow + wt_stall*stall + wt_val*val)
        return (float(clamp(score, -100.0, 100.0)), parts)

    def should_drop(self, feats: Dict) -> bool:
        # Hard disqualifiers for safety: extreme ATR, huge drawdown, absurd extension over AVWAP
        atr = safe(feats.get("ATRpct"), 0.0)
        dd  = safe(feats.get("drawdown_pct"), 0.0)
        px  = safe(feats.get("price"), 0.0)
        avwap = safe(feats.get("AVWAP252"), None)
        if atr is not None and atr >= 12.0: return True
        if dd is not None and dd <= -75.0:  return True
        if avwap and px and px > avwap * 1.8: return True
        return False

    def score_universe(self, ranked: List[Tuple[str,str,Dict,float,Dict]]):
        # Compute and attach stage2 scores + parts + tags
        self.fit_cross_section(ranked)
        scored = []
        for row in ranked:
            t, n, f, s1, meta = row
            if self.should_drop(f):  # hard safety
                continue
            s2, parts = self.composite_score(f)
            tags = list(meta.get("tags", []))
            # derive stall tag again for visibility
            rsi = safe(f.get("RSI14"), None); vs50 = safe(f.get("vsSMA50"), 0.0)
            v20 = safe(f.get("vol_vs20"), 0.0); macd = safe(f.get("MACD_hist"), 0.0)
            r60 = safe(f.get("r60"), 0.0); r120 = safe(f.get("r120"), 0.0)
            accel = ((r60 - r120) / 20.0) if (r60 is not None and r120 is not None) else 0.0
            if (rsi is not None and 55 <= rsi <= 70) and (vs50 > 0) and (((v20 or 0) <= -20) or (macd < 0)):
                if "stall_risk" not in tags: tags.append("stall_risk")
            meta2 = dict(meta)
            meta2["parts_s2"] = parts
            meta2["tags"] = tags
            scored.append((t, n, f, s2, meta2))
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored

def _has_pe(row) -> bool:
    feats = row[2]
    for k in ("val_PE","PE","pe","pe_hint"):
        try:
            if feats.get(k) is not None and float(feats.get(k)) > 0: return True
        except Exception:
            pass
    return False

def _tier_of_row(row) -> str:
    feats = row[2]
    t = (feats.get("liq_tier") or "").lower()
    if t in {"small","large"}: return t
    adv = safe(feats.get("avg_dollar_vol_20d"), None)
    if adv is None: return "small"
    return "large" if adv >= float(os.getenv("TR_LARGE_ADV_USD", "30000000")) else "small"

def pick_top_stratified(
    ranked: List[Tuple[str,str,Dict,float,Dict]],
    total: int = 30,
    min_small: int = 15,
    min_large: int = 15,
    pe_min: int = 8
) -> List[Tuple[str,str,Dict,float,Dict]]:
    """
    Stage-2: stratified pick from Stage-1 'ranked' (t, name, feats, score, meta).
    Preserves pipeline expectations and quotas. Applies small late-stage/ stall protections.
    """
    P = RankerParams(total=total, min_small=min_small, min_large=min_large, require_pe_min=pe_min)
    ranker = RobustRanker(P)
    scored = ranker.score_universe(ranked)

    # enforce quotas
    small = [r for r in scored if _tier_of_row(r) == "small"]
    large = [r for r in scored if _tier_of_row(r) == "large"]

    pick_s = small[:min_small]
    pick_l = large[:min_large]

    taken = set([r[0] for r in (pick_s + pick_l)])
    remainder = [r for r in scored if r[0] not in taken]
    tail = remainder[:max(0, total - len(pick_s) - len(pick_l))]
    picked = (pick_s + pick_l + tail)[:total]

    # PE presence floor; if insufficient, swap in more with PE from remainder
    if pe_min and sum(1 for r in picked if _has_pe(r)) < pe_min:
        need = pe_min - sum(1 for r in picked if _has_pe(r))
        pool = [r for r in remainder if _has_pe(r) and r[0] not in {x[0] for x in picked}]
        picked += pool[:max(0, need)]
        # trim if exceeded total
        picked = picked[:total]

    picked.sort(key=lambda x: x[3], reverse=True)
    return picked

__all__ = ["RankerParams","HardFilter","RobustRanker","pick_top_stratified"]
