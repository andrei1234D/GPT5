# scripts/trash_ranker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import os
import logging
import numpy as np

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
        return 0.5
    try:
        x = float(x)
    except Exception:
        return 0.5
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
                pump_d5=self.pump_d5 + 999,
            )
        if m == "strict":
            return dict(
                min_price=self.min_price,
                max_atr_pct=self.max_atr_pct - 2.0,
                max_vssma200_neg=self.max_vssma200_neg + 5.0,
                max_drawdown_neg=self.max_drawdown_neg + 5.0,
                pump_rsi=self.pump_rsi - 2.0,
                pump_vol_vs20=self.pump_vol_vs20 - 50.0,
                pump_d5=self.pump_d5 - 2.0,
            )
        if m == "normal":
            return dict(
                min_price=self.min_price,
                max_atr_pct=self.max_atr_pct - 1.0,
                max_vssma200_neg=self.max_vssma200_neg + 2.0,
                max_drawdown_neg=self.max_drawdown_neg + 2.0,
                pump_rsi=self.pump_rsi - 1.0,
                pump_vol_vs20=self.pump_vol_vs20 - 25.0,
                pump_d5=self.pump_d5 - 1.0,
            )
        # loose default
        return dict(
            min_price=self.min_price,
            max_atr_pct=self.max_atr_pct + 1.5,
            max_vssma200_neg=self.max_vssma200_neg - 2.0,
            max_drawdown_neg=self.max_drawdown_neg - 2.0,
            pump_rsi=self.pump_rsi + 1.5,
            pump_vol_vs20=self.pump_vol_vs20 + 50.0,
            pump_d5=self.pump_d5 + 2.0,
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

        # Trend leaders get a little ATR grace
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
            if self.mode in {"off", "loose"} and trend_leader:
                return False, "pump_pattern_soft_keep"
            return True, "pump pattern (RSI/vol/d5)"

        # Valuation blow-off only if mania momentum present
        pe  = safe(feats.get("val_PE"), None)
        ps  = safe(feats.get("val_PS"), None)
        peg = safe(feats.get("val_PEG"), None)
        if ((pe is not None and pe >= 80) or (ps is not None and ps >= 40) or (peg is not None and peg >= 5.0)):
            if (rsi is not None and rsi >= t["pump_rsi"]) and (v20 is not None and v20 >= t["pump_vol_vs20"]) and (d5 is not None and d5 >= t["pump_d5"]):
                if self.mode in {"off", "loose"} and trend_leader:
                    return False, "valuation-blowoff-soft-keep"
                return True, "valuation-blowoff (PE/PS/PEG extreme + RSI/vol spike)"

        # Optional EMA rollover hard filter
        if self.ema_rollover_drop:
            e50  = safe(feats.get("EMA50"), None)
            e200 = safe(feats.get("EMA200"), None)
            px   = safe(feats.get("price"), None)
            if e50 and e200 and px:
                if (e50 < e200) and (px < e50) and self.mode in {"normal", "strict"}:
                    return True, "ema_rollover (px<EMA50<EMA200)"
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
        keys = ["vsSMA50","vsSMA200","d20","RSI14","MACD_hist","ATRpct","drawdown_pct","vol_vs20","r60","r120",
                "vsEMA50","vsEMA200","EMA50_slope_5d"]
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
        try:
            vs50  = safe(feats.get("vsSMA50"), 0.0)
            vs200 = safe(feats.get("vsSMA200"), 0.0)
            macd  = safe(feats.get("MACD_hist"), 0.0)
            rsi   = safe(feats.get("RSI14"),   None)
            rsi_band = tri_sweetspot(rsi, lo=35, mid_lo=47, mid_hi=68, hi=75)

            # EMA-based fields
            vsem50  = safe(feats.get("vsEMA50"), None)
            vsem200 = safe(feats.get("vsEMA200"), None)
            e50s    = safe(feats.get("EMA50_slope_5d"), None)

            trend = (
                0.28*(self._zs("vsSMA200", vs200)/4.0) +
                0.20*(self._zs("vsSMA50",  vs50) /4.0) +
                0.20*(self._zs("vsEMA200", vsem200)/4.0) +
                0.17*(self._zs("vsEMA50",  vsem50) /4.0) +
                0.15*(clamp(macd,-1.5,1.5)/1.5) +
                0.10*rsi_band
            )

            d20  = safe(feats.get("d20"), 0.0)
            r60  = safe(feats.get("r60"), 0.0)
            r120 = safe(feats.get("r120"), 0.0)
            is20h = feats.get("is_20d_high")
            near20 = 1.0 if (is20h or (d20 is not None and d20 >= 8 and safe(feats.get("RSI14"), 50) <= 72)) else 0.0
            momo = (
                0.35*(self._zs("d20",  d20) /4.0) +
                0.27*(self._zs("r60",  r60) /4.0) +
                0.23*(self._zs("r120", r120)/4.0) +
                0.10*(self._zs("EMA50_slope_5d", e50s)/4.0) +
                0.05*near20
            )

            sma50 = safe(feats.get("SMA50"), None)
            sma200= safe(feats.get("SMA200"), None)
            px    = safe(feats.get("price"), None)
            avwap = safe(feats.get("AVWAP252"), None)
            e50   = safe(feats.get("EMA50"), None)
            e200  = safe(feats.get("EMA200"), None)

            align = 0.0
            if sma50 is not None and sma200 is not None:
                align += 0.5 if (sma50 > sma200) else -0.2
            if e50 and e200:
                align += 0.6 if (e50 > e200) else -0.25
            if px and e50:
                align += 0.25 if (px > e50) else -0.15
            if px and avwap:
                align += 0.25 if (px > avwap) else -0.10
            if px and e50 and e200:
                if px > e50 > e200: align += 0.20
                elif px < e50 < e200: align -= 0.18
            struct = clamp(align, -1.0, 1.0)

            atrp = safe(feats.get("ATRpct"), None)
            dd   = safe(feats.get("drawdown_pct"), None)
            v20  = safe(feats.get("vol_vs20"), None)
            atr_pen = 0.0 if atrp is None else (-clamp((atrp - P.atr_soft_cap)/P.atr_soft_cap, 0.0, 1.0))
            dd_pen  = 0.0 if dd   is None else (-clamp((abs(min(dd,0.0)) - abs(P.dd_soft_cap))/25.0, 0.0, 1.0))
            vol_pen = 0.0 if v20  is None else (-clamp((v20 - P.vol20_soft_cap)/P.vol20_soft_cap, 0.0, 1.0))

            # Mean-reversion risk if far above EMA50
            ext_pen = 0.0
            if vsem50 is not None and vsem50 >= 20:
                ext_pen = -clamp((vsem50 - 20)/40.0, 0.0, 0.3)  # up to -0.3

            # --- ATH guard (near 52w high + very hot) ---
            ath_guard = int(os.getenv("ATH_GUARD", "1")) != 0
            ath_sev = 0.0
            ath_relief = 1.0
            if ath_guard:
                near_pct   = float(os.getenv("ATH_NEAR_PCT", "1.0"))   # within 1% of 52w high
                min_rsi    = float(os.getenv("ATH_MIN_RSI", "80"))     # RSI threshold
                min_vs50   = float(os.getenv("ATH_MIN_VS50", "25"))    # vsEMA50 threshold
                vol_relief = float(os.getenv("ATH_VOL_RELIEF", "60"))  # strong volume reduces penalty
                dd0 = safe(feats.get("drawdown_pct"), None)

                if (dd0 is not None and dd0 >= -near_pct and
                    rsi is not None and rsi >= min_rsi and
                    vsem50 is not None and vsem50 >= min_vs50):
                    ath_sev = clamp(
                        0.5 * ((rsi - min_rsi) / 10.0) + 0.5 * ((vsem50 - min_vs50) / 25.0),
                        0.0, 1.0
                    )
                    if v20 is not None and v20 >= vol_relief:
                        ath_relief = 0.5  # participation halves the penalty

            # include ATH component inside stability slightly
            ath_pen_stab = -0.6 * ath_sev * ath_relief if ath_sev > 0 else 0.0

            stability = clamp(
                0.45*atr_pen +
                0.20*dd_pen  +
                0.10*vol_pen +
                0.10*ext_pen +
                0.15*ath_pen_stab,
                -1.0, 0.0
            )

            # Blowoff detector (EMA-aware)
            blow = 0.0
            if (rsi is not None and rsi >= 80) and (v20 is not None and v20 >= 200) and (d20 is not None and d20 >= 15):
                blow = -1.0
            elif (rsi is not None and rsi >= 85):
                blow = -0.7
            if vsem50 is not None and vsem50 >= 25 and blow < 0:
                blow = min(-1.0, blow - 0.2)

            # Harsher blowoffs under extreme valuations
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

            # Final ATH haircut so the effect is material on the headline score
            if ath_sev > 0:
                haircut = float(os.getenv("ATH_SCORE_HAIRCUT", "22"))  # points on -100..+100 scale
                scr = clamp(scr - haircut * ath_sev * ath_relief, -100.0, 100.0)

            if self.verbose and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Ranker] parts trend=%.3f momo=%.3f struct=%.3f stab=%.3f blow=%.3f value=%.3f -> score=%.2f (ath_sev=%.2f relief=%.2f)",
                    parts["trend"], parts["momo"], parts["struct"], parts["stability"], parts["blowoff"], parts["value"],
                    scr, ath_sev, ath_relief
                )
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
    True if row has a positive, finite val_PE.
    """
    f = row[2]
    pe = f.get("val_PE")
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

    # Helpers
    def pool_not_picked():
        return [r for r in ranked if r[0] not in picked_ids]

    def count_pe(rows):
        return sum(1 for r in rows if _has_pe(r))

    # Fill the rest, biasing toward meeting pe_min
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
        nxt = remaining[0]
        picked.append(nxt)
        picked_ids.add(nxt[0])

    # Improve PE coverage with swaps while honoring tier quotas
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

    picked.sort(key=lambda x: x[3], reverse=True)
    return picked[:total]

__all__ = [
    "HardFilter",
    "RankerParams",
    "RobustRanker",
    "pick_top_stratified",
]
