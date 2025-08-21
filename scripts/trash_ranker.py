# scripts/trash_ranker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import os
import logging
import numpy as np

# --- BEGIN: env loader for local testing -------------------------------------
import os, sys, re, io

def _apply_kv_cli_overrides(argv: list) -> list:
    """
    Consume bare KEY=VALUE tokens from argv and set os.environ.
    Return argv with those tokens removed.
    """
    out = [argv[0]]
    for tok in argv[1:]:
        if "=" in tok and not tok.startswith("--"):
            k, v = tok.split("=", 1)
            k = k.strip()
            if k:
                os.environ[k] = v
            continue
        out.append(tok)
    return out

def _to_percent_points(x: Optional[float], *thresholds: float) -> Optional[float]:
    """
    Normalize possibly-fractional inputs to percentage points.
    If thresholds are clearly in percent space (>2), and |x|<=2, treat x as a fraction and *100.
    """
    if x is None:
        return None
    try:
        xv = float(x)
    except Exception:
        return None
    # Heuristic: only rescale when thresholds look like percent points (e.g., 20/35/50)
    if thresholds and max(thresholds) > 2.0 and abs(xv) <= 2.0:
        return xv * 100.0
    return xv


def _parse_gha_env_from_text(yml_text: str, step_name: str | None) -> dict:
    """
    Best-effort extraction of the 'env:' mapping from a GitHub Actions step.

    Strategy:
    1) With PyYAML: search ALL jobs. If step_name is given, return that step's env
       merged with the job's env. Otherwise, pick the step with the largest
       (job_env ∪ step_env).
    2) Fallback (no PyYAML): scan the file for ALL 'env:' blocks and pick the largest.
       If step_name is given, prefer the first env: that appears under that step.
    """
    # ------------ Preferred path: structured parse ------------
    try:
        import yaml  # type: ignore
        doc = yaml.safe_load(yml_text) or {}
        jobs = (doc or {}).get("jobs", {}) or {}

        # If exact step requested: search all jobs
        if step_name:
            for job_id, job in jobs.items():
                job_env = (job or {}).get("env") or {}
                for st in ((job or {}).get("steps") or []) or []:
                    if ((st or {}).get("name") or "").strip() == step_name:
                        step_env = (st or {}).get("env") or {}
                        merged = {**job_env, **step_env}
                        return {str(k): "" if v is None else str(v) for k, v in merged.items()}

        # Otherwise: pick the largest merged env across all jobs/steps
        best = {}
        for job_id, job in jobs.items():
            job_env = (job or {}).get("env") or {}
            for st in ((job or {}).get("steps") or []) or []:
                step_env = (st or {}).get("env") or {}
                merged = {**job_env, **step_env}
                if len(merged) > len(best):
                    best = merged
        if best:
            return {str(k): "" if v is None else str(v) for k, v in best.items()}
    except Exception:
        pass  # fall through to text scan

    # ------------ Fallback: plain text scan ------------
    import re
    lines = yml_text.splitlines()

    def _grab_env_block(start_idx: int) -> dict:
        """Given index of a line that is exactly 'env:', capture its indented block."""
        env_line = lines[start_idx]
        env_indent = len(env_line) - len(env_line.lstrip(" "))
        out = {}
        for j in range(start_idx + 1, len(lines)):
            raw = lines[j]
            if not raw.strip():
                continue
            ind = len(raw) - len(raw.lstrip(" "))
            if ind <= env_indent:
                break
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)\s*$", raw)
            if not m:
                continue
            k = m.group(1)
            v = m.group(2).strip()
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            out[k] = v
        return out

    # If a step name was requested, find that step then the next env: under it
    if step_name:
        name_pat = re.compile(r"^\s*name\s*:\s*(.+?)\s*$")
        step_idx = -1
        for i, ln in enumerate(lines):
            m = name_pat.match(ln)
            if m and m.group(1).strip() == step_name:
                step_idx = i
                break
        if step_idx >= 0:
            # find first subsequent 'env:' line
            for i in range(step_idx + 1, len(lines)):
                if re.match(r"^\s*env\s*:\s*$", lines[i]):
                    blk = _grab_env_block(i)
                    if blk:
                        return blk

    # Otherwise, examine ALL env blocks and pick the largest
    best = {}
    for i, ln in enumerate(lines):
        if re.match(r"^\s*env\s*:\s*$", ln):
            blk = _grab_env_block(i)
            if len(blk) > len(best):
                best = blk
    return best

def _apply_gha_env(path: str, step_name: str | None) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception as e:
        print(f"[env] Could not read YAML at {path}: {e}", flush=True)
        return {}
    env_map = _parse_gha_env_from_text(txt, step_name)
    for k, v in env_map.items():
        os.environ[k] = v
    if env_map:
        print(f"[env] Loaded {len(env_map)} vars from '{os.path.basename(path)}'"
              + (f" step '{step_name}'" if step_name else ""), flush=True)
    else:
        print(f"[env] No env block found in '{os.path.basename(path)}'", flush=True)
    return env_map

def _consume_env_flags(argv: list) -> list:
    """
    Supports:
      --gha-env PATH
      --gha-step NAME
    Removes those flags from argv after applying.
    """
    out = [argv[0]]
    i = 1
    gha_path = None
    gha_step = None
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--gha-env="):
            gha_path = tok.split("=", 1)[1]
            i += 1
            continue
        if tok == "--gha-env" and (i + 1) < len(argv):
            gha_path = argv[i + 1]
            i += 2
            continue
        if tok.startswith("--gha-step="):
            gha_step = tok.split("=", 1)[1]
            i += 1
            continue
        if tok == "--gha-step" and (i + 1) < len(argv):
            gha_step = argv[i + 1]
            i += 2
            continue
        out.append(tok)
        i += 1
    # env file via flag takes precedence
    if gha_path:
        _apply_gha_env(gha_path, gha_step or None)
    # also honor an environment variable GHA_ENV_YML if set
    elif os.getenv("GHA_ENV_YML"):
        _apply_gha_env(os.getenv("GHA_ENV_YML"), os.getenv("GHA_ENV_STEP") or None)
    return out

# run loaders before the rest of the script sees argv
sys.argv = _apply_kv_cli_overrides(sys.argv)
sys.argv = _consume_env_flags(sys.argv)
# --- END: env loader for local testing ---------------------------------------


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
    if rsi is None or vs50 is None:
        return 0
    A_RSI = _env_float("OVERHEAT_A_RSI", 78.0)
    B_RSI = _env_float("OVERHEAT_B_RSI", 83.0)
    C_RSI = _env_float("OVERHEAT_C_RSI", 87.0)
    A_V50 = _env_float("OVERHEAT_A_VS50", 20.0)
    B_V50 = _env_float("OVERHEAT_B_VS50", 35.0)
    C_V50 = _env_float("OVERHEAT_C_VS50", 50.0)

    r = float(rsi)
    # normalize vsSMA50 to percentage points if it was provided as a fraction
    v = _to_percent_points(vs50, A_V50, B_V50, C_V50)

    if v is None:
        return 0
    if r >= C_RSI and v >= C_V50: return 3
    if r >= B_RSI and v >= B_V50: return 2
    if r >= A_RSI and v >= A_V50: return 1
    return 0

def _probe_gate(feats: Dict) -> Tuple[bool, int]:
    if not _env_bool("ALLOW_BLOWOFF_PROBE", True):
        return False, 0
    rsi  = safe(feats.get("RSI14"), None)
    vs50_raw = safe(feats.get("vsSMA50"), None)
    atr  = safe(feats.get("ATRpct"), None)
    v20  = safe(feats.get("vol_vs20"), None)

    lvl = _overheat_level(rsi, vs50_raw)
    ev_ok  = (atr is not None) and (atr >= _env_float("PROBE_MIN_EV", 4.0))
    vol_ok = (v20 is None) or (v20 <= _env_float("PROBE_MAX_VOL_SPIKE", 150.0))

    if _env_bool("RANKER_DEBUG_PROBE", False):
        # Also show the normalized vs50 used for gating:
        A_V50 = _env_float("OVERHEAT_A_VS50", 20.0)
        B_V50 = _env_float("OVERHEAT_B_VS50", 35.0)
        C_V50 = _env_float("OVERHEAT_C_VS50", 50.0)
        vs50_pp = _to_percent_points(vs50_raw, A_V50, B_V50, C_V50)
        logger.info(
            "[probe] rsi=%.2f vs50=%.4f (pp=%.2f) atr=%.2f v20=%.2f | lvl=%d ev_ok=%s vol_ok=%s allow=%s",
            (rsi if rsi is not None else float('nan')),
            (vs50_raw if vs50_raw is not None else float('nan')),
            (vs50_pp if vs50_pp is not None else float('nan')),
            (atr if atr is not None else float('nan')),
            (v20 if v20 is not None else float('nan')),
            lvl, ev_ok, vol_ok, _env_bool("ALLOW_BLOWOFF_PROBE", True)
        )

    if lvl == 0:
        return False, 0
    if ev_ok and vol_ok:
        return True, lvl
    return False, lvl


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
                return False, f"probe_keep_L{probe_lvl}"
            if self.mode in {"off", "loose"} and trend_leader:
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
                if self.mode in {"off", "loose"} and trend_leader:
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


# ---------- composite score ----------
@dataclass
class RankerParams:
    # weights (keep valuation relatively small to avoid cheap/expensive bias)
    w_trend: float = 0.30
    w_momo: float = 0.32
    w_struct: float = 0.16  # a touch more structure to surface "buys"
    w_stab: float = 0.12    # slightly less stability headwind to avoid over-pruning
    w_blowoff: float = 0.02
    w_value: float = 0.08   # reduced valuation influence

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
        try:
            # trend inputs
            vs50  = safe(feats.get("vsSMA50"), 0.0)
            vs200 = safe(feats.get("vsSMA200"), 0.0)
            macd  = safe(feats.get("MACD_hist"), 0.0)
            rsi   = safe(feats.get("RSI14"),   None)
            rsi_band = tri_sweetspot(rsi, lo=40, mid_lo=52, mid_hi=68, hi=75)

            # EMA-based fields
            vsem50  = safe(feats.get("vsEMA50"), None)
            vsem200 = safe(feats.get("vsEMA200"), None)
            e50s    = safe(feats.get("EMA50_slope_5d"), None)

            # Fair value & anchor
            px = safe(feats.get("price"), None)
            fva = safe(feats.get("fva_hint"), None)
            value = self._value_tilt(feats)

            # FVA term (small influence by design)
            fva_term = 0.0
            if fva and px:
                disc = (fva - px) / max(abs(fva), 1e-9) * 100.0  # +% means px below anchor
                if disc < 0:
                    fva_term = -clamp(abs(disc)/25.0, 0.0, 1.0) * 0.35
                elif disc > 0:
                    fva_term =  clamp(disc/25.0, 0.0, 1.0) * 0.15

            # KO penalty with momentum-carry discount
            ko_pen = 0.0
            ko_pct = _env_float("QS_FVA_KO_PCT", 0.0)  # 0 disables
            d20_for_ko = safe(feats.get("d20"), None)
            if ko_pct > 0 and (px is not None) and (fva is not None) and px > fva:
                gap = (px - fva) / max(abs(fva), 1e-9) * 100.0
                extended = ((rsi is not None and rsi >= 72) or
                            (vsem50 is not None and vsem50 >= 40) or
                            (d20_for_ko is not None and d20_for_ko >= 15))
                if gap >= ko_pct and extended:
                    base_ko = -min(0.6, max(0.0, gap - ko_pct) * 0.012)  # contributes inside stability
                    # acceleration calc for discount decision
                    z_r60  = self._zs("r60", safe(feats.get("r60"), 0.0))
                    z_r120 = self._zs("r120", safe(feats.get("r120"), 0.0))
                    accel_z = clamp(((z_r60 - z_r120) / 2.0), -1.0, 1.0)
                    strong_trend = (safe(feats.get("r60"), 0.0) >= 30.0) and (vs50 is not None and vs50 >= 15.0)
                    discount = 1.0
                    if strong_trend and accel_z > 0.10:
                        discount = _env_float("KO_MOMO_DISCOUNT", 0.50)
                        if accel_z >= 0.35:
                            discount = _env_float("KO_MOMO_DISCOUNT_STRONG", 0.35)
                    ko_pen = base_ko * discount

            # trend score
            trend = (
                0.28*(self._zs("vsSMA200", vs200)/4.0) +
                0.20*(self._zs("vsSMA50",  vs50) /4.0) +
                0.20*(self._zs("vsEMA200", vsem200)/4.0) +
                0.17*(self._zs("vsEMA50",  vsem50) /4.0) +
                0.15*(clamp(macd,-1.5,1.5)/1.5) +
                0.10*rsi_band
            )
            trend = clamp(trend, -1.0, 1.0)

            # momentum
            d20  = safe(feats.get("d20"), 0.0)
            r60  = safe(feats.get("r60"), 0.0)
            r120 = safe(feats.get("r120"), 0.0)
            is20h = feats.get("is_20d_high")

            near20 = 0.0
            if is20h or (d20 is not None and d20 >= 8 and safe(feats.get("RSI14"), 50) <= 72):
                near20 = 1.0
            elif (d20 is not None and 3.0 <= d20 <= 7.0 and rsi is not None and rsi <= 65 and
                  vsem200 is not None and vsem200 <= 10.0):
                near20 = 0.6

            z_r60  = self._zs("r60", r60)
            z_r120 = self._zs("r120", r120)
            accel_z = clamp(((z_r60 - z_r120) / 2.0), -1.0, 1.0)

            momo = (
                0.33*(self._zs("r60",  r60) /4.0) +
                0.25*(self._zs("r120", r120)/4.0) +
                0.24*(self._zs("d20",  d20) /4.0) +
                0.10*(self._zs("EMA50_slope_5d", e50s)/4.0) +
                0.05*near20 +
                0.08*accel_z
            )
            try:
                knee  = _env_float("TR_CHASE_KNEE", 35.0)
                slope = _env_float("TR_CHASE_SLOPE", 0.006)
                cap   = _env_float("TR_CHASE_MAX", 0.18)
            except Exception:
                knee, slope, cap = 35.0, 0.006, 0.18
            if r60 is not None and r60 > knee:
                momo -= min(cap, (r60 - knee) * slope)
            momo = clamp(momo, -1.0, 1.0)

            # structure
            sma50 = safe(feats.get("SMA50"), None)
            sma200= safe(feats.get("SMA200"), None)
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

            # AVWAP premium headwind (QS_STRUCT_PREM_CAP points on 0..100 scale)
            prem_cap_pts = _env_float("QS_STRUCT_PREM_CAP", 0.0)
            if prem_cap_pts > 0 and px and avwap:
                prem = (px / avwap) - 1.0  # >0 means above AVWAP
                if prem > 0:
                    prem_cap = prem_cap_pts / 100.0
                    struct -= min(prem_cap, prem * (prem_cap / 0.20))

            # continuation boost
            v20_for_boost = safe(feats.get("vol_vs20"), None)
            if (vsem50 is not None and 3.0 <= vsem50 <= 18.0) and (vsem200 is not None and vsem200 >= -2.0) and \
               (rsi is not None and 52.0 <= rsi <= 68.0) and (e50s is not None and e50s > 0.0):
                if (v20_for_boost is None) or (-20.0 <= v20_for_boost <= 180.0):
                    struct = clamp(struct + 0.08, -1.0, 1.0)

            # stability (inverse risk)
            atrp = safe(feats.get("ATRpct"), None)
            dd   = safe(feats.get("drawdown_pct"), None)
            v20  = safe(feats.get("vol_vs20"), None)
            atr_pen = 0.0 if atrp is None else (-clamp((atrp - P.atr_soft_cap)/P.atr_soft_cap, 0.0, 1.0))
            dd_pen  = 0.0 if dd   is None else (-clamp((abs(min(dd,0.0)) - abs(P.dd_soft_cap))/25.0, 0.0, 1.0))
            vol_pen = 0.0 if v20  is None else (-clamp((v20 - P.vol20_soft_cap)/P.vol20_soft_cap, 0.0, 1.0))

            # Mean-reversion risk if far above EMA50
            ext_pen = 0.0
            if vsem50 is not None and vsem50 >= 20:
                ext_pen = -clamp((vsem50 - 20)/40.0, 0.0, 0.3)

            # ATH guard
            ath_guard = int(os.getenv("ATH_GUARD", "1")) != 0
            ath_sev = 0.0
            ath_relief = 1.0
            if ath_guard:
                near_pct   = _env_float("ATH_NEAR_PCT", 1.0)   # within X% of 52w high
                min_rsi    = _env_float("ATH_MIN_RSI", 80.0)
                min_vs50   = _env_float("ATH_MIN_VS50", 25.0)
                vol_relief = _env_float("ATH_VOL_RELIEF", 60.0)

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

            # distribution / stall penalties
            stall_pen = 0.0
            if accel_z < -0.25:
                stall_pen -= min(0.08, (abs(accel_z) - 0.25) * 0.10)
            if (rsi is not None and 55 <= rsi <= 75) and (macd < 0.0) and (vs50 > 0.0):
                stall_pen -= 0.08
            if (rsi is not None and 55 <= rsi <= 70) and (vs50 > 0.0) and (v20 is not None and v20 <= -20):
                stall_pen -= 0.04

            # stability aggregates (include ko_pen)
            stability = clamp(
                0.45*atr_pen +
                0.20*dd_pen  +
                0.10*vol_pen +
                0.10*ext_pen +
                0.10*stall_pen +
                0.20*ko_pen,
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

            # valuation (with FVA term)
            value = clamp(value + fva_term, -1.0, 1.0)

            # ★ Apply probe add-back on struct/stability (normalized: PROBE_ADD_BACK_X / 100)
            probe_ok, probe_lvl = _probe_gate(feats)
            if probe_ok:
                add_pts = _env_float("PROBE_ADD_BACK_A", 4.0) if probe_lvl == 1 else \
                          _env_float("PROBE_ADD_BACK_B", 7.0) if probe_lvl == 2 else \
                          _env_float("PROBE_ADD_BACK_C", 10.0)
                add = clamp(add_pts / 100.0, 0.0, 0.20)  # max +0.20 on the -1..+1 scale
                share = _env_float("PROBE_STRUCT_SHARE", 0.45)
                struct = clamp(struct + add * share, -1.0, 1.0)
                stability = clamp(stability + add * (1.0 - share), -1.0, 0.0)

            parts = {
                "trend": trend,
                "momo": momo,
                "struct": struct,
                "stability": stability,
                "blowoff": blow,
                "value": value,
                # debug extras (not weighted)
                "accel_z": accel_z,
                "ko_pen": ko_pen,
                "probe_ok": bool01(probe_ok),
                "probe_lvl": float(probe_lvl) if probe_ok else 0.0,
            }
            w = {
                "trend": P.w_trend,
                "momo": P.w_momo,
                "struct": P.w_struct,
                "stability": P.w_stab,
                "blowoff": P.w_blowoff,
                "value": P.w_value,
            }
            active_w = sum(w.values()) or 1.0
            score_unit = sum(w[k] * parts[k] for k in w.keys())
            scr = clamp((score_unit / active_w) * 100.0, -100.0, 100.0)

            # Final ATH haircut on the headline score
            if 'ath_sev' in locals() and ath_sev > 0:
                haircut = _env_float("ATH_SCORE_HAIRCUT", 22.0)
                scr = clamp(scr - haircut * ath_sev * ath_relief, -100.0, 100.0)

            if self.verbose and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Ranker] parts trend=%.3f momo=%.3f struct=%.3f stab=%.3f blow=%.3f value=%.3f -> score=%.2f (accel=%.3f ko=%.3f probe=%s/L%d)",
                    parts["trend"], parts["momo"], parts["struct"], parts["stability"], parts["blowoff"], parts["value"], scr,
                    parts["accel_z"], parts["ko_pen"], str(bool(probe_ok)), int(probe_lvl)
                )
            return (scr, parts)
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
        A = Aggressive, B = Balanced (default), C = Conservative.
        Controlled by RANKER_PROFILE (or SELECTION_MODE) env.
        """
        prof = (os.getenv("RANKER_PROFILE", os.getenv("SELECTION_MODE", "B")) or "B").upper()
        P = self.params
        if prof == "A":
            # More momentum/trend, looser stability
            P.w_trend, P.w_momo, P.w_struct, P.w_stab, P.w_blowoff, P.w_value = 0.32, 0.36, 0.16, 0.10, 0.02, 0.04
            P.atr_soft_cap, P.vol20_soft_cap, P.dd_soft_cap = 6.8, 300.0, -40.0
        elif prof == "C":
            # Value/stability forward; harsher blowoff
            P.w_trend, P.w_momo, P.w_struct, P.w_stab, P.w_blowoff, P.w_value = 0.26, 0.28, 0.16, 0.22, 0.04, 0.04
            P.atr_soft_cap, P.vol20_soft_cap, P.dd_soft_cap = 5.5, 220.0, -30.0
        else:
            # B = defaults set in dataclass
            pass

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
