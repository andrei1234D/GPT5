# scripts/prompt_blocks.py
import os
from typing import Tuple, Optional
import math


# -------- robust numeric helpers (handle numpy, Decimal, strings, NaN/inf) -------- #
def _to_float(x):
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def _fmt_num(x):
    xf = _to_float(x)
    return "N/A" if xf is None else f"{xf:.2f}"

def _fmt_pct(x):
    xf = _to_float(x)
    return "N/A" if xf is None else f"{xf:.2f}%"

def _clamp(v, lo, hi):
    v = _to_float(v)
    if v is None: return None
    return max(lo, min(hi, v))

def _yn(b: bool) -> str:
    return "Yes" if b else "No"
# ---------------------------------------------------------------------------------- #

# --------- shared gates (mirror rankers) --------- #
def _overheat_level(rsi: Optional[float], vs50: Optional[float]) -> int:
    if rsi is None or vs50 is None:
        return 0
    A_RSI = float(os.getenv("OVERHEAT_A_RSI", "78"))
    B_RSI = float(os.getenv("OVERHEAT_B_RSI", "83"))
    C_RSI = float(os.getenv("OVERHEAT_C_RSI", "87"))
    A_V50 = float(os.getenv("OVERHEAT_A_VS50", "20"))
    B_V50 = float(os.getenv("OVERHEAT_B_VS50", "35"))
    C_V50 = float(os.getenv("OVERHEAT_C_VS50", "50"))
    r, v = float(rsi), float(vs50)
    if r >= C_RSI and v >= C_V50: return 3
    if r >= B_RSI and v >= B_V50: return 2
    if r >= A_RSI and v >= A_V50: return 1
    return 0

def _probe_gate(feats: dict) -> Tuple[bool, int]:
    allow = (os.getenv("ALLOW_BLOWOFF_PROBE", "1").lower() in {"1","true","yes"})
    if not allow:
        return False, 0
    rsi  = _to_float(feats.get("RSI14"))
    vs50 = _to_float(feats.get("vsSMA50"))
    atr  = _to_float(feats.get("ATRpct"))
    v20  = _to_float(feats.get("vol_vs20"))
    lvl = _overheat_level(rsi, vs50)
    if lvl == 0: return False, 0
    ev_ok  = (atr is not None) and (atr >= float(os.getenv("PROBE_MIN_EV", "4")))
    vol_ok = (v20 is None) or (v20 <= float(os.getenv("PROBE_MAX_VOL_SPIKE", "150")))
    return (ev_ok and vol_ok), lvl

# ---------- ranker-aligned “setup” flags ---------- #
def _flag_early_turn(feats: dict) -> Tuple[bool, dict]:
    """
    Mirror of the 'early turn' idea used in ranking:
      - RSI 47..63
      - EMA50 5d slope >= 2
      - modest participation (vol_vs20 ~110..220) if present
      - price >= EMA20
      - vsEMA200 in a 'near/just below' band [-12..+8]
    """
    rsi      = _to_float(feats.get("RSI14"))
    e50s     = _to_float(feats.get("EMA50_slope_5d"))
    v20      = _to_float(feats.get("vol_vs20"))
    px       = _to_float(feats.get("price"))
    e20      = _to_float(feats.get("EMA20"))
    vsem200  = _to_float(feats.get("vsEMA200"))

    lo_rsi   = float(os.getenv("EARLY_TURN_RSI_LO", "47.0"))
    hi_rsi   = float(os.getenv("EARLY_TURN_RSI_HI", "63.0"))
    min_slope= float(os.getenv("EARLY_TURN_MIN_E50_SLOPE", "2.0"))
    lo_v200  = float(os.getenv("EARLY_TURN_VS200_LO", "-12.0"))
    hi_v200  = float(os.getenv("EARLY_TURN_VS200_HI", "8.0"))
    lo_vol   = float(os.getenv("EARLY_TURN_VOL_LO", "110.0"))
    hi_vol   = float(os.getenv("EARLY_TURN_VOL_HI", "220.0"))

    rsi_ok   = (rsi is not None) and (lo_rsi <= rsi <= hi_rsi)
    slope_ok = (e50s is not None) and (e50s >= min_slope)
    vol_ok   = (v20 is None) or (lo_vol <= v20 <= hi_vol)
    px_ok    = (px is not None) and (e20 is not None) and (px >= e20)
    near200  = (vsem200 is not None) and (lo_v200 <= vsem200 <= hi_v200)

    ok = bool(rsi_ok and slope_ok and vol_ok and px_ok and near200)
    dbg = {
        "RSI14": rsi, "EMA50_slope_5d": e50s, "vol_vs20": v20,
        "price": px, "EMA20": e20, "vsEMA200": vsem200,
        "rsi_gate": rsi_ok, "slope_gate": slope_ok, "vol_gate": vol_ok,
        "px_gate": px_ok, "near200_gate": near200,
    }
    return ok, dbg

def _flag_coil_setup(feats: dict) -> Tuple[bool, dict]:
    """
    Coil/setup similar to the “setup_protect” tag in the scorer:
      - vsSMA50 in [-4, 10], vsSMA200 >= -2
      - RSI in [45, 62]
      - ATRpct <= QS_SETUP_ATR_MAX (default 6)
      - vol_vs20 in [-60, 40]
    """
    vs50 = _to_float(feats.get("vsSMA50"))
    vs200= _to_float(feats.get("vsSMA200"))
    rsi  = _to_float(feats.get("RSI14"))
    atr  = _to_float(feats.get("ATRpct"))
    v20  = _to_float(feats.get("vol_vs20"))

    atr_max = float(os.getenv("QS_SETUP_ATR_MAX", "6.0"))

    ok = (
        (vs50 is not None and -4.0 <= vs50 <= 10.0) and
        (vs200 is not None and vs200 >= -2.0) and
        (rsi is not None and 45.0 <= rsi <= 62.0) and
        (atr is not None and atr <= atr_max) and
        (v20 is None or -60.0 <= v20 <= 40.0)
    )
    dbg = {"vsSMA50": vs50, "vsSMA200": vs200, "RSI14": rsi, "ATRpct": atr, "vol_vs20": v20, "ATR_max": atr_max}
    return bool(ok), dbg

def _flag_continuation(feats: dict) -> Tuple[bool, dict]:
    """
    In-motion but not extended:
      - vsEMA50 in [3, 18], vsEMA200 >= -2
      - RSI in [52, 68]
      - EMA50_slope_5d > 0
      - vol_vs20 <= 180 (if present)
    """
    vsem50 = _to_float(feats.get("vsEMA50"))
    vsem200= _to_float(feats.get("vsEMA200"))
    rsi    = _to_float(feats.get("RSI14"))
    e50s   = _to_float(feats.get("EMA50_slope_5d"))
    v20    = _to_float(feats.get("vol_vs20"))

    ok = (
        (vsem50 is not None and 3.0 <= vsem50 <= 18.0) and
        (vsem200 is not None and vsem200 >= -2.0) and
        (rsi is not None and 52.0 <= rsi <= 68.0) and
        (e50s is not None and e50s > 0.0) and
        (v20 is None or v20 <= 180.0)
    )
    dbg = {"vsEMA50": vsem50, "vsEMA200": vsem200, "RSI14": rsi, "EMA50_slope_5d": e50s, "vol_vs20": v20}
    return bool(ok), dbg

# ---------- small helpers for plan inputs ---------- #
def _derive_ev_pct(feats: dict, proxies: dict) -> float:
    ev = _to_float(proxies.get("expected_volatility_pct"))
    if ev is None:
        atr = _to_float(feats.get("ATRpct"))
        if atr is None:
            return 2.0
        ev = _clamp(atr, 1.0, 6.0)
    return float(ev)

def _derive_fva_hint(feats: dict, proxies: dict) -> Optional[float]:
    fva = _to_float(proxies.get("fva_hint"))
    if fva is None:
        fva = _to_float(feats.get("FVA_HINT"))
    if fva is None:
        fva = _to_float(feats.get("AVWAP252"))
    return fva

def _ath_guard(feats: dict) -> Tuple[bool, dict]:
    if os.getenv("ATH_GUARD", "1") not in {"1","true","TRUE","True"}:
        return False, {"active": False}
    near_pct   = float(os.getenv("ATH_NEAR_PCT", "1.0"))
    min_rsi    = float(os.getenv("ATH_MIN_RSI", "80"))
    min_vs50   = float(os.getenv("ATH_MIN_VS50", "25"))   # vsEMA50 threshold
    vol_relief = float(os.getenv("ATH_VOL_RELIEF", "60"))
    dd  = _to_float(feats.get("drawdown_pct"))
    rsi = _to_float(feats.get("RSI14"))
    v50 = _to_float(feats.get("vsEMA50"))
    v20 = _to_float(feats.get("vol_vs20"))
    trig = (dd is not None and dd >= -near_pct and
            rsi is not None and rsi >= min_rsi and
            v50 is not None and v50 >= min_vs50)
    relief = (v20 is not None and v20 >= vol_relief)
    return bool(trig), {
        "active": True, "triggered": bool(trig),
        "NEAR_PCT": near_pct, "MIN_RSI": min_rsi, "MIN_VS50": min_vs50, "VOL_RELIEF": vol_relief,
        "relief_applies": relief
    }

def _ko_context(feats: dict, fva: Optional[float]) -> Tuple[Optional[float], bool]:
    px  = _to_float(feats.get("price"))
    rsi = _to_float(feats.get("RSI14"))
    v50 = _to_float(feats.get("vsEMA50"))
    d20 = _to_float(feats.get("d20"))
    if px is None or fva is None or fva == 0:
        return None, False
    gap_pct = (px - fva) / abs(fva) * 100.0
    extended = ((rsi is not None and rsi >= 72) or
                (v50 is not None and v50 >= 40) or
                (d20 is not None and d20 >= 15))
    return gap_pct, extended

# ---------- main builder ---------- #
def build_prompt_block(
    t: str, name: str, feats: dict, proxies: dict, fund_proxy: dict,
    cat: dict, earn_sev: int, fm: dict, baseline_hints: dict, baseline_str: str,
    pe_hint: Optional[float] = None,
) -> Tuple[str, dict]:
    """
    Builds the per-ticker block for GPT and returns (block_text, debug_dict).
    - Existing line/label order is preserved.
    - New, ranker-aligned hints are appended at the end (SETUP_FLAGS, RISK_GUARDS, KO_CONTEXT, PLAN_SPEC).
    - Robust formatting so numpy.float64 etc. don't print as N/A.
    """
    fm = dict(fm or {})
    fund_proxy = dict(fund_proxy or {})
    proxies = dict(proxies or {})
    cat = dict(cat or {})


    fm["PE"]  = feats.get("val_PE")   if feats.get("val_PE")  is not None else fm.get("PE")
    fm["PEG"] = feats.get("val_PEG")  if feats.get("val_PEG") is not None else fm.get("PEG")
    fm["YoY_Growth"] = feats.get("val_YoY")  # ensure YoY is always carried

    # ---- Fill valuation fields from feats[...] if missing in fm ----
    def _pick(key: str, feat_key: str):
        if fm.get(key) is None:
            fm[key] = feats.get(feat_key)

    _pick("PE",         "val_PE")
    _pick("PS",         "val_PS")
    _pick("EV_EBITDA",  "val_EV_EBITDA")
    _pick("EV_REV",     "val_EV_REV")
    _pick("PEG",        "val_PEG")
    _pick("FCF_YIELD",  "val_FCF_YIELD")  # may be negative; that's fine

    # Normalize to plain finite floats
    for k in ("PE", "PS", "EV_EBITDA", "EV_REV", "PEG", "FCF_YIELD"):
        fm[k] = _to_float(fm.get(k))

    # Default pe_hint from the (now) completed fm if not provided
    if pe_hint is None:
        pe_hint = fm.get("PE")

    # --- Availability flags ---
    has_fund_proxies = any(k in fund_proxy for k in (
        "GROWTH_TECH", "MARGIN_TREND_TECH", "FCF_TREND_TECH", "OP_EFF_TREND_TECH"
    ))
    fundamentals_availability = "FUNDAMENTALS=PARTIAL" if has_fund_proxies else "FUNDAMENTALS=MISSING"

    val_keys = ["PE", "PS", "EV_EBITDA", "EV_REV", "FCF_YIELD", "PEG"]
    val_present_ct = sum(1 for k in val_keys if fm.get(k) is not None)
    has_val_tech = (proxies.get("fva_hint") is not None) or (proxies.get("valuation_history") not in (None, 0))
    valuation_availability = "VALUATION=PARTIAL" if (val_present_ct > 0 or has_val_tech) else "VALUATION=MISSING"

    # ---------- CATALYSTS AVAILABILITY ----------
    cat_keys = ("TECH_BREAKOUT", "TECH_BREAKDOWN", "DIP_REVERSAL")
    has_any_cat_key = any(k in cat for k in cat_keys)
    has_nonzero_cat = any((cat.get(k) or 0) != 0 for k in cat_keys)
    has_earn_hint = (earn_sev or 0) != 0
    has_breakout_timing_source = feats.get("is_20d_high") is not None  # we *can* infer timing context
    force_partial = os.getenv("FORCE_CATALYSTS_PARTIAL", "1").lower() in {"1", "true", "yes"}
    if has_any_cat_key or has_earn_hint or has_breakout_timing_source or force_partial:
        catalysts_availability = "CATALYSTS=PARTIAL"
    else:
        catalysts_availability = "CATALYSTS=MISSING"
    # -------------------------------------------------------------------

    # --- derive plan inputs/hints aligned with rankers ---
    fva_hint = _derive_fva_hint(feats, proxies)
    ev_pct   = _derive_ev_pct(feats, proxies)
    early_ok, early_dbg = _flag_early_turn(feats)
    coil_ok,  coil_dbg  = _flag_coil_setup(feats)
    cont_ok,  cont_dbg  = _flag_continuation(feats)
    probe_ok, probe_lvl = _probe_gate(feats)
    ath_trig, ath_dbg   = _ath_guard(feats)
    ko_gap_pct, ko_ext  = _ko_context(feats, fva_hint)

    # --- Valuation fields line (labels must remain unchanged) ---
    val_fields = (
    "VALUATION_FIELDS: "
    f"PE={_fmt_num(fm.get('PE'))}; "
    f"YoY_Growth={_fmt_pct(fm.get('YoY_Growth'))}; "
    f"EV_EBITDA={_fmt_num(fm.get('EV_EBITDA'))}; "
    f"EV_REV={_fmt_num(fm.get('EV_REV'))}; "
    f"PS={_fmt_num(fm.get('PS'))}; "
    f"FCF_YIELD={_fmt_pct(fm.get('FCF_YIELD'))}; "
    f"PEG={_fmt_num(fm.get('PEG'))}"
)

    def sgn(k: int) -> str:
        return ("+" + str(k)) if k > 0 else ("-" + str(abs(k)) if k < 0 else "0")

    # Catalyst lines shown to GPT
    catalyst_line = "TECH_BREAKOUT={}; TECH_BREAKDOWN={}; DIP_REVERSAL={}; EARNINGS_SOON={}".format(
        sgn(cat.get("TECH_BREAKOUT", 0)), sgn(cat.get("TECH_BREAKDOWN", 0)),
        sgn(cat.get("DIP_REVERSAL", 0)), sgn(earn_sev)
    )
    timing_tb = "TECH_BREAKOUT={}".format(
        "Today" if (cat.get("TECH_BREAKOUT", 0) > 0 and feats.get("is_20d_high")) else "None"
    )

    # --------- Compose prompt block text (existing order preserved) ----------
    block_text = (
        "TICKER: {t}\n"
        "COMPANY: {name}\n"
        "CURRENT_PRICE: ${price}\n"
        "vsSMA20: {vs20}\n"
        "vsSMA50: {vs50}\n"
        "vsSMA200: {vs200}\n"
        "SMA50: {sma50}\n"
        "AVWAP252: {avwap}\n"
        "RSI14: {rsi}\n"
        "MACD_hist: {macd}\n"
        "ATR%: {atr}\n"
        "Drawdown%: {dd}\n"
        "5d%: {d5}\n"
        "20d%: {d20}\n"
        "60d%: {r60}\n"
        "Vol_vs_20d%: {v20}\n"
        "EMA20: {ema20}\n"
        "EMA50: {ema50}\n"
        "EMA200: {ema200}\n"
        "vsEMA50: {vsem50}\n"
        "vsEMA200: {vsem200}\n"
        "EMA50_slope_5d%: {ema50s}\n"
        "{val_fields}\n"
        "BASELINE_HINTS: {baselines}\n"
        "PROXIES: MARKET_TREND={mt}; REL_STRENGTH={rs}; BREADTH_VOLUME={bv}; VALUATION_HISTORY={vh}; RISK_VOLATILITY={rv}; RISK_DRAWDOWN={rd}\n"
        "PROXIES_FUNDAMENTALS: GROWTH_TECH={gt}; MARGIN_TREND_TECH={mtf}; FCF_TREND_TECH={ft}; OP_EFF_TREND_TECH={ot}\n"
        "PROXIES_CATALYSTS: {cat_line}\n"
        "CATALYST_TIMING_HINTS: {timing_tb}\n"
        "EXPECTED_VOLATILITY_PCT: {ev}\n"
        "FVA_HINT: {fva}\n"
        "PE_HINT: {pe}\n"
        "SUGGESTED_BONUSES: {bon}\n"
        # ---- appended ranker-aligned hints (new) ----
        "SETUP_FLAGS: EARLY_TURN={early}; COIL_SETUP={coil}; CONTINUATION_OK={cont}\n"
        "RISK_GUARDS: ATH_GUARD_ACTIVE={ath_active}; ATH_TRIGGERED={ath_trig}; PROBE_ELIGIBLE={probe_ok}; PROBE_LEVEL={probe_lvl}\n"
        "KO_CONTEXT: GAP_PCT={ko_gap}; EXTENDED={ko_ext}\n"
        # ---- explicit plan contract (one line; LLM uses this strictly) ----
        "PLAN_SPEC: Use FVA as anchor (|FVA−PRICE| ≤ 25% if CONTINUATION_OK or Certainty ≥ 80%). Let EV=clamp(ATR%,1..6). Buy=FVA×(1−0.8×EV/100)…FVA×(1+0.8×EV/100); Stop=FVA×(1−2.0×EV/100); Target=FVA×(1+3.0×EV/100). Fix conflicts: if stop≥buy_low → stop=min(buy_low×0.99,FVA×(1−2.2×EV/100)); if target≤buy_high → target=max(buy_high×1.05,FVA×(1+3.2×EV/100)). Round $ to 2 decimals and output exactly: Buy X–Y; Stop Z; Target T; Max hold time: ≤ 1 year (Anchor: $FVA).\n"
    ).format(
        t=t, name=name,
        price=_fmt_num(feats.get("price")),
        vs20=_fmt_pct(feats.get("vsSMA20")), vs50=_fmt_pct(feats.get("vsSMA50")), vs200=_fmt_pct(feats.get("vsSMA200")),
        sma50=_fmt_num(feats.get("SMA50")), avwap=_fmt_num(feats.get("AVWAP252")),
        rsi=_fmt_num(feats.get("RSI14")), macd=_fmt_num(feats.get("MACD_hist")), atr=_fmt_pct(feats.get("ATRpct")),
        dd=_fmt_pct(feats.get("drawdown_pct")), d5=_fmt_pct(feats.get("d5")), d20=_fmt_pct(feats.get("d20")), r60=_fmt_pct(feats.get("r60")),
        v20=_fmt_pct(feats.get("vol_vs20")),
        ema20=_fmt_num(feats.get("EMA20")), ema50=_fmt_num(feats.get("EMA50")), ema200=_fmt_num(feats.get("EMA200")),
        vsem50=_fmt_pct(feats.get("vsEMA50")), vsem200=_fmt_pct(feats.get("vsEMA200")),
        ema50s=_fmt_pct(feats.get("EMA50_slope_5d")),
        val_fields=val_fields,
        funds_avail=fundamentals_availability,
        cats_avail=catalysts_availability,
        vals_avail=valuation_availability,
        baselines=baseline_str,
        mt=proxies.get("market_trend"), rs=proxies.get("relative_strength"),
        bv=proxies.get("breadth_volume"), vh=proxies.get("valuation_history"),
        rv=proxies.get("risk_volatility"), rd=proxies.get("risk_drawdown"),
        gt=("+" + str(fund_proxy.get("GROWTH_TECH", 0))) if fund_proxy.get("GROWTH_TECH", 0) > 0 else str(fund_proxy.get("GROWTH_TECH", 0)),
        mtf=("+" + str(fund_proxy.get("MARGIN_TREND_TECH", 0))) if fund_proxy.get("MARGIN_TREND_TECH", 0) > 0 else str(fund_proxy.get("MARGIN_TREND_TECH", 0)),
        ft=("+" + str(fund_proxy.get("FCF_TREND_TECH", 0))) if fund_proxy.get("FCF_TREND_TECH", 0) > 0 else str(fund_proxy.get("FCF_TREND_TECH", 0)),
        ot=("+" + str(fund_proxy.get("OP_EFF_TREND_TECH", 0))) if fund_proxy.get("OP_EFF_TREND_TECH", 0) > 0 else str(fund_proxy.get("OP_EFF_TREND_TECH", 0)),
        cat_line=catalyst_line, timing_tb=timing_tb,
        ev=_fmt_num(ev_pct),
        fva=_fmt_num(fva_hint),
        pe=_fmt_num(pe_hint),
        bon=proxies.get("suggested_bonuses", "NONE"),
        early=_yn(early_ok), coil=_yn(coil_ok), cont=_yn(cont_ok),
        ath_active=_yn(bool(ath_dbg.get("active"))), ath_trig=_yn(ath_trig),
        probe_ok=_yn(probe_ok),
        probe_lvl=("A" if probe_lvl==1 else "B" if probe_lvl==2 else "C" if probe_lvl==3 else "None"),
        ko_gap=("N/A" if ko_gap_pct is None else f"{ko_gap_pct:.2f}%"),
        ko_ext=_yn(bool(ko_ext)),
    )

    # --------- Debug payload (expanded) ----------
    proxies_catalysts_full = {**cat, "EARNINGS_SOON": earn_sev}
    catalyst_timing_hints = {
        "TECH_BREAKOUT": ("Today" if (cat.get("TECH_BREAKOUT", 0) > 0 and feats.get("is_20d_high")) else "None")
    }

    val_fields_dict = {
    "PE": fm.get("PE"),
    "PE_SECTOR": None,
    "EV_EBITDA": fm.get("EV_EBITDA"),
    "EV_REV": fm.get("EV_REV"),
    "PS": fm.get("PS"),
    "FCF_YIELD_pct": fm.get("FCF_YIELD"),
    "PEG": fm.get("PEG"),
    "YoY_Growth": fm.get("YoY_Growth"),
    "VAL_FIELDS_PRESENT_COUNT": val_present_ct,
    "PE_PRESENT_BOOL": (fm.get("PE") is not None and fm.get("PE") > 0),
}
    indicators_dict = {
        "vsSMA20_pct": feats.get("vsSMA20"),
        "vsSMA50_pct": feats.get("vsSMA50"),
        "vsSMA200_pct": feats.get("vsSMA200"),
        "SMA50": feats.get("SMA50"),
        "AVWAP252": feats.get("AVWAP252"),
        "RSI14": feats.get("RSI14"),
        "MACD_hist": feats.get("MACD_hist"),
        "ATR_pct": feats.get("ATRpct"),
        "Drawdown_pct": feats.get("drawdown_pct"),
        "d5_pct": feats.get("d5"),
        "d20_pct": feats.get("d20"),
        "r60_pct": feats.get("r60"),
        "Vol_vs_20d_pct": feats.get("vol_vs20"),
        "EMA20": feats.get("EMA20"),
        "EMA50": feats.get("EMA50"),
        "EMA200": feats.get("EMA200"),
        "vsEMA50_pct": feats.get("vsEMA50"),
        "vsEMA200_pct": feats.get("vsEMA200"),
        "EMA50_slope_5d_pct": feats.get("EMA50_slope_5d"),
    }

    debug_dict = {
        "TICKER": t,
        "COMPANY": name,
        "CURRENT_PRICE": feats.get("price"),

        "FEATURES_RAW": feats,
        "INDICATORS_BLOCK": indicators_dict,

        "BASELINE_HINTS_DICT": baseline_hints,
        "BASELINE_HINTS_STR": baseline_str,

        "VALUATION_FIELDS_DICT": val_fields_dict,
        "PROXIES_BLOCK": {
            "MARKET_TREND": proxies.get("market_trend"),
            "REL_STRENGTH": proxies.get("relative_strength"),
            "BREADTH_VOLUME": proxies.get("breadth_volume"),
            "VALUATION_HISTORY": proxies.get("valuation_history"),
            "RISK_VOLATILITY": proxies.get("risk_volatility"),
            "RISK_DRAWDOWN": proxies.get("risk_drawdown"),
            "EXPECTED_VOLATILITY_PCT": ev_pct,
            "FVA_HINT": fva_hint,
            "SUGGESTED_BONUSES": proxies.get("suggested_bonuses"),
        },
        "PROXIES_FUNDAMENTALS_BLOCK": fund_proxy,
        "PROXIES_CATALYSTS_BLOCK": proxies_catalysts_full,
        "CATALYST_TIMING_HINTS_BLOCK": catalyst_timing_hints,

        "DATA_AVAILABILITY_FLAGS": {
            "fundamentals": fundamentals_availability,
            "catalysts": catalysts_availability,
            "valuation": valuation_availability,
            "has_nonzero_catalyst": has_nonzero_cat,
        },

        # Ranker-aligned introspection:
        "RANKING_TIER": feats.get("liq_tier"),
        "SETUP_FLAGS": {
            "EARLY_TURN": early_ok,
            "EARLY_TURN_DEBUG": early_dbg,
            "COIL_SETUP": coil_ok,
            "COIL_SETUP_DEBUG": coil_dbg,
            "CONTINUATION_OK": cont_ok,
            "CONTINUATION_DEBUG": cont_dbg,
        },
        "BLOWOFF_PROBE": {
            "allowed": bool(probe_ok),
            "level": probe_lvl,
            "env": {
                "ALLOW_BLOWOFF_PROBE": os.getenv("ALLOW_BLOWOFF_PROBE", "1"),
                "PROBE_MIN_EV": os.getenv("PROBE_MIN_EV", "4"),
                "PROBE_MAX_VOL_SPIKE": os.getenv("PROBE_MAX_VOL_SPIKE", "150"),
                "OVERHEAT_RSIs": {
                    "A": os.getenv("OVERHEAT_A_RSI", "78"),
                    "B": os.getenv("OVERHEAT_B_RSI", "83"),
                    "C": os.getenv("OVERHEAT_C_RSI", "87"),
                },
            },
        },
        "ATH_GUARD": ath_dbg,
        "KO_CONTEXT": {"gap_pct": ko_gap_pct, "extended": ko_ext},

        "PROMPT_BLOCK": block_text,
    }
    print(f"[DEBUG VALS] {t}: PE={fm.get('PE')}, YoY={fm.get('YoY_Growth')}, PEG={fm.get('PEG')}")
    return block_text, debug_dict
