# scripts/prompt_blocks.py
import os
from typing import Tuple, Optional
import math

BASELINE_HINTS = {
    "MKT_SECTOR": 110,
    " Quality (Tech Proxies)": 130,  # (kept as-is to avoid downstream diffs)
    "Near-Term Catalysts": 75,
    "Technical Valuation": 110,
    "RISKS": 25,
}

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
# ---------------------------------------------------------------------------------- #

def build_prompt_block(
    t: str, name: str, feats: dict, proxies: dict, fund_proxy: dict,
    cat: dict, earn_sev: int, fm: dict, baseline_hints: dict, baseline_str: str,
    pe_hint: Optional[float] = None,
) -> Tuple[str, dict]:
    """
    Builds the per-ticker block for GPT and returns (block_text, debug_dict).
    - Fills valuation fields from feats['val_*'] when fm is missing entries.
    - Uses robust formatting so numpy.float64 etc. don't print as N/A.
    """
    fm = dict(fm or {})
    fund_proxy = dict(fund_proxy or {})
    proxies = dict(proxies or {})
    cat = dict(cat or {})

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

    # ---------- CATALYSTS AVAILABILITY (fix: zeros ≠ missing) ----------
    # Treat “keys present but all zero” as PARTIAL (we *do* have data; it says no active signals now).
    # Only mark MISSING if we truly lack the keys *and* have no timing/earnings hints.
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

    # --- Valuation fields line ---
    val_fields = (
        "VALUATION_FIELDS: "
        f"PE={_fmt_num(fm.get('PE'))}; "
        f"PE_SECTOR=N/A; "
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

    # --------- Compose prompt block text (all values formatted) ----------
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
        "DATA_AVAILABILITY: {funds_avail}; {cats_avail}; {vals_avail}\n"
        "BASELINE_HINTS: {baselines}\n"
        "PROXIES: MARKET_TREND={mt}; REL_STRENGTH={rs}; BREADTH_VOLUME={bv}; VALUATION_HISTORY={vh}; RISK_VOLATILITY={rv}; RISK_DRAWDOWN={rd}\n"
        "PROXIES_FUNDAMENTALS: GROWTH_TECH={gt}; MARGIN_TREND_TECH={mtf}; FCF_TREND_TECH={ft}; OP_EFF_TREND_TECH={ot}\n"
        "PROXIES_CATALYSTS: {cat_line}\n"
        "CATALYST_TIMING_HINTS: {timing_tb}\n"
        "EXPECTED_VOLATILITY_PCT: {ev}\n"
        "FVA_HINT: {fva}\n"
        "PE_HINT: {pe}\n"
        "SUGGESTED_BONUSES: {bon}\n"
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
        ev=_fmt_num(proxies.get("expected_volatility_pct")),
        fva=_fmt_num(proxies.get("fva_hint")),
        pe=_fmt_num(pe_hint),
        bon=proxies.get("suggested_bonuses", "NONE"),
    )

    # --------- Debug payload ----------
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
            "EXPECTED_VOLATILITY_PCT": proxies.get("expected_volatility_pct"),
            "FVA_HINT": proxies.get("fva_hint"),
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

        "PROMPT_BLOCK": block_text,
    }

    return block_text, debug_dict
