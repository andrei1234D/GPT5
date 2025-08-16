# scripts/prompt_blocks.py
from typing import Tuple, Optional

BASELINE_HINTS = {
    "MKT_SECTOR": 200,
    "FUNDAMENTALS": 125,
    "CATALYSTS": 100,
    "VALUATION": 50,
    "RISKS": 50,
}

def _fmt_pct(x):
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"

def _fmt_num(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"

def build_prompt_block(
    t: str, name: str, feats: dict, proxies: dict, fund_proxy: dict,
    cat: dict, earn_sev: int, fm: dict, baseline_hints: dict, baseline_str: str,
    pe_hint: Optional[float] = None,
) -> Tuple[str, dict]:
    """
    Builds the per-ticker block for GPT and returns (block_text, debug_dict).
    - Marks DATA_AVAILABILITY for FUNDAMENTALS, CATALYSTS, VALUATION explicitly.
    - Fills valuation fields from feats['val_*'] when fm is missing entries.
    """
    fm = dict(fm or {})

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

    # Default pe_hint from the (now) completed fm if not provided
    if pe_hint is None:
        pe_hint = fm.get("PE")

    # --- Availability flags ---
    # FUNDAMENTALS: mark PARTIAL if we have any tech-fund proxies present (keys exist), else MISSING.
    has_fund_proxies = any(k in fund_proxy for k in (
        "GROWTH_TECH", "MARGIN_TREND_TECH", "FCF_TREND_TECH", "OP_EFF_TREND_TECH"
    ))
    fundamentals_availability = "FUNDAMENTALS=PARTIAL" if has_fund_proxies else "FUNDAMENTALS=MISSING"

    # VALUATION: PARTIAL if we have any of the simple ratios OR the technical valuation hints (FVA or ValHist).
    val_keys = ["PE", "PS", "EV_EBITDA", "EV_REV", "FCF_YIELD", "PEG"]
    val_present_ct = sum(1 for k in val_keys if fm.get(k) is not None)
    has_val_tech = (proxies.get("fva_hint") is not None) or (proxies.get("valuation_history") not in (None, 0))
    valuation_availability = "VALUATION=PARTIAL" if (val_present_ct > 0 or has_val_tech) else "VALUATION=MISSING"

    # CATALYSTS: PARTIAL if any catalyst severity non-zero, earnings sev > 0, or timing suggests breakout.
    has_cat_signal = (
        any(abs(cat.get(k, 0)) > 0 for k in ("TECH_BREAKOUT", "TECH_BREAKDOWN", "DIP_REVERSAL")) or
        (earn_sev or 0) > 0 or
        bool(feats.get("is_20d_high"))
    )
    catalysts_availability = "CATALYSTS=PARTIAL" if has_cat_signal else "CATALYSTS=MISSING"

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

    # --------- Compose prompt block text ----------
    block_text = (
        "TICKER: {t}\n"
        "COMPANY: {name}\n"
        "CURRENT_PRICE: ${price}\n"
        "vsSMA20: {vs20}%\n"
        "vsSMA50: {vs50}%\n"
        "vsSMA200: {vs200}%\n"
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
        "vsEMA50: {vsem50}%\n"
        "vsEMA200: {vsem200}%\n"
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
        price=feats.get("price"),
        vs20=feats.get("vsSMA20"), vs50=feats.get("vsSMA50"), vs200=feats.get("vsSMA200"),
        sma50=feats.get("SMA50"), avwap=feats.get("AVWAP252"),
        rsi=feats.get("RSI14"), macd=feats.get("MACD_hist"), atr=feats.get("ATRpct"),
        dd=feats.get("drawdown_pct"), d5=feats.get("d5"), d20=feats.get("d20"), r60=feats.get("r60"),
        v20=feats.get("vol_vs20"),
        ema20=feats.get("EMA20"), ema50=feats.get("EMA50"), ema200=feats.get("EMA200"),
        vsem50=feats.get("vsEMA50"), vsem200=feats.get("vsEMA200"),
        ema50s=feats.get("EMA50_slope_5d"),
        val_fields=val_fields,
        funds_avail=fundamentals_availability,
        cats_avail=catalysts_availability,     # <-- ensure catalysts availability is emitted
        vals_avail=valuation_availability,
        baselines=baseline_str,
        mt=proxies["market_trend"], rs=proxies["relative_strength"],
        bv=proxies["breadth_volume"], vh=proxies["valuation_history"],
        rv=proxies["risk_volatility"], rd=proxies["risk_drawdown"],
        gt=("+" + str(fund_proxy.get("GROWTH_TECH", 0))) if fund_proxy.get("GROWTH_TECH", 0) > 0 else str(fund_proxy.get("GROWTH_TECH", 0)),
        mtf=("+" + str(fund_proxy.get("MARGIN_TREND_TECH", 0))) if fund_proxy.get("MARGIN_TREND_TECH", 0) > 0 else str(fund_proxy.get("MARGIN_TREND_TECH", 0)),
        ft=("+" + str(fund_proxy.get("FCF_TREND_TECH", 0))) if fund_proxy.get("FCF_TREND_TECH", 0) > 0 else str(fund_proxy.get("FCF_TREND_TECH", 0)),
        ot=("+" + str(fund_proxy.get("OP_EFF_TREND_TECH", 0))) if fund_proxy.get("OP_EFF_TREND_TECH", 0) > 0 else str(fund_proxy.get("OP_EFF_TREND_TECH", 0)),
        cat_line=catalyst_line, timing_tb=timing_tb,
        ev=proxies.get("expected_volatility_pct"),
        fva=proxies.get("fva_hint") if proxies.get("fva_hint") is not None else "N/A",
        pe=(f"{pe_hint:.2f}" if isinstance(pe_hint, (int, float)) else "N/A"),
        bon=proxies.get("suggested_bonuses", "NONE"),
    )

    # --------- Debug payload (define helper dicts BEFORE using) ----------
    proxies_catalysts_full = {**(cat or {}), "EARNINGS_SOON": earn_sev}
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

        "PROMPT_BLOCK": block_text,
    }

    return block_text, debug_dict
