# scripts/prompt_blocks.py
from typing import Tuple, Optional

BASELINE_HINTS = {
    "MKT_SECTOR": 200,
    "FUNDAMENTALS": 125,
    "CATALYSTS": 100,
    "VALUATION": 50,
    "RISKS": 50,
}

def _fmt_pct(x): return f"{x:.2f}%" if isinstance(x,(int,float)) else "N/A"
def _fmt_num(x): return f"{x:.2f}" if isinstance(x,(int,float)) else "N/A"

def build_prompt_block(
    t: str, name: str, feats: dict, proxies: dict, fund_proxy: dict,
    cat: dict, earn_sev: int, fm: dict, baseline_hints: dict, baseline_str: str, pe_hint: Optional[float] = None,
) -> Tuple[str, dict]:
    # --- Availability flags ---
    # We’re not supplying GAAP fundamentals; keep FUNDAMENTALS=MISSING so GPT uses baseline for that category.
    fundamentals_availability = "FUNDAMENTALS=MISSING"

    # Count actual valuation fields present
    val_keys = ["PE", "PS", "EV_EBITDA", "EV_REV", "FCF_YIELD", "PEG"]
    val_present_ct = sum(1 for k in val_keys if fm.get(k) is not None)
    valuation_availability = "VALUATION=PARTIAL" if val_present_ct > 0 else "VALUATION=MISSING"

    data_availability = f"{fundamentals_availability}; {valuation_availability}"

    # --- Valuation fields line (now includes EV_REV) ---
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

    def sgn(k:int)->str: return ("+"+str(k)) if k>0 else ("-"+str(abs(k)) if k<0 else "0")
    catalyst_line = "TECH_BREAKOUT={}; TECH_BREAKDOWN={}; DIP_REVERSAL={}; EARNINGS_SOON={}".format(
        sgn(cat.get("TECH_BREAKOUT",0)), sgn(cat.get("TECH_BREAKDOWN",0)), sgn(cat.get("DIP_REVERSAL",0)), sgn(earn_sev)
    )
    timing_tb = "TECH_BREAKOUT={}".format(
        "Today" if (cat.get("TECH_BREAKOUT",0)>0 and feats.get("is_20d_high")) else "None"
    )

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
        "{val_fields}\n"
        "DATA_AVAILABILITY: {data_av}\n"
        "BASELINE_HINTS: {baselines}\n"
        "PROXIES: MARKET_TREND={mt}; REL_STRENGTH={rs}; BREADTH_VOLUME={bv}; VALUATION_HISTORY={vh}; RISK_VOLATILITY={rv}; RISK_DRAWDOWN={rd}\n"
        "PROXIES_FUNDAMENTALS: GROWTH_TECH={gt}; MARGIN_TREND_TECH={mtf}; FCF_TREND_TECH={ft}; OP_EFF_TREND_TECH={ot}\n"
        "PROXIES_CATALYSTS: {cat_line}\n"
        "CATALYST_TIMING_HINTS: {timing_tb}\n"
        "EXPECTED_VOLATILITY_PCT: {ev}\n"
        "FVA_HINT: {fva}\n"
        "PE_HINT: {pe}\n"
        "SUGGESTED_BONUSES: {bon}\n".format(
            t=t, name=name,
            price=feats.get("price"),
            vs20=feats.get("vsSMA20"), vs50=feats.get("vsSMA50"), vs200=feats.get("vsSMA200"),
            sma50=feats.get("SMA50"), avwap=feats.get("AVWAP252"),
            rsi=feats.get("RSI14"), macd=feats.get("MACD_hist"), atr=feats.get("ATRpct"),
            dd=feats.get("drawdown_pct"), d5=feats.get("d5"), d20=feats.get("d20"), r60=feats.get("r60"),
            v20=feats.get("vol_vs20"),
            val_fields=val_fields,
            data_av=data_availability,
            baselines=baseline_str,
            mt=proxies["market_trend"], rs=proxies["relative_strength"],
            bv=proxies["breadth_volume"], vh=proxies["valuation_history"],
            rv=proxies["risk_volatility"], rd=proxies["risk_drawdown"],
            gt=("+"+str(fund_proxy["GROWTH_TECH"])) if fund_proxy["GROWTH_TECH"]>0 else str(fund_proxy["GROWTH_TECH"]),
            mtf=("+"+str(fund_proxy["MARGIN_TREND_TECH"])) if fund_proxy["MARGIN_TREND_TECH"]>0 else str(fund_proxy["MARGIN_TREND_TECH"]),
            ft=("+"+str(fund_proxy["FCF_TREND_TECH"])) if fund_proxy["FCF_TREND_TECH"]>0 else str(fund_proxy["FCF_TREND_TECH"]),
            ot=("+"+str(fund_proxy["OP_EFF_TREND_TECH"])) if fund_proxy["OP_EFF_TREND_TECH"]>0 else str(fund_proxy["OP_EFF_TREND_TECH"]),
            cat_line=catalyst_line, timing_tb=timing_tb,
            ev=proxies["expected_volatility_pct"],
            fva=proxies["fva_hint"] if proxies["fva_hint"] is not None else "N/A",
            pe=(f"{pe_hint:.2f}" if isinstance(pe_hint,(int,float)) else "N/A"),
            bon=proxies["suggested_bonuses"],
        )
    )

    # --- Debug payload (include all valuation fields, including EV_REV) ---
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
    }
    proxies_catalysts_full = {**cat, "EARNINGS_SOON": earn_sev}
    catalyst_timing_hints = {
        "TECH_BREAKOUT": ("Today" if (cat.get("TECH_BREAKOUT",0) > 0 and feats.get("is_20d_high")) else "None")
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
            "MARKET_TREND": proxies["market_trend"],
            "REL_STRENGTH": proxies["relative_strength"],
            "BREADTH_VOLUME": proxies["breadth_volume"],
            "VALUATION_HISTORY": proxies["valuation_history"],
            "RISK_VOLATILITY": proxies["risk_volatility"],
            "RISK_DRAWDOWN": proxies["risk_drawdown"],
            "EXPECTED_VOLATILITY_PCT": proxies["expected_volatility_pct"],
            "FVA_HINT": proxies["fva_hint"],
            "SUGGESTED_BONUSES": proxies["suggested_bonuses"],
        },
        "PROXIES_FUNDAMENTALS_BLOCK": fund_proxy,
        "PROXIES_CATALYSTS_BLOCK": proxies_catalysts_full,
        "CATALYST_TIMING_HINTS_BLOCK": catalyst_timing_hints,

        "PE_HINT": pe_hint, 
        "PROMPT_BLOCK": block_text,
    }

    return block_text, debug_dict
