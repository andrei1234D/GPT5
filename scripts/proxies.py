# scripts/proxies.py
import pandas as pd
from features import fetch_history, compute_indicators

def get_spy_ctx() -> dict:
    try:
        spy_hist = fetch_history(["SPY"])
        df = spy_hist.get("SPY")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("SPY history empty")
        feats = compute_indicators(df)
        return {
            "vsSMA50": feats.get("vsSMA50") or 0.0,
            "vsSMA200": feats.get("vsSMA200") or 0.0,
            "r60": feats.get("r60") or 0.0,
        }
    except Exception:
        return {"vsSMA50": 0.0, "vsSMA200": 0.0, "r60": 0.0}

def _sev_from_return(pct: float, cuts=(2, 5, 10, 15)) -> int:
    a = abs(pct or 0.0)
    if a >= cuts[3]: return 5
    if a >= cuts[2]: return 4
    if a >= cuts[1]: return 3
    if a >= cuts[0]: return 2
    return 1

def _sev_from_value(a: float, cuts=(2, 4, 6, 9)) -> int:
    x = abs(a or 0.0)
    if x >= cuts[3]: return 5
    if x >= cuts[2]: return 4
    if x >= cuts[1]: return 3
    if x >= cuts[0]: return 2
    return 1

def derive_proxies(feats: dict, spy: dict) -> dict:
    mt_sign = 1 if (spy["vsSMA50"] > 0 and spy["vsSMA200"] > 0 and spy["r60"] > 0) else (-1 if (spy["vsSMA50"] < 0 and spy["vsSMA200"] < 0 and spy["r60"] < 0) else 0)
    mt_sev = 3 if mt_sign != 0 else 1
    market_trend = f"{'+' if mt_sign>0 else '-' if mt_sign<0 else ''}{mt_sev}"

    rs_60 = (feats.get("r60") or 0.0) - (spy["r60"] or 0.0)
    rs_sign = 1 if rs_60 > 0 else (-1 if rs_60 < 0 else 0)
    rs_sev = _sev_from_return(rs_60, (2, 5, 10, 15))
    rel_strength = f"{'+' if rs_sign>0 else '-' if rs_sign<0 else ''}{rs_sev}"

    vv20 = feats.get("vol_vs20")
    if vv20 is None:
        breadth_volume = "0"
    else:
        sign = 1 if vv20 > 10 else (-1 if vv20 < -10 else 0)
        sev = 1 if sign == 0 else (2 if abs(vv20) < 25 else 3 if abs(vv20) < 50 else 4)
        breadth_volume = f"{'+' if sign>0 else '-' if sign<0 else ''}{sev if sign!=0 else 0}"

    vs200 = feats.get("vsSMA200")
    if vs200 is None:
        valuation_hist = "0"
    else:
        sign = 1 if vs200 < 0 else (-1 if vs200 > 0 else 0)
        sev = _sev_from_return(vs200, (3, 7, 12, 20))
        valuation_hist = f"{'+' if sign>0 else '-' if sign<0 else ''}{sev if sign!=0 else 0}"

    atrp = feats.get("ATRpct")
    dd_mag = -(feats.get("drawdown_pct") or 0.0)
    risk_vol = str(_sev_from_value(atrp or 0.0))
    risk_drawdown = str(_sev_from_value(dd_mag or 0.0))

    bonuses = []
    drawdown = abs(feats.get("drawdown_pct") or 0.0)
    if 5.0 <= drawdown <= 12.0:
        bonuses.append("DIP_5_12")
    if feats.get("is_20d_high") and (vv20 is not None and vv20 > 20):
        bonuses.append("BREAKOUT_VOL_CONF")

    ev = feats.get("ATRpct") or 4.0
    ev = max(1.0, min(ev, 6.0))
    sma50 = feats.get("SMA50"); avwap = feats.get("AVWAP252")
    fva_hint = None
    if sma50 and avwap:
        fva_hint = round((sma50 + avwap) / 2.0, 2)
    elif sma50:
        fva_hint = round(sma50, 2)
    elif avwap:
        fva_hint = round(avwap, 2)

    return dict(
        market_trend=market_trend,
        relative_strength=rel_strength,
        breadth_volume=breadth_volume,
        valuation_history=valuation_hist,
        risk_volatility=risk_vol,
        risk_drawdown=risk_drawdown,
        expected_volatility_pct=round(ev, 2),
        fva_hint=fva_hint,
        suggested_bonuses=",".join(bonuses) if bonuses else "NONE",
    )

def fund_proxies_from_feats(feats: dict) -> dict:
    def sev_from(p, cuts):
        a = abs(p or 0.0)
        if a >= cuts[3]: return 5
        if a >= cuts[2]: return 4
        if a >= cuts[1]: return 3
        if a >= cuts[0]: return 2
        return 1

    r60  = feats.get("r60") or 0.0
    r120 = feats.get("r120") or 0.0
    vs200 = feats.get("vsSMA200")
    rsi  = feats.get("RSI14") or 50
    macd = feats.get("MACD_hist") or 0.0
    atrp = feats.get("ATRpct")

    g_raw  = (r60 + r120) / 2.0
    g_sign = 1 if g_raw > 0 else (-1 if g_raw < 0 else 0)
    g_sev  = sev_from(g_raw, (5, 10, 20, 30)) * g_sign if g_sign != 0 else 0

    mt_sign = 1 if (macd > 0 and rsi > 55) else (-1 if (macd < 0 and rsi < 45) else 0)
    m_sev = 0
    if mt_sign != 0:
        macd_metric = abs(macd)
        m_sev = (1 if macd_metric < 0.2 else
                 2 if macd_metric < 0.5 else
                 3 if macd_metric < 1.0 else
                 4 if macd_metric < 1.5 else
                 5)
        m_sev *= mt_sign

    f_sev = 0
    if vs200 is not None and vs200 != 0:
        f_sign = 1 if vs200 < 0 else -1
        f_sev = sev_from(vs200, (3, 7, 12, 20)) * f_sign

    o_sev = 0
    if atrp is not None:
        if atrp <= 3:    o_sev = +3
        elif atrp <= 5:  o_sev = +2
        elif atrp <= 7:  o_sev = -2
        elif atrp <= 10: o_sev = -3
        else:            o_sev = -4

    def clip5(x): return max(-5, min(5, int(x)))
    return {
        "GROWTH_TECH":       clip5(g_sev),
        "MARGIN_TREND_TECH": clip5(m_sev),
        "FCF_TREND_TECH":    clip5(f_sev),
        "OP_EFF_TREND_TECH": clip5(o_sev),
    }

def catalyst_severity_from_feats(feats: dict) -> dict:
    vs50 = feats.get("vsSMA50") or 0.0
    vs200 = feats.get("vsSMA200") or 0.0
    rsi = feats.get("RSI14") or 50
    macd = feats.get("MACD_hist") or 0.0
    vv20 = feats.get("vol_vs20")
    is_high = bool(feats.get("is_20d_high"))
    dd = abs(feats.get("drawdown_pct") or 0.0)
    d5 = feats.get("d5") or 0.0

    tech_breakout = 0
    if is_high:
        if vv20 is None: tech_breakout = 1
        elif vv20 > 100: tech_breakout = 5
        elif vv20 > 50:  tech_breakout = 4
        elif vv20 > 20:  tech_breakout = 3
        elif vv20 > 5:   tech_breakout = 2
        else:            tech_breakout = 1

    tech_breakdown = 0
    if vs50 < -3 and macd < 0 and rsi < 45:
        tech_breakdown = -4 if (vs200 < -5 or rsi < 35) else -2

    dip_reversal = 0
    if 5 <= dd <= 12 and (rsi >= 50 or d5 > 0):
        dip_reversal = 3 if dd >= 10 else 2 if dd >= 8 else 1

    return {"TECH_BREAKOUT": tech_breakout, "TECH_BREAKDOWN": tech_breakdown, "DIP_REVERSAL": dip_reversal}
