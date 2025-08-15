# scripts/quick_ai_scorer.py
from typing import Dict

SEV = {"+5":5, "+4":4, "+3":3, "+2":2, "+1":1, "0":0, "-1":-1, "-2":-2, "-3":-3, "-4":-4, "-5":-5}

def _sev(x):
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str): return float(SEV.get(x.strip(), 0))
    return 0.0

def quick_ai_score(feats: Dict, proxies: Dict) -> float:
    """
    Ultra-fast composite using only your cheap signals:
      - proxy core (MARKET_TREND, REL_STRENGTH, etc.)
      - light technical nudges (vsSMA50/200, MACD, RSI, d20)
      - soft risk trims (ATR, drawdown)
    Returns ~[-100, 100]. Missing-safe.
    """
    # 1) proxy core
    mt   = _sev(proxies.get("MARKET_TREND"))
    rs   = _sev(proxies.get("REL_STRENGTH"))
    br   = _sev(proxies.get("BREADTH_VOLUME"))
    valh = _sev(proxies.get("VALUATION_HISTORY"))
    rv   = _sev(proxies.get("RISK_VOLATILITY"))
    rd   = _sev(proxies.get("RISK_DRAWDOWN"))

    core = (10.0*mt + 12.0*rs + 8.0*br + 6.0*valh) - (8.0*rv + 10.0*rd)  # ~[-200, 200]

    # 2) light technicals
    vs50  = float(feats.get("vsSMA50") or 0.0)
    vs200 = float(feats.get("vsSMA200") or 0.0)
    macd  = float(feats.get("MACD_hist") or 0.0)
    rsi   = float(feats.get("RSI14") or 50.0)
    d20   = float(feats.get("d20") or 0.0)
    atrp  = float(feats.get("ATRpct") or 4.0)
    dd    = float(feats.get("drawdown_pct") or 0.0)

    tech = (
        1.6*max(min(vs50, 40.0), -20.0) +
        2.0*max(min(vs200, 40.0), -20.0) +
        9.0*max(min(macd, 1.5), -1.5) +
        0.8*(rsi - 50.0) +
        1.2*max(min(d20, 20.0), -15.0)
    )
    if atrp > 6.0: tech -= (atrp - 6.0) * 2.0
    if -12.0 <= dd <= -5.0 and vs200 > 0 and 45 <= rsi <= 68:
        tech += 8.0

    # 3) blend & clamp
    score = 0.55*(core/2.0) + 0.45*tech
    return float(max(min(score, 100.0), -100.0))
