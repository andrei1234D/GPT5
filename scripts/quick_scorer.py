# scripts/quick_scorer.py
from typing import Dict, Tuple, List

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe(x, d=0.0):
    try:
        xf = float(x)
        if xf != xf or xf == float("inf") or xf == float("-inf"):
            return d
        return xf
    except:
        return d

def quick_score(feats: Dict) -> Tuple[float, Dict]:
    """
    Simplified quick scorer (~12-14 rules).
    Neutral between small/large caps, purely technical.
    Returns: (score, breakdown dict)
    """

    score = 0
    parts = {}

    # Core indicators
    price   = safe(feats.get("price"))
    vs20    = safe(feats.get("vsSMA20"))
    vs50    = safe(feats.get("vsSMA50"))
    vs200   = safe(feats.get("vsSMA200"))
    vsE50   = safe(feats.get("vsEMA50"))
    vsE200  = safe(feats.get("vsEMA200"))
    rsi     = safe(feats.get("RSI14"))
    slope50 = safe(feats.get("EMA50_slope_5d"))
    atr     = safe(feats.get("ATRpct"))
    d20     = safe(feats.get("20d%"))
    d60     = safe(feats.get("60d%"))
    macd    = safe(feats.get("MACD_hist"))
    v20     = safe(feats.get("vol_vs20"))
    dd      = safe(feats.get("drawdown_pct"))

    # === RULES ===
    # 1. RSI sweet spot
    if 50 <= rsi <= 65:
        score += 12; parts["RSI_sweet"] = 12
    elif 65 < rsi <= 70:
        score += 4; parts["RSI_mid"] = 4
    elif rsi > 75:
        score -= 15; parts["RSI_hot_pen"] = -15

    # 2. EMA50 slope = trend strength
    if 0.5 <= slope50 <= 4:
        score += 10; parts["EMA50_slope_good"] = 10
    elif slope50 > 8:
        score -= 8; parts["EMA50_too_steep"] = -8

    # 3. Above EMA200
    if vsE200 > 0:
        score += 8; parts["above_EMA200"] = 8
    else:
        score -= 6; parts["below_EMA200"] = -6

    # 4. Price vs EMA50
    if -5 <= vsE50 <= 20:
        score += 6; parts["EMA50_near"] = 6
    elif vsE50 > 40:
        score -= 10; parts["EMA50_far"] = -10

    # 5. SMA alignment
    if vs50 > 0 and vs200 > 0:
        score += 8; parts["SMA_alignment"] = 8
    elif vs50 < 0 and vs200 < 0:
        score -= 6; parts["SMA_downtrend"] = -6

    # 6. ATR sanity
    if atr <= 6:
        score += 6; parts["ATR_low"] = 6
    elif atr > 10:
        score -= 10; parts["ATR_high"] = -10

    # 7. MACD
    if macd > 0:
        score += 5; parts["MACD_pos"] = 5
    elif macd < 0 and rsi > 55:
        score -= 5; parts["MACD_neg"] = -5

    # 8. 20d performance
    if 5 <= d20 <= 25:
        score += 8; parts["20d_gain_ok"] = 8
    elif d20 > 50:
        score -= 12; parts["20d_too_hot"] = -12

    # 9. 60d performance
    if 10 <= d60 <= 50:
        score += 6; parts["60d_gain_ok"] = 6
    elif d60 > 80:
        score -= 12; parts["60d_too_hot"] = -12

    # 10. Volume vs 20d avg
    if -30 <= v20 <= 100:
        score += 4; parts["vol_normal"] = 4
    elif v20 < -60:
        score -= 8; parts["vol_too_low"] = -8
    elif v20 > 200:
        score -= 10; parts["vol_spike"] = -10

    # 11. Breakout near 20d high
    if d20 > 8 and rsi >= 55 and atr <= 7:
        score += 10; parts["breakout_near"] = 10

    # 12. Recovery coil
    if -25 <= dd <= -5 and rsi >= 50:
        score += 6; parts["coil_recovery"] = 6

    # 13. Early-turn coil setup
    if (vs50 >= -5 and vs50 <= 12) and (vs200 > -3) and (45 <= rsi <= 62):
        score += 12; parts["early_turn"] = 12

    # 14. Reject parabolics
    if (d20 > 60 or d60 > 120) and rsi > 75:
        score -= 20; parts["parabolic_reject"] = -20

    return score, parts


def rank_stage1(universe: List[Tuple[str, str, Dict]], keep=200):
    """
    Run simplified quick scorer over universe.
    universe: list of (ticker, name, feats)
    Returns: list of (ticker, name, feats, score, breakdown)
    """
    scored = []
    for t, n, f in universe:
        s, parts = quick_score(f)
        scored.append((t, n, f, s, parts))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:keep]
