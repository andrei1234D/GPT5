# scripts/scoring.py
from typing import Dict, Tuple

def base_importance_score(feats: Dict) -> float:
    """
    Programmatic, fast ranker to go from ~2000 -> 200.
    """
    def safe(x):
        try:
            if x is None: return 0.0
            if x != x: return 0.0
            return float(x)
        except: return 0.0

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    vs50 = safe(feats.get("vsSMA50"))
    vs200 = safe(feats.get("vsSMA200"))
    rsi = safe(feats.get("RSI14"))
    d20 = safe(feats.get("d20"))
    dd = safe(feats.get("drawdown_pct"))
    atrp = safe(feats.get("ATRpct"))
    macd = safe(feats.get("MACD_hist"))

    score = 0.0
    score += clamp(vs50, -20, 40) * 2.0
    score += clamp(vs200, -20, 40) * 2.0
    score += clamp(d20, -15, 20) * 2.0
    score += clamp(rsi - 50, -30, 30) * 1.5
    score += clamp(macd, -1.5, 1.5) * 10.0
    if -12 <= dd <= -5 and vs200 > 0 and 45 <= rsi <= 65:
        score += 25
    if atrp and atrp > 3.5:
        score -= (atrp - 3.5) * 2.0

    return score


def composite_importance_score(feats: Dict) -> Tuple[float, Dict[str, float]]:
    """
    New composite score in [-100, 100], plus breakdown dict.
    Lazy-imports the ranker so running `python scripts/notify.py` works.
    """
    # Ensure sibling imports work when running as a plain script
    import os, sys
    HERE = os.path.dirname(__file__)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)