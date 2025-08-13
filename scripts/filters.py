# scripts/filters.py

# Make sure the "scripts" folder (this file's dir) is on sys.path
# so absolute imports like "from trash_ranker import HardFilter" work
# when you run:  python scripts/notify.py
import os, sys
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Import the ranker (absolute only). If this fails, we want the real error,
# not a fallback to a relative import that breaks outside a package.
from trash_ranker import HardFilter

_hard = HardFilter()

def is_garbage(feats: dict) -> bool:
    """
    Backwards-compatible default. Keeps your old invariants, now upgraded:
    - price < $3 (small-cap friendly)
    - ATR% > 12%
    - vsSMA200 < -45%
    - 52w drawdown < -70%
    - classic pump (RSI>=83 & vol_vs20>=400 & d5>=18)
    """
    return _hard.is_garbage(feats)

def daily_index_filter(feats: dict, context: dict) -> bool:
    """
    Second-stage filter with your daily context (bench/sector/breadth).
    Keep exactly as you had it.
    """
    if context.get("bench_trend") == "down" and feats.get("vsSMA50", 0) < -5:
        return False
    if context.get("breadth50", 0) < 40 and feats.get("vsSMA200", 0) < 0:
        return False
    return True
