# scripts/pe_fetcher.py
from typing import Dict, List, Optional
import yfinance as yf
from aliases import apply_alias, load_aliases_csv

def _safe_pe(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v > 0 and v < 5000 else None  # clamp nonsense
    except Exception:
        return None

def fetch_pe_for_top(symbol: str):
    tk = yf.Ticker(symbol)
    fi = getattr(tk, "fast_info", None) or {}

    v = _safe_pe(fi.get("trailing_pe"))
    if v is not None: return v, "trailing"

    info = {}
    try:
        info = tk.get_info() or {}
    except Exception:
        info = tk.info or {}

    v = _safe_pe(info.get("trailingPE"))
    if v is not None: return v, "trailing"

    v = _safe_pe(info.get("forwardPE"))
    if v is not None: return v, "forward"

    return None, "N/A"