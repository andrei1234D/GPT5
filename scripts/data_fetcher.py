# scripts/data_fetcher.py
from __future__ import annotations
import os, time, math, json, logging
from typing import Dict, List, Optional, Tuple

import yfinance as yf

# Optional alias helpers. If you don't have these files, we no-op.
try:
    from aliases import apply_alias, load_aliases_csv
except Exception:
    def apply_alias(t: str, extra: Dict[str,str]|None=None) -> str: return t
    def load_aliases_csv(path: str) -> Dict[str,str]: return {}

# ---------- logging ----------
def _maybe_configure_logging():
    level_name = (os.getenv("PE_LOG_LEVEL") or os.getenv("FEATURES_LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_maybe_configure_logging()
logger = logging.getLogger("data_fetcher")

# ---------- symbol normalization (consistent with features.py) ----------
_YH_SUFFIXES = {
    "L","DE","F","SW","PA","AS","BR","LS","MC","MI","VI","ST","HE","CO","OL","WA","PR",
    "TO","V","AX","HK","T","SI","KS","KQ","NS","BO","JO","SA","MX","NZ","DU","AD","SR","TA",
    "SS","SZ","TW","IR"
}
def _normalize_symbol_for_yahoo(sym: str) -> str:
    s = (sym or "").strip().upper().replace(" ", "-")
    if not s:
        return s
    if "." in s:
        head, tail = s.rsplit(".", 1)
        if tail in _YH_SUFFIXES:
            if head.isdigit() and tail in {"T","HK","TW","SS","SZ"}:
                head = head.zfill(4)
            return f"{head}.{tail}"
        s = s.replace(".", "-")
        return s
    return s

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        f = float(x)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except Exception:
        return None

# Simple JSON cache on disk to reduce repeated lookups within a day
_CACHE_PATH = os.getenv("PE_CACHE_PATH", "data/pe_cache.json")
_CACHE_TTL_S = int(os.getenv("PE_CACHE_TTL_S", "86400"))  # 1 day

def _load_cache() -> Dict[str, dict]:
    try:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_cache(cache: Dict[str, dict]) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def _from_cache(cache: Dict[str, dict], t: str) -> Optional[Tuple[Optional[float], str]]:
    row = cache.get(t)
    if not isinstance(row, dict): return None
    ts = row.get("ts")
    if not isinstance(ts, (int, float)): return None
    if time.time() - float(ts) > _CACHE_TTL_S: return None
    return (row.get("pe"), row.get("source") or "unknown")

def _put_cache(cache: Dict[str, dict], t: str, pe: Optional[float], source: str) -> None:
    cache[t] = {"ts": time.time(), "pe": pe, "source": source}

def _extract_pe_version_safe(tk: yf.Ticker) -> Tuple[Optional[float], str]:
    """
    Try to get *trailing* P/E first, then forward, across yfinance versions.
    Returns (pe_value or None, "trailing"|"forward"|"none").
    """
    # 1) fast_info (cheap)
    try:
        fi = getattr(tk, "fast_info", None)
        if isinstance(fi, dict):
            for key in ("trailingPE", "trailing_pe", "pe_ratio"):
                pe = _safe_float(fi.get(key))
                if pe and pe > 0:
                    return pe, "trailing"
            for key in ("forwardPE", "forward_pe"):
                pe = _safe_float(fi.get(key))
                if pe and pe > 0:
                    return pe, "forward"
    except Exception:
        pass

    # 2) info or get_info (slower)
    info = None
    try:
        if hasattr(tk, "get_info"):
            info = tk.get_info()
        else:
            info = tk.info
    except Exception as e:
        logger.debug("info fetch failed: %r", e)

    if isinstance(info, dict):
        pe = _safe_float(info.get("trailingPE"))
        if pe and pe > 0:
            return pe, "trailing"
        pe = _safe_float(info.get("forwardPE"))
        if pe and pe > 0:
            return pe, "forward"

    return None, "none"

def fetch_pe_for_top(tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Fetch trailing P/E (preferred) or forward P/E as a fallback.
    Returns {ticker: pe or None}. Never throws.
    """
    if os.getenv("DISABLE_PE_FETCH", "").strip().lower() in {"1","true","yes"}:
        return {t: None for t in tickers}

    extra_aliases = {}
    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}

    cache = _load_cache()
    out: Dict[str, Optional[float]] = {}
    delay = float(os.getenv("PE_FETCH_DELAY_S", "0.15"))

    for t in tickers:
        # Cache
        cached = _from_cache(cache, t)
        if cached is not None:
            pe, src = cached
            out[t] = _safe_float(pe)
            logger.debug("[pe] cache %s -> %s (%s)", t, out[t], src)
            continue

        ali = apply_alias(t, extra_aliases)
        yh = _normalize_symbol_for_yahoo(ali)

        try:
            # IMPORTANT: do NOT pass a requests.Session to yfinance here
            tk = yf.Ticker(yh)
            pe, src = _extract_pe_version_safe(tk)
            out[t] = pe
            _put_cache(cache, t, pe, src)
            logger.info("[pe] %s (%s) -> %s (%s)", t, yh, "N/A" if pe is None else f"{pe:.2f}", src)
        except Exception as e:
            out[t] = None
            _put_cache(cache, t, None, "error")
            logger.warning("[pe] fetch fail %s (%s): %r", t, yh, e)

        time.sleep(delay)

    _save_cache(cache)
    return out

# simple CLI
if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "ACQ.TO", "ANIP"]
    res = fetch_pe_for_top(tickers)
    print(json.dumps(res, indent=2))
