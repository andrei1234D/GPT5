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

# ---------- helpers ----------
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

def _fmt(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.2f}"

def _as_dict(obj) -> dict:
    if isinstance(obj, dict): return obj
    try:
        return dict(obj)
    except Exception:
        return {}

def _count_numeric(d: Dict[str, Optional[float]]) -> int:
    return sum(1 for v in (d or {}).values() if _safe_float(v) is not None)

# --------- cache helpers ---------
def _load_cache(path: str) -> Dict[str, dict]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_cache(path: str, cache: Dict[str, dict]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def _from_cache(cache: Dict[str, dict], t: str, ttl: int) -> Optional[dict]:
    row = cache.get(t)
    if not isinstance(row, dict): return None
    ts = row.get("ts")
    if not isinstance(ts, (int, float)): return None
    if time.time() - float(ts) > ttl: return None
    return row

def _put_cache(cache: Dict[str, dict], t: str, payload: dict) -> None:
    cache[t] = {"ts": time.time(), **payload}

# ---------- price / size helpers ----------
def _get_price_fast(tk: yf.Ticker) -> Optional[float]:
    try:
        fi = _as_dict(getattr(tk, "fast_info", None))
        for k in ("lastPrice", "last_price", "last_close", "last_close_price"):
            p = _safe_float(fi.get(k))
            if p is not None:
                return p
    except Exception:
        pass
    try:
        h = tk.history(period="1d")
        if hasattr(h, "empty") and not h.empty:
            return _safe_float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _get_market_cap(tk: yf.Ticker, info: Optional[dict]) -> Optional[float]:
    try:
        fi = _as_dict(getattr(tk, "fast_info", None))
        mc = _safe_float(fi.get("market_cap"))
        if mc: return mc
    except Exception:
        pass
    if isinstance(info, dict):
        mc = _safe_float(info.get("marketCap"))
        if mc: return mc
        px = _safe_float(info.get("currentPrice")) or _get_price_fast(tk)
        shares = _safe_float(info.get("sharesOutstanding"))
        if px and shares:
            return px * shares
    return None

def _get_enterprise_value(info: Optional[dict]) -> Optional[float]:
    if isinstance(info, dict):
        ev = _safe_float(info.get("enterpriseValue"))
        if ev: return ev
        mcap = _safe_float(info.get("marketCap"))
        total_debt = _safe_float(info.get("totalDebt"))
        cash = _safe_float(info.get("cash") or info.get("cashAndCashEquivalents"))
        if mcap and total_debt is not None and cash is not None:
            return mcap + total_debt - cash
    return None

# ---------- PE (with EPS compute fallback) ----------
def _extract_pe_with_compute(tk: yf.Ticker) -> Tuple[Optional[float], str]:
    try:
        fi = _as_dict(getattr(tk, "fast_info", None))
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

    info = None
    try:
        info = tk.get_info() if hasattr(tk, "get_info") else tk.info
    except Exception as e:
        logger.debug("info fetch failed: %r", e)
        info = None

    if isinstance(info, dict):
        pe = _safe_float(info.get("trailingPE"))
        if pe and pe > 0: return pe, "trailing"
        pe = _safe_float(info.get("forwardPE"))
        if pe and pe > 0: return pe, "forward"

    price = _get_price_fast(tk)
    if isinstance(info, dict) and price:
        teps = _safe_float(info.get("trailingEps"))
        if teps and teps > 0: return price / teps, "trailing_computed"
        feps = _safe_float(info.get("forwardEps"))
        if feps and feps > 0: return price / feps, "forward_computed"

    return None, "none"

# ---------- FCF Yield helper ----------
def _calc_fcf_yield_pct(tk: yf.Ticker, info: Optional[dict], fi_dict: Optional[dict]) -> Optional[float]:
    fcf = _safe_float((info or {}).get("freeCashflow") or (info or {}).get("freeCashFlow") or (info or {}).get("freeCashFlowTTM"))
    if fcf is None:
        try:
            cf = getattr(tk, "cashflow", None)
            if cf is not None and hasattr(cf, "index") and "Free Cash Flow" in cf.index:
                series = cf.loc["Free Cash Flow"]
                if len(series) > 0:
                    fcf = _safe_float(series.iloc[0])
        except Exception:
            pass

    mcap = _get_market_cap(tk, info)
    if mcap is None:
        px = None
        fi = _as_dict(fi_dict)
        px = _safe_float(fi.get("lastPrice") or fi.get("last_price") or fi.get("last_close"))
        if px is None:
            px = _get_price_fast(tk)
        shares = _safe_float((info or {}).get("sharesOutstanding"))
        if px and shares:
            mcap = px * shares

    if (mcap is not None and mcap > 0) and (fcf is not None):
        return 100.0 * (fcf / mcap)
    return None

# ---------- Full valuation extractor ----------
def _extract_valuations_version_safe(tk: yf.Ticker) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    vals: Dict[str, Optional[float]] = { "PE": None, "PS": None, "EV_EBITDA": None, "EV_REV": None, "PEG": None, "FCF_YIELD": None }
    srcs: Dict[str, str] = {k: "none" for k in vals.keys()}

    info = None
    try:
        info = tk.get_info() if hasattr(tk, "get_info") else tk.info
    except Exception as e:
        logger.debug("info fetch failed: %r", e)
        info = None

    fi = _as_dict(getattr(tk, "fast_info", None))

    # PE
    pe, pe_src = _extract_pe_with_compute(tk)
    vals["PE"], srcs["PE"] = pe, pe_src

    # PS
    if isinstance(info, dict):
        ps = _safe_float(info.get("priceToSalesTrailing12Months") or info.get("priceToSales"))
        if ps and ps > 0:
            vals["PS"], srcs["PS"] = ps, "info"
        else:
            price = _get_price_fast(tk)
            rps = _safe_float(info.get("revenuePerShareTTM") or info.get("revenuePerShare"))
            if price and rps and rps > 0:
                vals["PS"], srcs["PS"] = price / rps, "computed_rps"
            else:
                mc = _get_market_cap(tk, info)
                tot_rev = _safe_float(info.get("totalRevenue"))
                if mc and tot_rev and tot_rev > 0:
                    vals["PS"], srcs["PS"] = mc / tot_rev, "computed_mc_rev"

    # EV/EBITDA
    if isinstance(info, dict):
        ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
        if ev_ebitda and ev_ebitda > 0:
            vals["EV_EBITDA"], srcs["EV_EBITDA"] = ev_ebitda, "info"
        else:
            ev = _get_enterprise_value(info)
            ebitda = _safe_float(info.get("ebitda"))
            if ev and ebitda and ebitda > 0:
                vals["EV_EBITDA"], srcs["EV_EBITDA"] = ev / ebitda, "computed"

    # EV/Revenue
    if isinstance(info, dict):
        ev_rev = _safe_float(info.get("enterpriseToRevenue"))
        if ev_rev and ev_rev > 0:
            vals["EV_REV"], srcs["EV_REV"] = ev_rev, "info"
        else:
            ev = _get_enterprise_value(info)
            tot_rev = _safe_float(info.get("totalRevenue"))
            if ev and tot_rev and tot_rev > 0:
                vals["EV_REV"], srcs["EV_REV"] = ev / tot_rev, "computed"

    # PEG
    if isinstance(info, dict):
        peg = _safe_float(info.get("pegRatio") or info.get("trailingPegRatio"))
        if peg and peg > 0:
            vals["PEG"], srcs["PEG"] = peg, "info"

    # FCF Yield
    vals["FCF_YIELD"] = _calc_fcf_yield_pct(tk, info, fi)
    srcs["FCF_YIELD"] = "computed" if vals["FCF_YIELD"] is not None else "none"

    return vals, srcs

# ---------- Public: fetch just PE ----------
_PE_CACHE_PATH = os.getenv("PE_CACHE_PATH", "data/pe_cache.json")
_PE_CACHE_TTL_S = int(os.getenv("PE_CACHE_TTL_S", "86400"))  # 1 day

def fetch_pe_for_top(tickers: List[str]) -> Dict[str, Optional[float]]:
    if os.getenv("DISABLE_PE_FETCH", "").strip().lower() in {"1","true","yes"}:
        return {t: None for t in tickers}

    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}

    cache = _load_cache(_PE_CACHE_PATH)
    logger.debug("PE_CACHE_PATH=%s", _PE_CACHE_PATH)

    out: Dict[str, Optional[float]] = {}
    delay = float(os.getenv("PE_FETCH_DELAY_S", "0.15"))
    max_retries = int(os.getenv("YF_MAX_RETRIES", "3"))
    retry_sleep = float(os.getenv("YF_RETRY_SLEEP", "2.5"))

    for t in tickers:
        cached = _from_cache(cache, t, _PE_CACHE_TTL_S)
        if cached is not None and "pe" in cached:
            pe_cached = _safe_float(cached.get("pe"))
            if pe_cached and pe_cached > 0:
                out[t] = pe_cached
                logger.debug("[pe] cache %s -> %s (%s)", t, _fmt(out[t]), cached.get("source") or "unknown")
                continue  # accept only positive numeric cache
            else:
                logger.debug("[pe] cache %s present but invalid (%s) -> refetch", t, cached.get("pe"))

        yh = _normalize_symbol_for_yahoo(apply_alias(t, extra_aliases))

        pe, src = None, "none"
        for attempt in range(1, max_retries + 1):
            try:
                tk = yf.Ticker(yh)
                pe, src = _extract_pe_with_compute(tk)
                if pe and pe > 0:
                    break
            except Exception as e:
                msg = str(e)
                if "Rate" in msg or "Too Many Requests" in msg:
                    time.sleep(retry_sleep * attempt)
                    continue
            time.sleep(0.05)

        out[t] = pe
        # write cache only if we have a meaningful value
        if pe and pe > 0:
            _put_cache(cache, t, {"pe": pe, "source": src})
        logger.info("[pe] %s (%s) -> %s (%s)", t, yh, "N/A" if pe is None else f"{pe:.2f}", src)
        time.sleep(delay)

    _save_cache(_PE_CACHE_PATH, cache)
    return out

# ---------- Full valuations fetch ----------
_VAL_CACHE_PATH = os.getenv("VAL_CACHE_PATH", "data/valuations_cache.json")
_VAL_CACHE_TTL_S = int(os.getenv("VAL_CACHE_TTL_S", "86400"))  # 1 day

def fetch_valuations_for_top(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Fetch: PE, PS, EV_EBITDA, EV_REV, PEG, FCF_YIELD (percent)
    Cache is used only if it contains >= VAL_MIN_NUMERIC_CACHE numeric fields (default: 1).
    """
    if os.getenv("DISABLE_VAL_FETCH", "").strip().lower() in {"1","true","yes"}:
        return {t: {"PE": None, "PS": None, "EV_EBITDA": None, "EV_REV": None, "PEG": None, "FCF_YIELD": None} for t in tickers}

    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}

    cache = _load_cache(_VAL_CACHE_PATH)
    logger.debug("VAL_CACHE_PATH=%s", _VAL_CACHE_PATH)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    delay = float(os.getenv("VAL_FETCH_DELAY_S", os.getenv("PE_FETCH_DELAY_S", "0.15")))
    min_numeric_cache = int(os.getenv("VAL_MIN_NUMERIC_CACHE", "1"))
    max_retries = int(os.getenv("YF_MAX_RETRIES", "3"))
    retry_sleep = float(os.getenv("YF_RETRY_SLEEP", "2.5"))

    keys = ["PE","PS","EV_EBITDA","EV_REV","PEG","FCF_YIELD"]

    for t in tickers:
        # --- read cache (accept only if it has enough numeric fields) ---
        cached = _from_cache(cache, t, _VAL_CACHE_TTL_S)
        if cached is not None and isinstance(cached.get("vals"), dict):
            vals_raw = cached["vals"]
            vals_norm = {k: _safe_float(vals_raw.get(k)) for k in keys}
            if _count_numeric(vals_norm) >= min_numeric_cache:
                out[t] = vals_norm
                logger.debug("[val] cache %s -> %s", t, {k:_fmt(v) for k,v in out[t].items()})
                continue
            else:
                logger.debug("[val] cache %s present but insufficient numeric fields -> refetch", t)

        yh = _normalize_symbol_for_yahoo(apply_alias(t, extra_aliases))

        vals, srcs = None, None
        for attempt in range(1, max_retries + 1):
            try:
                tk = yf.Ticker(yh)
                vals, srcs = _extract_valuations_version_safe(tk)
                if _count_numeric(vals) >= 1:
                    break  # good enough
            except Exception as e:
                msg = str(e)
                if "Rate" in msg or "Too Many Requests" in msg:
                    time.sleep(retry_sleep * attempt)  # backoff
                    continue
            time.sleep(0.05)

        if vals is None:
            vals = {k: None for k in keys}
        out[t] = vals

        # write cache only if we have at least one numeric value
        if _count_numeric(vals) >= min_numeric_cache:
            _put_cache(cache, t, {"vals": vals, "sources": (srcs or {})})

        logger.info(
            "[val] %s (%s) -> PE=%s PS=%s EV/EBITDA=%s EV/REV=%s PEG=%s FCF_YIELD=%s%%",
            t, yh,
            _fmt(vals.get("PE")), _fmt(vals.get("PS")), _fmt(vals.get("EV_EBITDA")), _fmt(vals.get("EV_REV")),
            _fmt(vals.get("PEG")), _fmt(vals.get("FCF_YIELD"))
        )
        time.sleep(delay)

    _save_cache(_VAL_CACHE_PATH, cache)
    return out

# simple CLIs
if __name__ == "__main__":
    import sys, json as _json
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "KOD", "RDDT"]
    vals = fetch_valuations_for_top(tickers)
    print(_json.dumps(vals, indent=2))
