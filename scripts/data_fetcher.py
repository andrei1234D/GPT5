# scripts/data_fetcher.py
from __future__ import annotations
import os, time, math, json, logging
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import hashlib
import pickle
from pathlib import Path
import pandas as pd

# Optional alias helpers. If you don't have these files, we no-op.
try:
    from aliases import apply_alias, load_aliases_csv, save_aliases_csv
except Exception:
    def apply_alias(t: str, extra: Dict[str,str]|None=None) -> str: return t
    def load_aliases_csv(path: str) -> Dict[str,str]: return {}
    def save_aliases_csv(path: str, mapping: Dict[str,str]) -> None: return None

# ---------- logging ----------
def _maybe_configure_logging():
    level_name = (os.getenv("PE_LOG_LEVEL") or os.getenv("FEATURES_LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_maybe_configure_logging()
logger = logging.getLogger("data_fetcher")

# ---------- env knobs ----------
YF_MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "3"))
YF_RETRY_SLEEP = float(os.getenv("YF_RETRY_SLEEP", "1.0"))

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

def _fmt(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.2f}"

def _is_rate_limited(err: Exception) -> bool:
    s = f"{type(err).__name__}: {err}"
    return ("Rate limit" in s) or ("Too Many Requests" in s) or ("YFRateLimitError" in s)

def _coerce_cache_scalar(v):
    """Turn 'N/A'/'NA'/'' -> None; numeric strings -> float; leave floats/None as-is."""
    if v is None: return None
    if isinstance(v, (int, float)):
        return _safe_float(v)
    if isinstance(v, str):
        vs = v.strip().upper()
        if vs in {"N/A", "NA", ""}:
            return None
        # try numeric
        try:
            return _safe_float(float(v))
        except Exception:
            return None
    # anything else -> None
    return None

def _sanitize_val_block(d: dict) -> dict:
    """Coerce any cached 'vals' dict fields to float|None and drop string 'N/A'."""
    if not isinstance(d, dict):
        return {}
    out = {}
    for k in ["PE","PS","EV_EBITDA","EV_REV","PEG","FCF_YIELD"]:
        out[k] = _coerce_cache_scalar(d.get(k))
    return out

# --------- cache helpers ----------
_PE_CACHE_PATH = os.getenv("PE_CACHE_PATH", "data/pe_cache.json")
_VAL_CACHE_PATH = os.getenv("VAL_CACHE_PATH", "data/valuations_cache.json")
_PE_CACHE_TTL_S = int(os.getenv("PE_CACHE_TTL_S", "86400"))     # 1 day
_VAL_CACHE_TTL_S = int(os.getenv("VAL_CACHE_TTL_S", "86400"))   # 1 day

def _load_cache(path: str) -> Dict[str, dict]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                # SANITIZE in-place: convert any "N/A" to None; numeric strings -> floats
                dirty = False
                for t, row in list(data.items()):
                    if not isinstance(row, dict):
                        dirty = True
                        data.pop(t, None)
                        continue
                    # PE cache shape: {"ts":..,"pe":..., "source":...}
                    if "pe" in row:
                        new_pe = _coerce_cache_scalar(row.get("pe"))
                        if new_pe != row.get("pe"):
                            row["pe"] = new_pe
                            dirty = True
                    # VAL cache shape: {"ts":..,"vals":{...}, "sources": {...}}
                    if "vals" in row and isinstance(row["vals"], dict):
                        new_vals = _sanitize_val_block(row["vals"])
                        # if original had any strings -> we mark dirty
                        if any(isinstance(v, str) for v in row["vals"].values()):
                            dirty = True
                        row["vals"] = new_vals
                if dirty:
                    try:
                        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                        with open(path, "w", encoding="utf-8") as g:
                            json.dump(data, g)
                        logger.info("[cache] sanitized bad 'N/A' strings and rewrote %s", path)
                    except Exception:
                        pass
                return data
    except Exception as e:
        logger.debug("[cache] load error %s: %r", path, e)
    return {}

def _save_cache(path: str, cache: Dict[str, dict]) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.debug("[cache] save error %s: %r", path, e)

def _from_cache(cache: Dict[str, dict], t: str, ttl: int, path: str) -> Optional[dict]:
    row = cache.get(t)
    if not isinstance(row, dict): return None
    ts = row.get("ts")
    if not isinstance(ts, (int, float)): return None
    if time.time() - float(ts) > ttl: return None
    # Guard: if this cached row contains only None in vals, treat as miss
    if "vals" in row and isinstance(row["vals"], dict):
        vals = _sanitize_val_block(row["vals"])
        # if any original value was a string, consider it stale to force a refetch
        if any(isinstance(v, str) for v in row["vals"].values()):
            logger.debug("[val] cached row had strings for %s; ignoring and refetching (path=%s)", t, path)
            return None
        if all(v is None for v in vals.values()):
            return None
    if "pe" in row and _safe_float(row.get("pe")) is None:
        return None
    return row

def _put_cache(cache: Dict[str, dict], t: str, payload: dict) -> None:
    cache[t] = {"ts": time.time(), **payload}

# ---------- price / size helpers ----------
def _get_price_fast(tk: yf.Ticker) -> Optional[float]:
    # fast_info
    try:
        fi = getattr(tk, "fast_info", None)
        if isinstance(fi, dict):
            for k in ("lastPrice", "last_price", "last_close"):
                p = _safe_float(fi.get(k))
                if p is not None:
                    return p
    except Exception:
        pass
    # history fallback
    try:
        h = tk.history(period="1d")
        if hasattr(h, "empty") and not h.empty:
            return _safe_float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _get_market_cap(tk: yf.Ticker, info: Optional[dict]) -> Optional[float]:
    # fast_info first
    try:
        fi = getattr(tk, "fast_info", None)
        if isinstance(fi, dict):
            mc = _safe_float(fi.get("market_cap"))
            if mc: return mc
    except Exception:
        pass
    if isinstance(info, dict):
        mc = _safe_float(info.get("marketCap"))
        if mc: return mc
        # compute fallback: price * sharesOutstanding
        px = _safe_float(info.get("currentPrice"))
        if px is None:
            px = _get_price_fast(tk)
        shares = _safe_float(info.get("sharesOutstanding"))
        if px and shares:
            return px * shares
    return None

def _get_enterprise_value(info: Optional[dict], tk: Optional[yf.Ticker]=None) -> Optional[float]:
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
    # 1) fast_info
    try:
        fi = getattr(tk, "fast_info", None)
        if isinstance(fi, dict):
            for key in ("trailingPE", "trailing_pe", "pe_ratio"):
                pe = _safe_float(fi.get(key))
                if pe and pe > 0: return pe, "trailing"
            for key in ("forwardPE", "forward_pe"):
                pe = _safe_float(fi.get(key))
                if pe and pe > 0: return pe, "forward"
    except Exception:
        pass

    # 2) info (retry for rate-limit)
    info = None
    last_err = None
    for i in range(max(1, YF_MAX_RETRIES)):
        try:
            info = tk.get_info() if hasattr(tk, "get_info") else tk.info
            break
        except Exception as e:
            last_err = e
            if _is_rate_limited(e) and i < YF_MAX_RETRIES - 1:
                logger.debug("[pe] rate-limited; retrying %d/%d", i+1, YF_MAX_RETRIES-1)
                time.sleep(YF_RETRY_SLEEP)
                continue
            logger.debug("info fetch failed: %r", e)
            info = None
            break

    if isinstance(info, dict):
        pe = _safe_float(info.get("trailingPE"))
        if pe and pe > 0: return pe, "trailing"
        pe = _safe_float(info.get("forwardPE"))
        if pe and pe > 0: return pe, "forward"

    # 3) compute
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
        if isinstance(fi_dict, dict):
            px = _safe_float(fi_dict.get("lastPrice") or fi_dict.get("last_price"))
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
    vals: Dict[str, Optional[float]] = {
        "PE": None, "PS": None, "EV_EBITDA": None, "EV_REV": None, "PEG": None, "FCF_YIELD": None,
    }
    srcs: Dict[str, str] = {k: "none" for k in vals.keys()}

    # info with retry
    info = None
    last_err = None
    for i in range(max(1, YF_MAX_RETRIES)):
        try:
            info = tk.get_info() if hasattr(tk, "get_info") else tk.info
            break
        except Exception as e:
            last_err = e
            if _is_rate_limited(e) and i < YF_MAX_RETRIES - 1:
                logger.debug("[val] rate-limited; retrying %d/%d", i+1, YF_MAX_RETRIES-1)
                time.sleep(YF_RETRY_SLEEP)
                continue
            logger.debug("info fetch failed: %r", e)
            info = None
            break

    fi = getattr(tk, "fast_info", None)
    if not isinstance(fi, dict):
        fi = None

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
            ev = _get_enterprise_value(info, tk)
            ebitda = _safe_float(info.get("ebitda"))
            if ev and ebitda and ebitda > 0:
                vals["EV_EBITDA"], srcs["EV_EBITDA"] = ev / ebitda, "computed"

    # EV/Revenue
    if isinstance(info, dict):
        ev_rev = _safe_float(info.get("enterpriseToRevenue"))
        if ev_rev and ev_rev > 0:
            vals["EV_REV"], srcs["EV_REV"] = ev_rev, "info"
        else:
            ev = _get_enterprise_value(info, tk)
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
def fetch_pe_for_top(tickers: List[str]) -> Dict[str, Optional[float]]:
    if os.getenv("DISABLE_PE_FETCH", "").strip().lower() in {"1","true","yes"}:
        return {t: None for t in tickers}

    extra_aliases = {}
    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}
    alias_updates: Dict[str, str] = {}
    alias_updates: Dict[str, str] = {}

    cache = _load_cache(_PE_CACHE_PATH)
    logger.debug("[pe] cache_path=%s", os.path.abspath(_PE_CACHE_PATH))
    out: Dict[str, Optional[float]] = {}
    delay = float(os.getenv("PE_FETCH_DELAY_S", "0.15"))

    for t in tickers:
        cached = _from_cache(cache, t, _PE_CACHE_TTL_S, _PE_CACHE_PATH)
        if cached is not None and "pe" in cached:
            out[t] = _safe_float(cached.get("pe"))
            logger.debug("[pe] cache %s -> %s (%s)", t, _fmt(out[t]), cached.get("source") or "unknown")
            continue

        ali = apply_alias(t, extra_aliases)
        yh = _normalize_symbol_for_yahoo(ali)

        try:
            tk = yf.Ticker(yh)
            pe, src = _extract_pe_with_compute(tk)
            out[t] = pe
            # only cache success (not rate-limit empties)
            _put_cache(cache, t, {"pe": pe, "source": src})
            logger.info("[pe] %s (%s) -> %s (%s)", t, yh, "N/A" if pe is None else f"{pe:.2f}", src)
        except Exception as e:
            out[t] = None
            if _is_rate_limited(e):
                logger.warning("[pe] rate-limited for %s (%s); NOT caching empty result", t, yh)
            else:
                logger.warning("[pe] fetch fail %s (%s): %r (NOT caching empty)", t, yh, e)
        time.sleep(delay)

    _save_cache(_PE_CACHE_PATH, cache)
    return out

# ---------- Full valuations fetch ----------
def fetch_valuations_for_top(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if os.getenv("DISABLE_VAL_FETCH", "").strip().lower() in {"1","true","yes"}:
        return {t: {"PE": None, "PS": None, "EV_EBITDA": None, "EV_REV": None, "PEG": None, "FCF_YIELD": None} for t in tickers}

    extra_aliases = {}
    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}

    cache = _load_cache(_VAL_CACHE_PATH)
    logger.debug("[val] cache_path=%s", os.path.abspath(_VAL_CACHE_PATH))
    out: Dict[str, Dict[str, Optional[float]]] = {}
    delay = float(os.getenv("VAL_FETCH_DELAY_S", os.getenv("PE_FETCH_DELAY_S", "0.15")))

    for t in tickers:
        cached = _from_cache(cache, t, _VAL_CACHE_TTL_S, _VAL_CACHE_PATH)
        if cached is not None and isinstance(cached.get("vals"), dict):
            vals_clean = _sanitize_val_block(cached["vals"])
            out[t] = vals_clean
            logger.debug("[val] cache %s -> %s", t, {k:_fmt(v) for k,v in vals_clean.items()})
            continue

        ali = apply_alias(t, extra_aliases)
        yh = _normalize_symbol_for_yahoo(ali)

        try:
            tk = yf.Ticker(yh)
            vals, srcs = _extract_valuations_version_safe(tk)
            out[t] = vals
            # cache only if at least one field present (avoid caching all-None due to transient errors)
            if any(v is not None for v in vals.values()):
                _put_cache(cache, t, {"vals": vals, "sources": srcs})
            else:
                logger.debug("[val] %s (%s) returned all None; NOT caching", t, yh)
            logger.info(
                "[val] %s (%s) -> PE=%s PS=%s EV/EBITDA=%s EV/REV=%s PEG=%s FCF_YIELD=%s%%",
                t, yh,
                _fmt(vals["PE"]), _fmt(vals["PS"]), _fmt(vals["EV_EBITDA"]), _fmt(vals["EV_REV"]),
                _fmt(vals["PEG"]), _fmt(vals["FCF_YIELD"])
            )
        except Exception as e:
            out[t] = {"PE": None, "PS": None, "EV_EBITDA": None, "EV_REV": None, "PEG": None, "FCF_YIELD": None}
            if _is_rate_limited(e):
                logger.warning("[val] rate-limited for %s (%s); NOT caching empty result", t, yh)
            else:
                logger.warning("[val] fetch fail %s (%s): %r (NOT caching empty)", t, yh, e)
        time.sleep(delay)

    _save_cache(_VAL_CACHE_PATH, cache)
    return out


# ---------- Cached history downloader (batch-aware) ----------
YF_HISTORY_CACHE_DIR = os.getenv("YF_HISTORY_CACHE_DIR", "data/yf_history_cache")
YF_HISTORY_CACHE_TTL_S = int(os.getenv("YF_HISTORY_CACHE_TTL_S", "86400"))
# Large Yahoo batch requests are unreliable; chunk or disable batching.
YF_BATCH_SIZE = int(os.getenv("YF_BATCH_SIZE", "200"))
YF_DISABLE_BATCH = os.getenv("YF_DISABLE_BATCH", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _cache_path_for(ticker: str, period: Optional[str], start: Optional[str], end: Optional[str], interval: str) -> Path:
    key = f"{ticker}__{period or (start or '')}__{end or ''}__{interval}"
    # safe filename
    name = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(YF_HISTORY_CACHE_DIR) / f"{ticker}_{name}.pkl"


def download_history_cached_dict(
    tickers: List[str],
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = False,
    group_by: Optional[str] = "ticker",
    threads: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Return a dict {ticker: DataFrame} with cached per-ticker history.

    Only downloads missing or expired tickers in a single yf.download batch when possible.
    """
    os.makedirs(YF_HISTORY_CACHE_DIR, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []
    norm_map: Dict[str, str] = {}

    # normalize symbols and decide cache hits
    extra_aliases = {}
    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}

    for t in tickers:
        ali = apply_alias(t, extra_aliases)
        ysym = _normalize_symbol_for_yahoo(ali)
        norm_map[t] = ysym
        p = _cache_path_for(t, period, start, end, interval)
        if p.exists() and (time.time() - p.stat().st_mtime) <= YF_HISTORY_CACHE_TTL_S:
            try:
                df = pd.read_pickle(p)
                # Skip empty cache entries (force re-download)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out[t] = df
                    continue
            except Exception:
                pass
        to_download.append(t)

    def _drop_all_nan_rows(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return df
        try:
            if df.empty:
                return df
            return df.dropna(how="all")
        except Exception:
            return df

    if to_download:
        def _download_single(sym: str) -> pd.DataFrame:
            try:
                return yf.download(
                    tickers=sym,
                    period=period,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=progress,
                    group_by=group_by,
                    threads=False,
                )
            except Exception as e:
                print(f"[data_fetcher][WARN] yf.download single failed for {sym}: {e}")
                return pd.DataFrame()

        def _try_suffixes(orig_sym: str, sym: str) -> pd.DataFrame:
            # Try common Yahoo suffixes. If sym already has a suffix, strip it first.
            base = sym.split(".", 1)[0] if "." in sym else sym
            if not base:
                return pd.DataFrame()
            cur_suf = sym.split(".", 1)[1] if "." in sym else None
            for suf in sorted(_YH_SUFFIXES):
                if cur_suf and suf == cur_suf:
                    continue
                trial = f"{base}.{suf}"
                df_try = _download_single(trial)
                df_try = _drop_all_nan_rows(df_try)
                if df_try is not None and not df_try.empty:
                    print(f"[data_fetcher] resolved {sym} -> {trial}")
                    alias_updates[orig_sym] = trial
                    return df_try
            return pd.DataFrame()

        def _chunks(lst: List[str], n: int) -> List[List[str]]:
            if n <= 0:
                return [lst]
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        if YF_DISABLE_BATCH or YF_BATCH_SIZE <= 1:
            for orig in to_download:
                ysym = norm_map.get(orig)
                df_raw = _download_single(ysym)
                df_raw = _drop_all_nan_rows(df_raw)
                try:
                    pd.to_pickle(df_raw, _cache_path_for(orig, period, start, end, interval))
                except Exception:
                    pass
                out[orig] = df_raw
            return out

        # Chunked batch downloads to avoid Yahoo batch instability
        for chunk in _chunks(to_download, YF_BATCH_SIZE):
            syms = [norm_map[t] for t in chunk]
            try:
                print(f"[data_fetcher] yf.download batch for {len(syms)} symbols")
                data = yf.download(
                    tickers=" ".join(syms),
                    period=period,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=progress,
                    group_by=group_by,
                    threads=threads,
                )
            except Exception as e:
                print(f"[data_fetcher][WARN] yf.download batch failed: {e}")
                data = None

            # Extract per-ticker frames from the batch result
            for orig in chunk:
                ysym = norm_map.get(orig)
                df_raw = None
                try:
                    if data is None:
                        df_raw = _download_single(ysym)
                    elif isinstance(data.columns, pd.MultiIndex):
                        # try (ticker, field) orientation
                        try:
                            df_raw = data.xs(ysym, axis=1, level=0, drop_level=True)
                        except Exception:
                            try:
                                df_raw = data[ysym]
                            except Exception:
                                df_raw = pd.DataFrame()
                    else:
                        # single-ticker or single-frame
                        df_raw = data.copy()
                except Exception:
                    df_raw = pd.DataFrame()

                # If batch result missing this symbol, retry single
                if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                    df_raw = _download_single(ysym)
                # If batch result is all-NaN rows (common in large batches), drop and retry single
                df_raw = _drop_all_nan_rows(df_raw)
                if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                    df_raw = _download_single(ysym)
                # If still empty, try common suffixes
                if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                    df_raw = _try_suffixes(orig, ysym)
                    df_raw = _drop_all_nan_rows(df_raw)

            # write to cache if we have something
            try:
                if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                    pd.to_pickle(df_raw, _cache_path_for(orig, period, start, end, interval))
            except Exception:
                pass
            out[orig] = df_raw

    if alias_updates:
        try:
            save_aliases_csv("data/aliases.csv", alias_updates)
            logger.info("[data_fetcher] saved %d alias updates to data/aliases.csv", len(alias_updates))
        except Exception:
            pass
    return out

# simple CLIs
if __name__ == "__main__":
    import sys, json as _json
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "KOD", "RDDT"]
    vals = fetch_valuations_for_top(tickers)
    print(_json.dumps(vals, indent=2))
