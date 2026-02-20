# scripts/data_fetcher.py
from __future__ import annotations
import os, time, logging, random
from typing import Dict, List, Optional

import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import hashlib
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

def _is_rate_limit(err: object) -> bool:
    s = str(err)
    return isinstance(err, YFRateLimitError) or "RateLimit" in s or "Too Many Requests" in s

def _sleep_backoff(attempt: int) -> None:
    base = max(YF_RATE_LIMIT_SLEEP, YF_RETRY_SLEEP, 0.1)
    exp = min(attempt, 10)
    sleep_s = min(base * (2 ** exp), 60.0)
    sleep_s += random.random() * 0.25 * sleep_s
    time.sleep(sleep_s)

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

# ---------- Cached history downloader (batch-aware) ----------
YF_HISTORY_CACHE_DIR = os.getenv("YF_HISTORY_CACHE_DIR", "data/yf_history_cache")
YF_HISTORY_CACHE_TTL_S = int(os.getenv("YF_HISTORY_CACHE_TTL_S", "86400"))
# Large Yahoo batch requests are unreliable; chunk or disable batching.
YF_BATCH_SIZE = int(os.getenv("YF_BATCH_SIZE", os.getenv("YF_CHUNK_SIZE", "200")))
YF_BATCH_SLEEP = float(os.getenv("YF_BATCH_SLEEP", "0"))
YF_DISABLE_BATCH = os.getenv("YF_DISABLE_BATCH", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
YF_BATCH_FALLBACK_ALL = os.getenv("YF_BATCH_FALLBACK_ALL", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
# Rate-limit handling
YF_RATE_LIMIT_RETRIES = int(os.getenv("YF_RATE_LIMIT_RETRIES", "2"))
YF_RATE_LIMIT_SLEEP = float(os.getenv("YF_RATE_LIMIT_SLEEP", "2.5"))
# Limit suffix attempts per symbol (0 = all)
YF_SUFFIX_MAX = int(os.getenv("YF_SUFFIX_MAX", "0"))
YF_SUFFIX_PRIORITY = os.getenv("YF_SUFFIX_PRIORITY", "")


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
    # de-duplicate tickers while preserving order
    if tickers:
        seen = set()
        uniq = []
        for t in tickers:
            tt = (t or "").strip().upper()
            if not tt or tt in seen:
                continue
            seen.add(tt)
            uniq.append(tt)
        tickers = uniq

    out: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []
    norm_map: Dict[str, str] = {}
    rate_limit_hit = False
    rate_limited_tickers: set[str] = set()

    def _mark_rate_limit(err: object, tickers: List[str] | None = None) -> bool:
        nonlocal rate_limit_hit
        if _is_rate_limit(err):
            rate_limit_hit = True
            if tickers:
                rate_limited_tickers.update([t.upper() for t in tickers if t])
            return True
        return False

    def _check_shared_rate_limit(tickers: List[str] | None = None) -> bool:
        nonlocal rate_limit_hit
        try:
            import yfinance.shared as shared  # type: ignore
            errs = getattr(shared, "_ERRORS", None)
            if not errs:
                return False
            if isinstance(errs, dict):
                vals = list(errs.values())
                try:
                    shared._ERRORS = {}
                except Exception:
                    pass
            else:
                vals = list(errs)
                try:
                    shared._ERRORS = {}
                except Exception:
                    pass
            for e in vals:
                if _is_rate_limit(e):
                    rate_limit_hit = True
                    if tickers:
                        rate_limited_tickers.update([t.upper() for t in tickers if t])
                    return True
        except Exception:
            return False
        return False

    # normalize symbols and decide cache hits
    extra_aliases = {}
    try:
        extra_aliases = load_aliases_csv("data/aliases.csv")
    except Exception:
        extra_aliases = {}
    alias_updates: Dict[str, str] = {}

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
        def _download_single(sym: str, mark_sym: str | None = None) -> pd.DataFrame:
            attempt = 0
            while True:
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
                    is_rate_limit = _mark_rate_limit(e, [mark_sym or sym])
                    if is_rate_limit and (YF_RATE_LIMIT_RETRIES < 0 or attempt < YF_RATE_LIMIT_RETRIES):
                        _sleep_backoff(attempt)
                        attempt += 1
                        continue
                    print(f"[data_fetcher][WARN] yf.download single failed for {sym}: {e}")
                    return pd.DataFrame()

        def _try_suffixes(orig_sym: str, sym: str) -> pd.DataFrame:
            # Try common Yahoo suffixes. If sym already has a suffix, strip it first.
            base = sym.split(".", 1)[0] if "." in sym else sym
            if not base:
                return pd.DataFrame()
            cur_suf = sym.split(".", 1)[1] if "." in sym else None
            if YF_SUFFIX_PRIORITY:
                priority = [s.strip().upper() for s in YF_SUFFIX_PRIORITY.split(",") if s.strip()]
            else:
                priority = [
                    "TO","V","L","AX","PA","AS","DE","F","SW","MI","MC","HK","T",
                    "SI","KS","KQ","NS","BO","SA","BR","MX","NZ","IR","HE","ST",
                    "OL","CO","WA","PR","VI","AD","SR","TA","SS","SZ","TW","DU",
                ]
            base_list = [s for s in priority if s in _YH_SUFFIXES]
            suffixes = base_list + [s for s in sorted(_YH_SUFFIXES) if s not in base_list]
            if YF_SUFFIX_MAX > 0:
                suffixes = suffixes[:YF_SUFFIX_MAX]
            for suf in suffixes:
                if cur_suf and suf == cur_suf:
                    continue
                trial = f"{base}.{suf}"
                df_try = _download_single(trial, orig_sym)
                df_try = _drop_all_nan_rows(df_try)
                if df_try is not None and not df_try.empty:
                    print(f"[data_fetcher] resolved {sym} -> {trial}")
                    alias_updates[orig_sym] = trial
                    try:
                        save_aliases_csv("data/aliases.csv", {orig_sym: trial})
                    except Exception:
                        pass
                    return df_try
            if cur_suf:
                df_try = _download_single(base, orig_sym)
                df_try = _drop_all_nan_rows(df_try)
                if df_try is not None and not df_try.empty:
                    print(f"[data_fetcher] resolved {sym} -> {base}")
                    alias_updates[orig_sym] = base
                    try:
                        save_aliases_csv("data/aliases.csv", {orig_sym: base})
                    except Exception:
                        pass
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
            data = None
            attempt = 0
            while True:
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
                    if _check_shared_rate_limit(chunk):
                        raise YFRateLimitError("rate limited")
                    break
                except Exception as e:
                    is_rate_limit = _mark_rate_limit(e, chunk)
                    if is_rate_limit and (YF_RATE_LIMIT_RETRIES < 0 or attempt < YF_RATE_LIMIT_RETRIES):
                        _sleep_backoff(attempt)
                        attempt += 1
                        continue
                    print(f"[data_fetcher][WARN] yf.download batch failed: {e}")
                    data = None
                    break

            # First pass: extract per-ticker frames from batch only
            batch_frames: Dict[str, pd.DataFrame] = {}
            batch_missing = []
            for orig in chunk:
                ysym = norm_map.get(orig)
                df_raw = pd.DataFrame()
                if data is not None:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
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
                df_raw = _drop_all_nan_rows(df_raw)
                if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                    batch_missing.append(orig)
                    df_raw = pd.DataFrame()
                batch_frames[orig] = df_raw

            # If any missing in batch, optionally re-fetch all tickers singly (slow but robust)
            if batch_missing and YF_BATCH_FALLBACK_ALL:
                for orig in chunk:
                    ysym = norm_map.get(orig)
                    df_raw = _download_single(ysym, orig)
                    df_raw = _drop_all_nan_rows(df_raw)
                    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                        df_raw = _try_suffixes(orig, ysym)
                        df_raw = _drop_all_nan_rows(df_raw)
                    try:
                        if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                            pd.to_pickle(df_raw, _cache_path_for(orig, period, start, end, interval))
                    except Exception:
                        pass
                    out[orig] = df_raw
            else:
                for orig in chunk:
                    ysym = norm_map.get(orig)
                    df_raw = batch_frames.get(orig, pd.DataFrame())
                    # If batch result missing this symbol, retry single
                    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                        df_raw = _download_single(ysym, orig)
                    # If batch result is all-NaN rows, drop and retry single
                    df_raw = _drop_all_nan_rows(df_raw)
                    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                        df_raw = _download_single(ysym, orig)
                    # If still empty, try common suffixes
                    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                        df_raw = _try_suffixes(orig, ysym)
                        df_raw = _drop_all_nan_rows(df_raw)
                    try:
                        if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                            pd.to_pickle(df_raw, _cache_path_for(orig, period, start, end, interval))
                    except Exception:
                        pass
                    out[orig] = df_raw

            if YF_BATCH_SLEEP > 0:
                time.sleep(YF_BATCH_SLEEP)

    if alias_updates:
        try:
            save_aliases_csv("data/aliases.csv", alias_updates)
            logger.info("[data_fetcher] saved %d alias updates to data/aliases.csv", len(alias_updates))
        except Exception:
            pass
    download_history_cached_dict.last_rate_limit = rate_limit_hit  # type: ignore[attr-defined]
    download_history_cached_dict.last_rate_limited_tickers = sorted(rate_limited_tickers)  # type: ignore[attr-defined]
    return out

