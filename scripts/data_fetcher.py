# scripts/data_fetcher.py
from __future__ import annotations
import os, time, logging
from typing import Dict, List, Optional

import yfinance as yf
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
            if cur_suf:
                df_try = _download_single(base)
                df_try = _drop_all_nan_rows(df_try)
                if df_try is not None and not df_try.empty:
                    print(f"[data_fetcher] resolved {sym} -> {base}")
                    alias_updates[orig_sym] = base
                    return df_try
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

