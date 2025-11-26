# scripts/trend_applier.py
from __future__ import annotations
import json, time, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from yfinance.exceptions import YFRateLimitError
from data_fetcher import download_history_cached_dict

CACHE_PATH = Path("data/market_trend.json")
CACHE_TTL_HOURS = 6

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _read_cache() -> Optional[str]:
    try:
        if not CACHE_PATH.exists():
            return None
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        ts = datetime.fromisoformat(obj.get("timestamp"))
        if datetime.utcnow() - ts <= timedelta(hours=CACHE_TTL_HOURS):
            return obj.get("trend")
        return None
    except Exception:
        return None

def _write_cache(trend: str) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump({"trend": trend, "timestamp": _now_iso()}, f)
    except Exception:
        pass  # cache is best-effort

def _download_single(symbol: str, period: str = "300d", retries: int = 5) -> pd.DataFrame:
    """
    Robust single-symbol download with backoff; returns OHLCV df or empty df.
    """
    # Use the cached batch downloader to minimize calls
    try:
        got = download_history_cached_dict([symbol], period=period, interval="1d", auto_adjust=True)
        df = got.get(symbol)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()

def _compute_trend(close: pd.Series) -> str:
    """
    Bullish if Close>SMA200 and SMA50>SMA200
    Bearish if Close<SMA200 and SMA50<SMA200
    Else Neutral. Need >=200 obs, else Neutral.
    """
    if close is None or close.empty or len(close) < 200:
        return "Neutral"
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    last_close = float(close.iloc[-1])
    last_sma50 = float(sma50.iloc[-1])
    last_sma200 = float(sma200.iloc[-1])
    if (np.isnan(last_sma50)) or (np.isnan(last_sma200)):
        return "Neutral"
    if last_close > last_sma200 and last_sma50 > last_sma200:
        return "Bullish"
    if last_close < last_sma200 and last_sma50 < last_sma200:
        return "Bearish"
    return "Neutral"

def detect_market_trend(symbols: Iterable[str] = ("SPY", "^GSPC")) -> str:
    """
    Try each symbol in order; on first usable dataset compute regime.
    If all fail, return Neutral.
    """
    for sym in symbols:
        df = _download_single(sym)
        if df is None or df.empty:
            continue
        close = df.get("Close")
        if close is None or close.empty:
            continue
        trend = _compute_trend(close)
        return trend
    return "Neutral"

def apply_market_env() -> str:
    """
    Public entry: use short-lived cache; on miss compute and cache.
    Never raise—always returns a valid string.
    """
    cached = _read_cache()
    if cached:
        return cached
    trend = detect_market_trend(("SPY", "^GSPC"))
    _write_cache(trend)
    return trend
