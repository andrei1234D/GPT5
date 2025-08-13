# scripts/features.py
import math
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd
import yfinance as yf
import os

CHUNK_SIZE = int(os.getenv("YF_CHUNK_SIZE", "60"))
MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("YF_RETRY_SLEEP", "2.0"))
# ---------------------------- constants ---------------------------- #
FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

# Valid Yahoo exchange suffixes we want to preserve
_YH_SUFFIXES = {
    "L","DE","F","SW","PA","AS","BR","LS","MC","MI","VI","ST","HE","CO","OL","WA","PR",
    "TO","V","AX","HK","T","SI","KS","KQ","NS","BO","JO","SA","MX","NZ","DU","AD","SR","TA",
    "SS","SZ","TW","IR"
}

# ---------------------------- indicators --------------------------- #
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def _macd_hist(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def _atr_pct_from_hl_close(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    need = {"High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    tr = (df["High"] - df["Low"]).abs()
    atr = tr.rolling(period).mean()
    return (atr / df["Close"]) * 100

def _pct(a: float, b: float) -> float:
    if b == 0 or pd.isna(a) or pd.isna(b):
        return float("nan")
    return (a / b - 1.0) * 100.0

def _anchored_vwap(px: pd.Series, vol: pd.Series, lookback: int = 252) -> Optional[float]:
    if px is None or vol is None:
        return None
    if len(px) < 5 or len(px) != len(vol):
        return None
    n = min(lookback, len(px))
    p = px.iloc[-n:]
    v = vol.iloc[-n:]
    denom = float(v.sum())
    if denom == 0 or math.isnan(denom):
        return None
    return float((p * v).sum() / denom)

# ---------------------------- symbol utils ------------------------- #
def _normalize_symbol_for_yahoo(sym: str) -> str:
    """
    Keep real Yahoo exchange suffixes. If no real suffix, convert inner dot (class share)
    to hyphen (BRK.B -> BRK-B). Zero-pad numeric tickers for some markets.
    """
    s = (sym or "").strip().upper().replace(" ", "-")
    if not s:
        return s
    if "." in s:
        head, tail = s.rsplit(".", 1)
        if tail in _YH_SUFFIXES:
            if head.isdigit() and tail in {"T", "HK", "TW", "SS", "SZ"}:
                head = head.zfill(4)
            return f"{head}.{tail}"
        # Not a real exchange suffix → treat as US class-share separator
        s = s.replace(".", "-")
        return s
    # No dot → nothing special (US or already suffixless)
    return s

# ---------------------------- frame utils -------------------------- #
def _normalize_ohlcv(df: pd.DataFrame, ticker_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Return a single-ticker OHLCV frame with columns in FIELDS subset.
    Handles:
      - Single-level columns (already good)
      - MultiIndex (ticker, field) or (field, ticker)
      - Flattened names like 'SPY_Close'
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # Single-level columns already contain fields
    if isinstance(df.columns, pd.Index) and ("Close" in df.columns or "Adj Close" in df.columns):
        return df

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        # Case A: (ticker, field)
        if "Close" in set(lv1) or "Adj Close" in set(lv1):
            # choose the hint if present, otherwise first ticker
            tickers = list(dict.fromkeys(lv0))
            key = ticker_hint if ticker_hint in tickers else tickers[0]
            sub = df.xs(key, axis=1, level=0, drop_level=True)
            return sub

        # Case B: (field, ticker)
        if "Close" in set(lv0) or "Adj Close" in set(lv0):
            tickers = list(dict.fromkeys(lv1))
            key = ticker_hint if ticker_hint in tickers else tickers[0]
            sub = df.xs(key, axis=1, level=1, drop_level=True)
            return sub

        # Fallback: flatten "<whatever>_<Field>"
        flat = df.copy()
        flat.columns = ["_".join(map(str, c)).strip() for c in flat.columns.values]
        candidates = [c for c in flat.columns if c.endswith("_Close") or c.endswith("_Adj Close")]
        if candidates:
            prefix = candidates[0].rsplit("_", 1)[0]
            cols = [f"{prefix}_{k}" for k in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            existing = [c for c in cols if c in flat.columns]
            if existing:
                out = flat[existing].copy()
                out.columns = [c.split("_", 1)[1] for c in existing]
                return out

    # Last resort: return as-is
    return df

# ---------------------------- data fetch --------------------------- #
def fetch_history(
    tickers: List[str],
    period: str = "270d",
    chunk_size: int = 60,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """
    Returns {original_ticker: normalized OHLCV DataFrame}.
    - Preserves real Yahoo suffixes.
    - Batches requests to reduce rate-limit errors.
    - Auto-adjusted prices to keep units consistent (splits/dividends).
    """
    # Normalize symbols for Yahoo
    yh_map = {t: _normalize_symbol_for_yahoo(t) for t in tickers}

    out: Dict[str, pd.DataFrame] = {}
    keys = list(yh_map.keys())
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i + chunk_size]
        chunk_syms = [yh_map[k] for k in chunk_keys]

        # Retry a few times on transient errors
        attempt = 0
        data = None
        while attempt < max_retries:
            try:
                data = yf.download(
                    tickers=" ".join(chunk_syms),
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                break
            except Exception:
                attempt += 1
                if attempt < max_retries:
                    time.sleep(retry_sleep * attempt)
                else:
                    data = None

        # If download failed entirely, skip this chunk
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            continue

        # For each ticker in this chunk, extract + normalize its own frame
        for orig in chunk_keys:
            yh = yh_map[orig]
            try:
                df_norm = _normalize_ohlcv(data, ticker_hint=yh)
                if isinstance(df_norm, pd.DataFrame) and not df_norm.empty:
                    # keep just known fields, dropna
                    cols = [c for c in df_norm.columns if c in FIELDS]
                    if cols:
                        out[orig] = df_norm[cols].dropna(how="all")
            except Exception:
                # try a one-off single download as a fallback
                try:
                    single = yf.download(
                        tickers=yh,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=False,
                    )
                    df1 = _normalize_ohlcv(single)
                    cols = [c for c in df1.columns if c in FIELDS]
                    if cols:
                        out[orig] = df1[cols].dropna(how="all")
                except Exception:
                    pass

    return out

# ---------------------------- compute ------------------------------ #
def compute_indicators(df: pd.DataFrame) -> dict:
    # Normalize shape first
    df = _normalize_ohlcv(df)
    cols = set(df.columns)

    # Prefer adjusted if present; with auto_adjust=True, Close is already adjusted
    if "Adj Close" in cols:
        px = df["Adj Close"]
    elif "Close" in cols:
        px = df["Close"]
    else:
        raise ValueError(f"No Close/Adj Close column found. Columns: {list(df.columns)}")

    vol = df["Volume"] if "Volume" in cols else pd.Series(index=px.index, dtype=float)

    sma20 = px.rolling(20).mean()
    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    rsi14 = _rsi(px, 14)
    macdh = _macd_hist(px)
    atrp_series = _atr_pct_from_hl_close(df, 14)  # may be None
    high_252 = px.rolling(252, min_periods=50).max()
    low_252 = px.rolling(252, min_periods=50).min()
    avwap252 = _anchored_vwap(px, vol, 252) if "Volume" in cols else None

    last = -1
    cur = float(px.iloc[last])

    def _safe_last(s: pd.Series) -> Optional[float]:
        try:
            v = float(s.iloc[last])
            return None if math.isnan(v) else v
        except Exception:
            return None

    feats = dict(
        price=round(cur, 2),

        vsSMA20=round(_pct(cur, float(sma20.iloc[last])), 2),
        vsSMA50=round(_pct(cur, float(sma50.iloc[last])), 2),
        vsSMA200=round(_pct(cur, float(sma200.iloc[last])), 2),

        SMA20=round(_safe_last(sma20), 2) if _safe_last(sma20) is not None else None,
        SMA50=round(_safe_last(sma50), 2) if _safe_last(sma50) is not None else None,
        SMA200=round(_safe_last(sma200), 2) if _safe_last(sma200) is not None else None,
        AVWAP252=round(avwap252, 2) if avwap252 is not None and not math.isnan(avwap252) else None,

        RSI14=int(round(rsi14.iloc[last])) if not math.isnan(rsi14.iloc[last]) else None,
        MACD_hist=round(float(macdh.iloc[last]), 3) if not math.isnan(macdh.iloc[last]) else None,
        ATRpct=(
            round(float(atrp_series.iloc[last]), 2)
            if isinstance(atrp_series, pd.Series) and not math.isnan(atrp_series.iloc[last])
            else None
        ),

        drawdown_pct=round(_pct(cur, float(high_252.iloc[last])), 2),
        dist_to_52w_low=round(_pct(cur, float(low_252.iloc[last])), 2),

        d5=round(_pct(cur, float(px.iloc[-5])), 2) if len(px) >= 5 else None,
        d20=round(_pct(cur, float(px.iloc[-20])), 2) if len(px) >= 20 else None,
        r60=round(_pct(cur, float(px.iloc[-60])), 2) if len(px) >= 60 else None,
        r120=round(_pct(cur, float(px.iloc[-120])), 2) if len(px) >= 120 else None,

        is_20d_high=bool(cur >= (float(px.rolling(20).max().iloc[last]) * 0.999)) if len(px) >= 20 else None,
        vol_vs20=(
            round(_pct(float(vol.iloc[last]), float(vol.rolling(20).mean().iloc[last])), 2)
            if "Volume" in cols and not pd.isna(vol.rolling(20).mean().iloc[last])
            else None
        ),
    )
    return feats

# ---------------------------- API ---------------------------------- #
def build_features(
    universe: List[Tuple[str, str]],
    batch_size: int = 150,
    period: str = "270d",
) -> Dict[str, Dict]:
    """
    Returns:
      { 'TICKER': {'company': 'Company Name', 'features': {...}}, ... }
    """
    out: Dict[str, Dict] = {}
    tickers = [t for t, _ in universe]
    name_map = dict(universe)

    # Fetch in batches; fetch_history does its own internal sub-chunking
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i + batch_size]
        if not chunk:
            continue

        hist = fetch_history(
            chunk,
            period=period,
            chunk_size=CHUNK_SIZE,
            max_retries=MAX_RETRIES,
            retry_sleep=RETRY_SLEEP,
        )

        for t in chunk:
            df = hist.get(t)
            if df is None or df.empty:
                continue
            try:
                feats = compute_indicators(df)
            except Exception:
                # Skip tickers we cannot normalize/compute
                continue
            out[t] = {"company": name_map.get(t, t), "features": feats}

    return out
