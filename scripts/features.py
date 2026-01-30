# scripts/features.py
from __future__ import annotations

import csv
import logging
import math
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from aliases import apply_alias, load_aliases_csv  # optional; safe if file missing

# ---------------------------- logging ------------------------------ #
def _maybe_configure_logging():
    # Prefer FEATURES_LOG_LEVEL, otherwise reuse RANKER_LOG_LEVEL
    level_name = (os.getenv("FEATURES_LOG_LEVEL") or os.getenv("RANKER_LOG_LEVEL") or "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


_maybe_configure_logging()

# Quiet 3rd-party DEBUG noise; keep our own logs verbose
for _name in ("yfinance", "peewee", "curl_cffi.requests", "urllib3", "requests"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger("features")

FEATURES_VERBOSE = os.getenv("FEATURES_VERBOSE", "").strip().lower() in {"1", "true", "yes"}

def _fmt(x, fmt=".2f"):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "None"
        return format(float(x), fmt)
    except Exception:
        return str(x)


# ---------------------------- config ------------------------------- #
CHUNK_SIZE = int(os.getenv("YF_CHUNK_SIZE", "60"))
MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "5"))
RETRY_SLEEP = float(os.getenv("YF_RETRY_SLEEP", "2.0"))
MIN_ROWS = int(os.getenv("YF_MIN_ROWS", "40"))

# IMPORTANT:
# This file is for **runtime** yfinance failures during feature building.
# Do NOT mix it with your long-lived "universe hygiene" rejects used by clean_universe.py.
REJECTS_PATH = os.getenv("FEATURES_REJECTS_PATH", "data/universe_rejects_runtime.csv")
REJECTS_MODE = (os.getenv("FEATURES_REJECTS_MODE", "overwrite") or "overwrite").lower().strip()
# REJECTS_MODE:
#   - overwrite: replace file each run (recommended)
#   - append: append new rows (keeps growing)
#   - daily: create per-day file data/universe_rejects_runtime_YYYY-MM-DD.csv
if REJECTS_MODE not in {"overwrite", "append", "daily"}:
    REJECTS_MODE = "overwrite"


# ---------------------------- constants ---------------------------- #
FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

# ---------- symbol normalization (consistent with data_fetcher.py) ---------- #
_YH_SUFFIXES = {
    "L", "DE", "F", "SW", "PA", "AS", "BR", "LS", "MC", "MI", "VI", "ST", "HE", "CO", "OL", "WA", "PR",
    "TO", "V", "AX", "HK", "T", "SI", "KS", "KQ", "NS", "BO", "JO", "SA", "MX", "NZ", "DU", "AD", "SR", "TA",
    "SS", "SZ", "TW", "IR"
}


# ---------------------------- EMA helpers -------------------------- #
def _ema(series: pd.Series, length: int) -> pd.Series:
    # min_periods ensures early values are NaN until we have enough data
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def add_ema_features(hist: pd.DataFrame, feats: dict) -> None:
    """Attach EMA20/50/200, vsEMA50/200, and a tiny EMA50 slope to feats."""
    if hist is None or hist.empty or "Close" not in hist:
        feats["EMA20"] = feats["EMA50"] = feats["EMA200"] = None
        feats["vsEMA50"] = feats["vsEMA200"] = None
        feats["EMA50_slope_5d"] = None
        return

    c = hist["Close"].astype(float)

    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200 = _ema(c, 200)

    def last_val(s: pd.Series) -> Optional[float]:
        try:
            v = s.iloc[-1]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    feats["EMA20"] = last_val(ema20)
    feats["EMA50"] = last_val(ema50)
    feats["EMA200"] = last_val(ema200)

    px = feats.get("price")
    e50 = feats.get("EMA50")
    e200 = feats.get("EMA200")

    feats["vsEMA50"] = ((px / e50) - 1.0) * 100.0 if (px and e50) else None
    feats["vsEMA200"] = ((px / e200) - 1.0) * 100.0 if (px and e200) else None

    # 5-trading-day slope on EMA50 (small momentum hint)
    try:
        if (
            len(ema50) >= 6
            and pd.notna(ema50.iloc[-6])
            and pd.notna(ema50.iloc[-1])
            and float(ema50.iloc[-6]) != 0.0
        ):
            feats["EMA50_slope_5d"] = (float(ema50.iloc[-1]) / float(ema50.iloc[-6]) - 1.0) * 100.0
        else:
            feats["EMA50_slope_5d"] = None
    except Exception:
        feats["EMA50_slope_5d"] = None


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
        if FEATURES_VERBOSE:
            logger.debug(f"[ind] ATR skipped: missing {need - set(df.columns)}")
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
        s = s.replace(".", "-")
        return s
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
            tickers = list(dict.fromkeys(lv0))
            key = ticker_hint if (ticker_hint in tickers) else tickers[0]
            if FEATURES_VERBOSE:
                logger.debug(f"[norm] MultiIndex (ticker, field) -> key={key}")
            sub = df.xs(key, axis=1, level=0, drop_level=True)
            return sub

        # Case B: (field, ticker)
        if "Close" in set(lv0) or "Adj Close" in set(lv0):
            tickers = list(dict.fromkeys(lv1))
            key = ticker_hint if (ticker_hint in tickers) else tickers[0]
            if FEATURES_VERBOSE:
                logger.debug(f"[norm] MultiIndex (field, ticker) -> key={key}")
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
                if FEATURES_VERBOSE:
                    logger.debug(f"[norm] Flattened -> prefix={prefix}, fields={existing}")
                out = flat[existing].copy()
                out.columns = [c.split("_", 1)[1] for c in existing]
                return out

    # Last resort: return as-is
    return df


# ---------------------------- rejects writer ----------------------- #
def _rejects_output_path() -> str:
    if REJECTS_MODE == "daily":
        day = time.strftime("%Y-%m-%d")
        base, ext = os.path.splitext(REJECTS_PATH)
        return f"{base}_{day}{ext or '.csv'}"
    return REJECTS_PATH


def _write_rejects(rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    os.makedirs("data", exist_ok=True)
    path = _rejects_output_path()

    mode = "a"
    header_needed = True
    if REJECTS_MODE == "overwrite":
        mode = "w"
        header_needed = True
    else:
        header_needed = not os.path.exists(path)

    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["ticker", "alias_used", "yf_symbol", "reason", "ts"])
        for r in rows:
            w.writerow([r.get("ticker", ""), r.get("alias_used", ""), r.get("yf_symbol", ""), r.get("reason", ""), r.get("ts", "")])

    logger.info(f"[fetch] rejects wrote: {len(rows)} -> {path}")


# ---------------------------- data fetch --------------------------- #
def fetch_history(
    tickers: List[str],
    period: str = "270d",
    chunk_size: int = 60,
    max_retries: int = 5,
    retry_sleep: float = 2.0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, str]]]:
    """
    Returns:
      (hist_map, meta_map)

    hist_map:
      {original_ticker: normalized OHLCV DataFrame}

    meta_map (always, even for rejects):
      {original_ticker: {'alias_used': str, 'yf_symbol': str}}

    Rejects are written into REJECTS_PATH (default: data/universe_rejects_runtime.csv).
    """
    t0 = time.time()
    extra_aliases = load_aliases_csv("data/aliases.csv")  # optional

    # Map original -> (alias_used -> yf_symbol)
    meta: Dict[str, Dict[str, str]] = {}
    yh_map: Dict[str, str] = {}
    for t in tickers:
        alias_used = apply_alias(t, extra_aliases)
        yf_sym = _normalize_symbol_for_yahoo(alias_used)
        yh_map[t] = yf_sym
        meta[t] = {"alias_used": alias_used, "yf_symbol": yf_sym}

    out: Dict[str, pd.DataFrame] = {}
    rejects: List[Dict[str, str]] = []

    def _usable_df(df_in: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if not isinstance(df_in, pd.DataFrame) or df_in.empty:
            return None
        cols = [c for c in df_in.columns if c in FIELDS]
        if not cols:
            return None
        df2 = df_in[cols].dropna(how="all")
        if df2.empty or df2.shape[0] < MIN_ROWS:
            return None
        return df2

    for i in range(0, len(tickers), chunk_size):
        chunk_keys = tickers[i:i + chunk_size]
        chunk_syms = [yh_map[k] for k in chunk_keys]

        # Retry bounded (NOT infinite), with exponential backoff
        data = None
        last_err = None
        for attempt in range(1, max_retries + 1):
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
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Some errors are effectively permanent; don't spin.
                if "404" in msg or "not found" in msg or "invalid" in msg:
                    break
                sleep_s = retry_sleep * attempt
                logger.warning(f"[fetch] transient error (attempt {attempt}/{max_retries}): {e!r} sleep={sleep_s:.1f}s")
                time.sleep(sleep_s)

        if data is None and last_err is not None:
            logger.warning(f"[fetch] batch download failed: {last_err!r}")

        for orig in chunk_keys:
            yf_sym = yh_map[orig]
            alias_used = meta[orig]["alias_used"]

            df_norm = _normalize_ohlcv(data, ticker_hint=yf_sym) if data is not None else None
            df_use = _usable_df(df_norm)

            if df_use is not None:
                out[orig] = df_use
                continue

            # fallback single ticker
            single_ok = False
            last_single_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    single = yf.download(
                        tickers=yf_sym,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=False,
                    )
                    df1 = _normalize_ohlcv(single)
                    df_use = _usable_df(df1)
                    if df_use is not None:
                        out[orig] = df_use
                        single_ok = True
                    break
                except Exception as e:
                    last_single_err = e
                    msg = str(e).lower()
                    if "404" in msg or "not found" in msg or "invalid" in msg:
                        break
                    time.sleep(retry_sleep * attempt)

            if single_ok:
                continue

            reason = "404/empty-or-short"
            if last_single_err is not None:
                # Keep it short; avoids huge CSV rows
                reason = f"download-error:{type(last_single_err).__name__}"
            rejects.append(
                {
                    "ticker": orig,
                    "alias_used": alias_used,
                    "yf_symbol": yf_sym,
                    "reason": reason,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            if FEATURES_VERBOSE:
                logger.debug(f"[fetch] reject {orig} -> {yf_sym} (alias={alias_used}): {reason}")

    _write_rejects(rejects)

    logger.info(
        f"[fetch] finished in {time.time()-t0:.2f}s ok={len(out)} rejects={len(rejects)} / universe={len(tickers)}"
    )
    return out, meta


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

    # --- Liquidity proxies for tiering (cheap) ---
    try:
        avg_vol20 = float(vol.rolling(20).mean().iloc[last]) if "Volume" in cols else None
    except Exception:
        avg_vol20 = None
    avg_dollar_vol20 = (avg_vol20 * cur) if (avg_vol20 is not None and cur is not None) else None

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
        # Liquidity proxies
        avg_vol_20d=round(avg_vol20, 0) if avg_vol20 is not None else None,
        avg_dollar_vol_20d=round(avg_dollar_vol20, 2) if avg_dollar_vol20 is not None else None,
    )

    # Add EMA-derived features (uses Close column from df)
    add_ema_features(df, feats)
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

    Notes:
      - Also includes debug fields: alias_used, yf_symbol, history_ok, data_rows.
      - Rejects go to REJECTS_PATH (default: data/universe_rejects_runtime.csv).
    """
    t0 = time.time()
    out: Dict[str, Dict] = {}
    tickers = [t for t, _ in universe]
    name_map = dict(universe)

    total = len(tickers)
    fetched = computed = skipped_df = skipped_compute = 0

    logger.info(f"[build] start universe={total} batch_size={batch_size} period={period}")

    for i in range(0, total, batch_size):
        chunk = tickers[i:i + batch_size]
        if not chunk:
            continue

        try:
            hist, meta = fetch_history(
                chunk,
                period=period,
                chunk_size=CHUNK_SIZE,
                max_retries=MAX_RETRIES,
                retry_sleep=RETRY_SLEEP,
            )
        except Exception as e:
            logger.warning(f"[build] fetch failed for batch {i//batch_size+1}: {e!r}")
            continue

        fetched += len(hist)
        logger.info(f"[build] batch {i//batch_size+1}/{(total+batch_size-1)//batch_size} fetched={len(hist)}/{len(chunk)}")

        for t in chunk:
            df = hist.get(t)
            if df is None or df.empty:
                skipped_df += 1
                continue

            try:
                feats = compute_indicators(df)
                # attach meta for downstream debugging / alias inspection
                m = meta.get(t, {})
                feats["alias_used"] = m.get("alias_used")
                feats["yf_symbol"] = m.get("yf_symbol")
                feats["history_ok"] = True
                feats["data_rows"] = int(df.shape[0]) if isinstance(df, pd.DataFrame) else None

                out[t] = {"company": name_map.get(t, t), "features": feats}
                computed += 1
            except Exception as e:
                skipped_compute += 1
                logger.warning(f"[build] compute failed {t}: {e!r}")

    logger.info(
        "[build] done in %.2fs | fetched=%d computed=%d skipped_df=%d skipped_compute=%d total=%d",
        time.time()-t0, fetched, computed, skipped_df, skipped_compute, total
    )
    return out
