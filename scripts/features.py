# scripts/features.py
import math
import time
from typing import Dict, List, Tuple, Optional
import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from aliases import apply_alias, load_aliases_csv  # optional; safe if file missing

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

    ema20  = _ema(c, 20)
    ema50  = _ema(c, 50)
    ema200 = _ema(c, 200)

    def last_val(s: pd.Series) -> Optional[float]:
        try:
            v = s.iloc[-1]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    feats["EMA20"]  = last_val(ema20)
    feats["EMA50"]  = last_val(ema50)
    feats["EMA200"] = last_val(ema200)

    px   = feats.get("price")
    e50  = feats.get("EMA50")
    e200 = feats.get("EMA200")

    feats["vsEMA50"]  = ((px / e50)  - 1.0) * 100.0 if (px and e50)  else None
    feats["vsEMA200"] = ((px / e200) - 1.0) * 100.0 if (px and e200) else None

    # 5-trading-day slope on EMA50 (small momentum hint)
    try:
        if len(ema50) >= 6 and pd.notna(ema50.iloc[-6]) and pd.notna(ema50.iloc[-1]) and float(ema50.iloc[-6]) != 0.0:
            feats["EMA50_slope_5d"] = (float(ema50.iloc[-1]) / float(ema50.iloc[-6]) - 1.0) * 100.0
        else:
            feats["EMA50_slope_5d"] = None
    except Exception:
        feats["EMA50_slope_5d"] = None

# ---------------------------- logging ------------------------------ #
def _maybe_configure_logging():
    # Prefer FEATURES_LOG_LEVEL, otherwise reuse RANKER_LOG_LEVEL
    level_name = (os.getenv("FEATURES_LOG_LEVEL") or os.getenv("RANKER_LOG_LEVEL") or "").upper().strip()
    if not level_name:
        return
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_maybe_configure_logging()
# quiet 3rd-party DEBUG noise; keep our own logs verbose
for _name in ("yfinance", "peewee", "curl_cffi.requests", "urllib3", "requests"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger("features")

FEATURES_VERBOSE = os.getenv("FEATURES_VERBOSE", "").strip().lower() in {"1", "true", "yes"}
FEATURES_LOG_EVERY = int(os.getenv("FEATURES_LOG_EVERY", "300"))

def _fmt(x, fmt=".2f"):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "None"
        return format(float(x), fmt)
    except Exception:
        return str(x)

# ---------------------------- config ------------------------------- #
CHUNK_SIZE = int(os.getenv("YF_CHUNK_SIZE", "60"))
MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("YF_RETRY_SLEEP", "2.0"))

# ---------------------------- constants ---------------------------- #
FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

# ---------- symbol normalization (consistent with data_fetcher.py) ---------- #
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
            key = ticker_hint if ticker_hint in tickers else tickers[0]
            if FEATURES_VERBOSE:
                logger.debug(f"[norm] MultiIndex (ticker, field) -> key={key}")
            sub = df.xs(key, axis=1, level=0, drop_level=True)
            return sub

        # Case B: (field, ticker)
        if "Close" in set(lv0) or "Adj Close" in set(lv0):
            tickers = list(dict.fromkeys(lv1))
            key = ticker_hint if ticker_hint in tickers else tickers[0]
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
    - Applies aliases (static + data/aliases.csv if present).
    - Batches requests to reduce rate-limit errors.
    - Auto-adjusted prices to keep units consistent (splits/dividends).
    - Accepts only frames with >= YF_MIN_ROWS (default 40) rows after cleaning.
    """
    t0 = time.time()
    min_rows = int(os.getenv("YF_MIN_ROWS", "40"))

    extra_aliases = load_aliases_csv("data/aliases.csv")  # optional external file

    # apply aliases first, then normalize for Yahoo
    alias_applied: Dict[str, str] = {}
    yh_map: Dict[str, str] = {}
    for t in tickers:
        ali = apply_alias(t, extra_aliases)
        if ali != t:
            alias_applied[t] = ali
        yh_map[t] = _normalize_symbol_for_yahoo(ali)

    if FEATURES_VERBOSE and alias_applied:
        for k, v in alias_applied.items():
            logger.info(f"[alias] {k} -> {v}")

    out: Dict[str, pd.DataFrame] = {}
    rejects: List[Tuple[str, str]] = []  # (ticker, reason)
    keys = list(yh_map.keys())

    if FEATURES_VERBOSE:
        logger.info(f"[fetch] start period={period} universe={len(keys)} chunk_size={chunk_size} retries={max_retries} min_rows={min_rows}")

    # totals across all chunks
    total_ok = total_fb_ok = total_empty = total_rejects = 0

    def _usable_df(df_in: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Keep only OHLCV cols, drop all-NaN rows, require min_rows."""
        if not isinstance(df_in, pd.DataFrame) or df_in.empty:
            return None
        cols = [c for c in df_in.columns if c in FIELDS]
        if not cols:
            return None
        df2 = df_in[cols].dropna(how="all")
        if df2 is None or df2.empty or df2.shape[0] < min_rows:
            return None
        return df2

    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i + chunk_size]
        chunk_syms = [yh_map[k] for k in chunk_keys]
        if FEATURES_VERBOSE:
            logger.info(
                f"[fetch] chunk {i//chunk_size+1}/{(len(keys)+chunk_size-1)//chunk_size} size={len(chunk_keys)}"
            )

        # 🔥 added: throttle between chunks
        if i > 0:
            sleep_s = float(os.getenv("YF_SLEEP_BETWEEN_CHUNKS", "1.0"))
            if FEATURES_VERBOSE:
                logger.info(f"[fetch] sleeping {sleep_s:.1f}s before next chunk…")
            time.sleep(sleep_s)

        # Retry the batch on transient errors
        attempt = 0
        data = None
        while attempt < max_retries:
            try:
                t1 = time.time()
                data = yf.download(
                    tickers=" ".join(chunk_syms),
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                if FEATURES_VERBOSE:
                    logger.debug(
                        f"[fetch] download ok in {time.time()-t1:.2f}s "
                        f"(shape={getattr(data,'shape',None)})"
                    )
                break
            except Exception as e:
                attempt += 1
                if FEATURES_VERBOSE:
                    logger.warning(f"[fetch] attempt {attempt} failed: {e!r}")
                if attempt < max_retries:
                    time.sleep(retry_sleep * attempt)
                else:
                    data = None

        # Retry the batch on transient errors
        attempt = 0
        data = None
        while attempt < max_retries:
            try:
                t1 = time.time()
                data = yf.download(
                    tickers=" ".join(chunk_syms),
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                if FEATURES_VERBOSE:
                    logger.debug(f"[fetch] download ok in {time.time()-t1:.2f}s (shape={getattr(data,'shape',None)})")
                break
            except Exception as e:
                attempt += 1
                if FEATURES_VERBOSE:
                    logger.warning(f"[fetch] attempt {attempt} failed: {e!r}")
                if attempt < max_retries:
                    time.sleep(retry_sleep * attempt)
                else:
                    data = None

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            if FEATURES_VERBOSE:
                logger.warning("[fetch] chunk failed: empty data after retries")

        ok_in_chunk = empty_in_chunk = fb_ok_in_chunk = rej_in_chunk = 0

        for orig in chunk_keys:
            yh = yh_map[orig]
            try:
                df_norm = _normalize_ohlcv(data, ticker_hint=(yh or "").strip()) if data is not None else None
                df_use = _usable_df(df_norm)
                if df_use is not None:
                    out[orig] = df_use
                    ok_in_chunk += 1
                    total_ok += 1
                    if FEATURES_VERBOSE and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[fetch] ok {orig}->{yh} rows={df_use.shape[0]} cols={list(df_use.columns)}")
                    continue
                else:
                    empty_in_chunk += 1
                    total_empty += 1
                    if FEATURES_VERBOSE:
                        logger.debug(f"[fetch] empty/too-few-rows {orig}->{yh} (trying fallback)")
                # force fallback
                raise ValueError("normalize-empty")
            except Exception:
                # single-ticker fallback
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
                    df_use = _usable_df(df1)
                    if df_use is not None:
                        out[orig] = df_use
                        fb_ok_in_chunk += 1
                        total_fb_ok += 1
                        if FEATURES_VERBOSE:
                            logger.debug(f"[fetch] fallback ok {orig} rows={df_use.shape[0]}")
                    else:
                        rejects.append((orig, "404/empty-or-short"))
                        rej_in_chunk += 1
                        total_rejects += 1
                        if FEATURES_VERBOSE:
                            logger.debug(f"[fetch] reject {orig}: empty/short after fallback")
                except Exception as e2:
                    rejects.append((orig, f"fetch-error:{e2.__class__.__name__}"))
                    rej_in_chunk += 1
                    total_rejects += 1
                    if FEATURES_VERBOSE:
                        logger.debug(f"[fetch] reject {orig}: {e2!r}")

                # 🔥 added: throttle after each single fallback request
                sleep_fb = float(os.getenv("YF_SLEEP_FALLBACK", "0.5"))
                if sleep_fb > 0:
                    if FEATURES_VERBOSE:
                        logger.debug(f"[fetch] sleeping {sleep_fb:.1f}s after fallback for {orig}")
                    time.sleep(sleep_fb)
        if FEATURES_VERBOSE:
            logger.info(
                f"[fetch] chunk done ok={ok_in_chunk} fb_ok={fb_ok_in_chunk} "
                f"empty={empty_in_chunk} rejects={rej_in_chunk}/{len(chunk_keys)}"
            )

    # persist rejects (append-safe, once per function call)
    if rejects:
        import csv
        os.makedirs("data", exist_ok=True)
        path = "data/universe_rejects.csv"
        header_needed = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["ticker", "reason", "ts"])
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            for tkr, r in rejects:
                w.writerow([tkr, r, ts])
        if FEATURES_VERBOSE:
            logger.info(f"[fetch] rejects wrote: {len(rejects)} -> {path}")

    if FEATURES_VERBOSE:
        logger.info(
            f"[fetch] finished in {time.time()-t0:.2f}s "
            f"ok={total_ok} fb_ok={total_fb_ok} empty={total_empty} rejects={total_rejects} / universe={len(keys)}"
        )
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

    if FEATURES_VERBOSE and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[ind] rows={len(px)} hasVol={'Volume' in cols} minDate={px.index.min()} maxDate={px.index.max()}")

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

    # Optional: local liquidity tier only when TIER_POLICY says THRESH.
    # When TIER_POLICY=TOPK_ADV, tiering is decided globally in rank_stage1 using top-K ADV.
    try:
        tier_policy = (os.getenv("TIER_POLICY", "THRESH") or "THRESH").upper()
        if tier_policy == "THRESH":
            hi = float(os.getenv("TIER_LARGE_USD", "50000000"))   # $50M default
            mid = float(os.getenv("TIER_MEDIUM_USD", "10000000")) # $10M default
            adv = feats["avg_dollar_vol_20d"] or 0.0
            feats["liq_tier"] = "LARGE" if adv >= hi else ("MID" if adv >= mid else "SMALL")
        else:
            feats["liq_tier"] = None
    except Exception:
        feats["liq_tier"] = None

    if FEATURES_VERBOSE and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[ind] snap px=%s SMA50=%s SMA200=%s RSI=%s MACD=%s ATR=%s dd=%s r60=%s v20=%s",
            _fmt(cur), _fmt(feats["SMA50"]), _fmt(feats["SMA200"]),
            feats["RSI14"], _fmt(feats["MACD_hist"], ".3f"), _fmt(feats["ATRpct"]),
            _fmt(feats["drawdown_pct"]), _fmt(feats["r60"]), _fmt(feats["vol_vs20"])
        )

    # Add EMA-derived features (uses Close column from df)
    try:
        add_ema_features(df, feats)
    except Exception as e:
        if FEATURES_VERBOSE:
            logger.debug(f"[ind] EMA calc failed: {e!r}")

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
    t0 = time.time()
    out: Dict[str, Dict] = {}
    tickers = [t for t, _ in universe]
    name_map = dict(universe)

    total = len(tickers)
    fetched = 0
    computed = 0
    skipped_df = 0
    skipped_compute = 0

    if FEATURES_VERBOSE:
        logger.info(f"[build] start universe={total} batch_size={batch_size} period={period}")

    # Fetch in batches; fetch_history does its own internal sub-chunking
    for i in range(0, total, batch_size):
        chunk = tickers[i:i + batch_size]
        if not chunk:
            continue

        t1 = time.time()
        hist = fetch_history(
            chunk,
            period=period,
            chunk_size=CHUNK_SIZE,
            max_retries=MAX_RETRIES,
            retry_sleep=RETRY_SLEEP,
        )
        fetched += len(hist)

        if FEATURES_VERBOSE:
            logger.info(f"[build] batch {i//batch_size+1}/{(total+batch_size-1)//batch_size} fetched={len(hist)}/{len(chunk)} in {time.time()-t1:.2f}s")

        for idx, t in enumerate(chunk, 1):
            df = hist.get(t)
            if df is None or df.empty:
                skipped_df += 1
                if FEATURES_VERBOSE and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[build] skip (no df) {t}")
                continue
            try:
                feats = compute_indicators(df)
                out[t] = {"company": name_map.get(t, t), "features": feats}
                computed += 1
                if FEATURES_VERBOSE and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[build] ok %s px=%s vs50=%s vs200=%s rsi=%s macd=%s dd=%s atr=%s",
                        t,
                        _fmt(feats.get("price")), _fmt(feats.get("vsSMA50")),
                        _fmt(feats.get("vsSMA200")), feats.get("RSI14"),
                        _fmt(feats.get("MACD_hist"), ".3f"),
                        _fmt(feats.get("drawdown_pct")), _fmt(feats.get("ATRpct"))
                    )
            except Exception as e:
                skipped_compute += 1
                if FEATURES_VERBOSE:
                    logger.warning(f"[build] compute failed {t}: {e!r}")

            # heartbeat
            if FEATURES_VERBOSE and FEATURES_LOG_EVERY and (idx % FEATURES_LOG_EVERY == 0):
                logger.info(f"[build] progress batch {i//batch_size+1}: {idx}/{len(chunk)} tickers processed")

    if FEATURES_VERBOSE:
        logger.info(
            "[build] done in %.2fs | fetched=%d computed=%d skipped_df=%d skipped_compute=%d total=%d",
            time.time()-t0, fetched, computed, skipped_df, skipped_compute, total
        )

    return out
