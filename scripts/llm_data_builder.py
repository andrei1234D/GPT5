# scripts/llm_data_builder.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ==========================================================
# Helpers
# ==========================================================
def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def _to_float_or_none(v):
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def compute_RSI(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = safe_div(avg_gain.values, (avg_loss.values + 1e-12))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes:
      - Volatility_30D / Volatility_252D (annualized)
      - High_30D / Low_30D
      - High_52W / Low_52W
      - SMA20
    """
    df = df.copy()
    Close = _as_float_series(df, "Close")
    High = _as_float_series(df, "High")
    Low = _as_float_series(df, "Low")

    ret = Close.pct_change(fill_method=None)

    df["Volatility_30D"] = ret.rolling(30, min_periods=20).std() * np.sqrt(252.0)
    df["Volatility_252D"] = ret.rolling(252, min_periods=200).std() * np.sqrt(252.0)

    df["High_30D"] = High.rolling(30, min_periods=20).max()
    df["Low_30D"] = Low.rolling(30, min_periods=20).min()
    df["High_52W"] = High.rolling(252, min_periods=200).max()
    df["Low_52W"] = Low.rolling(252, min_periods=200).min()

    df["SMA20"] = Close.rolling(20, min_periods=10).mean()
    return df


def compute_growth_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your provided feature engineering, adapted to a single-ticker time series DF with
    columns: Open, High, Low, Close, Volume.
    """
    df = compute_volatility_features(df)

    Close = _as_float_series(df, "Close")
    High52 = _as_float_series(df, "High_52W")
    Low52 = _as_float_series(df, "Low_52W")
    High30 = _as_float_series(df, "High_30D")
    Low30 = _as_float_series(df, "Low_30D")
    Vol30 = _as_float_series(df, "Volatility_30D")
    Vol252 = _as_float_series(df, "Volatility_252D")
    SMA20 = _as_float_series(df, "SMA20")

    Volume = _as_float_series(df, "Volume") if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    # Momentum
    df["Momentum_63D"] = safe_div(Close.values, Close.shift(63).values)
    df["Momentum_126D"] = safe_div(Close.values, Close.shift(126).values)
    df["Momentum_252D"] = safe_div(Close.values, Close.shift(252).values)

    # AMT
    df["AMT"] = np.where(df["Momentum_126D"] > 1.1, df["Momentum_252D"], 0.0)

    # SMC
    smc_part1 = safe_div(df["Momentum_252D"].values, Vol30.values)
    smc_part2 = safe_div(Close.values, High52.values)
    df["SMC"] = smc_part1 * smc_part2

    # TSS
    df["TSS"] = (df["Momentum_63D"] + df["Momentum_126D"] + df["Momentum_252D"]) / 3.0

    # RMI
    num = safe_div(Close.values, Low30.values)
    denom = safe_div(High30.values, Low30.values)
    df["RMI"] = safe_div(num, denom + 1e-6)

    # ABS
    range_52 = High52 - Low52
    abs_ratio = safe_div((Close - Low52).values, range_52.values)
    df["ABS"] = abs_ratio * Vol30.values

    # VAM
    df["VAM"] = safe_div(df["Momentum_63D"].values, 1.0 + Vol30.values)

    # RSE
    ret = Close.pct_change(fill_method=None)
    roll_mean = ret.rolling(63, min_periods=20).mean()
    roll_std = ret.rolling(63, min_periods=20).std()
    df["RSE"] = safe_div(roll_mean.values, roll_std.values)

    # CBP
    df["CBP"] = safe_div(Vol30.values, Vol252.values)

    # SMA slope 3M
    sma_past = SMA20.shift(63)
    df["SMA_Slope_3M"] = safe_div((SMA20 - sma_past).values, sma_past.values)

    # Returns
    df["Ret_5D"] = Close.pct_change(5)
    df["Ret_10D"] = Close.pct_change(10)

    # Positions
    range_30 = High30 - Low30
    df["pos_52w"] = safe_div((Close - Low52).values, range_52.values)
    df["pos_30d"] = safe_div((Close - Low30).values, range_30.values)

    # Volume features
    vol_roll_mean_20 = Volume.rolling(20, min_periods=5).mean()
    df["Volume_SMA20"] = vol_roll_mean_20
    df["Volume_Trend"] = safe_div(Volume.values, (vol_roll_mean_20.values + 1e-6))

    # RSI
    df["RSI_14"] = compute_RSI(Close, window=14)

    # Builder extras
    df["avg_close_past_3_days"] = Close.rolling(3, min_periods=1).mean()
    df["avg_volatility_30D"] = _as_float_series(df, "Volatility_30D").rolling(30, min_periods=20).mean()
    df["current_price"] = Close

    return df


def _cs_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - mu) / (sd + 1e-12)


def _cs_rank01(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    # percentile rank in [0,1]
    return x.rank(pct=True, method="average")


def _attach_cross_sectional(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Adds *_cs_z and *_cs_rank columns to match your feature list.
    """
    df = df_latest.copy()

    cs_sources = [
        "Momentum_63D", "Momentum_126D", "Momentum_252D",
        "Volatility_30D", "Volatility_252D",
        "RSE", "CBP", "SMC", "TSS", "VAM", "ABS", "AMT",
        "pos_52w", "pos_30d",
    ]

    for col in cs_sources:
        df[f"{col}_cs_z"] = _cs_z(df[col])
        df[f"{col}_cs_rank"] = _cs_rank01(df[col])

    return df


def _pick_common_asof_date(data_by_ticker: Dict[str, pd.DataFrame]) -> pd.Timestamp:
    """
    To keep cross-sectional features meaningful, align all tickers to a common date.
    We choose the minimum of each ticker's last available date (common-asof).
    """
    last_dates = []
    for t, d in data_by_ticker.items():
        if d is None or d.empty:
            continue
        idx = d.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            last_dates.append(idx.max())
    if not last_dates:
        raise RuntimeError("No valid ticker histories to compute a common as-of date.")
    return min(last_dates)


def build_llm_today_data(
    stage2_path: str = "data/stage2_merged.csv",
    out_path: str = "data/LLM_today_data.jsonl",
    top_n: int =10,
    history_years: int = 2,
) -> List[str]:
    """
    Builds LLM_today_data.jsonl but now it contains YOUR tabular features (not prompts).
    Records are aligned to a common as-of date across tickers so cs_* features are comparable.
    """
    stage2_path = Path(stage2_path)
    out_path = Path(out_path)

    if not stage2_path.exists():
        raise FileNotFoundError(f"{stage2_path} not found")

    df_stage2 = pd.read_csv(stage2_path)

    if "merged_final_score" in df_stage2.columns:
        score_col = "merged_final_score"
    elif "merged_score" in df_stage2.columns:
        score_col = "merged_score"
    else:
        raise KeyError("Neither 'merged_final_score' nor 'merged_score' column found in stage2_merged.csv")

    df_stage2 = df_stage2.sort_values(score_col, ascending=False)

    oversample_factor = int(os.getenv("LLM_OVERSAMPLE_FACTOR", "4"))
    candidate_count = max(top_n * oversample_factor, top_n + 10)
    tickers = df_stage2["ticker"].dropna().astype(str).drop_duplicates().head(candidate_count).tolist()
    print(f"[LLM_DATA] Selected {len(tickers)} candidate tickers (oversample_factor={oversample_factor}).")

    chunk_size = int(os.getenv("LLM_YF_CHUNK_SIZE", "50"))
    batch_sleep = float(os.getenv("LLM_YF_BATCH_SLEEP", "0.5"))

    # Download histories
    data_by_ticker: Dict[str, pd.DataFrame] = {}
    skip_reasons: Dict[str, str] = {}

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]

        # Use yfinance multi-download if possible
        try:
            # group_by="ticker" gives nested columns; we handle both cases below
            raw = yf.download(batch, period=f"{history_years}y", interval="1d", auto_adjust=False, progress=False, group_by="ticker")
        except Exception as e:
            raw = None
            for t in batch:
                skip_reasons[t] = f"yfinance batch download failed: {repr(e)}"

        for t in batch:
            if raw is None:
                continue

            df_t = None
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.get_level_values(0):
                        df_t = raw[t].copy()
                else:
                    # If only one ticker, yfinance may return a single-level DF
                    df_t = raw.copy()

                if df_t is None or df_t.empty:
                    skip_reasons[t] = "empty history"
                    continue

                required = {"Open", "High", "Low", "Close"}
                if not required.issubset(set(map(str, df_t.columns))):
                    skip_reasons[t] = f"missing OHLC columns: {list(df_t.columns)}"
                    continue

                df_t = df_t.dropna(subset=["Open", "High", "Low", "Close"])
                if df_t.empty:
                    skip_reasons[t] = "only NaNs after dropping OHLC"
                    continue

                # Standardize index as DatetimeIndex
                if not isinstance(df_t.index, pd.DatetimeIndex):
                    df_t.index = pd.to_datetime(df_t.index, errors="coerce")

                data_by_ticker[t] = df_t
            except Exception as e:
                skip_reasons[t] = f"failed to normalize history: {repr(e)}"

        time.sleep(batch_sleep)

    if not data_by_ticker:
        raise RuntimeError("No ticker histories produced – all downloads failed?")

    common_asof = _pick_common_asof_date(data_by_ticker)
    print(f"[LLM_DATA] Common as-of date: {common_asof.date()} (min last-date across tickers)")

    # Compute latest-row features per ticker as-of common date
    rows = []
    written = []

    for t, hist in data_by_ticker.items():
        try:
            # Use data up to common_asof
            hist2 = hist.loc[hist.index <= common_asof].copy()
            if hist2.empty:
                skip_reasons[t] = "no data up to common as-of date"
                continue

            feats = compute_growth_formulas(hist2)

            latest_dt = feats.index.max()
            latest = feats.loc[latest_dt].copy()

            # Ensure "Month"
            month = int(pd.to_datetime(latest_dt).month)

            row = {"Ticker": t, "Date": int(pd.to_datetime(latest_dt).value // 10**6), "Month": month}

            # Pull the exact base features (non-cs) from latest
            base_cols = [
                "Volatility_30D", "Volatility_252D", "High_52W", "Low_52W", "High_30D", "Low_30D",
                "Momentum_63D", "Momentum_126D", "Momentum_252D",
                "AMT", "SMC", "TSS", "RMI", "ABS", "VAM", "RSE", "CBP",
                "SMA_Slope_3M", "Ret_5D", "Ret_10D", "pos_52w", "pos_30d",
                "Volume_SMA20", "Volume_Trend", "RSI_14",
                "avg_close_past_3_days", "avg_volatility_30D", "current_price",
            ]

            for c in base_cols:
                row[c] = _to_float_or_none(latest.get(c))

            rows.append(row)
            written.append(t)
        except Exception as e:
            skip_reasons[t] = f"feature computation failed: {repr(e)}"

        if len(rows) >= top_n:
            break

    if not rows:
        raise RuntimeError("No feature rows built – feature computation failed for all tickers?")

    df_latest = pd.DataFrame(rows).set_index("Ticker", drop=False)

    # Add cross-sectional features
    df_latest = _attach_cross_sectional(df_latest)

    # Final: write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df_latest.iterrows():
            rec = {k: (None if (pd.isna(v) or v is None) else (float(v) if isinstance(v, (int, float, np.floating)) else v))
                   for k, v in r.to_dict().items()}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    print(f"[LLM_DATA] Wrote {len(df_latest)} records → {out_path}")

    # Diagnostics
    logs_path = Path("logs")
    logs_path.mkdir(parents=True, exist_ok=True)
    diag_path = logs_path / "llm_stage2_reconciliation.json"
    diag = {
        "stage2_path": str(stage2_path),
        "top_n": top_n,
        "history_years": history_years,
        "oversample_factor": oversample_factor,
        "candidate_count": candidate_count,
        "requested_candidates": tickers,
        "produced_count": int(len(df_latest)),
        "produced_tickers": written,
        "common_asof_date": str(common_asof),
        "skipped_reasons": skip_reasons,
    }
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[LLM_DATA] Wrote diagnostics → {diag_path}")

    return written


if __name__ == "__main__":
    build_llm_today_data()
