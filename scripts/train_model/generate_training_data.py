# -*- coding: utf-8 -*-
"""
generate_training_data_v9_rich_features.py
------------------------------------------
✅ Based on v8 fast sequential
✅ Keeps all original volatility + growth formulas
✅ Adds richer alpha-style features:
   - Short-term returns (Ret_1D, Ret_5D, Ret_10D)
   - Range position (pos_52w, pos_30d)
   - Volume features (Volume_Z20, Volume_SMA20, Volume_Trend)
   - RSIs (RSI_14, RSI_2)
   - Date features (DayOfWeek, Month)
   - Cross-sectional z-scores & ranks per Date for key factors
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================
INPUT_CSV   = "../../data/universe_clean.csv"
OUTPUT_DIR  = Path("/LLM_Training_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "full_market_2007_2025.parquet"
TEMP_PATH   = OUTPUT_DIR / "temp_checkpoint.parquet"

START_DATE  = "2007-01-01"
END_DATE    = "2025-01-01"
MAX_TICKERS = 5000
BATCH_SIZE  = 60      # 60 tickers per batch
DOWNLOAD_RETRIES = 5  # retries per batch in safe_download


# ==========================================================
# UTILITIES
# ==========================================================
def collapse(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.loc[:, ~df.columns.duplicated()]
    for c in df.columns:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
    return df


def s(df, col):
    if col not in df.columns:
        raise KeyError(col)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val.squeeze(), errors="coerce")


def safe_div(num, denom, eps=1e-6):
    """
    Numerically safe division: returns NaN when denom is too small or NaN.
    Prevents inf and crazy blowups from near-zero denominators.
    """
    denom_safe = denom.copy()
    denom_safe = denom_safe.replace([np.inf, -np.inf], np.nan)
    return num / (denom_safe.where(denom_safe.abs() > eps, np.nan))


def compute_RSI(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Classic RSI calculation on close prices.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rs   = safe_div(gain, loss)
    rsi  = 100 - (100 / (1 + rs))
    return rsi


# ==========================================================
# CALCULATIONS (full v8 functionality + extra alpha features)
# ==========================================================
def compute_volatility_features(df):
    df = collapse(df)
    Close, High, Low = s(df, "Close"), s(df, "High"), s(df, "Low")

    # Drop obviously invalid price rows for this ticker
    valid_mask = (Close > 0) & (High > 0) & (Low > 0)
    df = df.loc[valid_mask].copy()
    Close = Close.loc[valid_mask]
    High  = High.loc[valid_mask]
    Low   = Low.loc[valid_mask]

    # Daily returns
    ret = Close.pct_change(fill_method=None)

    df["Volatility_30D"]  = ret.rolling(30,  min_periods=10).std()
    df["Volatility_252D"] = ret.rolling(252, min_periods=60).std()
    df["High_52W"]        = High.rolling(252, min_periods=60).max()
    df["Low_52W"]         = Low.rolling(252, min_periods=60).min()
    df["High_30D"]        = High.rolling(30,  min_periods=10).max()
    df["Low_30D"]         = Low.rolling(30,  min_periods=10).min()
    df["SMA20"]           = Close.rolling(20, min_periods=10).mean()

    # We keep Volume as-is (may be used later)
    return df


def compute_growth_formulas(df):
    df = compute_volatility_features(df)

    Close, High52, Low52, High30, Low30, Vol30, Vol252, SMA20 = [
        s(df, c) for c in [
            "Close","High_52W","Low_52W","High_30D","Low_30D",
            "Volatility_30D","Volatility_252D","SMA20"
        ]
    ]
    Volume = s(df, "Volume") if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    # --- Momentum measures ---
    m63_den  = Close.shift(63)
    m126_den = Close.shift(126)
    m252_den = Close.shift(252)

    df["Momentum_63D"]  = safe_div(Close, m63_den)
    df["Momentum_126D"] = safe_div(Close, m126_den)
    df["Momentum_252D"] = safe_div(Close, m252_den)
    df["Momentum_1Y"]   = df["Momentum_252D"]

    # --- Adaptive Momentum Trigger (AMT) ---
    df["AMT"] = np.where(df["Momentum_126D"] > 1.1, df["Momentum_252D"], 0.0)

    # --- Smart Momentum Composite (SMC) ---
    smc_part1 = safe_div(df["Momentum_252D"], Vol30)
    smc_part2 = safe_div(Close, High52)
    df["SMC"] = smc_part1 * smc_part2

    # --- Trend Strength Score (TSS) ---
    df["TSS"] = (df["Momentum_63D"] + df["Momentum_126D"] + df["Momentum_252D"]) / 3.0

    # --- Recovery Momentum Index (RMI) ---
    num   = safe_div(Close, Low30)
    denom = safe_div(High30, Low30)
    df["RMI"] = safe_div(num, denom + 1e-6)

    # --- ATR Breakout Signal (ABS) ---
    range_52 = High52 - Low52
    abs_ratio = safe_div(Close - Low52, range_52)
    df["ABS"] = abs_ratio * Vol30

    # --- Volatility-Adjusted Momentum (VAM) ---
    df["VAM"] = safe_div(df["Momentum_63D"], 1.0 + Vol30)

    # --- Rolling Sharpe-like Efficiency (RSE) ---
    ret = Close.pct_change(fill_method=None)
    roll_mean = ret.rolling(63, min_periods=20).mean()
    roll_std  = ret.rolling(63, min_periods=20).std()
    df["RSE"] = safe_div(roll_mean, roll_std)

    # --- Compression Break Potential (CBP) ---
    df["CBP"] = safe_div(Vol30, Vol252)

    # --- SMA Slope 3M ---
    sma_past = SMA20.shift(63)
    df["SMA_Slope_3M"] = safe_div(SMA20 - sma_past, sma_past)

    # ======================================================
    # EXTRA FEATURES (short-term, volume, range, RSI, etc.)
    # ======================================================

    # Short-term returns
    df["Ret_1D"]  = ret
    df["Ret_5D"]  = Close.pct_change(5)
    df["Ret_10D"] = Close.pct_change(10)

    # Range position features
    range_30 = High30 - Low30
    df["pos_52w"] = safe_div(Close - Low52, range_52)
    df["pos_30d"] = safe_div(Close - Low30, range_30)

    # Volume-based features
    vol_roll_mean_20 = Volume.rolling(20, min_periods=5).mean()
    vol_roll_std_20  = Volume.rolling(20, min_periods=5).std()
    df["Volume_SMA20"]  = vol_roll_mean_20
    df["Volume_Z20"]    = safe_div(Volume - vol_roll_mean_20, vol_roll_std_20 + 1e-6)
    df["Volume_Trend"]  = safe_div(Volume, vol_roll_mean_20)

    # RSI signals
    df["RSI_14"] = compute_RSI(Close, window=14)
    df["RSI_2"]  = compute_RSI(Close, window=2)

    return collapse(df)


# ==========================================================
# DOWNLOADER
# ==========================================================
def safe_download(batch, start, end, retries=DOWNLOAD_RETRIES):
    """
    Robust batch downloader using yfinance with:
    - group_by="ticker" → columns (ticker, field)
    - multiple retries with cooldown
    """
    for attempt in range(retries):
        try:
            return yf.download(
                batch,
                start=start,
                end=end,
                group_by="ticker",
                progress=False,
                auto_adjust=True,   # adjusted prices
            )
        except Exception as e:
            print(f"[WARN] Download failed (attempt {attempt+1}/{retries}): {e}")
            sleep_time = 10 + random.uniform(5, 15)
            print(f"[WAIT] Cooling down for {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    print("[ERROR] All retries failed for this batch.")
    return pd.DataFrame()


# ==========================================================
# CROSS-SECTIONAL FEATURES
# ==========================================================
def add_cross_sectional_features(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional z-scores and percentile ranks per Date
    for key factor columns.
    """
    full_df = full_df.copy()
    if "Date" not in full_df.columns:
        return full_df

    # Ensure Date is datetime
    full_df["Date"] = pd.to_datetime(full_df["Date"])

    cs_cols = [
        "Momentum_63D", "Momentum_126D", "Momentum_252D", "Momentum_1Y",
        "Volatility_30D", "Volatility_252D",
        "RSE", "CBP", "SMC", "TSS", "VAM", "ABS", "AMT",
        "pos_52w", "pos_30d",
        "Ret_1D", "Ret_5D", "Ret_10D",
        "Volume_Z20"
    ]

    grp = full_df.groupby("Date")

    for col in cs_cols:
        if col not in full_df.columns:
            continue
        col_series = full_df[col]
        mean = grp[col].transform("mean")
        std  = grp[col].transform("std")

        full_df[col + "_cs_z"] = (col_series - mean) / (std + 1e-6)
        full_df[col + "_cs_rank"] = grp[col].transform(lambda x: x.rank(pct=True))

    return full_df


# ==========================================================
# MAIN EXECUTION (sequential, 60 per batch, full v9 features)
# ==========================================================
def main():
    dfu = pd.read_csv(INPUT_CSV)
    tickers = dfu["ticker"].dropna().unique().tolist()[:MAX_TICKERS]
    print(f"[INFO] Starting data generation for {len(tickers)} tickers (batch={BATCH_SIZE})")
    print(f"[INFO] Interval: {START_DATE} → {END_DATE}\n")

    combined_records = []
    batch_index = 0

    for i in range(0, len(tickers), BATCH_SIZE):
        batch_index += 1
        batch = tickers[i : i + BATCH_SIZE]
        print(f"[BATCH {batch_index}] downloading {len(batch)} tickers: {batch[0]} .. {batch[-1]}")

        try:
            data = safe_download(batch, start=START_DATE, end=END_DATE)
        except Exception as e:
            print(f"[ERROR] Batch download failed unexpectedly: {e}")
            continue

        if data.empty:
            print(f"[WARN] Batch {batch[0]}..{batch[-1]} returned empty data.")
            time.sleep(1)
            continue

        # Multi-ticker: expect MultiIndex (ticker, field)
        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                try:
                    if (t, "Close") not in data.columns:
                        continue

                    df_t = data[t].dropna().copy()
                    if df_t.empty:
                        continue

                    df_t = compute_growth_formulas(df_t)
                    df_t["Ticker"] = t
                    df_t["Date"] = df_t.index

                    keep = [
                        "Date","Ticker","Close",
                        "Volatility_30D","Volatility_252D",
                        "High_52W","Low_52W","High_30D","Low_30D",
                        "Momentum_63D","Momentum_126D","Momentum_252D","Momentum_1Y",
                        "AMT","SMC","TSS","RMI",
                        "ABS","VAM","RSE","CBP","SMA_Slope_3M",
                        "Ret_1D","Ret_5D","Ret_10D",
                        "pos_52w","pos_30d",
                        "Volume","Volume_SMA20","Volume_Z20","Volume_Trend",
                        "RSI_14","RSI_2",
                    ]
                    df_t = df_t[[c for c in keep if c in df_t.columns]]
                    combined_records.append(df_t)
                    print(f"[OK] Processed {t} ({len(df_t)} rows kept).")

                except Exception as e:
                    print(f"[ERROR] {t}: {e}")
                    continue

        else:
            # Single-ticker edge case (batch size 1)
            t = batch[0]
            try:
                if "Close" not in data.columns:
                    print(f"[WARN] No Close column for {t}")
                else:
                    df_t = data.dropna().copy()
                    if not df_t.empty:
                        df_t = compute_growth_formulas(df_t)
                        df_t["Ticker"] = t
                        df_t["Date"] = df_t.index

                        keep = [
                            "Date","Ticker","Close",
                            "Volatility_30D","Volatility_252D",
                            "High_52W","Low_52W","High_30D","Low_30D",
                            "Momentum_63D","Momentum_126D","Momentum_252D","Momentum_1Y",
                            "AMT","SMC","TSS","RMI",
                            "ABS","VAM","RSE","CBP","SMA_Slope_3M",
                            "Ret_1D","Ret_5D","Ret_10D",
                            "pos_52w","pos_30d",
                            "Volume","Volume_SMA20","Volume_Z20","Volume_Trend",
                            "RSI_14","RSI_2",
                        ]
                        df_t = df_t[[c for c in keep if c in df_t.columns]]
                        combined_records.append(df_t)
                        print(f"[OK] Processed {t}")
            except Exception as e:
                print(f"[ERROR] {t}: {e}")

        # Checkpoint every 5 batches
        if batch_index % 5 == 0 and combined_records:
            temp_df = pd.concat(combined_records, axis=0, ignore_index=True)
            temp_df.to_parquet(TEMP_PATH, index=False)
            print(f"[CHECKPOINT] Saved {len(temp_df):,} rows to {TEMP_PATH}")

        # Be nice to Yahoo
        time.sleep(1)

    if not combined_records:
        print("[ERROR] No data collected. Exiting.")
        return

    print(f"[INFO] Concatenating {len(combined_records)} per-ticker DataFrames...")
    full_df = pd.concat(combined_records, axis=0, ignore_index=True)

    # ------------------------------------------------------
    # Add date-based features
    # ------------------------------------------------------
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    full_df["DayOfWeek"] = full_df["Date"].dt.weekday
    full_df["Month"]     = full_df["Date"].dt.month

    # ------------------------------------------------------
    # Add cross-sectional features (per Date)
    # ------------------------------------------------------
    print("[INFO] Adding cross-sectional z-scores and ranks...")
    full_df = add_cross_sectional_features(full_df)

    print(f"[INFO] Saving full parquet to {OUTPUT_PATH} ...")
    full_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Saved {len(full_df):,} rows → {OUTPUT_PATH}")

    if TEMP_PATH.exists():
        TEMP_PATH.unlink()
        print("[INFO] Temporary checkpoint removed.")


# ==========================================================
if __name__ == "__main__":
    main()
