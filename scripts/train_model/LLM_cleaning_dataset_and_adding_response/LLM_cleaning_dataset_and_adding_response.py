# -*- coding: utf-8 -*-
"""
LLM_cleaning_dataset_and_adding_response_v2.py
----------------------------------------------
✅ Compatible with the new dataset containing:
   - Growth / volatility / range / extra features
   - Date + Ticker
✅ Computes 126-day forward reward score (0–1000)
✅ Adds basic rolling averages for smoothing
✅ Removes redundant/fundamental fields
✅ Drops any rows containing NaN/inf (per ticker)
✅ Saves to final parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype  # (unused now, but harmless)

# ==========================================================
# CONFIG
# ==========================================================
INPUT_PATH  = Path("../LLM_Training_data/full_market_2007_2025_new_features.parquet")
OUTPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_with_response_new_features.parquet")
TEMP_PATH   = Path("../LLM_Training_data/temp_checkpoint.parquet")

HOLD_DAYS = 126
SAVE_EVERY = 100

# ==========================================================
# LOAD DATA
# ==========================================================
print(f"[INFO] Loading enhanced dataset from {INPUT_PATH} ...")
df = pd.read_parquet(INPUT_PATH)
df = df.reset_index(drop=False)

if "Date" not in df.columns:
    raise KeyError("❌ No 'Date' column found in dataset!")

# Handle Date column
if np.issubdtype(df["Date"].dtype, np.number):
    print("[INFO] Converting numeric timestamps → datetime (ms).")
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", errors="coerce")
else:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
print(f"[INFO] Loaded {len(df):,} rows for {df['Ticker'].nunique()} tickers.")

# ==========================================================
# REWARD FUNCTION
# ==========================================================
def compute_rewards_for_ticker(df_t: pd.DataFrame) -> pd.DataFrame:
    closes = df_t["Close"].to_numpy()
    n = len(closes)
    scores = np.zeros(n, dtype=float)

    for i in range(n - HOLD_DAYS):
        window = closes[i + 1 : i + HOLD_DAYS]
        if len(window) == 0:
            continue

        price_today = closes[i]
        max_future = np.max(window)
        min_future = np.min(window)

        gain = (max_future - price_today) / price_today
        drawdown = (min_future - price_today) / price_today
        day_to_peak = np.argmax(window)

        if drawdown < -0.25 and gain < 0.25:
            score = 0.0
        else:
            score = min(gain * 1000.0, 1000.0)
            score *= (0.5 + 0.5 * (HOLD_DAYS - day_to_peak) / HOLD_DAYS)
            if drawdown < -0.3:
                score *= 0.8

        scores[i] = score

    df_t["score"] = scores
    return df_t

# ==========================================================
# APPLY PER TICKER + CHECKPOINTING
# ==========================================================
out, processed = [], 0

if TEMP_PATH.exists():
    print("[INFO] Resuming from checkpoint...")
    df_done = pd.read_parquet(TEMP_PATH)
    done_tickers = set(df_done["Ticker"].unique())
else:
    df_done, done_tickers = pd.DataFrame(), set()

for ticker, group in tqdm(df.groupby("Ticker"), desc="Scoring tickers"):
    if ticker in done_tickers or len(group) < HOLD_DAYS:
        continue

    result = compute_rewards_for_ticker(group.copy())

    # 🔹 keep only positive-score rows for this ticker
    result = result[result["score"] > 0]

    # 🔹 drop NaN/inf early for this ticker
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(how="any")

    if result.empty:
        continue

    out.append(result)
    processed += 1

    if processed % SAVE_EVERY == 0:
        temp_df = pd.concat(out + [df_done], ignore_index=True)
        temp_df.to_parquet(TEMP_PATH, index=False)
        print(f"[CHECKPOINT] Saved {len(temp_df):,} rows after {processed} tickers.")

df_final = pd.concat(out + [df_done], ignore_index=True) if out else df_done

print(f"[INFO] After scoring & per-ticker filtering: {len(df_final):,} rows")

# ==========================================================
# CLEANING AND BASIC SMOOTHING
# ==========================================================
# Drop unneeded columns
drop_cols = ["Adj Close", "Volume"]
df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns],
                         errors="ignore")

# Rolling averages for feature smoothing
if "Close" in df_final.columns:
    df_final["avg_close_past_3_days"] = (
        df_final.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

if "Volatility_30D" in df_final.columns:
    df_final["avg_volatility_30D"] = (
        df_final.groupby("Ticker")["Volatility_30D"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

# ==========================================================
# DROP FUNDAMENTAL COLUMNS (NOT PRESENT IN THIS VERSION)
# ==========================================================
drop_fundamentals = [
    "PE", "PEG", "PS", "PB", "DividendYield",
    "Beta", "MarketCap", "YoY_Growth",
]
df_final = df_final.drop(columns=[c for c in drop_fundamentals if c in df_final.columns],
                         errors="ignore")

# ==========================================================
# ADD CURRENT PRICE
# ==========================================================
df_final["current_price"] = df_final["Close"]
print(f"[INFO] current_price column added (copied from Close).")

# (No global NaN/inf cleanup – it was already done per ticker)

# ==========================================================
# SAVE FINAL DATASET
# ==========================================================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.to_parquet(OUTPUT_PATH, index=False)
if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"[SUCCESS] Saved cleaned dataset with {len(df_final):,} rows → {OUTPUT_PATH}")
print(f"[COLUMNS] {df_final.columns.tolist()}")
