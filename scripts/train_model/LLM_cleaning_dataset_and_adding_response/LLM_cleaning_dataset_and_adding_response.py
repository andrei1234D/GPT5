# -*- coding: utf-8 -*-
"""
LLM_cleaning_dataset_and_adding_response.py
-------------------------------------------
✅ Handles 'Date' correctly even if stored as index or numeric timestamp
✅ Computes reward scores (0–1000)
✅ Computes:
    - 3-day averages for Open/Close
    - 30-day averages for Low/High
    - 30-day volatility (range-based)
✅ Removes redundant columns
✅ Saves progress incrementally (checkpointing)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================
INPUT_PATH = Path("../LLM_Training_data/full_market_2007_2025.parquet")
OUTPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_with_response.parquet")
TEMP_PATH = Path("../LLM_Training_data/temp_checkpoint.parquet")
HOLD_DAYS = 126
SAVE_EVERY = 100

# ==========================================================
# LOAD DATA
# ==========================================================
print(f"[INFO] Loading full market dataset from {INPUT_PATH} ...")
df = pd.read_parquet(INPUT_PATH)
df = df.reset_index(drop=False)

if "Date" not in df.columns:
    raise KeyError("❌ No 'Date' column found in dataset!")

if np.issubdtype(df["Date"].dtype, np.number):
    print("[INFO] Converting numeric timestamps → datetime (ms).")
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", errors="coerce")
else:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
print(f"[INFO] Loaded {len(df):,} rows for {df['Ticker'].nunique()} tickers.")

# ==========================================================
# ADD PAST-DAY FEATURES
# ==========================================================
for col in ["Open", "High", "Low", "Close"]:
    if col in df.columns:
        df[f"Prev_{col}"] = df.groupby("Ticker")[col].shift(1)

# ==========================================================
# REWARD FUNCTION
# ==========================================================
def compute_rewards_for_ticker(df_t):
    closes = df_t["Close"].to_numpy()
    n = len(closes)
    scores = np.zeros(n)
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
            score = 0
        else:
            score = min(gain * 1000, 1000)
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
    out.append(result)
    processed += 1
    if processed % SAVE_EVERY == 0:
        temp_df = pd.concat(out + [df_done], ignore_index=True)
        temp_df.to_parquet(TEMP_PATH, index=False)
        print(f"[CHECKPOINT] Saved {len(temp_df):,} rows after {processed} tickers.")

df_final = pd.concat(out + [df_done], ignore_index=True) if out else df_done

# ==========================================================
# CLEAN & ADD ROLLING AVERAGES
# ==========================================================
df_final = df_final[df_final["score"] > 0]

# ✅ Renumește în datasetul final (nu în cel inițial)
rename_map = {
    "Low": "avg_low_raw",
    "High": "avg_high_raw",
    "Open": "open_raw",
    "Close": "close_raw"
}
df_final = df_final.rename(columns={k: v for k, v in rename_map.items() if k in df_final.columns})

# ✅ Drop redundant columns
drop_cols = ["Adj Close", "Volume"]
df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns], errors="ignore")

# ✅ Rolling averages
df_final["avg_open_past_3_days"] = df_final.groupby("Ticker")["Prev_Open"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df_final["avg_close_past_3_days"] = df_final.groupby("Ticker")["Prev_Close"].transform(lambda x: x.rolling(3, min_periods=1).mean())

df_final["avg_low_30"] = df_final.groupby("Ticker")["avg_low_raw"].transform(lambda x: x.rolling(30, min_periods=1).mean())
df_final["avg_high_30"] = df_final.groupby("Ticker")["avg_high_raw"].transform(lambda x: x.rolling(30, min_periods=1).mean())

# ✅ Volatility 30 zile
df_final["volatility_30"] = (df_final["avg_high_30"] - df_final["avg_low_30"]) / df_final["avg_low_30"]

df_final = df_final[df_final["score"] > 0]

# Drop unwanted columns
drop_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns], errors="ignore")

# Rename raw Low/High
rename_map = {}
if "Low" in df.columns:
    rename_map["Low"] = "avg_low_raw"
if "High" in df.columns:
    rename_map["High"] = "avg_high_raw"
df_final = df_final.rename(columns=rename_map)

# ==========================================================
# ADD ROLLING AVERAGES
# ==========================================================
# 3-day rolling for Open/Close
df_final["avg_open_past_3_days"] = df_final.groupby("Ticker")["Prev_Open"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df_final["avg_close_past_3_days"] = df_final.groupby("Ticker")["Prev_Close"].transform(lambda x: x.rolling(3, min_periods=1).mean())

# 30-day rolling for Low/High
df_final["avg_low_30"] = df_final.groupby("Ticker")["avg_low_raw"].transform(lambda x: x.rolling(30, min_periods=1).mean())
df_final["avg_high_30"] = df_final.groupby("Ticker")["avg_high_raw"].transform(lambda x: x.rolling(30, min_periods=1).mean())

# 30-day volatility (percentage range)
df_final["volatility_30"] = (df_final["avg_high_30"] - df_final["avg_low_30"]) / df_final["avg_low_30"]

# ==========================================================
# REMOVE FUNDAMENTAL COLUMNS
# ==========================================================
drop_fundamentals = [
    "PE", "PEG", "PS", "PB", "DividendYield", 
    "Beta", "MarketCap", "YoY_Growth"
]

df_final = df_final.drop(columns=[c for c in drop_fundamentals if c in df_final.columns], errors="ignore")

print(f"[INFO] Removed fundamental columns: {drop_fundamentals}")
print(f"[INFO] Remaining columns: {df_final.columns.tolist()}")

# ==========================================================
# ADD CURRENT PRICE COLUMN
# ==========================================================
if "close_raw" in df_final.columns:
    df_final["current_price"] = df_final["close_raw"]
else:
    print("[WARNING] 'close_raw' column not found — could not create 'current_price'.")


# ==========================================================
# FINAL SANITY FILTER (very small but very effective)
# ==========================================================
price_cols = ["open_raw", "avg_low_raw", "avg_high_raw", "close_raw"]
price_cols = [c for c in price_cols if c in df_final.columns]

if price_cols:
    before = len(df_final)
    df_final = df_final[
        (df_final[price_cols].gt(0.01).all(axis=1)) &
        (df_final[price_cols].lt(10000).all(axis=1))
    ]
    after = len(df_final)
    print(f"[INFO] Final sanity filter kept {after:,} / {before:,} rows.")



# ==========================================================
# SAVE
# ==========================================================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.to_parquet(OUTPUT_PATH, index=False)

if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"[SUCCESS] Saved cleaned dataset with {len(df_final):,} rows → {OUTPUT_PATH}")
print(f"[COLUMNS] {df_final.columns.tolist()}")
