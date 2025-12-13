# -*- coding: utf-8 -*-
"""
LLM_cleaning_dataset_and_adding_response.py
-------------------------------------------
✅ Handles 'Date' correctly even if stored as index or numeric timestamp
✅ Computes SELL/HOLD labels
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
OUTPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_SELL_with_response.parquet")
TEMP_PATH = Path("../LLM_Training_data/temp_checkpoint.parquet")
HOLD_DAYS = 126          # ~6 months lookahead
SAVE_EVERY = 100

# Trend-aware sell hyperparameters
EARLY_DAYS = 30          # window for early crash detection

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
# SELL LABEL FUNCTION (no score saved)
# ==========================================================
def compute_rewards_for_ticker(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    For each day within a ticker:
      - Look ahead HOLD_DAYS (~6 months).
      - Decide SELL (1) vs HOLD (0) based on MarketTrend-adjusted rules:

        MarketTrend == "Bullish":
          - EPS_PEAK = 0.03   (top 3% of future range)
          - CRASH_DROP = -0.23  (<= -23% in next EARLY_DAYS)
        MarketTrend == "Bearish":
          - EPS_PEAK = 0.07   (top 7% of future range)
          - CRASH_DROP = -0.18  (<= -18% in next EARLY_DAYS)
        MarketTrend == "Neutral" or other:
          - EPS_PEAK = 0.05
          - CRASH_DROP = -0.23

        SELL if:
          (upside_left_ratio <= EPS_PEAK)
          OR
          (early_drop <= CRASH_DROP)
    """
    closes = df_t["Close"].to_numpy()
    n = len(closes)

    labels = np.zeros(n, dtype=int)

    # Trend per row (default to "Neutral" if missing)
    trends = df_t.get("MarketTrend", pd.Series(["Neutral"] * n)).astype(str).to_numpy()

    for i in range(n - HOLD_DAYS):
        window = closes[i + 1 : i + HOLD_DAYS]
        if len(window) == 0:
            continue

        price_today = closes[i]
        if price_today <= 0 or not np.isfinite(price_today):
            continue

        trend = trends[i] if i < len(trends) else "Neutral"
        t = trend.lower()

        # Set trend-dependent parameters
        if t == "bullish":
            eps_peak = 0.03     # stricter: only very near top
            crash_drop = -0.23  # large crash matters
        elif t == "bearish":
            eps_peak = 0.07     # looser: take rallies more aggressively
            crash_drop = -0.18  # more sensitive to drawdowns
        else:  # "neutral" or any other
            eps_peak = 0.05
            crash_drop = -0.23

        max_future = float(np.max(window))
        min_future = float(np.min(window))

        # How much upside exists vs best future top?
        upside_best = (max_future - price_today) / price_today if price_today > 0 else 0.0

        if max_future > 0:
            upside_left_ratio = (max_future - price_today) / max_future
        else:
            upside_left_ratio = 1.0  # degenerate, treat as far from peak

        # Early crash detection window
        early_window = window[:EARLY_DAYS]
        if len(early_window) > 0:
            early_min = float(np.min(early_window))
            early_drop = (early_min - price_today) / price_today
        else:
            early_drop = 0.0

        # Conditions
        cond_peak = (upside_best > 0) and (upside_left_ratio <= eps_peak)
        cond_crash = early_drop <= crash_drop

        sell = cond_peak or cond_crash
        labels[i] = 1 if sell else 0

    df_t["SellLabel"] = labels
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
# ❌ We NO LONGER filter on score > 0 (no score column, we keep SELL and HOLD).

# ✅ Rename in final dataset (not in source)
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
df_final["avg_open_past_3_days"] = df_final.groupby("Ticker")["Prev_Open"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df_final["avg_close_past_3_days"] = df_final.groupby("Ticker")["Prev_Close"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

df_final["avg_low_30"] = df_final.groupby("Ticker")["avg_low_raw"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)
df_final["avg_high_30"] = df_final.groupby("Ticker")["avg_high_raw"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

# ✅ Volatility 30 zile
df_final["volatility_30"] = (df_final["avg_high_30"] - df_final["avg_low_30"]) / df_final["avg_low_30"]

# Drop unwanted OHLC columns (original names, if still present)
drop_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns], errors="ignore")

# Rename raw Low/High defensively (if still present in original df)
rename_map = {}
if "Low" in df.columns:
    rename_map["Low"] = "avg_low_raw"
if "High" in df.columns:
    rename_map["High"] = "avg_high_raw"
df_final = df_final.rename(columns=rename_map)

# ==========================================================
# ADD ROLLING AVERAGES (duplicated but kept to match your structure)
# ==========================================================
df_final["avg_open_past_3_days"] = df_final.groupby("Ticker")["Prev_Open"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df_final["avg_close_past_3_days"] = df_final.groupby("Ticker")["Prev_Close"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

df_final["avg_low_30"] = df_final.groupby("Ticker")["avg_low_raw"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)
df_final["avg_high_30"] = df_final.groupby("Ticker")["avg_high_raw"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

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
# SAVE
# ==========================================================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.to_parquet(OUTPUT_PATH, index=False)

if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"[SUCCESS] Saved cleaned dataset with {len(df_final):,} rows → {OUTPUT_PATH}")
print(f"[COLUMNS] {df_final.columns.tolist()}")
