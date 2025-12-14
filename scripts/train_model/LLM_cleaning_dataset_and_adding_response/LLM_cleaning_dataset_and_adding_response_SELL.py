# -*- coding: utf-8 -*-
"""
LLM_cleaning_dataset_and_adding_response.py
-------------------------------------------
✅ Handles 'Date' correctly even if stored as index or numeric timestamp
✅ Downloads S&P 500 (^GSPC) via yfinance and computes MarketTrend per row date
✅ Computes SELL/HOLD labels (trend-aware)
✅ Computes:
    - 3-day averages for Open/Close (based on previous day values)
    - 30-day averages for Low/High
    - 30-day volatility (range-based)
✅ Removes redundant columns
✅ Saves progress incrementally (checkpointing)

Notes:
- MarketTrend uses past-only SMA cross on ^GSPC (no future leakage).
- Rows that cannot be labeled (tail HOLD_DAYS) will remain 0 by default (kept as in your original logic).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yfinance as yf

# ==========================================================
# CONFIG
# ==========================================================
INPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_with_response_new_features.parquet")
OUTPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_SELL_with_response.parquet")
TEMP_PATH = Path("../LLM_Training_data/temp_checkpoint.parquet")

HOLD_DAYS = 126          # ~6 months lookahead
SAVE_EVERY = 100

# Trend-aware sell hyperparameters
EARLY_DAYS = 30          # window for early crash detection

# Market trend config (S&P 500)
SNP_TICKER = "^GSPC"
SMA_FAST = 50
SMA_SLOW = 200

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
# DOWNLOAD S&P AND COMPUTE MarketTrend PER DATE (yfinance)
# ==========================================================
print(f"[INFO] Downloading S&P proxy {SNP_TICKER} via yfinance to compute MarketTrend...")

dmin = df["Date"].min()
dmax = df["Date"].max()
if pd.isna(dmin) or pd.isna(dmax):
    raise ValueError("❌ Date column has no valid timestamps after conversion.")

# extra history for SMA_SLOW warmup; extra tail padding for alignment
start = (dmin - pd.Timedelta(days=365)).date()
end = (dmax + pd.Timedelta(days=5)).date()

snp = yf.download(
    SNP_TICKER,
    start=str(start),
    end=str(end),
    auto_adjust=False,
    progress=False,
    threads=True,
    group_by="column",
)

if snp is None or snp.empty:
    raise RuntimeError(f"❌ yfinance returned no data for {SNP_TICKER} from {start} to {end}.")

snp = snp.reset_index().rename(columns={"Date": "SNP_Date"})
snp["SNP_Date"] = pd.to_datetime(snp["SNP_Date"], errors="coerce")
snp = snp.sort_values("SNP_Date").dropna(subset=["SNP_Date"]).reset_index(drop=True)

snp_price_col = "Adj Close" if "Adj Close" in snp.columns else "Close"
if snp_price_col not in snp.columns:
    raise KeyError(f"❌ Expected Close/Adj Close in yfinance data for {SNP_TICKER}.")

snp["SNP_Close"] = pd.to_numeric(snp[snp_price_col], errors="coerce")
snp = snp.dropna(subset=["SNP_Close"]).reset_index(drop=True)

# Past-only MAs
snp["SMA_fast"] = snp["SNP_Close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
snp["SMA_slow"] = snp["SNP_Close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

bullish = (snp["SNP_Close"] > snp["SMA_slow"]) & (snp["SMA_fast"] > snp["SMA_slow"])
bearish = (snp["SNP_Close"] < snp["SMA_slow"]) & (snp["SMA_fast"] < snp["SMA_slow"])
snp["MarketTrend"] = np.where(bullish, "Bullish", np.where(bearish, "Bearish", "Neutral"))

# Merge trend onto every row by Date (use last prior trading day for weekends/holidays)
trend_by_date = snp[["SNP_Date", "MarketTrend"]].sort_values("SNP_Date")

df = df.sort_values("Date").reset_index(drop=True)
df = pd.merge_asof(
    df,
    trend_by_date,
    left_on="Date",
    right_on="SNP_Date",
    direction="backward",
    allow_exact_matches=True,
)
df = df.drop(columns=["SNP_Date"], errors="ignore")
df["MarketTrend"] = df["MarketTrend"].fillna("Neutral")

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
print("[INFO] MarketTrend added to all rows.")

# ==========================================================
# ADD PAST-DAY FEATURES
# ==========================================================
for col in ["Open", "High", "Low", "Close"]:
    if col in df.columns:
        df[f"Prev_{col}"] = df.groupby("Ticker")[col].shift(1)

# ==========================================================
# SELL LABEL FUNCTION (trend-aware; no score saved)
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

    Important fix:
      window = closes[i+1 : i+HOLD_DAYS+1]  # length HOLD_DAYS
    """
    closes = df_t["Close"].to_numpy(dtype=float)
    n = len(closes)

    labels = np.zeros(n, dtype=int)

    # Trend per row (default to "Neutral" if missing)
    trends = df_t.get("MarketTrend", pd.Series(["Neutral"] * n)).astype(str).to_numpy()

    for i in range(n - HOLD_DAYS):
        # FIXED: include HOLD_DAYS points (i+1 ... i+HOLD_DAYS)
        window = closes[i + 1 : i + HOLD_DAYS + 1]
        if window.size == 0:
            continue

        price_today = closes[i]
        if price_today <= 0 or not np.isfinite(price_today):
            continue

        trend = trends[i] if i < len(trends) else "Neutral"
        t = str(trend).lower()

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

        max_future = float(np.nanmax(window))

        # How much upside exists vs best future top?
        upside_best = (max_future - price_today) / price_today if price_today > 0 else 0.0

        if np.isfinite(max_future) and max_future > 0:
            upside_left_ratio = (max_future - price_today) / max_future
        else:
            upside_left_ratio = 1.0  # degenerate, treat as far from peak

        # Early crash detection window
        early_window = window[:EARLY_DAYS]
        if early_window.size > 0:
            early_min = float(np.nanmin(early_window))
            early_drop = (early_min - price_today) / price_today
        else:
            early_drop = 0.0

        # Conditions
        cond_peak = (upside_best > 0) and (upside_left_ratio <= eps_peak)
        cond_crash = early_drop <= crash_drop

        labels[i] = 1 if (cond_peak or cond_crash) else 0

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

# ✅ Rolling averages (computed once; removed duplicated block)
if "Prev_Open" in df_final.columns:
    df_final["avg_open_past_3_days"] = df_final.groupby("Ticker")["Prev_Open"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
if "Prev_Close" in df_final.columns:
    df_final["avg_close_past_3_days"] = df_final.groupby("Ticker")["Prev_Close"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

if "avg_low_raw" in df_final.columns:
    df_final["avg_low_30"] = df_final.groupby("Ticker")["avg_low_raw"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
if "avg_high_raw" in df_final.columns:
    df_final["avg_high_30"] = df_final.groupby("Ticker")["avg_high_raw"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

# ✅ Volatility 30 zile (safe denom)
if "avg_high_30" in df_final.columns and "avg_low_30" in df_final.columns:
    df_final["volatility_30"] = (df_final["avg_high_30"] - df_final["avg_low_30"]) / (df_final["avg_low_30"] + 1e-6)

# Drop unwanted OHLC columns (original names, if still present)
drop_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns], errors="ignore")

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
