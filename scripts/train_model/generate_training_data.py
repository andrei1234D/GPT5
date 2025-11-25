# -*- coding: utf-8 -*-
"""
generate_training_data_full_single.py — Full Daily Dataset Builder (Single Parquet)
----------------------------------------------------------------------------------
Downloads full daily OHLCV, technical indicators, and fundamentals
for every ticker between 2018–2025. Combines all tickers into a single
DataFrame and saves ONE big Parquet file:
    ../../data/full_market_2018_2025.parquet
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================================
# CONFIG
# ==========================================================
INPUT_CSV = "../../data/universe_clean.csv"
OUTPUT_DIR = Path("/LLM_Training_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "full_market_2018_2025.parquet"

MAX_TICKERS = 5311
BATCH_SIZE = 60
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
MAX_WORKERS = 10


# ==========================================================
# HELPERS
# ==========================================================

def _clean_num(v):
    """
    Convert yfinance numeric fields to finite floats or None.

    - Handles None
    - Handles 'Infinity', 'inf', 'NaN' as None
    - Forces float and drops +/-inf, nan
    """
    if v is None:
        return None

    if isinstance(v, str):
        v_str = v.strip().lower()
        if v_str in {"inf", "+inf", "-inf", "infinity", "-infinity", "nan"}:
            return None
        try:
            v = float(v_str)
        except Exception:
            return None

    try:
        v = float(v)
    except Exception:
        return None

    if not np.isfinite(v):
        return None
    return v


# ==========================================================
# MARKET TREND CONTEXT
# ==========================================================
def get_market_trend():
    spx = yf.download("^GSPC", start="2020-01-01", end="2025-01-01", progress=False)
    spx["SMA50"] = spx["Close"].rolling(50).mean()
    spx["SMA200"] = spx["Close"].rolling(200).mean()
    if len(spx) < 200:
        return "Neutral"
    last_close = float(spx["Close"].iloc[-1])
    last_sma50 = float(spx["SMA50"].iloc[-1])
    last_sma200 = float(spx["SMA200"].iloc[-1])
    if last_close > last_sma200 and last_sma50 > last_sma200:
        return "Bullish"
    elif last_close < last_sma200 and last_sma50 < last_sma200:
        return "Bearish"
    return "Neutral"


MARKET_TREND = get_market_trend()
print(f"[INFO] Market regime detected: {MARKET_TREND}")


# ==========================================================
# FUNDAMENTALS (ASYNC)
# ==========================================================
def fetch_fundamentals_single(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info

        fundamentals = {
            "PE": _clean_num(info.get("trailingPE")),
            "PEG": _clean_num(info.get("pegRatio")),
            "PS": _clean_num(info.get("priceToSalesTrailing12Months")),
            "PB": _clean_num(info.get("priceToBook")),
            "DividendYield": _clean_num(info.get("dividendYield")),
            "Beta": _clean_num(info.get("beta")),
            "MarketCap": _clean_num(info.get("marketCap")),
        }

        fin = tk.financials.T
        if "Total Revenue" in fin.columns and len(fin) >= 2:
            growth = fin["Total Revenue"].iloc[-1] / fin["Total Revenue"].iloc[-2] - 1
            fundamentals["YoY_Growth"] = _clean_num(growth)
        else:
            fundamentals["YoY_Growth"] = None

        return (ticker, fundamentals)

    except Exception:
        return (ticker, {})


def fetch_fundamentals_batch(batch):
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_fundamentals_single, t): t for t in batch}
        for f in as_completed(futures):
            t, res = f.result()
            results[t] = res
    return results


# ==========================================================
# TECHNICALS
# ==========================================================
def compute_indicators(df):
    df = df.copy()

    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    # RSI 14
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Volatility/ATR
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR%"] = df["ATR"] / df["Close"] * 100
    df["Volatility"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # Momentum & OBV
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    return df


# ==========================================================
# MAIN
# ==========================================================
def safe_download(batch, start, end, retries=5):
    for attempt in range(retries):
        try:
            return yf.download(
                batch,
                start=start,
                end=end,
                group_by="ticker",
                progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"[WARN] Download failed (attempt {attempt+1}/{retries}): {e}")
            sleep_time = 10 + random.uniform(5, 15)
            print(f"[WAIT] Cooling down for {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    print("[ERROR] All retries failed for this batch.")
    return pd.DataFrame()


def main():
    df_universe = pd.read_csv(INPUT_CSV)
    tickers = df_universe["ticker"].dropna().unique().tolist()[:MAX_TICKERS]
    print(f"[INFO] Building full dataset for {len(tickers)} tickers (batch={BATCH_SIZE})")

    combined_records = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"[INFO] Batch {i//BATCH_SIZE + 1}: downloading {len(batch)} tickers...")

        fundamentals = fetch_fundamentals_batch(batch)

        try:
            data = safe_download(batch, start=START_DATE, end=END_DATE)
        except Exception as e:
            print(f"[ERROR] Batch download failed: {e}")
            continue

        # MultiIndex case: columns like (ticker, 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                try:
                    if (t, "Close") not in data.columns:
                        continue
                    df_t = data[t].dropna().copy()
                    if df_t.empty:
                        continue

                    df_t = compute_indicators(df_t)

                    # Attach fundamentals
                    for k, v in fundamentals.get(t, {}).items():
                        df_t[k] = v

                    # Meta
                    df_t["MarketTrend"] = MARKET_TREND
                    df_t["Ticker"] = t

                    combined_records.append(df_t)
                    print(f"[OK] Processed {t}")
                except Exception as e:
                    print(f"[ERROR] {t}: {e}")
                    continue

        time.sleep(1)  # be nice to Yahoo

    if not combined_records:
        print("[ERROR] No data collected. Exiting.")
        return

    print(f"[INFO] Concatenating {len(combined_records)} per-ticker DataFrames...")
    full_df = pd.concat(combined_records, axis=0)
    full_df.index = pd.to_datetime(full_df.index)
    full_df.sort_index(inplace=True)

    # Add Year column for convenience
    full_df["Year"] = full_df.index.year

    # Ensure numeric columns are truly numeric and free of inf/NaN issues
    num_cols = [
        "PE",
        "PEG",
        "PS",
        "PB",
        "DividendYield",
        "Beta",
        "MarketCap",
        "YoY_Growth",
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI14",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "ATR",
        "ATR%",
        "Volatility",
        "Momentum",
        "OBV",
    ]

    for col in num_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
            full_df[col] = full_df[col].replace([np.inf, -np.inf], np.nan)

    print(f"[INFO] Saving single full parquet to {OUTPUT_PATH} ...")
    full_df.to_parquet(OUTPUT_PATH, index=True)
    print(f"[SAVED] {len(full_df)} rows → {OUTPUT_PATH}")
    print("[DONE] Full single-file dataset generated.")


if __name__ == "__main__":
    main()
