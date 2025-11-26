# scripts/llm_data_builder.py
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ==========================================================
# MARKET TREND CONTEXT (same logic as training)
# ==========================================================
def get_market_trend() -> str:
    spx = yf.download("^GSPC", start="2020-01-01", auto_adjust=False, progress=False)
    if spx.empty:
        return "Neutral"

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


# ==========================================================
# TECHNICALS (copied from training script)
# ==========================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_single_record(latest: pd.Series, market_trend: str) -> dict:
    ts = pd.to_datetime(latest["Date"])
    date_ms = int(ts.value // 10**6)

    def g(col, default=None):
        return latest.get(col, default)

    avg_low_30 = g("avg_low_30")
    avg_high_30 = g("avg_high_30")
    price = g("close_raw")

    if avg_low_30 is not None and avg_high_30 is not None:
        try:
            rng = float(avg_high_30) - float(avg_low_30)
            pos_30d = (float(price) - float(avg_low_30)) / (rng + 1e-9)
        except Exception:
            pos_30d = None
    else:
        pos_30d = None

    return {
        "Date": date_ms,
        "open_raw": _to_float_or_none(g("open_raw")),
        "avg_high_raw": _to_float_or_none(g("avg_high_raw")),
        "avg_low_raw": _to_float_or_none(g("avg_low_raw")),
        "close_raw": _to_float_or_none(g("close_raw")),
        "SMA20": _to_float_or_none(g("SMA20")),
        "SMA50": _to_float_or_none(g("SMA50")),
        "SMA200": _to_float_or_none(g("SMA200")),
        "EMA20": _to_float_or_none(g("EMA20")),
        "EMA50": _to_float_or_none(g("EMA50")),
        "EMA200": _to_float_or_none(g("EMA200")),
        "RSI14": _to_float_or_none(g("RSI14")),
        "MACD": _to_float_or_none(g("MACD")),
        "MACD_signal": _to_float_or_none(g("MACD_signal")),
        "MACD_hist": _to_float_or_none(g("MACD_hist")),
        "ATR": _to_float_or_none(g("ATR")),
        "ATR%": _to_float_or_none(g("ATR%")),
        "Volatility": _to_float_or_none(g("Volatility")),
        "Momentum": _to_float_or_none(g("Momentum")),
        "OBV": _to_float_or_none(g("OBV")),
        "MarketTrend": g("MarketTrend") or market_trend,
        "Ticker": g("Ticker"),
        "Year": int(g("Year")),
        "Prev_Open": _to_float_or_none(g("Prev_Open")),
        "Prev_High": _to_float_or_none(g("Prev_High")),
        "Prev_Low": _to_float_or_none(g("Prev_Low")),
        "Prev_Close": _to_float_or_none(g("Prev_Close")),
        "avg_open_past_3_days": _to_float_or_none(g("avg_open_past_3_days")),
        "avg_close_past_3_days": _to_float_or_none(g("avg_close_past_3_days")),
        "avg_low_30": _to_float_or_none(avg_low_30),
        "avg_high_30": _to_float_or_none(avg_high_30),
        "volatility_30": _to_float_or_none(g("volatility_30")),
        "current_price": _to_float_or_none(g("current_price")),
        "pos_30d": _to_float_or_none(pos_30d),
    }


def build_llm_today_data(
    stage2_path: str = "data/stage2_merged.csv",
    out_path: str = "data/LLM_today_data.jsonl",
    top_n: int = 50,
    history_years: int = 2,
) -> list[str]:
    stage2_path = Path(stage2_path)
    out_path = Path(out_path)

    if not stage2_path.exists():
        raise FileNotFoundError(f"{stage2_path} not found")

    df_stage2 = pd.read_csv(stage2_path)
    # Prefer the final merged score if present (keeps LLM selection aligned with final stage ranking)
    score_col = None
    if "merged_final_score" in df_stage2.columns:
        score_col = "merged_final_score"
    elif "merged_score" in df_stage2.columns:
        score_col = "merged_score"
    else:
        raise KeyError("Neither 'merged_final_score' nor 'merged_score' column found in stage2_merged.csv")

    df_stage2 = df_stage2.sort_values(score_col, ascending=False)

    # Oversample more aggressively to tolerate yfinance misses; we'll try until we write `top_n` records.
    oversample_factor = int(os.getenv("LLM_OVERSAMPLE_FACTOR", "4"))
    candidate_count = max(top_n * oversample_factor, top_n + 10)
    tickers = df_stage2["ticker"].dropna().astype(str).drop_duplicates().head(candidate_count).tolist()
    print(f"[LLM_DATA] Selected top {len(tickers)} candidate tickers for Brain input (oversample_factor={oversample_factor}).")

    market_trend = get_market_trend()
    print(f"[LLM_DATA] MarketTrend={market_trend}")

    out_records, written_tickers = [], []
    skip_reasons: dict[str, str] = {}

    chunk_size = int(os.getenv("LLM_YF_CHUNK_SIZE", "50"))
    batch_sleep = float(os.getenv("LLM_YF_BATCH_SLEEP", "0.5"))

    try:
        from data_fetcher import download_history_cached_dict
    except Exception:
        download_history_cached_dict = None

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
        if len(out_records) >= top_n:
            break

        if download_history_cached_dict:
            data_batch = download_history_cached_dict(batch, period=f"{history_years}y", interval="1d", auto_adjust=False)
        else:
            # fallback to individual downloads
            data_batch = {}
            for t in batch:
                try:
                    data_batch[t] = yf.download(t, period=f"{history_years}y", interval="1d", auto_adjust=False, progress=False)
                except Exception:
                    data_batch[t] = pd.DataFrame()

        for t in batch:
            if len(out_records) >= top_n:
                break

            data = data_batch.get(t)
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                reason = "no data returned by yfinance"
                print(f"[LLM_DATA][WARN] {reason} for {t}, skipping.")
                skip_reasons[t] = reason
                continue

            # If the batch downloader returned a per-ticker DataFrame already, use it.
            df_raw = data

            required_cols = {"Open", "High", "Low", "Close"}
            if not required_cols.issubset(set(map(str, df_raw.columns))):
                reason = f"missing OHLC columns: {list(df_raw.columns)}"
                print(f"[LLM_DATA][WARN] {reason} for {t}. Skipping.")
                skip_reasons[t] = reason
                continue

            df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close"])
            if df_raw.empty:
                reason = "only NaNs after dropping OHLC"
                print(f"[LLM_DATA][WARN] {reason} for {t}, skipping.")
                skip_reasons[t] = reason
                continue

            df_t = compute_indicators(df_raw)
            df_t = df_t.reset_index().rename(columns={"Date": "Date"})
            df_t["Ticker"] = t
            df_t["MarketTrend"] = market_trend
            df_t["Year"] = pd.to_datetime(df_t["Date"]).dt.year

            df_t["Prev_Open"] = df_t["Open"].shift(1)
            df_t["Prev_High"] = df_t["High"].shift(1)
            df_t["Prev_Low"] = df_t["Low"].shift(1)
            df_t["Prev_Close"] = df_t["Close"].shift(1)

            df_t = df_t.rename(
                columns={
                    "Open": "open_raw",
                    "High": "avg_high_raw",
                    "Low": "avg_low_raw",
                    "Close": "close_raw",
                }
            )

            df_t["avg_open_past_3_days"] = df_t["Prev_Open"].rolling(3, min_periods=1).mean()
            df_t["avg_close_past_3_days"] = df_t["Prev_Close"].rolling(3, min_periods=1).mean()
            df_t["avg_low_30"] = df_t["avg_low_raw"].rolling(30, min_periods=1).mean()
            df_t["avg_high_30"] = df_t["avg_high_raw"].rolling(30, min_periods=1).mean()
            df_t["volatility_30"] = (df_t["avg_high_30"] - df_t["avg_low_30"]) / df_t["avg_low_30"]
            df_t["current_price"] = df_t["close_raw"]

            latest = df_t.sort_values("Date").iloc[-1]
            rec = _build_single_record(latest, market_trend)
            out_records.append(rec)
            written_tickers.append(t)

        time.sleep(batch_sleep)

    if not out_records:
        raise RuntimeError("No LLM records produced – all downloads failed?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    print(f"[LLM_DATA] Wrote {len(out_records)} records → {out_path}")
    if len(out_records) < top_n:
        print(f"[LLM_DATA][WARN] Only {len(out_records)}/{top_n} built due to missing data.")
    # Write a diagnostics/reconciliation file mapping stage2 candidates -> produced tickers and skip reasons
    diag = {
        "stage2_top_n": df_stage2["ticker"].dropna().astype(str).drop_duplicates().head(top_n).tolist(),
        "stage2_candidates_count": candidate_count,
        "oversample_factor": oversample_factor,
        "stage2_candidates": tickers,
        "produced_count": len(out_records),
        "produced_tickers": written_tickers,
        "skipped_reasons": skip_reasons,
        "market_trend": market_trend,
    }

    logs_path = Path("logs")
    logs_path.mkdir(parents=True, exist_ok=True)
    diag_path = logs_path / "llm_stage2_reconciliation.json"
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"[LLM_DATA] Wrote diagnostics → {diag_path}")

    return written_tickers


if __name__ == "__main__":
    build_llm_today_data()
