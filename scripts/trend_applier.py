# trend_applier.py
import os
import yfinance as yf
import pandas as pd
import logging

log = logging.getLogger("trend_applier")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def detect_market_trend(symbol="SPY", lookback=200):
    df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    
    last = df.iloc[-1]
    trend = "neutral"

    # ✅ Force scalars to avoid any Series-vs-Series comparison weirdness
    close = float(last["Close"])
    sma50 = float(last["SMA50"])
    sma200 = float(last["SMA200"])

    if close > sma50 and sma50 > sma200:
        trend = "up"
    elif close < sma50 and sma50 < sma200:
        trend = "down"
    elif sma50 > sma200 and close < sma50:
        trend = "pullback"
    elif sma50 < sma200 and close > sma50:
        trend = "recovering"

    df["above_20dma"] = df["Close"] > df["Close"].rolling(20).mean()
    breadth = df["above_20dma"].tail(20).mean() * 100

    return {
        "trend": trend,
        "close": last["Close"],
        "sma50": last["SMA50"],
        "sma200": last["SMA200"],
        "breadth": breadth
    }

def apply_market_env():
    """Detect market trend and apply env weightings dynamically."""
    ctx = detect_market_trend("SPY")
    trend = ctx["trend"]
    breadth = ctx["breadth"]

    log.info(f"[Market] trend={trend}, breadth={breadth:.1f}%")

    if trend == "up" and breadth > 60:
        log.info("[Market] Bullish regime → emphasizing momentum & trend")
        os.environ.update({
            "QS_W_TREND_LARGE": "0.55",
            "QS_W_MOMO_LARGE":  "0.35",
            "QS_W_STRUCT_LARGE": "0.08",
            "QS_W_RISK_LARGE":   "0.02",
            "QS_MOMO_CHASE_PEN_MAX": "10",
            "QS_SETUP_RSI_HI": "66",
            "QS_SETUP_RSI_LO": "48"
        })
    elif trend == "down" or breadth < 40:
        log.info("[Market] Bearish regime → defensive scoring")
        os.environ.update({
            "QS_W_TREND_LARGE": "0.35",
            "QS_W_MOMO_LARGE":  "0.25",
            "QS_W_STRUCT_LARGE": "0.25",
            "QS_W_RISK_LARGE":   "0.15",
            "QS_MOMO_CHASE_PEN_MAX": "25",
            "QS_SETUP_RSI_HI": "58",
            "QS_SETUP_RSI_LO": "40"
        })
    else:
        log.info("[Market] Neutral regime → balanced weights")
        os.environ.update({
            "QS_W_TREND_LARGE": "0.45",
            "QS_W_MOMO_LARGE":  "0.30",
            "QS_W_STRUCT_LARGE": "0.20",
            "QS_W_RISK_LARGE":   "0.05",
            "QS_MOMO_CHASE_PEN_MAX": "15",
            "QS_SETUP_RSI_HI": "62",
            "QS_SETUP_RSI_LO": "45"
        })

    return trend
