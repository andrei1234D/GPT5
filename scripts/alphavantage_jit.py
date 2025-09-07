# scripts/alphavantage_jit.py
import os
import requests

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def get_news_sentiment_bulk(tickers: list[str], days: int = 7, limit: int = 50):
    """
    Fetch latest news sentiment for multiple tickers in one API call.
    Returns a dict: {ticker: [articles...]}.
    """
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY in environment")

    tickers_str = ",".join(tickers)
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers_str,
        "apikey": ALPHAVANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": limit,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    out = {t: [] for t in tickers}
    for item in data.get("feed", []):
        for related in item.get("ticker_sentiment", []):
            sym = related.get("ticker")
            if sym in out:
                out[sym].append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "published_at": item.get("time_published"),
                    "sentiment": item.get("overall_sentiment_label"),
                })
    return out
