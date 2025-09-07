import os
import requests

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def get_news_sentiment_bulk(ticker: str, limit: int = 5):
    """
    Fetch latest news sentiment for a given ticker from Alpha Vantage.
    Returns a list of dicts with title, source, published_at, sentiment.
    Ensures we only include articles where this ticker appears in ticker_sentiment.
    """
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY in environment")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": limit,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "feed" not in data:
        return []

    results = []
    for item in data["feed"]:
        # ensure ticker is actually relevant in this article
        for related in item.get("ticker_sentiment", []):
            if related.get("ticker") == ticker:
                results.append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "published_at": item.get("time_published"),
                    "sentiment": item.get("overall_sentiment_label"),
                })
                break  # don’t duplicate the same article
    return results
