import os
import requests

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def get_news_sentiment_bulk(tickers: list[str], limit: int = 5):
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY in environment")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "apikey": ALPHAVANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": limit,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = {t: [] for t in tickers}
    if "feed" not in data:
        return results

    for item in data["feed"]:
        for related in item.get("ticker_sentiment", []):
            if related.get("ticker") in results:
                results[related["ticker"]].append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "published_at": item.get("time_published"),
                    "sentiment": item.get("overall_sentiment_label"),
                })
    return results
