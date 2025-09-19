# alphavantage_jit.py
import os
import requests
from datetime import datetime, timedelta, UTC

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def get_news_sentiment_bulk(tickers: list[str], days: int = 7, limit: int = 50):
    """
    Fetch news sentiment for a list of tickers from Alpha Vantage.
    Filters by recency (days) and groups by ticker.
    Returns dict: {ticker: [articles...]}
    """
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY in environment")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "apikey": ALPHAVANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": limit,
    }

    print(f"[INFO] Fetching AlphaVantage news for: {', '.join(tickers)} (limit={limit}, days={days})")

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"AlphaVantage returned non-JSON: {resp.text[:200]}")

    # Init results map
    results = {t: [] for t in tickers}
    cutoff = datetime.now(UTC) - timedelta(days=days)

    if "feed" not in data:
        print("[WARN] No 'feed' key in AlphaVantage response")
        return results

    print(f"[INFO] Raw feed articles returned: {len(data['feed'])}")

    for item in data["feed"]:
        published_raw = item.get("time_published")
        if not published_raw:
            continue

        try:
            published_dt = datetime.strptime(published_raw, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except Exception:
            continue

        # Skip old articles
        if published_dt < cutoff:
            continue

        # Track matches
        matched = False
        for related in item.get("ticker_sentiment", []):
            sym = related.get("ticker")
            if sym in results:
                results[sym].append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "published_at": published_raw,
                    "sentiment": item.get("overall_sentiment_label"),
                })
                matched = True

        if not matched:
            print(f"[DEBUG] Article did not match any requested tickers: {item.get('title')}")

    # Console summary
    for t in tickers:
        print(f"[INFO] {t}: {len(results[t])} articles after filtering")

    return results
