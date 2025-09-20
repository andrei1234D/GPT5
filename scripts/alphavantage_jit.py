# alphavantage_jit.py
import os
import requests
from datetime import datetime, timedelta, UTC

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def _fetch_single_ticker(ticker: str, cutoff, limit: int = 50):
    """Helper: fetch news for a single ticker."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": limit,
    }

    print(f"[INFO] Fallback: Fetching AlphaVantage news for single ticker {ticker} (limit={limit})")

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Request failed for {ticker}: {e}")
        return []

    if "feed" not in data:
        print(f"[WARN] No 'feed' key for {ticker}. Raw response: {data}")
        return []

    articles = []
    for item in data["feed"]:
        published_raw = item.get("time_published")
        if not published_raw:
            continue
        try:
            published_dt = datetime.strptime(published_raw, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except Exception:
            continue
        if published_dt < cutoff:
            continue
        articles.append({
            "title": item.get("title"),
            "summary": item.get("summary"),
            "source": item.get("source"),
            "published_at": published_raw,
            "sentiment": item.get("overall_sentiment_label"),
        })

    print(f"[INFO] {ticker}: {len(articles)} articles after filtering")
    return articles


def get_news_sentiment_bulk(tickers: list[str], days: int = 7, limit: int = 50):
    """
    Fetch news sentiment for a list of tickers from Alpha Vantage.
    Tries bulk first, falls back to per-ticker if 'feed' is missing.
    Returns dict: {ticker: [articles...]}
    """
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY in environment")

    cutoff = datetime.now(UTC) - timedelta(days=days)
    results = {t: [] for t in tickers}

    # === Try bulk first ===
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "apikey": ALPHAVANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": limit,
    }

    print(f"[INFO] Fetching AlphaVantage news for: {', '.join(tickers)} (limit={limit}, days={days})")

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Bulk request failed: {e}")
        data = {}

    if "feed" in data:
        print(f"[INFO] Raw feed articles returned: {len(data['feed'])}")
        for item in data["feed"]:
            published_raw = item.get("time_published")
            if not published_raw:
                continue
            try:
                published_dt = datetime.strptime(published_raw, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            except Exception:
                continue
            if published_dt < cutoff:
                continue

            matched = False
            for related in item.get("ticker_sentiment", []):
                sym = related.get("ticker")
                if sym in results:
                    results[sym].append({
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "source": item.get("source"),
                        "published_at": published_raw,
                        "sentiment": item.get("overall_sentiment_label"),
                    })
                    matched = True
            if not matched:
                print(f"[DEBUG] Article did not match any requested tickers: {item.get('title')}")
        # Summarize results
        for t in tickers:
            print(f"[INFO] {t}: {len(results[t])} articles after filtering")
        return results

    # === Fallback: per-ticker ===
    print("[WARN] No 'feed' key in AlphaVantage bulk response → falling back to per-ticker fetch")
    for t in tickers:
        results[t] = _fetch_single_ticker(t, cutoff, limit)

    return results
