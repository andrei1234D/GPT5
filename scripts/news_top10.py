# scripts/news_top10.py
import os
import json
import openai
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pandas as pd

from alphavantage_jit import get_news_sentiment_bulk


def log(msg: str):
    print(msg, flush=True)


def fetch_bulk_news(tickers, days=7, limit=50):
    """Fetch news (bulk+fallback) and filter by cutoff date."""
    log(f"[INFO] Fetching bulk news for {len(tickers)} tickers, limit={limit}, days={days}")
    try:
        news_map = get_news_sentiment_bulk(tickers, days=days, limit=limit)

        # Detect possible AlphaVantage rate limit issues
        if not news_map:
            log("[WARN] AlphaVantage returned an empty news_map (possible API limit reached).")

        cutoff = datetime.now(UTC) - timedelta(days=days)
        filtered_map = {}

        for t in tickers:
            articles = news_map.get(t, [])
            if not articles:
                log(f"[WARN] No articles returned for {t} (could be API quota limit).")
            filtered = [
                a for a in articles
                if a.get("published_at")
                and datetime.strptime(a["published_at"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC) >= cutoff
            ]
            filtered_map[t] = filtered
            log(f"[DEBUG] {t}: {len(filtered)} articles after cutoff (raw: {len(articles)})")
            for a in filtered[:2]:  # show up to 2 samples
                log(f"[DEBUG] {t} Article: {a.get('title')} "
                    f"({a.get('sentiment')}, {a.get('source')}, {a.get('published_at')})")

        return filtered_map

    except Exception as e:
        log(f"[ERROR] Failed to fetch news: {e}")
        log("[HINT] This could be due to AlphaVantage API quota being exceeded.")
        # Return an explicit empty map for all requested tickers
        return {t: [] for t in tickers}


def format_news_prompt(tickers, news_map):
    """
    Format news for GPT summarizer with strict 'Impact' line.
    Preserve ticker order exactly as selected.
    """
    blocks = []
    for t in tickers:
        articles = news_map.get(t, [])
        if not articles:
            blocks.append(f"### {t}\n- N/A\nImpact: 0")
        else:
            blocks.append(
                f"### {t}\n" + "\n".join(
                    f"- {a.get('title')} — {a.get('summary', 'N/A')} "
                    f"({a.get('sentiment')}, {a.get('source')}, {a.get('published_at')})"
                    for a in articles[:3]
                )
            )
    formatted = "\n\n".join(blocks)
    log("[DEBUG] Formatted news prompt (first 1200 chars):\n" + formatted[:1200] + "...\n")
    return formatted


def main():
    # 1) Load ML-ranked top 10 (same as notify.py will use)
    top_path = Path("data/top10_ml.csv")
    if not top_path.exists():
        raise FileNotFoundError(f"{top_path} not found. It should be built by ml_rank_daily.py")

    top_df = pd.read_csv(top_path)
    if top_df.empty or "ticker" not in top_df.columns:
        raise RuntimeError("top10_ml.csv is empty or missing 'ticker'")

    tickers = top_df["ticker"].astype(str).str.strip().tolist()
    if not tickers:
        raise RuntimeError("No tickers found in top10_ml.csv")

    log(f"[INFO] ML Top-10 tickers: {', '.join(tickers)}")

    # 2) Fetch news for exactly these tickers
    days = int(os.getenv("NEWS_DAYS", "7"))
    limit = int(os.getenv("NEWS_LIMIT", "50"))
    news_map = fetch_bulk_news(tickers, days=days, limit=limit)

    # 3) Build the GPT-3.5 prompt
    prompt = """
You are a financial assistant.
For each ticker, do the following:
- Read all provided articles.
- Summarize the overall news sentiment in 1–2 concise bullet points.
- Assign ONE overall numerical impact score between -50 (very negative) and +50 (very positive).
- Do not write labels like 'headwind' or '?' — only provide a number.

Strict output rules:
- Always output exactly 1–2 summary bullets.
- Always end each block with "Impact: <number>".
- If no articles exist, output exactly:
### TICKER
- N/A
Impact: 0

Output format per ticker:

### TICKER
- summary bullet 1
- summary bullet 2
Impact: +/-XX
""".strip()

    prompt += "\n\n" + format_news_prompt(tickers, news_map)
    log("[DEBUG] Final prompt sent to GPT-3.5 (first 1500 chars):\n" + prompt[:1500] + "...\n")

    # 4) Call GPT-3.5 to condense & score impact
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.5,
    )

    output = response.choices[0].message.content.strip()
    log("[INFO] GPT-3.5 Output:\n" + output)

    # Validate format
    if "Impact:" not in output:
        log("[WARN] GPT output missing 'Impact:' lines, check prompt strictness.")

    # 5) Save to file for notify.py
    out_path = Path("data/news_summary_top10.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    log(f"[INFO] Saved GPT-3.5 news summary to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
