# scripts/news_top10.py
import os
import pandas as pd
import openai
from datetime import datetime, timedelta, UTC
from pathlib import Path
from alphavantage_jit import get_news_sentiment_bulk


def load_top10(path="data/stage2_merged.csv"):
    """Load top 10 tickers from stage2_merged.csv."""
    df = pd.read_csv(path)
    df = df.sort_values("merged_score", ascending=False)
    tickers = df.head(10)["ticker"].tolist()
    print(f"[INFO] Loaded Top-10 tickers from {path}: {', '.join(tickers)}")
    return tickers


def fetch_bulk_news(tickers, days=7, limit=50):
    """Fetch news (bulk+fallback) and filter by cutoff date."""
    print(f"[INFO] Fetching bulk news for {len(tickers)} tickers, limit={limit}, days={days}")
    try:
        news_map = get_news_sentiment_bulk(tickers, days=days, limit=limit)

        # Detect possible AlphaVantage rate limit issues
        if not news_map:
            print("[WARN] AlphaVantage returned an empty news_map (possible API limit reached).")
        else:
            for t, arts in news_map.items():
                if isinstance(arts, dict) and "Note" in arts.get("message", ""):
                    print(f"[WARN] API limit message detected for {t}: {arts['message']}")

        cutoff = datetime.now(UTC) - timedelta(days=days)
        filtered_map = {}

        for t, articles in news_map.items():
            if not articles:
                print(f"[WARN] No articles returned for {t} (could be API quota limit).")
            filtered = [
                a for a in articles
                if a.get("published_at")
                and datetime.strptime(a["published_at"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC) >= cutoff
            ]
            filtered_map[t] = filtered
            print(f"[DEBUG] {t}: {len(filtered)} articles after cutoff (raw: {len(articles)})")
            for a in filtered[:2]:  # show up to 2 samples
                print(f"[DEBUG] {t} Article: {a.get('title')} "
                      f"({a.get('sentiment')}, {a.get('source')}, {a.get('published_at')})")

        return filtered_map

    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
        print("[HINT] This could be due to AlphaVantage API quota being exceeded.")
        return {t: [] for t in tickers}


def format_news_prompt(news_map):
    """Format news for GPT summarizer with strict Impact field."""
    blocks = []
    for t, articles in news_map.items():
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
    print("[DEBUG] Formatted news prompt (first 1200 chars):\n", formatted[:1200], "...\n")
    return formatted


def main():
    tickers = load_top10()
    news_map = fetch_bulk_news(tickers)

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
"""
    prompt += "\n\n" + format_news_prompt(news_map)

    print("[DEBUG] Final prompt sent to GPT-3.5 (first 1500 chars):\n", prompt[:1500], "...\n")

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
    print("[INFO] GPT-3.5 Output:\n", output)

    # Validate format
    if "Impact:" not in output:
        print("[WARN] GPT output missing 'Impact:' lines, check prompt strictness.")

    # === Save to file for notify.py ===
    out_path = Path("data/news_summary_top10.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(output)
    print(f"[INFO] Saved GPT-3.5 news summary to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
