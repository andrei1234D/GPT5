# scripts/news_top10.py
import os
import pandas as pd
import openai
from datetime import datetime, timedelta, UTC
from alphavantage_jit import get_news_sentiment_bulk


def load_top10(path="data/stage2_merged.csv"):
    """Load top 10 tickers from stage2_merged.csv."""
    df = pd.read_csv(path)
    df = df.sort_values("merged_score", ascending=False)
    tickers = df.head(10)["ticker"].tolist()
    print(f"[INFO] Loaded Top-10 tickers from {path}: {', '.join(tickers)}")
    return tickers


def fetch_bulk_news(tickers, days=7, limit=50):
    """Fetch news (bulk+fallback) and filter by cutoff."""
    print(f"[INFO] Fetching bulk news for {len(tickers)} tickers, limit={limit}, days={days}")
    try:
        news_map = get_news_sentiment_bulk(tickers, days=days, limit=limit)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        filtered_map = {}

        for t, articles in news_map.items():
            filtered = [
                a for a in articles
                if a.get("published_at")
                and datetime.strptime(a["published_at"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC) >= cutoff
            ]
            filtered_map[t] = filtered
            print(f"[DEBUG] {t}: {len(filtered)} articles after cutoff")
            for a in filtered[:2]:  # show up to 2 samples
                print(f"[DEBUG] {t} Article: {a.get('title')} ({a.get('sentiment')}, {a.get('source')}, {a.get('published_at')})")

        return filtered_map

    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
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
                ) + "\nImpact: ?"
            )
    formatted = "\n\n".join(blocks)
    print("[DEBUG] Formatted news prompt:\n", formatted[:1200], "...\n")  # truncated for readability
    return formatted


def main():
    tickers = load_top10()
    news_map = fetch_bulk_news(tickers)

    prompt = """
You are a financial assistant.
For each ticker, summarize in 1–2 bullet points.
Assign a single overall impact score between -50 and +50.

RESPECT THE OUTPUT FORMAT EXACTLY.
Each block MUST end with one Impact line.

If no news is found, output exactly:
### TICKER
- N/A
Impact: 0

Output format per ticker:

### TICKER
- summary bullet 1
- summary bullet 2
Impact: +/- XX
    """
    prompt += "\n\n" + format_news_prompt(news_map)

    print("[DEBUG] Final prompt sent to GPT-3.5:\n", prompt[:1500], "...\n")

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


if __name__ == "__main__":
    main()
