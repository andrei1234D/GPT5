# scripts/news_top10.py
import os
import pandas as pd
import openai
from alphavantage_jit import get_news_sentiment_bulk

def load_top10(path="data/stage2_merged.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("merged_score", ascending=False)
    tickers = df.head(10)["ticker"].tolist()
    print(f"[INFO] Loaded Top-10 tickers from {path}: {', '.join(tickers)}")
    return tickers

def fetch_bulk_news(tickers, days=7, limit=50):
    print(f"[INFO] Fetching bulk news for {len(tickers)} tickers, limit={limit}, days={days}")
    try:
        news_map = get_news_sentiment_bulk(tickers, days=days, limit=limit)
        # Log results per ticker
        for t, articles in news_map.items():
            if not articles:
                print(f"[WARN] {t}: No articles found")
            else:
                print(f"[INFO] {t}: {len(articles)} articles pulled")
        return news_map
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
        return {t: [{"error": str(e)}] for t in tickers}

def format_news_prompt(news_map):
    blocks = []
    for t, articles in news_map.items():
        if not articles or "error" in articles[0]:
            blocks.append(f"### {t}\n- N/A")
        else:
            blocks.append(
                f"### {t}\n" + "\n".join(
                    f"- {a.get('title')} ({a.get('sentiment')}, {a.get('source')}, {a.get('published_at')})"
                    for a in articles[:3]
                )
            )
    formatted = "\n\n".join(blocks)
    print("[DEBUG] Formatted news prompt:\n", formatted)
    return formatted

def main():
    tickers = load_top10()
    news_map = fetch_bulk_news(tickers)

    prompt = """
You are a financial assistant.
Summarize the news sentiment for each ticker below in 1–2 concise bullet points.
Score the overall impact of each ticker as +50 to -50 (positive or negative).
If no relevant news, return exactly "N/A".
    """
    prompt += "\n\n" + format_news_prompt(news_map)

    print("[DEBUG] Final prompt sent to GPT:\n", prompt[:1000], "...\n")  # truncate for readability

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
    print("[INFO] GPT Output:\n", output)

if __name__ == "__main__":
    main()
