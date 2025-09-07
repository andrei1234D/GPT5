# scripts/news_top10.py
import os
import pandas as pd
import openai
from alphavantage_jit import get_news_sentiment_bulk

def load_top10(path="data/stage2_merged.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("merged_score", ascending=False)
    return df.head(10)["ticker"].tolist()

def fetch_bulk_news(tickers, days=7, limit=50):
    try:
        return get_news_sentiment_bulk(tickers, days=days, limit=limit)
    except Exception as e:
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
    return "\n\n".join(blocks)

def main():
    tickers = load_top10()
    news_map = fetch_bulk_news(tickers)

    prompt = """
You are a financial assistant.
Summarize the news sentiment for each ticker below in 1–2 concise bullet points.
Score the overall impact as +50 to -50 (positive or negative).
If no relevant news, return exactly "N/A".
    """
    prompt += "\n\n" + format_news_prompt(news_map)

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

    print(response.choices[0].message.content.strip())

if __name__ == "__main__":
    main()
