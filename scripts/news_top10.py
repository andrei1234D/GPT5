# scripts/news_top10.py
import os
import pandas as pd
import openai
from datetime import datetime, timedelta, UTC
from pathlib import Path

from trash_ranker import pick_top_stratified  # <-- align selection logic
from alphavantage_jit import get_news_sentiment_bulk


def log(msg: str):
    print(msg, flush=True)


def load_ranked_from_stage2(path: str = "data/stage2_merged.csv"):
    """
    Build a 'ranked' list usable by pick_top_stratified:
      [(ticker, name, feats_dict, score, parts), ...]
    Score preference: merged_final_score > merged_score
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    # Keep None for NaN (so optional fields remain missing, not NaN)
    df = df.where(pd.notnull(df), None)

    ranked = []
    for _, row in df.iterrows():
        feats = row.to_dict()
        t = feats.get("ticker") or feats.get("symbol")
        if not t:
            # Skip rows without a ticker
            continue
        name = feats.get("company") or feats.get("name") or t
        score = feats.get("merged_final_score")
        if score is None:
            score = feats.get("merged_score", 0.0)

        ranked.append((t, name, feats, float(score or 0.0), {}))

    # Sort by the same preference used in pick_top_stratified's fallback
    ranked.sort(
        key=lambda x: (
            x[2].get("merged_final_score")
            or x[2].get("_merged_score")
            or x[3]
        ),
        reverse=True,
    )
    log(f"[INFO] Loaded {len(ranked)} rows from {path}")
    return ranked


def select_top10_via_stratified(ranked):
    """
    Use the exact same quotas as notify.py via env overrides.
    Defaults: total=10, min_small=5, min_large=5, pe_min=5
    """
    total = int(os.getenv("STAGE2_TOTAL", "10"))
    min_small = int(os.getenv("STAGE2_MIN_SMALL", "5"))
    min_large = int(os.getenv("STAGE2_MIN_LARGE", "5"))
    pe_min = int(os.getenv("STAGE2_MIN_PE", "5"))

    picked = pick_top_stratified(
        ranked,
        total=total,
        min_small=min_small,
        min_large=min_large,
        pe_min=pe_min,
    )
    tickers = [t for (t, _, _, _, _) in picked]
    log(f"[INFO] Stratified Top-{total}: {', '.join(tickers)}")
    return picked, tickers


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
    # 1) Build ranked list from Stage-2 and select via the SAME stratified picker
    ranked = load_ranked_from_stage2("data/stage2_merged.csv")
    picked, tickers = select_top10_via_stratified(ranked)

    # Persist the exact selection for downstream validation/debug
    sel_path = Path("data/top10_selected.txt")
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    sel_path.write_text("\n".join(tickers), encoding="utf-8")
    log(f"[INFO] Wrote selected tickers to {sel_path}")

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
