# scripts/test_brain_ranker.py
import argparse
import json
import os
from pathlib import Path
import sys
import math
import pandas as pd

DEFAULT_JSONL = "data/LLM_today_data.jsonl"

def load_llm_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "Ticker" not in df.columns:
        raise ValueError("JSONL missing 'Ticker' column")
    return df

def fallback_brain(df: pd.DataFrame, top_k: int = 10):
    """
    Lightweight heuristic for wiring tests ONLY.
    Produces a 'BrainScore' ~ 0..1000 using a few technicals.
    """
    import numpy as np

    def nz(series, default=0.0):
        s = pd.to_numeric(series, errors="coerce")
        s = s.fillna(default)
        return s

    # Normalize helpful signals to 0..1
    rsi = nz(df.get("RSI14"), 50.0).clip(0, 100) / 100.0                 # higher better (sweet spot ~55-70, but ok)
    mom = nz(df.get("Momentum"), 0.0)
    # rescale momentum to 0..1 via rank
    mom_rank = mom.rank(pct=True)

    macd_hist = nz(df.get("MACD_hist"), 0.0)
    macd_rank = macd_hist.rank(pct=True)

    pos_30d = nz(df.get("pos_30d"), 0.5).clip(0, 1)                       # where price sits in 30d range
    vol30 = nz(df.get("volatility_30"), 0.03).clip(0.0, 0.25)             # cap so penalty isn’t extreme

    # Combine (weights are arbitrary, just for plumbing checks)
    score01 = (
        0.30 * rsi +
        0.25 * mom_rank +
        0.25 * macd_rank +
        0.15 * pos_30d +
        0.05 * (1.0 - (vol30 / 0.25))     # small penalty for very high vol
    )

    # Map to 0..1000
    brain_score = (score01.clip(0, 1) * 1000.0)

    out = (
        pd.DataFrame({
            "Ticker": df["Ticker"].astype(str),
            "BrainScore": brain_score,
            "RSI14": df.get("RSI14"),
            "Momentum": df.get("Momentum"),
            "MACD_hist": df.get("MACD_hist"),
            "pos_30d": df.get("pos_30d"),
            "volatility_30": df.get("volatility_30"),
        })
        .dropna(subset=["Ticker"])
        .sort_values("BrainScore", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return out

def main():
    ap = argparse.ArgumentParser(description="Test Brain ranker with LLM_today_data.jsonl")
    ap.add_argument("--path", default=DEFAULT_JSONL, help="Path to LLM_today_data.jsonl")
    ap.add_argument("--top_k", type=int, default=10, help="Top K")
    ap.add_argument("--save_csv", default="logs/brain_scores_latest.csv", help="Where to save scores CSV")
    args = ap.parse_args()

    jsonl_path = args.path
    if not os.path.exists(jsonl_path):
        print(f"[ERROR] JSONL not found: {jsonl_path}")
        sys.exit(1)

    df = load_llm_jsonl(jsonl_path)

    # Try your real Brain first
    used_real_brain = False
    try:
        from brain_ranker import rank_with_brain  # your implementation
        print("[INFO] brain_ranker.rank_with_brain found — using real Brain.")
        tickers_topk, scores_map = rank_with_brain(llm_data_path=jsonl_path, top_k=args.top_k)
        # Convert to DataFrame for consistent output/CSV
        rows = []
        for t in tickers_topk:
            rows.append({"Ticker": t, "BrainScore": float(scores_map.get(t, 0.0))})
        out = pd.DataFrame(rows).sort_values("BrainScore", ascending=False).reset_index(drop=True)
        used_real_brain = True
    except Exception as e:
        print(f"[WARN] Real Brain not available or failed ({e}). Falling back to heuristic.")
        out = fallback_brain(df, top_k=args.top_k)

    # Pretty print
    if not out.empty:
        print("\n[TOP PICKS]")
        print(out.to_string(index=False, formatters={"BrainScore": "{:.2f}".format}))
    else:
        print("[ERROR] No results produced.")

    # Save CSV for debugging
    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    print(f"\n[SAVED] {save_path} ({len(out)} rows)")

    # Also save JSON for quick machine consumption
    json_save = save_path.with_suffix(".json")
    out.to_json(json_save, orient="records", indent=2)
    print(f"[SAVED] {json_save}")

    # Helpful exit code: 0 if we got K rows
    if len(out) < args.top_k:
        sys.exit(2 if used_real_brain else 0)

if __name__ == "__main__":
    main()
