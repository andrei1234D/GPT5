# scripts/brain_ranker.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).parent / "train_model" / "LLM_bot" / "Brain" / "llm_signal_regression_model"
MODEL_PATH = MODEL_DIR / "model.safetensors"

def _load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Brain model directory not found: {MODEL_DIR}")

    print(f"[BRAIN] Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[BRAIN] Loaded on {device}")
    return tokenizer, model, device


def _safe_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def build_prompt_from_record(rec: dict) -> str:
    """
    Rebuilds the same style of prompt used during training.
    """
    ticker = rec.get("Ticker", "UNKNOWN")
    date = rec.get("Date", "N/A")  # ms timestamp, same as training after cleaning

    current_price = _safe_float(rec.get("current_price"))
    avg_open_3 = _safe_float(rec.get("avg_open_past_3_days"))
    avg_close_3 = _safe_float(rec.get("avg_close_past_3_days"))
    avg_low_30 = _safe_float(rec.get("avg_low_30"))
    avg_high_30 = _safe_float(rec.get("avg_high_30"))
    vol_30 = _safe_float(rec.get("volatility_30"))
    rsi14 = _safe_float(rec.get("RSI14"))
    macd = _safe_float(rec.get("MACD"))
    momentum = _safe_float(rec.get("Momentum"))
    mkt_trend = rec.get("MarketTrend", "Unknown")

    return (
        f"TICKER: {ticker}\n"
        f"DATE: {date}\n"
        f"CURRENT PRICE: {current_price:.2f}\n"
        f"AVG OPEN (3D): {avg_open_3:.2f}\n"
        f"AVG CLOSE (3D): {avg_close_3:.2f}\n"
        f"AVG LOW (30D): {avg_low_30:.2f}\n"
        f"AVG HIGH (30D): {avg_high_30:.2f}\n"
        f"VOLATILITY (30D): {vol_30:.4f}\n"
        f"RSI14: {rsi14:.2f}\n"
        f"MACD: {macd:.3f}\n"
        f"Momentum: {momentum:.2f}\n"
        f"Market Trend: {mkt_trend}"
    )


def _load_llm_today_records(llm_data_path: str) -> List[dict]:
    path = Path(llm_data_path)
    if not path.exists():
        raise FileNotFoundError(f"{llm_data_path} not found")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def rank_with_brain(
    llm_data_path: str = "data/LLM_today_data.jsonl",
    top_k: int = 10,
    batch_size: int = 8,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Run the Brain model on LLM_today_data and return:
      - ordered list of top_k tickers
      - dict {ticker: brain_score_raw}
    """
    records = _load_llm_today_records(llm_data_path)
    if not records:
        raise RuntimeError("No records in LLM_today_data.jsonl")

    tokenizer, model, device = _load_model()

    prompts: List[str] = []
    tickers: List[str] = []
    for rec in records:
        prompts.append(build_prompt_from_record(rec))
        tickers.append(rec.get("Ticker", "UNKNOWN"))

    scores: List[float] = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits  # [B, 1] for regression
            batch_scores = logits.squeeze(-1).detach().cpu().numpy().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

    if len(scores) != len(tickers):
        raise RuntimeError("Mismatch between scores and tickers length")

    # Map ticker -> score (if duplicates, keep max)
    ticker_score: Dict[str, float] = {}
    for t, s in zip(tickers, scores):
        s_float = float(s)
        if t not in ticker_score or s_float > ticker_score[t]:
            ticker_score[t] = s_float

    # Sort by score desc
    ordered = sorted(ticker_score.items(), key=lambda kv: kv[1], reverse=True)
    top_k = min(top_k, len(ordered))
    top_tickers = [t for (t, _) in ordered[:top_k]]
    top_scores = {t: ticker_score[t] for t in top_tickers}

    print("[BRAIN] Top tickers by Brain score:")
    for rank, t in enumerate(top_tickers, start=1):
        print(f"  #{rank}: {t} → {top_scores[t]:.2f}")

    return top_tickers, top_scores


if __name__ == "__main__":
    rank_with_brain()
