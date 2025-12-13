# scripts/brain_ranker.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Your runtime classes live here (and are required for inference)
from scorebot_runtime import ScoreBotSlim, CalibratedScoreBotSlim

# Optional: build today data if file missing
from llm_data_builder import build_llm_today_data


# -----------------------------
# Paths
# -----------------------------
SCRIPTS_DIR = Path(__file__).parent
SCOREBOT_PATH = (SCRIPTS_DIR / "train_model" / "LLM_bot" / "Brain" / "pseudo_score_mapping" / "scorebot_multiclass_top1_CALIBRATED_SLIM.pkl")

LLM_TODAY_JSONL = Path("data/LLM_today_data.jsonl")
OUT_JSONL = Path("data/LLM_today_scores.jsonl")
OUT_CSV = Path("data/brain_ranked_scores.csv")


# -----------------------------
# Feature list (authoritative)
# -----------------------------
REQUIRED_FEATURES = [
    "Volatility_30D", "Volatility_252D", "High_52W", "Low_52W", "High_30D", "Low_30D",
    "Momentum_63D", "Momentum_126D", "Momentum_252D",
    "AMT", "SMC", "TSS", "RMI", "ABS", "VAM", "RSE", "CBP",
    "SMA_Slope_3M",
    "Ret_5D", "Ret_10D",
    "pos_52w", "pos_30d",
    "Volume_SMA20", "Volume_Trend",
    "RSI_14",
    "Month",
    "Momentum_63D_cs_z", "Momentum_63D_cs_rank",
    "Momentum_126D_cs_z", "Momentum_126D_cs_rank",
    "Momentum_252D_cs_z", "Momentum_252D_cs_rank",
    "Volatility_30D_cs_z", "Volatility_30D_cs_rank",
    "Volatility_252D_cs_z", "Volatility_252D_cs_rank",
    "RSE_cs_z", "RSE_cs_rank",
    "CBP_cs_z", "CBP_cs_rank",
    "SMC_cs_z", "SMC_cs_rank",
    "TSS_cs_z", "TSS_cs_rank",
    "VAM_cs_z", "VAM_cs_rank",
    "ABS_cs_z", "ABS_cs_rank",
    "AMT_cs_z", "AMT_cs_rank",
    "pos_52w_cs_z", "pos_52w_cs_rank",
    "pos_30d_cs_z", "pos_30d_cs_rank",
    "avg_close_past_3_days",
    "avg_volatility_30D",
    "current_price",
]


def _load_scorebot(path: Path):
    """
    Handles the common failure mode where the pickle was created in Colab (__main__).
    We map those class names to the runtime classes in scorebot_runtime.
    """
    if not path.exists():
        raise FileNotFoundError(f"ScoreBot pickle not found: {path}")

    # Map __main__.ScoreBotSlim / __main__.CalibratedScoreBotSlim to our runtime classes
    main_mod = sys.modules.get("__main__")
    setattr(main_mod, "ScoreBotSlim", ScoreBotSlim)
    setattr(main_mod, "CalibratedScoreBotSlim", CalibratedScoreBotSlim)
    # Some earlier notebooks used these names; map them too (harmless if unused)
    setattr(main_mod, "CalibratedScoreBot", CalibratedScoreBotSlim)

    payload = joblib.load(str(path))
    bot = payload.get("bot")
    if bot is None:
        raise RuntimeError("Pickle payload missing 'bot' key.")

    feature_cols = payload.get("feature_cols") or getattr(bot, "feature_cols", None)
    if not feature_cols:
        raise RuntimeError("Pickle missing feature_cols and bot has no feature_cols.")

    return bot, list(feature_cols), payload


def _read_jsonl(path: Path) -> List[dict]:
    recs: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def rank_with_brain(
    stage2_path: str = "data/stage2_merged.csv",
    llm_data_path: str = "data/LLM_today_data.jsonl",
    top_k: int = 10,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Scores your candidate tickers using ScoreBotSlim and returns top_k tickers + pred_scores.
    Also writes:
      - data/LLM_today_scores.jsonl
      - data/brain_ranked_scores.csv
    """
    # Ensure today data exists
    llm_path = Path(llm_data_path)
    if not llm_path.exists():
        build_llm_today_data(stage2_path=stage2_path, out_path=str(llm_path), top_n=max(50, top_k * 5))

    bot, feature_cols, _payload = _load_scorebot(SCOREBOT_PATH)

    records = _read_jsonl(llm_path)
    if not records:
        raise RuntimeError(f"No records in {llm_path}")

    df = pd.DataFrame(records)

    # Ensure required features exist (fill missing with NaN)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # In production, fail fast: feature mismatch = wrong builder
        raise RuntimeError(f"LLM_today_data missing required features: {missing}")

    # Predict calibrated score in [0,1000]
    preds = bot.predict_df(df).values
    df["pred_score"] = np.clip(np.asarray(preds, dtype=float), 0.0, 1000.0)

    # Aggregate if duplicates: keep max score per ticker
    df["Ticker"] = df["Ticker"].astype(str)
    agg = df.groupby("Ticker", as_index=False)["pred_score"].max()
    agg = agg.sort_values("pred_score", ascending=False)

    top_k = min(top_k, len(agg))
    top_df = agg.head(top_k).copy()

    top_tickers = top_df["Ticker"].tolist()
    top_scores = {r["Ticker"]: float(r["pred_score"]) for _, r in top_df.iterrows()}

    print("[BRAIN] Top tickers by pred_score:")
    for i, t in enumerate(top_tickers, start=1):
        print(f"  #{i}: {t} → {top_scores[t]:.2f}")

    # Write full ordered list
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = [{"Ticker": r["Ticker"], "pred_score": float(r["pred_score"])} for _, r in agg.iterrows()]

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    import csv
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ticker", "pred_score"])
        for r in rows:
            w.writerow([r["Ticker"], f"{r['pred_score']:.6f}"])

    print("[BRAIN] Wrote ranked scores to:")
    print(f"       - {OUT_JSONL}")
    print(f"       - {OUT_CSV}")

    return top_tickers, top_scores


if __name__ == "__main__":
    rank_with_brain()
