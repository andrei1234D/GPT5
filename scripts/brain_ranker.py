# scripts/brain_ranker.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from scorebot_runtime import ScoreBotSlim, CalibratedScoreBotSlim
from llm_data_builder import build_llm_today_data


SCRIPTS_DIR = Path(__file__).resolve().parent
SCOREBOT_PATH = (
    SCRIPTS_DIR
    / "train_model"
    / "LLM_bot"
    / "Brain"
    / "pseudo_score_mapping"
    / "scorebot_multiclass_top1_CALIBRATED_SLIM.pkl"
)

OUT_JSONL = Path("data/LLM_today_scores.jsonl")
OUT_CSV = Path("data/brain_ranked_scores.csv")


def _load_scorebot(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"ScoreBot pickle not found: {path}")

    # Map notebook-defined classes to runtime equivalents
    main_mod = sys.modules.get("__main__")
    setattr(main_mod, "ScoreBotSlim", ScoreBotSlim)
    setattr(main_mod, "CalibratedScoreBotSlim", CalibratedScoreBotSlim)
    setattr(main_mod, "CalibratedScoreBot", CalibratedScoreBotSlim)

    payload = joblib.load(str(path))
    bot = payload.get("bot")
    if bot is None:
        raise RuntimeError("Pickle payload missing 'bot' key")

    feature_cols = payload.get("feature_cols") or getattr(bot, "feature_cols", None)
    if not feature_cols:
        raise RuntimeError("Missing feature_cols in pickle and bot")

    # IMPORTANT: if base_bot.models contains string paths, resolve relative to the pickle folder
    base = getattr(bot, "base_bot", None)
    if base is not None and hasattr(base, "model_root"):
        base.model_root = path.parent

    return bot, list(feature_cols)


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rank_with_brain(
    stage2_path: str = "data/stage2_merged.csv",
    llm_data_path: str = "data/LLM_today_data.jsonl",
    top_k: int = 10,
) -> Tuple[List[str], Dict[str, float]]:
    llm_path = Path(llm_data_path)
    if not llm_path.exists():
        build_llm_today_data(stage2_path=stage2_path, out_path=str(llm_path), top_n=max(50, top_k * 5))

    bot, feature_cols = _load_scorebot(SCOREBOT_PATH)

    records = _read_jsonl(llm_path)
    if not records:
        raise RuntimeError(f"No records in {llm_path}")

    df = pd.DataFrame(records)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"LLM_today_data missing required features: {missing}")

    preds = bot.predict_df(df).values
    df["pred_score"] = np.clip(preds.astype(float), 0.0, 1000.0)

    df["Ticker"] = df["Ticker"].astype(str)
    ranked = (
        df.groupby("Ticker", as_index=False)["pred_score"]
        .max()
        .sort_values("pred_score", ascending=False)
    )

    top_k = min(int(top_k), len(ranked))
    top_df = ranked.head(top_k)

    top_tickers = top_df["Ticker"].tolist()
    top_scores = {row["Ticker"]: float(row["pred_score"]) for _, row in top_df.iterrows()}

    print(f"[BRAIN] Top {top_k} tickers by pred_score:")
    for i, t in enumerate(top_tickers, start=1):
        print(f"  #{i}: {t} → {top_scores[t]:.2f}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for _, r in ranked.iterrows():
            f.write(json.dumps({"Ticker": r["Ticker"], "pred_score": float(r["pred_score"])}, separators=(",", ":")) + "\n")

    import csv
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ticker", "pred_score"])
        for _, r in ranked.iterrows():
            w.writerow([r["Ticker"], f"{r['pred_score']:.6f}"])

    print("[BRAIN] Outputs written:")
    print(f"  - {OUT_JSONL}")
    print(f"  - {OUT_CSV}")

    return top_tickers, top_scores


if __name__ == "__main__":
    rank_with_brain(top_k=10)
