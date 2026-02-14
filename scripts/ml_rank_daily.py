from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


FEATURES = [
    "QMI",
    "RLI",
    "STSI",
    "BRI",
    "TailwindScore",
    "TotalScore",
    "z_mom_6m",
    "z_mom_3m",
    "z_ema_stack",
    "z_ema50_slope",
    "z_vol_60",
    "z_mdd_3m",
    "z_sustain",
    "z_log_adv",
    "z_prox_high",
    "z_RS_spy_slope",
    "z_RS_sector_slope",
    "ema_stack",
    "ema50_slope",
    "RS_spy_slope90",
    "RS_sector_slope90",
]


def load_xgb_model(path: Path):
    import xgboost as xgb

    try:
        model = xgb.XGBRanker()
        model.load_model(str(path))
        return ("sk", model)
    except Exception:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return ("booster", booster)


def predict_xgb(model_pair, X: np.ndarray):
    kind, model = model_pair
    if kind == "sk":
        return model.predict(X)
    import xgboost as xgb

    dmat = xgb.DMatrix(X)
    return model.predict(dmat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/daily_scored.parquet")
    parser.add_argument("--keep", default="data/daily_keep_7/keep.parquet")
    parser.add_argument(
        "--model",
        default="scripts/train_model/LLM_bot/Brain/xgb_rank_fwd_end_6m.json",
    )
    parser.add_argument("--out", default="data/top10_ml.csv")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    data_df = pd.read_parquet(args.data)
    keep_df = pd.read_parquet(args.keep, columns=["date", "ticker"])
    keep_df["date"] = pd.to_datetime(keep_df["date"])
    keep_df["ticker"] = keep_df["ticker"].astype(str).str.strip()

    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df["ticker"] = data_df["ticker"].astype(str).str.strip()

    df = data_df.merge(keep_df, on=["date", "ticker"], how="inner")
    if df.empty:
        raise SystemExit("No rows after keep filter.")

    X = df.copy()
    for c in FEATURES:
        if c not in X.columns:
            X[c] = 0.0
    X = X[FEATURES]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model_pair = load_xgb_model(Path(args.model))
    pred = predict_xgb(model_pair, X_imp)
    df["pred_score"] = pred

    df = df.sort_values("pred_score", ascending=False).head(int(args.topk)).copy()
    df["rank"] = np.arange(1, len(df) + 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["date", "ticker", "company", "pred_score", "rank"] if c in df.columns]
    df[cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
