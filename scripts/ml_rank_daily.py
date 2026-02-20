from __future__ import annotations

import argparse
import json
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


def load_lgb_model(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/daily_scored.parquet")
    parser.add_argument("--keep", default="data/daily_keep_7/keep.parquet")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="data/top10_ml.csv")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--knobs", default="knobs/ml_knobs.json")
    args = parser.parse_args()

    model_type = "lightgbm"
    features = FEATURES
    model_path = args.model
    knobs_path = Path(args.knobs)
    if knobs_path.exists():
        try:
            knobs = json.loads(knobs_path.read_text(encoding="utf-8"))
            model_type = str(knobs.get("model_type", model_type)).lower()
            features = knobs.get("feature_cols", features)
            if not model_path:
                model_path = str(knobs.get("model_path", model_path))
            if args.topk is None and "topk" in knobs:
                args.topk = int(knobs.get("topk", 10))
        except Exception:
            pass

    if not model_path:
        model_path = "scripts/train_model/LLM_bot/Brain/xgb_rank_fwd_end_6m.json"
    if args.topk is None:
        args.topk = 10

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
    for c in features:
        if c not in X.columns:
            X[c] = 0.0
    X = X[features]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    if model_type in {"xgb", "xgboost"}:
        model_pair = load_xgb_model(Path(model_path))
        pred = predict_xgb(model_pair, X_imp)
    else:
        lgb_model = load_lgb_model(Path(model_path))
        pred = lgb_model.predict(X_imp)
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
