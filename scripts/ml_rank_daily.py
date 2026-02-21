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
    "z_vol_20",
    "z_vol_60",
    "z_vol_120",
    "z_vol_ratio_20_120",
    "z_vol_of_vol_20",
    "z_mdd_3m",
    "z_sustain",
    "z_log_adv",
    "z_log_adv_60",
    "z_adv_ratio_20_60",
    "z_prox_high",
    "z_vol_spike_20",
    "z_gap_up",
    "z_gap_up_on_vol",
    "z_pullback_quality",
    "z_RS_spy_slope",
    "z_RS_sector_slope",
    "market_trend",
    "market_trend_z",
    "market_regime",
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


def load_score_calibration(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bins = int(data.get("bins", 0))
    mu_adj = data.get("mu_adj")
    if not bins or not isinstance(mu_adj, list) or len(mu_adj) != bins:
        return None
    thresholds = data.get("thresholds", {})
    anchors = data.get("anchors", {})
    return {
        "bins": bins,
        "pred_min": float(data.get("pred_min", 0.0)),
        "pred_max": float(data.get("pred_max", 1.0)),
        "mu_adj": np.array(mu_adj, dtype=float),
        "mu_perfect": float(data.get("mu_perfect", thresholds.get("heavy_return", 1.5))),
        "thresholds": {
            "low_return": float(thresholds.get("low_return", 0.60)),
            "heavy_return": float(thresholds.get("heavy_return", 1.50)),
            "score_low": int(thresholds.get("score_low", 800)),
            "score_heavy": int(thresholds.get("score_heavy", 950)),
            "score_max": int(thresholds.get("score_max", 1000)),
        },
        "anchors": {
            "return_600": float(anchors.get("return_600", 0.20)),
            "return_800": float(anchors.get("return_800", 0.60)),
            "return_950": float(anchors.get("return_950", 1.50)),
        },
    }


def compute_gpt_score(pred: np.ndarray, calib: dict) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if pred.size == 0:
        return np.array([], dtype=int)
    bins = calib["bins"]
    pred_min = calib["pred_min"]
    pred_max = calib["pred_max"]
    if not np.isfinite(pred_min) or not np.isfinite(pred_max) or pred_max <= pred_min:
        return np.zeros(pred.shape, dtype=int)
    idx = ((pred - pred_min) / (pred_max - pred_min) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    mu = calib["mu_adj"][idx]
    t = calib["thresholds"]
    low_ret = float(t["low_return"])
    heavy_ret = float(t["heavy_return"])
    score_low = float(t["score_low"])
    score_heavy = float(t["score_heavy"])
    score_max = float(t["score_max"])
    mu_perfect = float(calib["mu_perfect"])
    a = calib["anchors"]
    r600 = float(a["return_600"])
    r800 = float(a["return_800"])
    r950 = float(a["return_950"])

    score = np.zeros_like(mu, dtype=float)
    # Piecewise anchors: 600->20%, 800->60%, 950->150%, 1000->perfection
    mask_0 = mu <= 0.0
    score[mask_0] = 0.0
    mask_a = (mu > 0.0) & (mu < r600)
    if r600 > 0:
        score[mask_a] = 600.0 * (mu[mask_a] / r600)
    mask_b = (mu >= r600) & (mu < r800)
    if r800 > r600:
        score[mask_b] = 600.0 + 200.0 * ((mu[mask_b] - r600) / (r800 - r600))
    mask_c = (mu >= r800) & (mu < r950)
    if r950 > r800:
        score[mask_c] = 800.0 + 150.0 * ((mu[mask_c] - r800) / (r950 - r800))
    mask_hi = mu >= r950
    if mu_perfect <= r950:
        score[mask_hi] = score_max
    else:
        ratio = (mu[mask_hi] - r950) / (mu_perfect - r950)
        ratio = np.clip(ratio, 0.0, 1.0)
        score[mask_hi] = score_heavy + (score_max - score_heavy) * ratio
    return np.clip(np.rint(score), 0, score_max).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/daily_scored.parquet")
    parser.add_argument("--keep", default="data/daily_keep_10/keep.parquet")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="data/top10_ml.csv")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--knobs", default="knobs/ml_knobs.json")
    parser.add_argument("--score-calib", default="knobs/score_calibration.json")
    args = parser.parse_args()

    model_type = "lightgbm"
    features = FEATURES
    model_path = args.model
    score_calib_path = Path(args.score_calib)
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
            score_calib_path = Path(
                knobs.get("score_calibration_path", score_calib_path)
            )
        except Exception:
            pass

    if not model_path:
        model_path = "scripts/train_model/LLM_bot/Brain/xgb_rank_fwd_end_6m.json"
    if args.topk is None:
        args.topk = 10

    if not score_calib_path.exists():
        alt = Path("live_bundle/ml/score_calibration.json")
        if alt.exists():
            score_calib_path = alt
    score_calib = load_score_calibration(score_calib_path)
    if score_calib is None:
        raise SystemExit(
            f"Score calibration not found or invalid: {score_calib_path}"
        )

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
    df["gpt_score"] = compute_gpt_score(df["pred_score"].to_numpy(), score_calib)

    df = df.sort_values("pred_score", ascending=False).head(int(args.topk)).copy()
    df["rank"] = np.arange(1, len(df) + 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        c
        for c in ["date", "ticker", "company", "pred_score", "gpt_score", "rank"]
        if c in df.columns
    ]
    df[cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
