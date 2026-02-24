import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from train_ml_ranker import add_calendar_fwd_max_dd  # noqa: E402
except Exception:
    sys.path.insert(0, r"D:\git repositorys\GPT5_sandbox\scripts")
    from train_ml_ranker import add_calendar_fwd_max_dd  # noqa: E402


def _load_score_calibration(path: Path) -> dict | None:
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
    p50 = data.get("p50")
    if not isinstance(p50, list) or len(p50) != bins:
        p50 = None
    thresholds = data.get("thresholds", {})
    anchors = data.get("anchors", {})
    return {
        "bins": bins,
        "pred_min": float(data.get("pred_min", 0.0)),
        "pred_max": float(data.get("pred_max", 1.0)),
        "mu_adj": np.array(mu_adj, dtype=float),
        "p50": None if p50 is None else np.array(p50, dtype=float),
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
            "return_800": float(anchors.get("return_800", 0.50)),
            "return_950": float(anchors.get("return_950", 1.50)),
        },
    }


def _compute_gpt_score(pred: np.ndarray, calib: dict) -> np.ndarray:
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
    mu_adj = calib["mu_adj"]
    p50 = calib.get("p50")
    if isinstance(p50, np.ndarray) and len(p50) == len(mu_adj):
        mu = p50[idx]
        mu = np.where(np.isfinite(mu), mu, mu_adj[idx])
    else:
        mu = mu_adj[idx]
    t = calib["thresholds"]
    score_heavy = float(t["score_heavy"])
    score_max = float(t["score_max"])
    mu_perfect = float(calib["mu_perfect"])
    a = calib["anchors"]
    r600 = float(a["return_600"])
    r800 = float(a["return_800"])
    r950 = float(a["return_950"])

    score = np.zeros_like(mu, dtype=float)
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
    base = Path(r"D:\git repositorys\GPT5_sandbox\data-15 years")
    data_path = base / "ohlcv_2010_2025_clean_scored.parquet"
    keep_path = base / "keep" / "best_locked" / "keep.parquet"

    knobs_path = Path(r"D:\git repositorys\GPT5\knobs\ml_knobs.json")
    calib_path = Path(r"D:\git repositorys\GPT5\knobs\score_calibration.json")

    knobs = json.loads(knobs_path.read_text())
    feats = knobs.get("feature_cols") or []
    feats = list(feats)

    base_cols = {
        "ticker",
        "date",
        "close",
        "adj_close",
        "market_trend_z",
        "market_regime",
        "TailwindScore",
        "z_mom_6m",
        "z_RS_spy_slope",
        "z_vol_spike_20",
    }
    cols = set(feats) | base_cols
    existing_cols = None
    try:
        import pyarrow.parquet as pq

        existing_cols = set(pq.ParquetFile(data_path).schema.names)
    except Exception:
        existing_cols = None
    if existing_cols:
        cols = [c for c in cols if c in existing_cols]
    else:
        cols = list(cols)

    df = pd.read_parquet(data_path, columns=cols)
    keep = pd.read_parquet(keep_path)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    keep["ticker"] = keep["ticker"].astype(str).str.strip()
    keep_tickers = set(keep["ticker"].unique())
    df = df[df["ticker"].isin(keep_tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])

    price_cols = ["ticker", "date", "close"]
    if "adj_close" in df.columns:
        price_cols.append("adj_close")
    price_df = df[price_cols].copy()
    fwd = add_calendar_fwd_max_dd(price_df, months=6, max_dd=0.5, hold_days=5)
    price_df["fwd_max_6m_dd"] = fwd
    df = df.merge(
        price_df[["ticker", "date", "fwd_max_6m_dd"]],
        on=["ticker", "date"],
        how="left",
    )

    if "combo_mom_trend" in feats and "combo_mom_trend" not in df.columns:
        if "z_mom_6m" in df.columns and "market_trend_z" in df.columns:
            df["combo_mom_trend"] = df["z_mom_6m"] * df["market_trend_z"]
    if "combo_rs_trend" in feats and "combo_rs_trend" not in df.columns:
        if "z_RS_spy_slope" in df.columns and "market_trend_z" in df.columns:
            df["combo_rs_trend"] = df["z_RS_spy_slope"] * df["market_trend_z"]
    if "combo_vol_trend" in feats and "combo_vol_trend" not in df.columns:
        if "z_vol_spike_20" in df.columns and "market_trend_z" in df.columns:
            df["combo_vol_trend"] = df["z_vol_spike_20"] * df["market_trend_z"]
    if "combo_tailwind_mom" in feats and "combo_tailwind_mom" not in df.columns:
        if "TailwindScore" in df.columns and "z_mom_6m" in df.columns:
            df["combo_tailwind_mom"] = df["TailwindScore"] * df["z_mom_6m"]
    if "combo_trend_regime" in feats and "combo_trend_regime" not in df.columns:
        if "market_trend_z" in df.columns and "market_regime" in df.columns:
            df["combo_trend_regime"] = df["market_trend_z"] * df["market_regime"]

    for f in feats:
        if f not in df.columns:
            df[f] = np.nan

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    model_path = str(knobs.get("model_path", ""))
    if model_path:
        mp = Path(model_path)
        if not mp.is_absolute():
            model_path = str(Path(r"D:\git repositorys\GPT5") / mp)
    model = lgb.Booster(model_file=str(model_path))
    pred = model.predict(X)

    calib = _load_score_calibration(calib_path)
    if calib is None:
        raise SystemExit("Invalid score_calibration.json")
    gpt_score = _compute_gpt_score(pred, calib)

    actual = df["fwd_max_6m_dd"].to_numpy()
    mask = np.isfinite(actual) & np.isfinite(gpt_score)
    gpt_score = gpt_score[mask]
    actual = actual[mask]

    score_bins = 100
    idx = (gpt_score / 1000.0 * score_bins).astype(int)
    idx = np.clip(idx, 0, score_bins - 1)

    q20 = np.full(score_bins, np.nan, dtype=float)
    q50 = np.full(score_bins, np.nan, dtype=float)
    q80 = np.full(score_bins, np.nan, dtype=float)
    mean = np.full(score_bins, np.nan, dtype=float)

    groups = pd.DataFrame({"idx": idx, "actual": actual}).groupby("idx")["actual"]
    qs = groups.quantile([0.2, 0.5, 0.8]).unstack()
    mu = groups.mean()

    for i in qs.index:
        q20[int(i)] = float(qs.loc[i, 0.2])
        q50[int(i)] = float(qs.loc[i, 0.5])
        q80[int(i)] = float(qs.loc[i, 0.8])
    for i in mu.index:
        mean[int(i)] = float(mu.loc[i])

    data = json.loads(calib_path.read_text(encoding="utf-8"))
    data["score_bins"] = score_bins
    data["score_min"] = 0
    data["score_max"] = 1000
    data["score_p20"] = [None if not np.isfinite(v) else float(v) for v in q20]
    data["score_p50"] = [None if not np.isfinite(v) else float(v) for v in q50]
    data["score_p80"] = [None if not np.isfinite(v) else float(v) for v in q80]
    data["score_mean"] = [None if not np.isfinite(v) else float(v) for v in mean]
    calib_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote score-based calibration to {calib_path}")


if __name__ == "__main__":
    main()
