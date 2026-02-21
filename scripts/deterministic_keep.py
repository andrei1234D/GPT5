from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_features_weights(features: str | list[str], weights: str | list[float]) -> tuple[list[str], list[float]]:
    if isinstance(features, list):
        feats = [str(f).strip() for f in features if str(f).strip()]
    else:
        feats = [f.strip() for f in str(features).split(",") if f.strip()]
    if not feats:
        raise ValueError("--features must include at least one column")
    if isinstance(weights, list):
        w = [float(x) for x in weights]
    elif weights:
        w = [float(x.strip()) for x in str(weights).split(",") if x.strip()]
    else:
        w = []
    if w and len(w) != len(feats):
        raise ValueError("--weights length must match --features length")
    if not w:
        w = [1.0] * len(feats)
    return feats, w


def parse_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def build_keep_df(
    df: pd.DataFrame,
    *,
    features: list[str],
    weights: list[float],
    raw_features: set[str],
    keep_pct: float,
    min_date_count: int,
    score_mode: str,
    higher_is_better: bool,
    min_sustain_all: float,
    min_sustain_20_50: float,
    min_sustain_50_200: float,
    min_sustain_price_200: float,
    min_adv20: float,
    market_trend_col: str,
    market_trend_keep_scale: float,
    market_regime_col: str,
    market_bull_scale: float,
    market_bear_scale: float,
    breadth_col: str,
    breadth_mode: str,
    breadth_keep_scale: float,
    breadth_center: float,
    min_keep_pct: float,
    max_keep_pct: float,
    drop_missing_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if min_sustain_all > 0 and "sustain_all" in df.columns:
        df = df[df["sustain_all"] >= min_sustain_all]
    if min_sustain_20_50 > 0 and "sustain_20_50" in df.columns:
        df = df[df["sustain_20_50"] >= min_sustain_20_50]
    if min_sustain_50_200 > 0 and "sustain_50_200" in df.columns:
        df = df[df["sustain_50_200"] >= min_sustain_50_200]
    if min_sustain_price_200 > 0 and "sustain_price_200" in df.columns:
        df = df[df["sustain_price_200"] >= min_sustain_price_200]
    if min_adv20 > 0 and "adv_20" in df.columns:
        df = df[pd.to_numeric(df["adv_20"], errors="coerce") >= min_adv20]

    comp = pd.Series(0.0, index=df.index)
    for feat, w in zip(features, weights):
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        if feat not in raw_features:
            if score_mode == "date_z":
                mean = s.groupby(df["date"]).transform("mean")
                std = s.groupby(df["date"]).transform("std", ddof=0).replace(0, np.nan)
                s = (s - mean) / std
            elif score_mode == "ticker_z":
                mean = s.groupby(df["ticker"]).transform("mean")
                std = s.groupby(df["ticker"]).transform("std", ddof=0).replace(0, np.nan)
                s = (s - mean) / std
        s = s.fillna(0.0)
        comp = comp + w * s

    df["composite"] = comp
    if drop_missing_features:
        df = df.dropna(subset=features)
    df = df.dropna(subset=["composite"])
    if df.empty:
        raise ValueError("No rows after composite calculation.")

    date_counts = df["date"].value_counts()
    valid_dates = date_counts[date_counts >= min_date_count].index
    df = df[df["date"].isin(valid_dates)].copy()
    if df.empty:
        raise ValueError("No rows after min-date-count filter.")

    keep_rows: list[pd.DataFrame] = []
    for _, g in df.groupby("date"):
        g = g.copy()
        ascending = not higher_is_better
        g["rank_pct"] = g["composite"].rank(method="first", ascending=ascending, pct=True)
        adj_keep_pct = keep_pct
        if market_trend_keep_scale != 0 and market_trend_col in g.columns:
            trend_val = pd.to_numeric(g[market_trend_col].iloc[0], errors="coerce")
            regime_val = None
            if market_regime_col and market_regime_col in g.columns:
                regime_val = pd.to_numeric(g[market_regime_col].iloc[0], errors="coerce")
            if np.isfinite(trend_val):
                if regime_val == 1:
                    adj_keep_pct = keep_pct * (1 + market_bull_scale * max(trend_val, 0))
                elif regime_val == -1:
                    adj_keep_pct = keep_pct * (1 - market_bear_scale * abs(min(trend_val, 0)))
                else:
                    adj_keep_pct = keep_pct * (1 + market_trend_keep_scale * trend_val)
                adj_keep_pct = max(min_keep_pct, min(max_keep_pct, adj_keep_pct))
        if breadth_keep_scale != 0 and breadth_col in g.columns:
            breadth_vals = pd.to_numeric(g[breadth_col], errors="coerce")
            if breadth_mode == "pos_frac":
                breadth = float((breadth_vals > 0).mean())
            else:
                breadth = float(breadth_vals.mean())
            if np.isfinite(breadth):
                adj_keep_pct = adj_keep_pct * (1 + breadth_keep_scale * (breadth - breadth_center))
                adj_keep_pct = max(min_keep_pct, min(max_keep_pct, adj_keep_pct))
        kept = g[g["rank_pct"] <= adj_keep_pct]

        kept_out = kept[["date", "ticker", "composite", "rank_pct"]].copy()
        kept_out["keep_pct"] = adj_keep_pct
        keep_rows.append(kept_out)

    keep_df = pd.concat(keep_rows, ignore_index=True)
    return keep_df, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/daily_scored.parquet")
    parser.add_argument("--out", default="data/daily_keep_10/keep.parquet")
    parser.add_argument("--knobs", default="knobs/keep_best_locked.json")
    parser.add_argument("--keep-pct", type=float, default=0.07)
    parser.add_argument("--min-date-count", type=int, default=0)
    parser.add_argument("--direction", choices=["asc", "desc"], default="asc")
    args = parser.parse_args()

    knobs_path = Path(args.knobs)
    if not knobs_path.exists():
        raise SystemExit(f"Knobs not found: {knobs_path}")
    knobs = json.loads(knobs_path.read_text(encoding="utf-8"))

    features, weights = parse_features_weights(knobs.get("features", []), knobs.get("weights", []))
    raw_features = set(parse_list(knobs.get("raw_features", [])))

    keep_pct = float(knobs.get("keep_pct", args.keep_pct))
    if keep_pct <= 0 or keep_pct >= 1:
        raise ValueError("--keep-pct must be between 0 and 1 (exclusive).")
    min_date_count = int(knobs.get("min_date_count", 50))
    if args.min_date_count and int(args.min_date_count) > 0:
        min_date_count = int(args.min_date_count)

    bull_scale = knobs.get("market_bull_scale", None)
    if bull_scale is None:
        bull_scale = knobs.get("market_trend_keep_scale", 0.2)
    bear_scale = knobs.get("market_bear_scale", None)
    if bear_scale is None:
        bear_scale = float(knobs.get("market_trend_keep_scale", 0.2)) * 1.5

    df = pd.read_parquet(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    keep_df, _ = build_keep_df(
        df,
        features=features,
        weights=weights,
        raw_features=raw_features,
        keep_pct=keep_pct,
        min_date_count=min_date_count,
        score_mode=str(knobs.get("score_mode", "date_z")),
        higher_is_better=bool(knobs.get("higher_is_better", True)),
        min_sustain_all=float(knobs.get("min_sustain_all", 0.0)),
        min_sustain_20_50=float(knobs.get("min_sustain_20_50", 0.0)),
        min_sustain_50_200=float(knobs.get("min_sustain_50_200", 0.0)),
        min_sustain_price_200=float(knobs.get("min_sustain_price_200", 0.0)),
        min_adv20=float(knobs.get("min_adv20", 0.0)),
        market_trend_col=str(knobs.get("market_trend_col", "market_trend_z")),
        market_trend_keep_scale=float(knobs.get("market_trend_keep_scale", 0.2)),
        market_regime_col=str(knobs.get("market_regime_col", "market_regime")),
        market_bull_scale=float(bull_scale),
        market_bear_scale=float(bear_scale),
        breadth_col=str(knobs.get("breadth_col", "mom_3m")),
        breadth_mode=str(knobs.get("breadth_mode", "pos_frac")),
        breadth_keep_scale=float(knobs.get("breadth_keep_scale", 0.3)),
        breadth_center=float(knobs.get("breadth_center", 0.5)),
        min_keep_pct=float(knobs.get("min_keep_pct", 0.02)),
        max_keep_pct=float(knobs.get("max_keep_pct", 0.2)),
        drop_missing_features=bool(knobs.get("drop_missing_features", False)),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} rows={len(keep_df)} keep_pct={keep_pct:.4f}")


if __name__ == "__main__":
    main()
