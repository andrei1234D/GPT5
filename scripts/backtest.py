# scripts/backtest.py
from __future__ import annotations
import argparse, math, datetime as dt
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

from .features import fetch_history, compute_indicators
from .trash_ranker import RobustRanker, HardFilter

# ---------- labeling rules ----------
def forward_label(
    px: pd.Series,
    bench: pd.Series,
    horizon_days: int = 60,
    max_dd_thresh: float = -20.0,     # -20% peak-to-trough within horizon
    rel_underperf: float = -10.0      # underperform SPY by <= -10% over horizon
) -> int:
    """
    Returns 1 if ticker is 'TRASH' at t0 based on forward horizon outcomes, else 0.
    Uses only info available at t0 to decide ex-ante.
    """
    if len(px) < horizon_days + 1 or len(bench) < horizon_days + 1:
        return 0
    # forward period from t0+1 to t0+horizon
    fwd = px.iloc[1:horizon_days+1]
    ret = float(fwd.iloc[-1] / px.iloc[0] - 1.0) * 100.0

    # max drawdown over the horizon (percent)
    cummax = fwd.cummax()
    dd = float(((fwd / cummax) - 1.0).min()) * 100.0  # negative

    # relative to SPY
    bret = float(bench.iloc[horizon_days] / bench.iloc[0] - 1.0) * 100.0
    rel = ret - bret

    is_trash = (dd <= max_dd_thresh) or (rel <= rel_underperf)
    return 1 if is_trash else 0

def _to_yyyymmdd(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")

# ---------- main backtest ----------
def run_backtest(
    universe_csv: str,
    start: str = "2018-01-01",
    end: str = None,
    rebalance: str = "M",         # evaluate monthly
    lookback_days: int = 270,     # for features
    horizon_days: int = 60,       # for labels
    top_keep: int = 10
) -> Dict:
    end = end or _to_yyyymmdd(pd.Timestamp.today())

    uni = pd.read_csv(universe_csv)
    tickers = list(uni["ticker"].astype(str).str.replace(".", "-", regex=False).values)
    # Always include SPY for benchmark
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    hist = yf.download(" ".join(tickers), start=start, end=end, interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)

    # Normalize per ticker
    def _norm_one(t) -> pd.DataFrame:
        try:
            df = hist[t]
            df = df.dropna()
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            # yfinance can vary shapes; try alt forms
            if isinstance(hist, pd.DataFrame) and "Close" in hist.columns:
                return hist.dropna()
            return pd.DataFrame()

    dates = pd.date_range(start=start, end=end, freq=rebalance)
    results = []
    hard = HardFilter()  # start with defaults; we’ll grid-scan later

    for d0 in dates:
        # need sufficient lookback
        t0 = pd.Timestamp(d0)
        t_minus = t0 - pd.Timedelta(days=lookback_days + 10)
        # slice window
        ranked_input = []
        price_map = {}
        for tkr in tickers:
            df = _norm_one(tkr)
            if df.empty: continue
            df = df[df.index <= t0]
            if df.shape[0] < 210:  # ensure SMAs exist
                continue
            df_win = df[df.index >= t_minus]
            try:
                feats = compute_indicators(df_win)
            except Exception:
                continue
            ranked_input.append((tkr, tkr, feats))
            price_map[tkr] = df["Adj Close"]

        if len(ranked_input) < 200:
            continue

        # score with ranker (uses cross-section medians)
        ranker = RobustRanker()
        scored = ranker.score_universe(ranked_input, context=None)

        # ground-truth labels using forward window from t0
        bench = price_map.get("SPY")
        if bench is None or bench[bench.index >= t0].shape[0] < (horizon_days + 1):
            continue

        # mark trash for any ticker with enough forward data
        truth = {}
        for tkr, _, feats, _, _ in scored:
            px = price_map.get(tkr)
            if px is None: continue
            fwd = px[px.index >= t0]
            if fwd.shape[0] >= (horizon_days + 1):
                truth[tkr] = forward_label(fwd, bench[bench.index >= t0], horizon_days=horizon_days)
        if not truth:
            continue

        # Evaluate two things:
        # A) Recall of trash removed by hard filter
        removed = [t for (t,_,f) in ranked_input if hard.is_garbage(f)]
        trash_set = {t for t,lab in truth.items() if lab == 1}
        kept_after_hard = [t for (t,_,f,_,_) in scored]  # these passed hard filters
        if not trash_set:
            continue

        recall_removed = len(set(removed) & trash_set) / max(1, len(trash_set))

        # B) Quality of survivors if we keep top_keep (should not be trash)
        survivors = [t for (t,_,_,_,_) in scored[:top_keep] if t in truth]
        survivor_bad = sum(truth[t] for t in survivors)  # count trash among survivors
        survivors_ok = len(survivors) - survivor_bad

        results.append(dict(
            date=_to_yyyymmdd(t0),
            n_universe=len(ranked_input),
            n_trash=len(trash_set),
            recall_removed=recall_removed,      # target >= 0.80
            survivors=len(survivors),
            survivors_ok=survivors_ok,
            survivors_trash=survivor_bad
        ))

    out = pd.DataFrame(results)
    if out.empty:
        return {"summary": {}, "by_date": out}
    summary = {
        "n_periods": int(out.shape[0]),
        "avg_recall_removed": float(out["recall_removed"].mean()),
        "median_recall_removed": float(out["recall_removed"].median()),
        "avg_survivor_trash_rate": float((out["survivors_trash"] / out["survivors"].replace(0, np.nan)).mean()),
    }
    return {"summary": summary, "by_date": out}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_csv", required=True)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end",   default=None)
    ap.add_argument("--rebalance", default="M")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--top_keep", type=int, default=10)
    args = ap.parse_args()

    res = run_backtest(
        universe_csv=args.universe_csv,
        start=args.start, end=args.end, rebalance=args.rebalance,
        horizon_days=args.horizon, top_keep=args.top_keep
    )
    print("== Summary ==")
    print(res["summary"])
    if isinstance(res["by_date"], pd.DataFrame):
        print(res["by_date"].head())
