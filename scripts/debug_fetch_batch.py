from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local scripts are importable
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from data_fetcher import download_history_cached_dict  # noqa: E402
from ml_rank_daily import FEATURES  # noqa: E402
from build_daily_dataset import (  # noqa: E402
    INDEX_TICKERS,
    SECTOR_ETFS,
    _load_bad_tickers,
    _to_ohlcv,
    add_stock_features,
    build_index_features,
    load_sector_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="data/universe_clean.csv")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--out", default="data/debug_batch.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--history-days", type=int, default=420)
    parser.add_argument("--min-rows", type=int, default=260)
    parser.add_argument("--include-sectors", action="store_true")
    parser.add_argument("--sector-mode", choices=["yfinance", "none"], default="yfinance")
    parser.add_argument("--sector-map", default="data/sector_map.json")
    parser.add_argument("--sector-sleep", type=float, default=0.05)
    parser.add_argument("--bad-tickers", default="data/bad_tickers.csv")
    args = parser.parse_args()

    tickers: list[str]
    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        uni = pd.read_csv(args.universe)
        if "ticker" not in uni.columns:
            raise ValueError("universe file must contain 'ticker'")
        uni["ticker"] = uni["ticker"].astype(str).str.strip()
        tickers = [t for t in uni["ticker"].tolist() if t]

    bad = _load_bad_tickers(args.bad_tickers)
    if bad:
        tickers = [t for t in tickers if t.upper() not in bad]

    tickers = tickers[: max(1, int(args.n))]

    extra = INDEX_TICKERS + (SECTOR_ETFS if args.include_sectors else [])
    all_tickers = sorted(set(tickers) | set(extra))

    period = f"{int(args.history_days)}d"
    hist_map = download_history_cached_dict(
        all_tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    rows = []
    for t in tickers:
        df = hist_map.get(t)
        if df is None or df.empty:
            continue
        ohlcv = _to_ohlcv(df)
        if ohlcv.empty or ohlcv.shape[0] < args.min_rows:
            continue
        ohlcv["ticker"] = t
        rows.append(ohlcv)

    if not rows:
        raise RuntimeError("No usable ticker history found for debug batch.")

    stock_df = pd.concat(rows, ignore_index=True)
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    stock_df = stock_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    sector_map = load_sector_map(tickers, args.sector_map, args.sector_mode, args.sector_sleep)
    index_features = build_index_features(hist_map, SECTOR_ETFS if args.include_sectors else [])

    spy_df = _to_ohlcv(hist_map.get("SPY", pd.DataFrame()))

    sector_frames = []
    for etf in SECTOR_ETFS:
        h = hist_map.get(etf)
        if h is None or h.empty:
            continue
        o = _to_ohlcv(h)
        if o.empty:
            continue
        o["sector_etf"] = etf
        sector_frames.append(o[["date", "sector_etf", "close"]].rename(columns={"close": "sector_close"}))
    sector_close_df = pd.concat(sector_frames, ignore_index=True) if sector_frames else pd.DataFrame()

    df = add_stock_features(stock_df, index_features, spy_df, sector_close_df, sector_map)

    # latest date snapshot
    last_date = df["date"].max()
    df = df[df["date"] == last_date].copy()
    df = df.sort_values("ticker").reset_index(drop=True)

    # ensure all requested tickers are present in output
    present = set(df["ticker"].astype(str).tolist())
    missing = [t for t in tickers if t not in present]
    if missing:
        filler = pd.DataFrame(
            {
                "date": [last_date] * len(missing),
                "ticker": missing,
            }
        )
        df = pd.concat([df, filler], ignore_index=True)

    cols = ["date", "ticker", "close", "volume"] + FEATURES
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out)} date={last_date.date()}")


if __name__ == "__main__":
    main()
