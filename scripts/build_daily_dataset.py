from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import time

import yfinance as yf

from data_fetcher import download_history_cached_dict


INDEX_TICKERS = ["SPY", "^VIX", "^TNX", "^IRX", "^RUT"]
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
SECTOR_TO_ETF = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # normalize columns
    ren = {}
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc == "adj close":
            ren[c] = "adj_close"
        elif lc == "close":
            ren[c] = "close"
        elif lc == "open":
            ren[c] = "open"
        elif lc == "high":
            ren[c] = "high"
        elif lc == "low":
            ren[c] = "low"
        elif lc == "volume":
            ren[c] = "volume"
    if ren:
        out = out.rename(columns=ren)
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index)
    out = out.reset_index()
    if "date" not in out.columns:
        if "Date" in out.columns:
            out = out.rename(columns={"Date": "date"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "date"})
        else:
            # fallback: assume first column is the date-like index
            first = out.columns[0]
            out = out.rename(columns={first: "date"})
    return out


def rolling_z(s: pd.Series, window: int = 252) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    return (s - mean) / std


def slope_log(s: pd.Series, window: int) -> pd.Series:
    return (np.log(s) - np.log(s.shift(window))) / window


def trailing_mdd(s: pd.Series, window: int) -> pd.Series:
    def mdd(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        running_max = np.maximum.accumulate(x)
        drawdown = x / running_max - 1.0
        return np.nanmin(drawdown)

    return s.rolling(window=window, min_periods=window).apply(mdd, raw=True)


def load_sector_map(tickers: List[str], path: str, mode: str, sleep_s: float) -> dict[str, str]:
    if mode == "none":
        return {}

    p = Path(path)
    try:
        cached = pd.read_json(p, typ="series").to_dict()
    except Exception:
        cached = {}

    updated = dict(cached)
    to_fetch = [t for t in tickers if t not in updated]

    if to_fetch:
        for t in to_fetch:
            try:
                info = yf.Ticker(t).get_info()
                sector = info.get("sector")
                if isinstance(sector, str) and sector.strip():
                    updated[t] = sector.strip()
            except Exception:
                pass
            if sleep_s:
                time.sleep(sleep_s)

        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            pd.Series(updated).to_json(p)
        except Exception:
            pass

    return updated


def _load_bad_tickers(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p)
        if "ticker" in df.columns:
            return set(df["ticker"].astype(str).str.strip().str.upper().tolist())
    except Exception:
        pass
    try:
        txt = p.read_text(encoding="utf-8")
        return set([t.strip().upper() for t in txt.splitlines() if t.strip()])
    except Exception:
        return set()


def _append_bad_tickers(path: str, tickers: List[str]) -> None:
    if not tickers:
        return
    p = Path(path)
    existing = _load_bad_tickers(path)
    new = set([t.strip().upper() for t in tickers if t and t.strip()])
    merged = sorted(existing | new)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": merged}).to_csv(p, index=False)

def build_index_features(hist_map: Dict[str, pd.DataFrame], sector_etfs: List[str]) -> pd.DataFrame:
    def get_series(t: str) -> pd.Series:
        df = hist_map.get(t)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        norm = _to_ohlcv(df)
        if norm.empty:
            return pd.Series(dtype=float)
        if "adj_close" in norm.columns:
            s = norm.set_index("date")["adj_close"]
        else:
            s = norm.set_index("date")["close"]
        return pd.to_numeric(s, errors="coerce")

    spx = get_series("SPY")
    vix = get_series("^VIX")
    tnx = get_series("^TNX")
    irx = get_series("^IRX")
    rut = get_series("^RUT")

    if spx.empty:
        return pd.DataFrame(columns=["date", "MRI", "VCI", "RORO", "YCSI", "STI_SPY"])

    spx_trend = slope_log(spx, 90)
    vix_risk = (vix - vix.rolling(60).mean()) / vix.rolling(60).mean()
    rate_shock = tnx - tnx.shift(20)

    spx_trend_z = rolling_z(spx_trend)
    vix_risk_z = rolling_z(vix_risk)
    rate_shock_z = rolling_z(rate_shock)

    mri = spx_trend_z - 0.8 * vix_risk_z - 0.3 * rate_shock_z
    vci = -vix_risk_z

    roro = rolling_z(slope_log(spx / rut.replace(0, np.nan), 60))
    ycsi = rolling_z(tnx - irx)

    def _sti(series: pd.Series) -> pd.Series:
        if series is None or series.empty:
            return pd.Series(dtype=float)
        mom_63 = series / series.shift(63) - 1.0
        slope_90 = slope_log(series, 90)
        vol_60 = np.log(series / series.shift(1)).rolling(60).std()
        return rolling_z(mom_63) + 0.5 * rolling_z(slope_90) - 0.5 * rolling_z(vol_60)

    sti_spy = _sti(spx)

    features = pd.DataFrame(
        {
            "date": spx.index,
            "MRI": mri,
            "VCI": vci,
            "RORO": roro,
            "YCSI": ycsi,
            "STI_SPY": sti_spy,
        }
    ).reset_index(drop=True)

    # Add sector STI columns (for SectorTailwind)
    for etf in sector_etfs:
        s = get_series(etf)
        if s.empty:
            continue
        features[f"STI_{etf}"] = _sti(s).values
    return features


def add_stock_features(
    stock_df: pd.DataFrame,
    index_features: pd.DataFrame,
    spy_close_df: pd.DataFrame | None,
    sector_close_df: pd.DataFrame | None,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    df = stock_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    g = df.groupby("ticker", group_keys=False)

    df["logret"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))
    df["mom_6m"] = g["close"].apply(lambda s: s / s.shift(126) - 1.0)
    df["mom_3m"] = g["close"].apply(lambda s: s / s.shift(63) - 1.0)
    df["vol_60"] = g["logret"].apply(lambda s: s.rolling(60).std())
    df["mdd_3m"] = g["close"].apply(lambda s: trailing_mdd(s, 63))

    df["ema20"] = g["close"].apply(lambda s: s.ewm(span=20, adjust=False).mean())
    df["ema50"] = g["close"].apply(lambda s: s.ewm(span=50, adjust=False).mean())
    df["ema200"] = g["close"].apply(lambda s: s.ewm(span=200, adjust=False).mean())
    df["ema20_50"] = df["ema20"] / df["ema50"] - 1.0
    df["ema50_200"] = df["ema50"] / df["ema200"] - 1.0
    df["dist_ema200"] = df["close"] / df["ema200"] - 1.0
    df["ema50_slope"] = g["ema50"].apply(lambda s: s / s.shift(20) - 1.0)
    df["ema200_slope"] = g["ema200"].apply(lambda s: s / s.shift(60) - 1.0)

    df["sustain_20_50"] = (df["ema20"] > df["ema50"]).groupby(
        df["ticker"], group_keys=False
    ).apply(lambda s: s.rolling(20).mean())
    df["sustain_50_200"] = (df["ema50"] > df["ema200"]).groupby(
        df["ticker"], group_keys=False
    ).apply(lambda s: s.rolling(20).mean())
    df["sustain_price_200"] = (df["close"] > df["ema200"]).groupby(
        df["ticker"], group_keys=False
    ).apply(lambda s: s.rolling(20).mean())
    df["sustain_all"] = df[["sustain_20_50", "sustain_50_200", "sustain_price_200"]].min(axis=1)

    df["adv_20"] = g["close"].apply(lambda s: s.rolling(20).mean()) * g["volume"].apply(
        lambda s: s.rolling(20).mean()
    )
    df["adv_60"] = g["close"].apply(lambda s: s.rolling(60).mean()) * g["volume"].apply(
        lambda s: s.rolling(60).mean()
    )
    df["log_adv_20"] = np.log1p(df["adv_20"])
    df["log_adv_60"] = np.log1p(df["adv_60"])

    df["proximity_high_126"] = df["close"] / g["close"].apply(lambda s: s.rolling(126).max()) - 1.0

    index_features = index_features.copy()
    index_features["date"] = pd.to_datetime(index_features["date"])
    index_features = index_features.drop_duplicates("date").sort_values("date")
    macro_cols = ["date", "MRI", "VCI", "RORO", "YCSI", "STI_SPY"]
    macro_cols = [c for c in macro_cols if c in index_features.columns]
    if len(macro_cols) > 1:
        macro = index_features[macro_cols].copy()
        for c in macro_cols:
            if c != "date":
                macro[c] = pd.to_numeric(macro[c], errors="coerce").astype("float32")
        df = df.merge(macro, on="date", how="left", sort=False, copy=False, validate="m:1")

    # Sector map + ETF
    df["sector"] = df["ticker"].map(sector_map)
    df["sector_etf"] = df["sector"].map(SECTOR_TO_ETF).fillna("SPY")

    # Sector tailwind (long format to avoid wide merge)
    sti_cols = [c for c in index_features.columns if c.startswith("STI_")]
    if sti_cols:
        sti_long = index_features[["date"] + sti_cols].melt(
            id_vars="date", var_name="sti_col", value_name="SectorTailwind"
        )
        sti_long["sector_etf"] = sti_long["sti_col"].str.replace("STI_", "", regex=False)
        sti_long = sti_long.drop(columns=["sti_col"])
        df = df.merge(sti_long, on=["date", "sector_etf"], how="left", sort=False, copy=False, validate="m:1")
    else:
        df["SectorTailwind"] = df.get("STI_SPY")

    # Merge SPY close from history (for RS features)
    if isinstance(spy_close_df, pd.DataFrame) and not spy_close_df.empty:
        spy_df = spy_close_df[["date", "close"]].rename(columns={"close": "SPY_close"}).copy()
        spy_df["date"] = pd.to_datetime(spy_df["date"])
        df = df.merge(spy_df, on="date", how="left", sort=False, copy=False, validate="m:1")
    else:
        df["SPY_close"] = np.nan

    df["SPY_close"] = df["SPY_close"].where(df["SPY_close"] > 0, np.nan)

    df["RS_spy_ratio"] = df["close"] / df["SPY_close"]
    df.loc[df["RS_spy_ratio"] <= 0, "RS_spy_ratio"] = np.nan
    df["RS_spy"] = np.log(df["RS_spy_ratio"])
    df["RS_spy_slope90"] = df.groupby("ticker", group_keys=False)["RS_spy_ratio"].apply(
        lambda s: slope_log(s, 90)
    )

    # Sector close merge + RS sector
    if isinstance(sector_close_df, pd.DataFrame) and not sector_close_df.empty:
        sc = sector_close_df.copy()
        sc["date"] = pd.to_datetime(sc["date"])
        df = df.merge(sc, on=["date", "sector_etf"], how="left", sort=False, copy=False, validate="m:1")
    else:
        df["sector_close"] = np.nan

    sector_close_filled = df["sector_close"].where(df["sector_close"] > 0, df["SPY_close"])
    df["RS_sector_ratio"] = df["close"] / sector_close_filled
    df.loc[df["RS_sector_ratio"] <= 0, "RS_sector_ratio"] = np.nan
    df["RS_sector"] = np.log(df["RS_sector_ratio"])
    df["RS_sector_slope90"] = df.groupby("ticker", group_keys=False)["RS_sector_ratio"].apply(
        lambda s: slope_log(s, 90)
    )

    def z_by_date(col: str) -> pd.Series:
        return df.groupby("date")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0))

    df["ema_stack"] = (df["dist_ema200"] + df["ema20_50"] + df["ema50_200"]) / 3.0
    df["z_mom_6m"] = z_by_date("mom_6m")
    df["z_mom_3m"] = z_by_date("mom_3m")
    df["z_ema_stack"] = z_by_date("ema_stack")
    df["z_ema50_slope"] = z_by_date("ema50_slope")
    df["z_vol_60"] = z_by_date("vol_60")
    df["abs_mdd_3m"] = df["mdd_3m"].abs()
    df["z_mdd_3m"] = z_by_date("abs_mdd_3m")
    df["z_sustain"] = z_by_date("sustain_all")
    df["z_log_adv"] = z_by_date("log_adv_20")
    df["z_prox_high"] = z_by_date("proximity_high_126")
    df["z_RS_spy_slope"] = z_by_date("RS_spy_slope90")
    df["z_RS_sector_slope"] = z_by_date("RS_sector_slope90")

    df["STSI"] = (
        0.35 * df["z_mom_6m"]
        + 0.20 * df["z_mom_3m"]
        + 0.20 * df["z_ema_stack"]
        + 0.15 * df["z_ema50_slope"]
        - 0.20 * df["z_vol_60"]
        - 0.20 * df["z_mdd_3m"]
    )

    df["RLI"] = 0.60 * df["z_RS_spy_slope"] + 0.40 * df["z_RS_sector_slope"]

    df["compression"] = -0.5 * df["z_vol_60"] - 0.5 * df["z_mdd_3m"]
    df["BRI"] = (
        0.35 * df["compression"]
        + 0.25 * df["z_log_adv"]
        + 0.25 * df["z_sustain"]
        + 0.15 * df["z_prox_high"]
    )

    df["QMI"] = 0.50 * df["z_sustain"] - 0.30 * df["z_vol_60"] - 0.20 * df["z_mdd_3m"]

    df["TailwindScore"] = (
        0.35 * df["STSI"]
        + 0.25 * df["SectorTailwind"]
        + 0.25 * df["MRI"]
        + 0.15 * df["VCI"]
    )

    df["TotalScore"] = 0.55 * df["STSI"] + 0.25 * df["RLI"] + 0.20 * df["TailwindScore"]

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="data/universe_clean.csv")
    parser.add_argument("--out", default="data/daily_scored.parquet")
    parser.add_argument("--history-days", type=int, default=420)
    parser.add_argument("--min-rows", type=int, default=260)
    parser.add_argument("--include-sectors", action="store_true")
    parser.add_argument("--sector-mode", choices=["yfinance", "none"], default="yfinance")
    parser.add_argument("--sector-map", default="data/sector_map.json")
    parser.add_argument("--sector-sleep", type=float, default=0.05)
    parser.add_argument("--bad-tickers", default="data/bad_tickers.csv")
    args = parser.parse_args()

    uni = pd.read_csv(args.universe)
    if "ticker" not in uni.columns:
        raise ValueError("universe file must contain 'ticker'")
    uni["ticker"] = uni["ticker"].astype(str).str.strip()
    tickers = [t for t in uni["ticker"].tolist() if t]
    bad = _load_bad_tickers(args.bad_tickers)
    if bad:
        tickers = [t for t in tickers if t.upper() not in bad]
    if not tickers:
        raise ValueError("No tickers found in universe")

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
    missing_universe = []
    minrows_bad = []
    for t in tickers:
        df = hist_map.get(t)
        if df is None or df.empty:
            missing_universe.append(t)
            continue
        ohlcv = _to_ohlcv(df)
        if ohlcv.empty or ohlcv.shape[0] < args.min_rows:
            minrows_bad.append(t)
            continue
        ohlcv["ticker"] = t
        rows.append(ohlcv)

    if not rows:
        raise RuntimeError("No usable ticker history found.")

    stock_df = pd.concat(rows, ignore_index=True)
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    stock_df = stock_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    sector_map = load_sector_map(tickers, args.sector_map, args.sector_mode, args.sector_sleep)

    index_features = build_index_features(hist_map, SECTOR_ETFS if args.include_sectors else [])
    spy_df = _to_ohlcv(hist_map.get("SPY", pd.DataFrame()))

    # Sector close (long format)
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

    # attach company names
    name_map = dict(zip(uni["ticker"], uni.get("company", "")))
    df["company"] = df["ticker"].map(name_map)

    # Keep only latest date across tickers
    last_date = df["date"].max()
    df = df[df["date"] == last_date].copy()
    df = df.sort_values("ticker").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} rows={len(df)} date={last_date.date()}")

    # Persist bad tickers so we skip them next run
    _append_bad_tickers(args.bad_tickers, missing_universe + minrows_bad)


if __name__ == "__main__":
    main()
