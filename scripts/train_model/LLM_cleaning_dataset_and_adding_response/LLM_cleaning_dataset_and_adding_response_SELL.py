# -*- coding: utf-8 -*-
"""
LLM_cleaning_dataset_and_adding_sellscore.py
--------------------------------------------
Replaces SellLabel with SellScore (regression-ready), tuned per market regime.

Key properties:
- Past-only features:
    * MarketTrend from ^GSPC SMA cross (past-only, backward date alignment)
    * Prev_* and rolling averages (past-only)
    * Volatility feature used for crash thresholds is past-only (rolling)
- Target/label-like computation is allowed to be forward-looking by design:
    * SellScore uses future windows to define "good time to sell" and "loss control"
- Deterministic parameter tuning (no ML training):
    * Tunes only w_peak and w_early per regime (w_crash fixed, crash threshold is vol-based)
    * Uses expanding time folds + embargo
    * Ensures training rows used for tuning do not require future prices in the validation period

Outputs:
- Writes parquet with SellScore column (float32), removes SellLabel if present.
- Writes JSON with tuned weights (w_peak, w_early) per regime and fixed w_crash config.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf


# ==========================================================
# CONFIG
# ==========================================================
INPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_SELL_new_features_with_response.parquet")
OUTPUT_PATH = Path("../LLM_Training_data/LLM_Training_data_SELL_new_features_with_sellscore.parquet")
TEMP_PATH = Path("../LLM_Training_data/temp_checkpoint_sellscore.parquet")
TUNED_PARAMS_PATH = Path("../LLM_Training_data/sellscore_tuned_params.json")

SAVE_EVERY = 2000

# Horizon and early preference
HOLD_DAYS = 126          # ~6 months lookahead (trading days)
EARLY_DAYS = 10          # "next 10 days"

# Peak sensitivity by regime
# Interpretation: peak condition activates when upside_left_ratio <= eps_peak (smaller = stricter)
EPS_PEAK_BY_REGIME = {
    1: 0.03,   # Bullish
    0: 0.05,   # Neutral
    -1: 0.07,  # Bearish
}


HOLD_DAYS_NEUTRAL_TUNE = 42   # ~2 months just for regime 0 tuning samples
MIN_FOLD_ROWS_DEFAULT = 5000
MIN_FOLD_ROWS_NEUTRAL = 1500


# Crash score saturation scale (beyond crash threshold)
D_CAP = 0.20

# Early gain saturation scale for S_early magnitude term
G_CAP = 0.20

# Market trend config (^GSPC)
SNP_TICKER = "^GSPC"
SMA_FAST = 50
SMA_SLOW = 200

# Tuning setup
N_FOLDS = 4
EMBARGO_CAL_DAYS = 210   # embargo to avoid overlap

# ==========================================================
# VOL-BASED CRASH THRESHOLD + FIXED w_crash
# ==========================================================
USE_VOL_BASED_CRASH = True

# Prefer an existing past-only volatility column. If missing, script computes one.
# Accepted examples: "Volatility_30D", "Volatility_252D", etc.
VOL_COL = "Volatility_30D"

# Crash threshold multiplier per market regime.
# Lower multiplier => more sensitive to drops (threshold smaller).
VOL_MULT_BY_REGIME = {
    -1: 2.0,  # bear: more sensitive
     0: 2.5,  # neutral
     1: 3.0,  # bull: less sensitive
}

# Clamp threshold to avoid absurd values (in return space).
CRASH_THR_MIN = 0.08   # 8%
CRASH_THR_MAX = 0.35   # 35%

# Fixed crash weight (not tuned; always non-zero)
FIXED_W_CRASH = 0.15

# Weight tuning resolution for w_peak (w_early is implied)
W_PEAK_STEP = 0.05  # 0.05 = 5% increments; deterministic and fast

# Utility objective weights (tuning target)
# y = gain_early - L_DELAY*(gain_horizon - gain_early)+ - L_RISK*drawdown_early
L_DELAY = 0.75
L_RISK = 1.25


# ==========================================================
# HELPERS
# ==========================================================

def pick_close_column(df: pd.DataFrame) -> str:
    """
    Pick the best available close-like column in the dataset.
    We need a per-row price series for forward-window scoring.
    """
    candidates = [
        "Close", "close",
        "close_raw", "Close_raw",
        "Adj Close", "AdjClose", "adj_close",
        "Price", "price", "current_price",
        "Last", "last"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        if "close" in str(c).lower():
            return c

    raise KeyError(
        "No close-like column found. Need a price column to compute SellScore.\n"
        f"Columns sample: {list(df.columns)[:50]}"
    )


def _standardize_yf_date_column(snp_df: pd.DataFrame) -> pd.DataFrame:
    snp_df = snp_df.reset_index()

    if isinstance(snp_df.columns, pd.MultiIndex):
        snp_df.columns = [
            "_".join([str(x) for x in col if str(x) not in ("", "None")]).strip("_")
            for col in snp_df.columns.to_list()
        ]
    else:
        snp_df.columns = [
            "_".join([str(x) for x in col if str(x) not in ("", "None")]).strip("_")
            if isinstance(col, tuple) else str(col)
            for col in snp_df.columns.to_list()
        ]

    candidates = ["SNP_Date", "Date", "Datetime", "date", "datetime", "index"]
    date_col = next((c for c in candidates if c in snp_df.columns), None)

    if date_col is None:
        for c in snp_df.columns:
            if np.issubdtype(snp_df[c].dtype, np.datetime64):
                date_col = c
                break

    if date_col is None:
        first = snp_df.columns[0]
        test = pd.to_datetime(snp_df[first], errors="coerce")
        if test.notna().any():
            date_col = first

    if date_col is None:
        raise KeyError(f"Could not find a usable date column after reset_index(). Columns: {list(snp_df.columns)}")

    snp_df["SNP_Date"] = pd.to_datetime(snp_df[date_col], errors="coerce")
    snp_df = snp_df.dropna(subset=["SNP_Date"]).sort_values("SNP_Date").reset_index(drop=True)

    if date_col != "SNP_Date":
        snp_df = snp_df.drop(columns=[date_col], errors="ignore")

    return snp_df


def _future_rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """For each i, returns max over arr[i : i+window] (inclusive of i)."""
    s = pd.Series(arr[::-1])
    return s.rolling(window, min_periods=1).max().to_numpy()[::-1]


def _future_rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr[::-1])
    return s.rolling(window, min_periods=1).min().to_numpy()[::-1]


def _safe_clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


def spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fast Spearman correlation via Pearson correlation of ranks.
    Deterministic, no scipy dependency.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size or x.size < 10:
        return np.nan

    rx = np.empty_like(x, dtype=np.float64)
    ry = np.empty_like(y, dtype=np.float64)
    rx[np.argsort(x, kind="mergesort")] = np.arange(x.size)
    ry[np.argsort(y, kind="mergesort")] = np.arange(y.size)

    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum())
    if denom == 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def ensure_volatility_column(df: pd.DataFrame, vol_col: str, close_col: str) -> pd.DataFrame:
    """
    Ensure a past-only volatility column exists.
    If VOL_COL is missing, compute a 30D realized vol (rolling std of log-returns).
    Result is in decimal space (e.g. 0.25 = 25%).
    """
    if vol_col in df.columns:
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        return df

    print(f"[WARNING] '{vol_col}' not found. Computing fallback realized volatility from '{close_col}' (rolling 30D).")
    # log returns per ticker (past-only)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    px = pd.to_numeric(df[close_col], errors="coerce")
    df["_lr"] = np.log(px).groupby(df["Ticker"]).diff()

    # rolling std of log returns, annualization not needed because we use it as a scale feature
    df[vol_col] = df.groupby("Ticker")["_lr"].transform(lambda s: s.rolling(30, min_periods=10).std())

    df = df.drop(columns=["_lr"], errors="ignore")

    # If still NaN-heavy, fill forward within ticker, then fill with cross-sectional median
    df[vol_col] = df.groupby("Ticker")[vol_col].ffill()
    med = float(df[vol_col].median(skipna=True)) if df[vol_col].notna().any() else 0.20
    df[vol_col] = df[vol_col].fillna(med)

    return df


def compute_crash_threshold_mag(vol: np.ndarray, regime: int) -> np.ndarray:
    """
    Returns per-row positive threshold magnitudes in return space.
    Uses only past-based volatility input.
    """
    vol = np.asarray(vol, dtype=np.float32)

    # Defensive: if vol is in percent (e.g. 25) convert to decimal (0.25)
    vol = np.where(vol > 3.0, vol / 100.0, vol)

    mult = float(VOL_MULT_BY_REGIME.get(int(regime), 2.5))
    thr = mult * vol

    return np.clip(thr, CRASH_THR_MIN, CRASH_THR_MAX).astype(np.float32)


def _make_time_folds(unique_dates: np.ndarray, n_folds: int):
    """
    Expanding folds on chronological dates.
    Returns list of (train_end_date, embargo_start_date) boundaries.
    """
    unique_dates = np.array(sorted(unique_dates))
    if len(unique_dates) < (n_folds + 2):
        n_folds = max(1, len(unique_dates) // 3)

    cut_idxs = np.linspace(0.55, 0.85, n_folds)
    fold_ends = [unique_dates[int(len(unique_dates) * x)] for x in cut_idxs]

    folds = []
    for train_end in fold_ends:
        embargo_start = pd.Timestamp(train_end) + pd.Timedelta(days=EMBARGO_CAL_DAYS)
        folds.append((pd.Timestamp(train_end), embargo_start))
    return folds


def build_tuning_segments(df: pd.DataFrame, regime: int, top_segments: int) -> pd.DataFrame:
    """
    Deterministically select eligible (Ticker, _seg) segments for tuning:
    - regime-constant segments
    - segment length >= min_len (regime-dependent)
    - rows have Date_horizon_end available
    """
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Date_horizon_end"] = df.groupby("Ticker")["Date"].shift(-HOLD_DAYS)
    df["_seg"] = df.groupby("Ticker")["MarketTrend"].apply(lambda s: s.ne(s.shift(1)).cumsum()).to_numpy()

    df_r = df[(df["MarketTrend"].astype("int8") == np.int8(regime)) & (df["Date_horizon_end"].notna())].copy()
    if df_r.empty:
        return df_r[["Ticker", "_seg"]].head(0)

    seg_sizes = df_r.groupby(["Ticker", "_seg"]).size()

    # Regime-specific minimum segment length
    # Neutral regimes tend to be shorter; relax to get “plenty” of samples.
    if int(regime) == 0:
        min_len = max(60, EARLY_DAYS + 5)     # much easier to satisfy than HOLD_DAYS+2
    else:
        min_len = (HOLD_DAYS + 2)

    good = seg_sizes[seg_sizes >= min_len].sort_values(ascending=False)
    if good.empty:
        return df_r[["Ticker", "_seg"]].head(0)

    return good.head(top_segments).reset_index()[["Ticker", "_seg"]]


def compute_primitives_for_tuning(
    df_all: pd.DataFrame,
    seg_ids: pd.DataFrame,
    eps_peak: float,
    stride: int,
    regime: int,
    hold_days: int,
    date_h_col: str
) -> pd.DataFrame:
    """
    Compute tuning primitives on FULL (Ticker,_seg) segments (no pre-stride),
    then downsample AFTER computing them.

    Outputs:
      Ticker, Date, Date_horizon_end,
      S_peak, S_early, S_crash, drawdown_mag, _utility_y, _seg
    """
    if seg_ids is None or seg_ids.empty:
        return pd.DataFrame()

    # Sort and ensure segmentation exists
    df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    if "_seg" not in df_all.columns:
        df_all["_seg"] = (
            df_all.groupby("Ticker")["MarketTrend"]
            .apply(lambda s: s.ne(s.shift(1)).cumsum())
            .to_numpy()
        )

    # Ensure horizon-end columns exist (created only once)
    if "Date_horizon_end_126" not in df_all.columns:
        df_all["Date_horizon_end_126"] = df_all.groupby("Ticker")["Date"].shift(-HOLD_DAYS)
    if "Date_horizon_end_42" not in df_all.columns:
        df_all["Date_horizon_end_42"] = df_all.groupby("Ticker")["Date"].shift(-HOLD_DAYS_NEUTRAL_TUNE)

    if date_h_col not in df_all.columns:
        raise KeyError(f"date_h_col='{date_h_col}' not found in df_all. Available: "
                       f"{[c for c in df_all.columns if c.startswith('Date_horizon_end')]}")

    # Fast filtering to chosen segments
    key = pd.MultiIndex.from_frame(seg_ids[["Ticker", "_seg"]])
    df_all_idx = df_all.set_index(["Ticker", "_seg"])
    df_sel = df_all_idx.loc[df_all_idx.index.intersection(key)].reset_index()
    if df_sel.empty:
        return pd.DataFrame()

    out_parts = []

    # Minimum rows sanity: need enough for early window + horizon window
    min_len_needed = max(EARLY_DAYS + 2, hold_days + 2)

    for (ticker, seg_id), g in df_sel.groupby(["Ticker", "_seg"], sort=False):
        if len(g) < min_len_needed:
            continue
        if "Close" not in g.columns:
            continue

        # If vol-based crash is enabled, keep only segments with volatility (or fill fallback below)
        has_vol = (VOL_COL in g.columns) if USE_VOL_BASED_CRASH else False

        closes = g["Close"].to_numpy(dtype=float)
        price = closes
        valid_price = np.isfinite(price) & (price > 0)

        # Future windows (horizon uses hold_days, not global HOLD_DAYS)
        fut_max_H_incl_i = _future_rolling_max(closes, hold_days + 1)
        fut_max_E_incl_i = _future_rolling_max(closes, EARLY_DAYS + 1)
        fut_min_E_incl_i = _future_rolling_min(closes, EARLY_DAYS + 1)

        # Shift to future-only (i+1..)
        fut_max_H = np.r_[fut_max_H_incl_i[1:], np.nan]
        fut_max_E = np.r_[fut_max_E_incl_i[1:], np.nan]
        fut_min_E = np.r_[fut_min_E_incl_i[1:], np.nan]

        # -------------------------
        # S_peak
        # -------------------------
        upside_best = np.where(valid_price & np.isfinite(fut_max_H), (fut_max_H - price) / price, np.nan)
        upside_left_ratio = np.where(
            np.isfinite(fut_max_H) & (fut_max_H > 0) & valid_price,
            (fut_max_H - price) / fut_max_H,
            np.nan
        )

        s_peak = np.where(
            valid_price
            & np.isfinite(upside_best) & (upside_best > 0)
            & np.isfinite(upside_left_ratio),
            _safe_clip(1.0 - (upside_left_ratio / (eps_peak + 1e-12)), 0.0, 1.0),
            0.0
        ).astype(np.float32)

        # -------------------------
        # drawdown_mag (EARLY window)
        # -------------------------
        early_drop = np.where(
            valid_price & np.isfinite(fut_min_E),
            (fut_min_E - price) / price,  # negative for drop
            np.nan
        )
        drawdown_mag = np.where(np.isfinite(early_drop), -early_drop, 0.0).astype(np.float32)

        # -------------------------
        # S_crash (vol-based threshold, past-only)
        # -------------------------
        if USE_VOL_BASED_CRASH:
            if has_vol:
                thr_mag_vec = compute_crash_threshold_mag(
                    g[VOL_COL].to_numpy(dtype=np.float32),
                    regime=int(regime),
                )
            else:
                # Fallback if vol column is missing in this segment: constant 15%
                thr_mag_vec = np.full(len(g), 0.15, dtype=np.float32)
        else:
            thr_mag_vec = np.full(len(g), 0.15, dtype=np.float32)

        s_crash = np.where(
            valid_price & np.isfinite(early_drop),
            _safe_clip((drawdown_mag - thr_mag_vec) / (D_CAP + 1e-12), 0.0, 1.0),
            0.0
        ).astype(np.float32)

        # -------------------------
        # Gains
        # -------------------------
        gainE = np.where(valid_price & np.isfinite(fut_max_E), (fut_max_E - price) / price, np.nan)
        gainH = np.where(valid_price & np.isfinite(fut_max_H), (fut_max_H - price) / price, np.nan)

        # -------------------------
        # S_early
        # -------------------------
        ratio = np.where(
            np.isfinite(gainE) & np.isfinite(gainH) & (gainH > 0),
            gainE / (gainH + 1e-12),
            0.0
        )
        mag = np.where(np.isfinite(gainE), gainE / (G_CAP + 1e-12), 0.0)
        s_early = (_safe_clip(ratio, 0.0, 1.0) * _safe_clip(mag, 0.0, 1.0)).astype(np.float32)

        # -------------------------
        # Utility y (tuning target)
        # -------------------------
        extra_late = np.where(np.isfinite(gainH) & np.isfinite(gainE), np.maximum(gainH - gainE, 0.0), 0.0)
        y = (
            np.nan_to_num(gainE, nan=0.0)
            - L_DELAY * extra_late
            - L_RISK * drawdown_mag
        ).astype(np.float32)

        # Output frame: unify the horizon-end column name
        base = g[["Ticker", "Date", date_h_col]].copy()
        base = base.rename(columns={date_h_col: "Date_horizon_end"})
        base["_seg"] = seg_id

        base["S_peak"] = s_peak
        base["S_early"] = s_early
        base["S_crash"] = s_crash
        base["drawdown_mag"] = drawdown_mag
        base["_utility_y"] = y

        # Downsample AFTER primitives
        if stride > 1:
            base["_i"] = np.arange(len(base), dtype=np.int32)
            base = base[base["_i"] % stride == 0].drop(columns=["_i"], errors="ignore")

        out_parts.append(base)

    if not out_parts:
        return pd.DataFrame()

    return pd.concat(out_parts, ignore_index=True)


def _compute_components_for_segment(
    df_t: pd.DataFrame,
    regime: int,
    eps_peak: float,
) -> pd.DataFrame:
    """
    Computes S_peak, S_crash, S_early (all in [0,1]) for a single constant-regime segment.
    Uses vol-based crash thresholds if enabled.
    """
    closes = df_t["Close"].to_numpy(dtype=float)

    fut_max_H_incl_i = _future_rolling_max(closes, HOLD_DAYS + 1)
    fut_max_E_incl_i = _future_rolling_max(closes, EARLY_DAYS + 1)
    fut_min_E_incl_i = _future_rolling_min(closes, EARLY_DAYS + 1)

    fut_max_H = np.r_[fut_max_H_incl_i[1:], np.nan]
    fut_max_E = np.r_[fut_max_E_incl_i[1:], np.nan]
    fut_min_E = np.r_[fut_min_E_incl_i[1:], np.nan]

    price = closes
    valid_price = np.isfinite(price) & (price > 0)

    upside_best = np.where(valid_price & np.isfinite(fut_max_H), (fut_max_H - price) / price, np.nan)
    upside_left_ratio = np.where(
        np.isfinite(fut_max_H) & (fut_max_H > 0) & valid_price,
        (fut_max_H - price) / fut_max_H,
        np.nan
    )

    s_peak = np.where(
        valid_price & np.isfinite(upside_best) & (upside_best > 0) & np.isfinite(upside_left_ratio),
        _safe_clip(1.0 - (upside_left_ratio / (eps_peak + 1e-12)), 0.0, 1.0),
        0.0
    ).astype(np.float32)

    early_drop = np.where(valid_price & np.isfinite(fut_min_E), (fut_min_E - price) / price, np.nan)
    drawdown_mag = np.where(np.isfinite(early_drop), -early_drop, 0.0).astype(np.float32)

    if USE_VOL_BASED_CRASH:
        if VOL_COL not in df_t.columns:
            raise KeyError(f"Missing volatility column '{VOL_COL}' required for vol-based crash thresholds.")
        thr_mag_vec = compute_crash_threshold_mag(df_t[VOL_COL].to_numpy(dtype=np.float32), regime=regime)
    else:
        thr_mag_vec = np.full(len(df_t), 0.15, dtype=np.float32)

    s_crash = np.where(
        valid_price & np.isfinite(early_drop),
        _safe_clip((drawdown_mag - thr_mag_vec) / (D_CAP + 1e-12), 0.0, 1.0),
        0.0
    ).astype(np.float32)

    gainE = np.where(valid_price & np.isfinite(fut_max_E), (fut_max_E - price) / price, np.nan)
    gainH = np.where(valid_price & np.isfinite(fut_max_H), (fut_max_H - price) / price, np.nan)

    ratio = np.where(np.isfinite(gainE) & np.isfinite(gainH) & (gainH > 0), gainE / (gainH + 1e-12), 0.0)
    mag = np.where(np.isfinite(gainE), gainE / (G_CAP + 1e-12), 0.0)
    s_early = (_safe_clip(ratio, 0.0, 1.0) * _safe_clip(mag, 0.0, 1.0)).astype(np.float32)

    df_t["S_peak"] = s_peak
    df_t["S_crash"] = s_crash
    df_t["S_early"] = s_early
    return df_t


def tune_params_by_regime(df_all: pd.DataFrame) -> dict:
    """
    Deterministic tuning:
    - crash threshold: volatility-based (NOT tuned)
    - w_crash: fixed (NOT tuned)
    - tune only w_peak (w_early implied), per regime
    - fold-based with embargo
    """
    # Sampling controls
    TOP_SEGMENTS_DEFAULT = 700
    TOP_SEGMENTS_NEUTRAL = 2500  # push regime 0 harder to ensure enough segments
    STRIDE = 5

    if FIXED_W_CRASH <= 0 or FIXED_W_CRASH >= 0.95:
        raise ValueError("FIXED_W_CRASH must be in (0, 0.95) to leave room for w_peak + w_early.")

    # Grid for w_peak only; w_early is implied.
    max_w_peak = 1.0 - FIXED_W_CRASH
    w_peak_grid = np.round(np.arange(0.0, max_w_peak + 1e-9, W_PEAK_STEP), 10).tolist()

    df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df_all["Date_horizon_end"] = df_all.groupby("Ticker")["Date"].shift(-HOLD_DAYS)

    unique_dates = df_all["Date"].dropna().unique()
    folds = _make_time_folds(unique_dates, N_FOLDS)

    results = {}

    for regime in [-1, 0, 1]:
        eps_peak = float(EPS_PEAK_BY_REGIME[regime])

        top_segments = TOP_SEGMENTS_NEUTRAL if regime == 0 else TOP_SEGMENTS_DEFAULT
        seg_ids = build_tuning_segments(df_all, regime=regime, top_segments=top_segments)
        use_h = HOLD_DAYS_NEUTRAL_TUNE if regime == 0 else HOLD_DAYS
        date_h_col = "Date_horizon_end_42" if regime == 0 else "Date_horizon_end_126"
        min_fold_rows = MIN_FOLD_ROWS_NEUTRAL if regime == 0 else MIN_FOLD_ROWS_DEFAULT

        if seg_ids.empty: 
            results[regime] = {
                "w_peak": None,
                "w_crash": float(FIXED_W_CRASH),
                "w_early": None,
                "cv_spearman": None,
                "n_rows": 0
            }
            print(f"[TUNING] Regime {regime}: empty sample (no eligible segments).")
            continue

        print(f"[TUNING] Regime {regime}: segments={len(seg_ids):,}, tickers={seg_ids['Ticker'].nunique()}")

        prim = compute_primitives_for_tuning(
            df_all=df_all,
            seg_ids=seg_ids,
            eps_peak=eps_peak,
            stride=STRIDE,
            regime=regime,
            hold_days=use_h,
            date_h_col=date_h_col
        )


        if prim.empty:
            results[regime] = {
                "w_peak": None,
                "w_crash": float(FIXED_W_CRASH),
                "w_early": None,
                "cv_spearman": None,
                "n_rows": 0
            }
            print(f"[TUNING] Regime {regime}: no primitives computed.")
            continue

        print(f"[TUNING] Regime {regime}: sample rows={len(prim):,}")

        def eligible_mask(train_end, embargo_start):
            return (
                (prim["Date"] <= train_end)
                & (prim["Date_horizon_end"].notna())
                & (prim["Date_horizon_end"] < embargo_start)
            )

        y_all = prim["_utility_y"].to_numpy(dtype=np.float32)
        sp_all = prim["S_peak"].to_numpy(dtype=np.float32)
        sc_all = prim["S_crash"].to_numpy(dtype=np.float32)
        se_all = prim["S_early"].to_numpy(dtype=np.float32)

        fold_bests = []

        for train_end, embargo_start in folds:
            m = eligible_mask(train_end, embargo_start).to_numpy()
            if m.sum() < min_fold_rows:
                continue

            y = y_all[m]
            sp = sp_all[m]
            sc = sc_all[m]
            se = se_all[m]

            local_best_corr = -1e18
            local_best_w_peak = None

            for w_peak in w_peak_grid:
                w_crash = float(FIXED_W_CRASH)
                w_early = float(1.0 - w_crash - w_peak)

                score = (w_peak * sp + w_crash * sc + w_early * se)
                corr = spearman_fast(score, y)

                if np.isfinite(corr) and corr > local_best_corr:
                    local_best_corr = corr
                    local_best_w_peak = float(w_peak)

            if local_best_w_peak is not None:
                fold_bests.append((local_best_corr, local_best_w_peak))

        if not fold_bests:
            # Deterministic fallback (reasonable, crash always active)
            w_peak = 0.20
            w_crash = float(FIXED_W_CRASH)
            w_early = float(1.0 - w_crash - w_peak)
            cv = None
        else:
            cv = float(np.mean([x[0] for x in fold_bests]))

            # Use median-best w_peak across folds for stability (fair + robust)
            w_peak = float(np.median([x[1] for x in fold_bests]))
            w_crash = float(FIXED_W_CRASH)
            w_early = float(1.0 - w_crash - w_peak)

        results[regime] = {
            "w_peak": float(w_peak),
            "w_crash": float(w_crash),
            "w_early": float(w_early),
            "cv_spearman": None if cv is None else float(cv),
            "n_rows": int(len(prim))
        }
        print(f"[TUNING] Regime {regime}: {results[regime]}")

    # Neutral regime fallback if still empty
    if results.get(0, {}).get("w_peak") is None:
        neg = results.get(-1, None)
        pos = results.get(1, None)

        if neg and pos and neg.get("w_peak") is not None and pos.get("w_peak") is not None:
            n_neg = max(int(neg.get("n_rows", 1)), 1)
            n_pos = max(int(pos.get("n_rows", 1)), 1)
            a = n_neg / (n_neg + n_pos)
            b = n_pos / (n_neg + n_pos)

            w_peak = float(a * neg["w_peak"] + b * pos["w_peak"])
            w_crash = float(FIXED_W_CRASH)
            w_early = float(1.0 - w_crash - w_peak)

            results[0] = {
                "w_peak": w_peak,
                "w_crash": w_crash,
                "w_early": w_early,
                "cv_spearman": None,
                "n_rows": 0
            }
            print("[TUNING] Regime 0: fallback to weighted avg of regimes -1 and +1.")
        else:
            w_peak = 0.18
            w_crash = float(FIXED_W_CRASH)
            w_early = float(1.0 - w_crash - w_peak)
            results[0] = {
                "w_peak": w_peak,
                "w_crash": w_crash,
                "w_early": w_early,
                "cv_spearman": None,
                "n_rows": 0
            }
            print("[TUNING] Regime 0: fallback to deterministic defaults (no -1/+1 available).")

    return results


def compute_sellscore_for_segment(df_t: pd.DataFrame, tuned: dict) -> pd.DataFrame:
    """
    Computes final SellScore per row for a single constant-regime segment.
    """
    regime = int(df_t["MarketTrend"].iloc[0])
    params = tuned.get(regime, None)

    if not params or params.get("w_peak") is None:
        df_t["SellScore"] = np.zeros(len(df_t), dtype=np.float32)
        return df_t

    w_peak = float(params["w_peak"])
    w_crash = float(params["w_crash"])
    w_early = float(params["w_early"])

    # Final numeric safety normalize
    s = w_peak + w_crash + w_early
    if s <= 0:
        w_peak, w_crash, w_early = 0.20, float(FIXED_W_CRASH), float(0.80 - FIXED_W_CRASH)
    else:
        w_peak /= s
        w_crash /= s
        w_early /= s

    eps_peak = float(EPS_PEAK_BY_REGIME[regime])

    df_t = _compute_components_for_segment(df_t, regime=regime, eps_peak=eps_peak)

    score = (w_peak * df_t["S_peak"] + w_crash * df_t["S_crash"] + w_early * df_t["S_early"]).astype(np.float32)
    df_t["SellScore"] = np.clip(score.to_numpy(dtype=np.float32), 0.0, 1.0)

    df_t = df_t.drop(columns=["S_peak", "S_crash", "S_early"], errors="ignore")
    return df_t


# ==========================================================
# LOAD DATA
# ==========================================================
print(f"[INFO] Loading dataset from {INPUT_PATH} ...")
df = pd.read_parquet(INPUT_PATH).reset_index(drop=False)

if "Date" not in df.columns:
    raise KeyError("No 'Date' column found in dataset.")

if np.issubdtype(df["Date"].dtype, np.number):
    print("[INFO] Converting numeric timestamps → datetime (ms).")
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", errors="coerce")
else:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

print(f"[INFO] Loaded {len(df):,} rows for {df['Ticker'].nunique()} tickers.")

CLOSE_COL = pick_close_column(df)
print(f"[INFO] Using '{CLOSE_COL}' as the close price source for SellScore computation.")

# Standardize to 'Close'
if CLOSE_COL != "Close":
    df["Close"] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
else:
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df = df.dropna(subset=["Close"])

# Ensure volatility column exists (past-only)
df = ensure_volatility_column(df, VOL_COL, close_col="Close")


# ==========================================================
# DOWNLOAD S&P AND COMPUTE MarketTrend PER DATE (yfinance)
# ==========================================================
print(f"[INFO] Downloading {SNP_TICKER} via yfinance to compute MarketTrend...")

dmin = df["Date"].min()
dmax = df["Date"].max()
start = (dmin - pd.Timedelta(days=365)).date()
end = (dmax + pd.Timedelta(days=5)).date()

snp = yf.download(
    SNP_TICKER,
    start=str(start),
    end=str(end),
    auto_adjust=False,
    progress=False,
    threads=True,
    group_by="column",
)

if snp is None or snp.empty:
    raise RuntimeError(f"yfinance returned no data for {SNP_TICKER} from {start} to {end}.")

snp = _standardize_yf_date_column(snp)

preferred_cols = [
    "Adj Close", f"Adj Close_{SNP_TICKER}",
    "Close",     f"Close_{SNP_TICKER}",
]
snp_price_col = next((c for c in preferred_cols if c in snp.columns), None)
if snp_price_col is None:
    for c in snp.columns:
        if c.startswith("Adj Close_") or c.startswith("Close_"):
            snp_price_col = c
            break
if snp_price_col is None:
    raise KeyError(f"Expected Close/Adj Close in yfinance data. Got columns: {list(snp.columns)}")

snp["SNP_Close"] = pd.to_numeric(snp[snp_price_col], errors="coerce")
snp = snp.dropna(subset=["SNP_Close"]).reset_index(drop=True)

# Past-only MAs
snp["SMA_fast"] = snp["SNP_Close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
snp["SMA_slow"] = snp["SNP_Close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

bullish = (snp["SNP_Close"] > snp["SMA_slow"]) & (snp["SMA_fast"] > snp["SMA_slow"])
bearish = (snp["SNP_Close"] < snp["SMA_slow"]) & (snp["SMA_fast"] < snp["SMA_slow"])
snp["MarketTrend"] = np.where(bullish, 1, np.where(bearish, -1, 0)).astype("int8")

if "SNP_Date" not in snp.columns:
    raise KeyError(f"[FATAL] snp is missing SNP_Date. Columns: {list(snp.columns)}")
if "MarketTrend" not in snp.columns:
    raise KeyError(f"[FATAL] snp is missing MarketTrend. Columns: {list(snp.columns)}")

# ==========================================================
# MAP MarketTrend ONTO df BY LAST PRIOR S&P TRADING DAY
# ==========================================================
trend_by_date = snp[["SNP_Date", "MarketTrend"]].dropna().copy()
trend_by_date["SNP_Date"] = pd.to_datetime(trend_by_date["SNP_Date"], errors="coerce").dt.tz_localize(None)
trend_by_date = trend_by_date.dropna(subset=["SNP_Date"]).sort_values("SNP_Date")
trend_by_date = trend_by_date.drop_duplicates(subset=["SNP_Date"], keep="last").reset_index(drop=True)

snp_dates = trend_by_date["SNP_Date"].to_numpy(dtype="datetime64[ns]")
snp_trend = trend_by_date["MarketTrend"].astype("int8").to_numpy()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
df_dates = df["Date"].to_numpy(dtype="datetime64[ns]")

idx = np.searchsorted(snp_dates, df_dates, side="right") - 1
df["MarketTrend"] = np.where(idx >= 0, snp_trend[idx], np.int8(0)).astype("int8")

print("[INFO] MarketTrend mapped onto df using backward date alignment (no merge_asof).")
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
print("[INFO] MarketTrend added to all rows (int8: -1,0,1).")


# ==========================================================
# ADD PAST-DAY FEATURES (no leakage)
# ==========================================================
for col in ["Open", "High", "Low", "Close"]:
    if col in df.columns:
        df[f"Prev_{col}"] = df.groupby("Ticker")[col].shift(1)


# ==========================================================
# TUNE PARAMETERS (deterministic)
# ==========================================================
print("[INFO] Tuning SellScore weights per regime (w_crash fixed, crash is volatility-based)...")
tuned = tune_params_by_regime(df)

TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(TUNED_PARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "config": {
                "HOLD_DAYS": HOLD_DAYS,
                "EARLY_DAYS": EARLY_DAYS,
                "EPS_PEAK_BY_REGIME": EPS_PEAK_BY_REGIME,
                "D_CAP": D_CAP,
                "G_CAP": G_CAP,
                "L_DELAY": L_DELAY,
                "L_RISK": L_RISK,
                "N_FOLDS": N_FOLDS,
                "EMBARGO_CAL_DAYS": EMBARGO_CAL_DAYS,
                "USE_VOL_BASED_CRASH": USE_VOL_BASED_CRASH,
                "VOL_COL": VOL_COL,
                "VOL_MULT_BY_REGIME": VOL_MULT_BY_REGIME,
                "CRASH_THR_MIN": CRASH_THR_MIN,
                "CRASH_THR_MAX": CRASH_THR_MAX,
                "FIXED_W_CRASH": FIXED_W_CRASH,
                "W_PEAK_STEP": W_PEAK_STEP,
            },
            "tuned_params_by_regime": tuned
        },
        f,
        indent=2
    )
print(f"[INFO] Saved tuned parameters to: {TUNED_PARAMS_PATH}")


# ==========================================================
# COMPUTE FINAL SellScore (checkpoint-safe)
# ==========================================================
out, processed = [], 0
if TEMP_PATH.exists():
    print("[INFO] Resuming from checkpoint...")
    df_done = pd.read_parquet(TEMP_PATH)
    done_keys = set(zip(df_done["Ticker"].astype(str), df_done["MarketTrend"].astype(int), df_done["Date"].astype("datetime64[ns]")))
else:
    df_done, done_keys = pd.DataFrame(), set()

# Compute per constant-regime segment
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
df["_regime_change"] = df.groupby("Ticker")["MarketTrend"].apply(lambda s: s.ne(s.shift(1)).cumsum()).to_numpy()

grouped = df.groupby(["Ticker", "_regime_change"], sort=False)

for (ticker, seg_id), g in tqdm(grouped, desc="Scoring segments"):
    if len(g) < (EARLY_DAYS + 10):
        continue

    # Skip if already done (conservative key)
    key = (str(ticker), int(g["MarketTrend"].iloc[0]), g["Date"].iloc[0].to_datetime64())
    if key in done_keys:
        continue

    result = compute_sellscore_for_segment(g.copy(), tuned=tuned)
    out.append(result)
    processed += 1

    if processed % SAVE_EVERY == 0:
        temp_df = pd.concat(out + [df_done], ignore_index=True)
        temp_df.to_parquet(TEMP_PATH, index=False)
        print(f"[CHECKPOINT] Saved {len(temp_df):,} rows after {processed} segments.")

df_scored = pd.concat(out + [df_done], ignore_index=True) if out else df_done
df_scored = df_scored.drop(columns=["_regime_change"], errors="ignore")


# ==========================================================
# CLEAN & ADD ROLLING AVERAGES (past-only; your original pattern)
# ==========================================================
df_scored = df_scored.drop(columns=["SellLabel"], errors="ignore")
df_scored = df_scored.drop(columns=["score"], errors="ignore")

rename_map = {
    "Low": "avg_low_raw",
    "High": "avg_high_raw",
    "Open": "open_raw",
    "Close": "close_raw",
}
df_scored = df_scored.rename(columns={k: v for k, v in rename_map.items() if k in df_scored.columns})

drop_cols = ["Adj Close", "Volume"]
df_scored = df_scored.drop(columns=[c for c in drop_cols if c in df_scored.columns], errors="ignore")

if "Prev_Open" in df_scored.columns:
    df_scored["avg_open_past_3_days"] = df_scored.groupby("Ticker")["Prev_Open"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
if "Prev_Close" in df_scored.columns:
    df_scored["avg_close_past_3_days"] = df_scored.groupby("Ticker")["Prev_Close"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

if "avg_low_raw" in df_scored.columns:
    df_scored["avg_low_30"] = df_scored.groupby("Ticker")["avg_low_raw"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
if "avg_high_raw" in df_scored.columns:
    df_scored["avg_high_30"] = df_scored.groupby("Ticker")["avg_high_raw"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

if "avg_high_30" in df_scored.columns and "avg_low_30" in df_scored.columns:
    df_scored["volatility_30"] = (df_scored["avg_high_30"] - df_scored["avg_low_30"]) / (df_scored["avg_low_30"] + 1e-6)

# Drop raw OHLC if any still exist
drop_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_scored = df_scored.drop(columns=[c for c in drop_cols if c in df_scored.columns], errors="ignore")

# Drop fundamentals
drop_fundamentals = [
    "PE", "PEG", "PS", "PB", "DividendYield",
    "Beta", "MarketCap", "YoY_Growth",
]
df_scored = df_scored.drop(columns=[c for c in drop_fundamentals if c in df_scored.columns], errors="ignore")

if "close_raw" in df_scored.columns:
    df_scored["current_price"] = df_scored["close_raw"]
else:
    print("[WARNING] 'close_raw' column not found — could not create 'current_price'.")

print(f"[INFO] Remaining columns: {df_scored.columns.tolist()}")


# ==========================================================
# SAVE
# ==========================================================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_scored.to_parquet(OUTPUT_PATH, index=False)

if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"[SUCCESS] Saved dataset with SellScore ({len(df_scored):,} rows) → {OUTPUT_PATH}")
print(f"[SUCCESS] Tuned params JSON → {TUNED_PARAMS_PATH}")
