# scripts/replay_backtest.py
"""
Replay a single ticker on a past date using your pipeline's scoring logic.

Usage (from repo root):
  python -m scripts.replay_backtest --ticker NVDA --asof 2025-02-20
  python scripts/replay_backtest.py --ticker NVDA --asof 2025-02-20
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pathlib import Path

# --- Make imports work both as "module" and "script" runs ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    # when run as a plain script: python scripts/replay_backtest.py
    from scripts.quick_scorer import quick_score
    from scripts.trash_ranker import RobustRanker, RankerParams
except Exception:
    # when run as a module: python -m scripts.replay_backtest
    from quick_scorer import quick_score
    from trash_ranker import RobustRanker, RankerParams

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please `pip install yfinance pandas numpy` in your environment.") from e

# ---------- helpers ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr

def atr_pct(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return (atr / df["Close"]) * 100.0

def anchored_vwap(close: pd.Series, volume: pd.Series, lookback: int = 252) -> pd.Series:
    price_vol = close * volume
    num = price_vol.rolling(lookback).sum()
    den = volume.rolling(lookback).sum()
    return (num / den).replace([np.inf, -np.inf], np.nan)

def _scalar(x):
    """Return a plain float if possible. Handles pandas Series/Index, numpy scalars/arrays gracefully."""
    try:
        if isinstance(x, pd.Series):
            if x.size == 0:
                return None
            xn = x.dropna()
            if xn.size == 0:
                return None
            return float(xn.iloc[-1])
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            return float(x.reshape(-1)[-1])
        val = float(x)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None

def pct(x):
    """Like _scalar() but returns None for non-finite."""
    return _scalar(x)

def _fmt(x, nd=2):
    try:
        v = _scalar(x)
        return "N/A" if v is None else f"{v:.{nd}f}"
    except Exception:
        return "N/A"

def _fmt_parts(d: dict) -> dict:
    """Round only numeric-like entries; leave strings/None as-is."""
    out = {}
    for k, v in (d or {}).items():
        try:
            fv = float(v)
            if math.isfinite(fv):
                out[k] = round(fv, 3)
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out

# ---------- deterministic plan builder (trend-aware) ----------
def build_plan(price, fva_hint, ev_pct, *, rsi, vsSMA50, vsSMA200,
               vol_vs20, is_20d_high, sma50=None, ema50=None):
    """
    Returns (plan_text, fva_used)

    Changes vs previous:
      • Trend-aware anchor catch-up: FVA := max(FVA_HINT, 0.98×SMA50, 0.96×EMA50)
      • Early-trend EV floor (broadens buy range on low-ATR megacaps):
            if 52≤RSI≤66 AND 0≤vsSMA50≤8 AND vsSMA200≥10 → EV := max(EV, PLAN_EARLY_TREND_EV_MIN=4.0)
      • Keep blowoff veto. Keep global ±20% FVA clamp unless very-strong breakout.
    """
    def g(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    price = _scalar(price) or 0.0
    rsi = _scalar(rsi) or 0.0
    vsSMA50 = _scalar(vsSMA50) or 0.0
    vsSMA200 = _scalar(vsSMA200) if vsSMA200 is not None else None
    vol_vs20 = _scalar(vol_vs20) or 0.0
    fva = _scalar(fva_hint) or 0.0
    sma50 = _scalar(sma50)
    ema50 = _scalar(ema50)

    # --- base EV (clamped) ---
    EV = max(1.0, min(6.0, float(_scalar(ev_pct) or 2.0)))

    # --- trend-aware anchor catch-up ---
    # Use fast trend anchors to avoid stale AVWAP during persistent uptrends
    anchors = [x for x in [
        fva,
        (0.98 * sma50) if sma50 is not None else None,
        (0.96 * ema50) if ema50 is not None else None,
    ] if x is not None and math.isfinite(x)]
    if anchors:
        fva = max(anchors)

    # Early-trend EV floor (widens buy band a bit on low-ATR names)
    early_trend = (52.0 <= rsi <= 66.0) and (0.0 <= vsSMA50 <= 8.0) and ((vsSMA200 is None) or (vsSMA200 >= 10.0))
    if early_trend:
        EV = max(EV, g("PLAN_EARLY_TREND_EV_MIN", 4.0))
        # gently nudge anchor toward price but keep a discount cushion
        # between 2%–8% below price proportional to vsSMA50 (0..8%)
        discount = 0.08 - (vsSMA50 / 8.0) * 0.06  # 8% -> 2%
        fva = max(fva, price * (1.0 - discount))

    # --- very-strong breakout override for 20% clamp ---
    very_strong = (bool(is_20d_high) and rsi >= 75 and vsSMA50 >= 20 and vol_vs20 >= 150)
    if not very_strong:
        lower = price * 0.80
        upper = price * 1.20
        fva = max(lower, min(upper, fva))

    # ---- Overheat/BLOWOFF veto (env-tunable) ----
    A_RSI  = g("OVERHEAT_A_RSI",  75)
    A_VS50 = g("OVERHEAT_A_VS50", 30)
    A_VOL  = g("OVERHEAT_A_VOL",  80)
    B_RSI  = g("OVERHEAT_B_RSI",  80)
    B_VS50 = g("OVERHEAT_B_VS50", 40)
    C_RSI  = g("OVERHEAT_C_RSI",  78)
    C_VS50 = g("OVERHEAT_C_VS50", 60)

    overheat = (
        (rsi >= A_RSI and vsSMA50 >= A_VS50 and vol_vs20 >= A_VOL) or
        (rsi >= B_RSI and vsSMA50 >= B_VS50) or
        (rsi >= C_RSI and vsSMA50 >= C_VS50)
    )
    if overheat:
        deep_discount_ok = price <= (fva * (1 - 0.03 * EV))
        if not deep_discount_ok:
            plan = f"No trade — momentum blowoff; wait for cooling. (Anchor: ${fva:.2f})"
            return plan, fva

    # --- standard plan from FVA & EV ---
    buy_lo = fva * (1 - 0.8*EV/100.0)
    buy_hi = fva * (1 + 0.8*EV/100.0)
    stop   = fva * (1 - 2.0*EV/100.0)
    target = fva * (1 + 3.0*EV/100.0)

    if stop >= buy_lo:
        stop = min(buy_lo*0.99, fva*(1 - 2.2*EV/100.0))
    if target <= buy_hi:
        target = max(buy_hi*1.05, fva*(1 + 3.2*EV/100.0))

    plan = f"Buy {buy_lo:.2f}–{buy_hi:.2f}; Stop {stop:.2f}; Target {target:.2f}; Max hold time: ≤ 1 year (Anchor: ${fva:.2f})"
    if target <= price:
        plan = f"No trade — extended; wait for pullback. (Anchor: ${fva:.2f})"
    elif price > buy_hi and price < target:
        plan += " (Wait for pullback into range.)"
    elif price < buy_lo and ((buy_lo - price)/price >= (2*EV)/100.0):
        plan += " (Accumulation zone)"
    return plan, fva

# ---------- feature builder ----------
def build_feats_from_history(df: pd.DataFrame) -> dict:
    # Work on a clean, monotonic index (drop dupes just in case)
    df = df[~df.index.duplicated(keep="last")].copy()

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    sma50 = sma(close, 50)
    sma200 = sma(close, 200)

    rsi14 = rsi_wilder(close, 14)
    atrp14 = atr_pct(df, 14)

    d5 = (close / close.shift(5) - 1.0) * 100.0
    d20 = (close / close.shift(20) - 1.0) * 100.0
    r60 = (close / close.shift(60) - 1.0) * 100.0
    r120 = (close / close.shift(120) - 1.0) * 100.0

    vol20 = vol.rolling(20).mean()
    vol_vs20 = (vol / vol20 - 1.0) * 100.0

    avwap252 = anchored_vwap(close, vol, 252)

    vsSMA20 = (close / sma(close, 20) - 1.0) * 100.0
    vsSMA50 = (close / sma50 - 1.0) * 100.0
    vsSMA200 = (close / sma200 - 1.0) * 100.0

    vsEMA50 = (close / ema50 - 1.0) * 100.0
    vsEMA200 = (close / ema200 - 1.0) * 100.0

    ema50_slope_5d = (ema50 / ema50.shift(5) - 1.0) * 100.0

    # drawdown vs 252d high
    roll_max = close.rolling(252).max()
    drawdown_pct = (close / roll_max - 1.0) * 100.0

    # is 20d high? robust single-boolean
    is_20d_high_series = (close >= close.rolling(20).max())
    is20_arr = is_20d_high_series.tail(1).astype("boolean").to_numpy()
    is20_flag = bool(is20_arr[0]) if is20_arr.size else False

    # avg dollar vol 20d
    adv20 = (close * vol).rolling(20).mean()

    feats = {
        "price": pct(close.iloc[-1]),
        "SMA50": pct(sma50.iloc[-1]),
        "SMA200": pct(sma200.iloc[-1]),
        "EMA20": pct(ema20.iloc[-1]),
        "EMA50": pct(ema50.iloc[-1]),
        "EMA200": pct(ema200.iloc[-1]),
        "AVWAP252": pct(avwap252.iloc[-1]),
        "RSI14": pct(rsi14.iloc[-1]),
        "ATRpct": pct(atrp14.iloc[-1]),
        "d5": pct(d5.iloc[-1]),
        "d20": pct(d20.iloc[-1]),
        "r60": pct(r60.iloc[-1]),
        "r120": pct(r120.iloc[-1]),
        "vol_vs20": pct(vol_vs20.iloc[-1]),
        "vsSMA20": pct(vsSMA20.iloc[-1]),
        "vsSMA50": pct(vsSMA50.iloc[-1]),
        "vsSMA200": pct(vsSMA200.iloc[-1]),
        "vsEMA50": pct(vsEMA50.iloc[-1]),
        "vsEMA200": pct(vsEMA200.iloc[-1]),
        "EMA50_slope_5d": pct(ema50_slope_5d.iloc[-1]),
        "drawdown_pct": pct(drawdown_pct.iloc[-1]),
        "is_20d_high": is20_flag,
        "avg_dollar_vol_20d": pct(adv20.iloc[-1]),
    }
    return feats

def _fetch_pe_fast(ticker: str):
    try:
        info = yf.Ticker(ticker).fast_info
        pe = getattr(info, "trailing_pe", None)
        if pe is None:
            pe = getattr(info, "pe_ratio", None)
        pe = float(pe) if pe is not None else None
        if pe and pe > 0 and math.isfinite(pe):
            return pe
    except Exception:
        return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--period_days", type=int, default=420, help="history window to fetch before as-of")
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()
    asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
    start = asof - timedelta(days=args.period_days)

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(asof + timedelta(days=1)).isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise SystemExit(f"No data for {ticker} up to {asof}")

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna().copy()

    feats = build_feats_from_history(df)

    # Base anchor hint = AVWAP252; the plan function will trend-adjust it
    fva_hint = feats.get("AVWAP252") or feats.get("SMA50")
    pe_hint = _fetch_pe_fast(ticker)

    # Stage-1
    s1_score, s1_parts = quick_score(feats, mode=os.getenv("STAGE1_MODE", "loose"))

    # Stage-2 robust ranker
    ranker = RobustRanker(params=RankerParams())
    ranker.fit_cross_section([feats])  # neutral z
    drop = ranker.should_drop(feats)
    s2_score, s2_parts = ranker.composite_score(feats)

    # Deterministic PLAN (trend-aware)
    price = feats.get("price")
    ev = max(1.0, min(6.0, feats.get("ATRpct") or 2.0))
    plan_text, fva_used = build_plan(
        price=price,
        fva_hint=fva_hint,
        ev_pct=ev,
        rsi=feats.get("RSI14"),
        vsSMA50=feats.get("vsSMA50"),
        vsSMA200=feats.get("vsSMA200"),
        vol_vs20=feats.get("vol_vs20"),
        is_20d_high=feats.get("is_20d_high"),
        sma50=feats.get("SMA50"),
        ema50=feats.get("EMA50"),
    )

    print("\n=== Replay Result ===")
    print(f"Ticker: {ticker}  |  As-of: {asof}")
    print(f"Price: ${_fmt(price)}  |  FVA_HINT(AVWAP252): ${_fmt(fva_hint)}  |  EV(ATR%): {_fmt(ev)}%")
    print(f"RSI14: {_fmt(feats.get('RSI14'))}  |  vsSMA50: {_fmt(feats.get('vsSMA50'))}%  |  vsSMA200: {_fmt(feats.get('vsSMA200'))}%  |  vol_vs20: {_fmt(feats.get('vol_vs20'))}%  |  is_20d_high: {feats.get('is_20d_high')}")

    if pe_hint is not None:
        print(f"PE_HINT (trailing): {_fmt(pe_hint)}")

    print("\nStage-1 quick_score:")
    print(f"  Score: {s1_score:.2f} | parts: {_fmt_parts(s1_parts)}")

    print("\nStage-2 RobustRanker:")
    print(f"  Dropped by hard filter? {drop}")
    print(f"  Score: {s2_score:.2f} | parts: {_fmt_parts(s2_parts)}")

    print("\nPLAN:")
    print(f"  {plan_text}")

    verdict = "No-trade (extended)" if plan_text.startswith("No trade") else \
              "Actionable BUY zone" if plan_text.startswith("Buy ") else "Neutral"
    print("\nSummary:")
    print(f"  Verdict: {verdict}")
    reasons = []
    if (feats.get("vsSMA50") or 0) > 18: reasons.append("Modest/strong overextension vs SMA50")
    if (feats.get("RSI14") or 0) >= 75: reasons.append("RSI very hot")
    if (price or 0) > (fva_used or 0) * 1.20: reasons.append("Price 20%+ above anchor")
    if (feats.get("vol_vs20") or 0) < -30: reasons.append("Weak participation (sub-avg volume)")
    if (feats.get("drawdown_pct") or -100) > -1: reasons.append("Near 52w high")
    if not reasons: reasons.append("None material")
    print("   - " + "; ".join(reasons))

if __name__ == "__main__":
    main()
