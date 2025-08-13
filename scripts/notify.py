import os
import re
import sys
import time
import json
from datetime import datetime

import pandas as pd
import requests
import pytz
import yfinance as yf  # earnings + fundamentals

from universe import load_universe
from features import build_features, fetch_history, compute_indicators
from filters import is_garbage, daily_index_filter
from scoring import base_importance_score
from prompts import SYSTEM_PROMPT_TOP20, USER_PROMPT_TOP20_TEMPLATE

TZ = pytz.timezone("Europe/Bucharest")
OPENAI_BASE = "https://api.openai.com/v1"

def log(m): print(m, flush=True)

def need_env(name):
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Missing env: {name}", flush=True); sys.exit(1)
    return v

OPENAI_API_KEY = need_env("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = need_env("DISCORD_WEBHOOK_URL")
force = os.getenv("FORCE_RUN", "").lower() in {"1","true","yes"}

# ---------- Strengthen system prompt so missing data stays NEUTRAL and proxies can move categories ----------
SYSTEM_PROMPT_TOP20 += """
INPUT EXTRAS:
- Each candidate block may include:
  • DATA_AVAILABILITY: which of {FUNDAMENTALS, VALUATION, RISKS, CATALYSTS} are MISSING or PARTIAL.
  • BASELINE_HINTS: exact baselines for each category.
  • VALUATION FIELDS (if present): PE, PE_SECTOR, EV_EBITDA, PS, FCF_YIELD, PEG.
  • PROXIES: simple, price/volume-only signals with 1–5 severity and a direction (+/-).
  • PROXIES_FUNDAMENTALS: {GROWTH_TECH, MARGIN_TREND_TECH, FCF_TREND_TECH, OP_EFF_TREND_TECH} as signed severities (−5..+5).
  • PROXIES_CATALYSTS: {TECH_BREAKOUT, TECH_BREAKDOWN, DIP_REVERSAL, EARNINGS_SOON} with signed severities like +1..+5 or -1..-5.
  • CATALYST_TIMING_HINTS: TECH_BREAKOUT=Today/None.
  • EXPECTED_VOLATILITY_PCT: derived from ATR%, use as 'Expected volatility' in the Certainty rule.
  • FVA_HINT: a technical fair-value anchor seed, derived only from supplied indicators.

MANDATORY HANDLING:
- If DATA_AVAILABILITY says a category is MISSING ⇒ set that category exactly to its BASELINE_HINTS value (do NOT set 0).
- If PARTIAL ⇒ use only the provided FIELDS and PROXIES for that category; keep unaddressed sub-factors at baseline.
- Never assume unknown = bad; unknown = baseline.
- PROXIES map 1–5 severity to the factor ranges you already have (e.g., 'Market trend +3' means add a positive amount near the mid of +10…+50).
- CATALYST PROXY MAPPING (apply to 'Near-Term Catalysts' base 100):
    • TECH_BREAKOUT (+1..+5)  ⇒ +10, +20, +30, +45, +60 (use stronger end if volume is explicitly strong).
    • DIP_REVERSAL (+1..+5)   ⇒ +8, +12, +18, +24, +30 (only if recent drawdown then firming momentum).
    • TECH_BREAKDOWN (-1..-5) ⇒ −10, −20, −30, −45, −60.
    • EARNINGS_SOON (+1..+5)  ⇒ +5, +8, +10, +12, +15.
  Timing multiplier applies for TECH_BREAKOUT timing: Today×1.50.
- FUNDAMENTALS PROXY MAPPING (start at 125 baseline; clamp total from these tech proxies to ±35 overall):
    • GROWTH_TECH (±1..±5)        → Revenue/EPS growth: ±5, ±10, ±15, ±25, ±35
    • MARGIN_TREND_TECH (±1..±5)  → Margin trend: ±4, ±8, ±12, ±20, ±30
    • FCF_TREND_TECH (±1..±5)     → FCF trend: ±4, ±8, ±12, ±20, ±30
    • OP_EFF_TREND_TECH (±1..±5)  → Operational efficiency: ±3, ±6, ±10, ±15, ±20
- You MAY start $FVA from FVA_HINT, adjusting modestly with the same indicators; do not invent external data.
"""

# ------- OpenAI helpers -------
def extract_output_text(data: dict) -> str | None:
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", "").strip()
    return None

def call_gpt5(system_msg: str, user_msg: str, max_tokens: int = 7000) -> str:
    url = f"{OPENAI_BASE}/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "gpt-5",
        "input": f"[System]: {system_msg}\n[User]: {user_msg}",
        "max_output_tokens": max_tokens,
        "temperature": 1.0
    }
    r = requests.post(url, headers=headers, json=body, timeout=180)
    r.raise_for_status()
    data = r.json()
    text = extract_output_text(data)
    if not text:
        raise RuntimeError(f"Empty output from Responses API: {str(data)[:400]}")
    return text

# ------- Personal bonus post-processor -------
BONUS_POINTS = {
    "AI_LEADER": 30,
    "DIP_5_12": 50,
    "EARNINGS_BEAT": 25,
    "INSIDER_BUYING_30D": 20,
    "ANALYST_UPGRADES_7D": 15,
    "SECTOR_ROTATION_TAILWIND": 10,
    "BREAKOUT_VOL_CONF": 20,
    "SHORT_SQUEEZE_SETUP": 15,
    "INSTITUTIONAL_ACCUMULATION": 15,
    "POSITIVE_OPTIONS_SKEW": 10,
    "DIVIDEND_SAFETY_GROWTH": 5,
    "ESG_MOMENTUM": 5,
}

RE_BASE = re.compile(r"^\s*6\)\s*Final base score:\s*(\d{1,4})\s*$", re.MULTILINE)
RE_BONUS_LINE = re.compile(r"^\s*3\)\s*Bonuses:\s*(.+)$", re.MULTILINE)
RE_PERSONAL = re.compile(r"^\s*7\)\s*Personal adjusted score:\s*.*$", re.MULTILINE)

def _parse_bonus_flags(bonus_line: str):
    raw = bonus_line.strip()
    if raw.upper() == "NONE": return []
    flags = [x.strip().upper().replace("-", "_") for x in raw.split(",")]
    return [f for f in flags if f in BONUS_POINTS]

def apply_personal_bonuses_to_text(gpt_text: str) -> str:
    blocks = re.split(r"\n\s*\n", gpt_text.strip())
    new_blocks = []
    for b in blocks:
        base_m = RE_BASE.search(b); bonus_m = RE_BONUS_LINE.search(b)
        if not base_m or not bonus_m:
            new_blocks.append(b); continue
        try:
            base = int(base_m.group(1))
        except Exception:
            new_blocks.append(b); continue

        flags = _parse_bonus_flags(bonus_m.group(1))
        bonus_sum = sum(BONUS_POINTS[f] for f in flags)
        adjusted = base + bonus_sum
        breakdown = " + ".join([f"{BONUS_POINTS[f]} {f}" for f in flags]) if flags else "+0"
        personal_line = f"7) Personal adjusted score: {adjusted} (Base {base}{(' + ' + breakdown) if flags else ''})".rstrip()

        if RE_PERSONAL.search(b):
            b2 = RE_PERSONAL.sub(personal_line, b, count=1)
        else:
            lines = b.splitlines()
            out_lines, inserted = [], False
            for line in lines:
                out_lines.append(line)
                if RE_BASE.match(line) and not inserted:
                    out_lines.append(personal_line); inserted = True
            b2 = "\n".join(out_lines)
        new_blocks.append(b2)
    return "\n\n".join(new_blocks)

# ------- Debug: post the exact inputs to Discord -------
def post_debug_inputs_to_discord(tickers: list[str], debug_inputs: dict[str, dict]):
    """Send a JSON dump of the exact inputs we gave the model for each selected ticker."""
    for t in tickers:
        data = debug_inputs.get(t)
        if not data:
            continue
        try:
            blob = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            safe = {k: (str(v) if not isinstance(v, (str, int, float, bool, dict, list)) else v) for k, v in (data or {}).items()}
            blob = json.dumps(safe, indent=2, ensure_ascii=False)

        # Discord embed description limit ~4096 chars; keep headroom
        if len(blob) > 3900:
            blob = blob[:3900] + "\n... (truncated)"

        payload = {
            "username": "Daily Stock Debug",
            "embeds": [{
                "title": f"Model inputs — {t}",
                "description": f"json\n{blob}\n"
            }]
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=60).raise_for_status()
        except Exception as e:
            log(f"[WARN] Debug Discord post failed for {t}: {repr(e)}")

# ------- Proxy engineering for missing fundamentals/valuation -------
BASELINE_HINTS = {
    "MKT_SECTOR": 200,
    "FUNDAMENTALS": 125,
    "CATALYSTS": 100,
    "VALUATION": 50,
    "RISKS": 50,
}

def clamp(n, lo, hi): return max(lo, min(hi, n))

def get_spy_ctx() -> dict:
    try:
        spy_hist = fetch_history(["SPY"])
        df = spy_hist.get("SPY")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("SPY history empty")
        feats = compute_indicators(df)
        return {
            "vsSMA50": feats.get("vsSMA50") or 0.0,
            "vsSMA200": feats.get("vsSMA200") or 0.0,
            "r60": feats.get("r60") or 0.0,
        }
    except Exception as e:
        log(f"[WARN] SPY context unavailable ({e}); using neutral market.")
        return {"vsSMA50": 0.0, "vsSMA200": 0.0, "r60": 0.0}

def severity_from_return(pct: float, pos_thresholds=(2, 5, 10, 15)) -> int:
    a = abs(pct or 0.0)
    if a >= pos_thresholds[3]: return 5
    if a >= pos_thresholds[2]: return 4
    if a >= pos_thresholds[1]: return 3
    if a >= pos_thresholds[0]: return 2
    return 1

def severity_from_value(a: float, cuts=(2, 4, 6, 9)) -> int:
    x = abs(a or 0.0)
    if x >= cuts[3]: return 5
    if x >= cuts[2]: return 4
    if x >= cuts[1]: return 3
    if x >= cuts[0]: return 2
    return 1

def derive_proxies(feats: dict, spy: dict) -> dict:
    # Market trend from SPY regime
    mt_sign = 1 if (spy["vsSMA50"] > 0 and spy["vsSMA200"] > 0 and spy["r60"] > 0) else (-1 if (spy["vsSMA50"] < 0 and spy["vsSMA200"] < 0 and spy["r60"] < 0) else 0)
    mt_sev = 3 if mt_sign != 0 else 1
    market_trend = f"{'+' if mt_sign>0 else '-' if mt_sign<0 else ''}{mt_sev}"

    # Relative strength 60d vs SPY
    rs_60 = (feats.get("r60") or 0.0) - (spy["r60"] or 0.0)
    rs_sign = 1 if rs_60 > 0 else (-1 if rs_60 < 0 else 0)
    rs_sev = severity_from_return(rs_60, (2, 5, 10, 15))
    rel_strength = f"{'+' if rs_sign>0 else '-' if rs_sign<0 else ''}{rs_sev}"

    # Breadth/volume proxy from vol_vs20
    vv20 = feats.get("vol_vs20")
    if vv20 is None:
        breadth_volume = "0"
    else:
        sign = 1 if vv20 > 10 else (-1 if vv20 < -10 else 0)
        sev = 1 if sign == 0 else (2 if abs(vv20) < 25 else 3 if abs(vv20) < 50 else 4)
        breadth_volume = f"{'+' if sign>0 else '-' if sign<0 else ''}{sev if sign!=0 else 0}"

    # Valuation (history-only) from vsSMA200 (discount=positive)
    vs200 = feats.get("vsSMA200")
    if vs200 is None:
        valuation_hist = "0"
    else:
        sign = 1 if vs200 < 0 else (-1 if vs200 > 0 else 0)
        sev = severity_from_return(vs200, (3, 7, 12, 20))
        valuation_hist = f"{'+' if sign>0 else '-' if sign<0 else ''}{sev if sign!=0 else 0}"

    # Risks from volatility (ATR%) and drawdown
    atrp = feats.get("ATRpct")
    dd_mag = -(feats.get("drawdown_pct") or 0.0)
    risk_vol = str(severity_from_value(atrp or 0.0))
    risk_drawdown = str(severity_from_value(dd_mag or 0.0))

    # Suggested bonuses (technical)
    bonuses = []
    drawdown = abs(feats.get("drawdown_pct") or 0.0)
    if 5.0 <= drawdown <= 12.0:
        bonuses.append("DIP_5_12")
    if feats.get("is_20d_high") and (vv20 is not None and vv20 > 20):
        bonuses.append("BREAKOUT_VOL_CONF")

    # Expected volatility & FVA hint (SMA50/AVWAP)
    ev = feats.get("ATRpct") or 4.0
    ev = max(1.0, min(ev, 6.0))  # clamp to 1–6%
    sma50 = feats.get("SMA50"); avwap = feats.get("AVWAP252")
    fva_hint = None
    if sma50 and avwap:
        fva_hint = round((sma50 + avwap) / 2.0, 2)
    elif sma50:
        fva_hint = round(sma50, 2)
    elif avwap:
        fva_hint = round(avwap, 2)

    return dict(
        market_trend=market_trend,
        relative_strength=rel_strength,
        breadth_volume=breadth_volume,
        valuation_history=valuation_hist,
        risk_volatility=risk_vol,
        risk_drawdown=risk_drawdown,
        expected_volatility_pct=round(ev, 2),
        fva_hint=fva_hint,
        suggested_bonuses=",".join(bonuses) if bonuses else "NONE",
    )

def fund_proxies_from_feats(feats: dict) -> dict:
    """
    Lightweight OHLCV-only heuristics to gently nudge Fundamentals.
    We keep magnitudes small and cap total later via the prompt.
    """
    def sev_from(p, cuts):
        a = abs(p or 0.0)
        if a >= cuts[3]: return 5
        if a >= cuts[2]: return 4
        if a >= cuts[1]: return 3
        if a >= cuts[0]: return 2
        return 1

    r60  = feats.get("r60") or 0.0
    r120 = feats.get("r120") or 0.0
    vs200 = feats.get("vsSMA200")
    rsi  = feats.get("RSI14") or 50
    macd = feats.get("MACD_hist") or 0.0
    atrp = feats.get("ATRpct")

    # GROWTH_TECH: average of 60d & 120d returns
    g_raw  = (r60 + r120) / 2.0
    g_sign = 1 if g_raw > 0 else (-1 if g_raw < 0 else 0)
    g_sev  = sev_from(g_raw, (5, 10, 20, 30)) * g_sign if g_sign != 0 else 0

    # MARGIN_TREND_TECH: momentum quality (MACD>0 & RSI>55 vs MACD<0 & RSI<45)
    mt_sign = 1 if (macd > 0 and rsi > 55) else (-1 if (macd < 0 and rsi < 45) else 0)
    m_sev = 0
    if mt_sign != 0:
        macd_metric = abs(macd)
        m_sev = (1 if macd_metric < 0.2 else
                 2 if macd_metric < 0.5 else
                 3 if macd_metric < 1.0 else
                 4 if macd_metric < 1.5 else
                 5)
        m_sev *= mt_sign

    # FCF_TREND_TECH: below 200d ~ positive value/FCF tailwind; above ~ headwind
    f_sev = 0
    if vs200 is not None and vs200 != 0:
        f_sign = 1 if vs200 < 0 else -1
        f_sev = sev_from(vs200, (3, 7, 12, 20)) * f_sign

    # OP_EFF_TREND_TECH: lower ATR% implies smoother execution; higher implies noise
    o_sev = 0
    if atrp is not None:
        if atrp <= 3:    o_sev = +3
        elif atrp <= 5:  o_sev = +2
        elif atrp <= 7:  o_sev = -2
        elif atrp <= 10: o_sev = -3
        else:            o_sev = -4

    def clip5(x): return max(-5, min(5, int(x)))
    return {
        "GROWTH_TECH":       clip5(g_sev),
        "MARGIN_TREND_TECH": clip5(m_sev),
        "FCF_TREND_TECH":    clip5(f_sev),
        "OP_EFF_TREND_TECH": clip5(o_sev),
    }

# ------- Earnings window (top-20 only, best-effort) -------
def fetch_next_earnings_days(tickers: list[str]) -> dict[str, int | None]:
    """Return days until next earnings for each ticker, or None if unknown."""
    out: dict[str, int | None] = {}
    today = datetime.now(TZ).date()
    for t in tickers:
        hy = t.replace(".", "-")
        days: int | None = None
        try:
            tk = yf.Ticker(hy)
            # Newer API
            try:
                df = tk.get_earnings_dates(limit=6)
                if df is not None and not df.empty:
                    if isinstance(df.index, pd.DatetimeIndex):
                        dates = [d.date() for d in df.index.to_pydatetime()]
                    elif "Earnings Date" in df.columns:
                        dates = [pd.to_datetime(x).date() for x in df["Earnings Date"]]
                    else:
                        dates = []
                    fut = [d for d in dates if d >= today]
                    if fut:
                        nd = min(fut)
                        days = (nd - today).days
            except Exception:
                pass
            # Fallback
            if days is None:
                cal = tk.calendar
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    val = None
                    if "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"].values[0]
                    elif "Earnings Date" in cal.columns:
                        val = cal["Earnings Date"].iloc[0]
                    if val is not None:
                        d = pd.to_datetime(val).date()
                        if d >= today:
                            days = (d - today).days
        except Exception:
            pass
        out[t] = days
    return out

# ------- Fundamentals & valuation (robust yfinance matchers) -------
import re as _re

def _find_row(df: pd.DataFrame, patterns: list[str]) -> pd.Series | None:
    """Case/space-insensitive row finder for yfinance frames."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    norm = {_re.sub(r"\s+", "", str(idx).lower()): idx for idx in df.index}
    for pat in patterns:
        key = _re.sub(r"\s+", "", pat.lower())
        for k, orig in norm.items():
            if key == k or key in k:
                return df.loc[orig]
    return None

def _latest_qyoy(series: pd.Series) -> float | None:
    try:
        s = series.dropna().astype(float).sort_index()
        if s.shape[0] < 5: return None
        prev, cur = float(s.iloc[-5]), float(s.iloc[-1])
        if prev == 0: return None
        return (cur/prev - 1.0) * 100.0
    except Exception:
        return None

def _ttm_sum(series: pd.Series) -> float | None:
    try:
        s = series.dropna().astype(float).sort_index()
        if s.shape[0] < 4: return None
        return float(s.iloc[-4:].sum())
    except Exception:
        return None

def _pct(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom == 0: return None
    return (numer/denom) * 100.0

def fetch_funda_valuation_for_top(tickers: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in tickers:
        hy = t.replace(".", "-")
        m = {
            "REV_GROWTH_YOY": None,
            "EPS_GROWTH_YOY": None,
            "GROSS_MARGIN": None,
            "OPER_MARGIN": None,
            "FCF_MARGIN": None,
            "DEBT_TO_EBITDA": None,
            "NET_CASH": None,
            "GUIDANCE_CHANGE": None,
            "OP_EFF_TREND": None,
            "PE": None,
            "PE_SECTOR": None,
            "EV_EBITDA": None,
            "PS": None,
            "FCF_YIELD": None,
            "PEG": None,
        }
        try:
            tk = yf.Ticker(hy)

            # Income-like
            qfin = getattr(tk, "quarterly_financials", None)
            rev = _find_row(qfin, ["Total Revenue", "Revenue"])
            gp  = _find_row(qfin, ["Gross Profit"])
            opi = _find_row(qfin, ["Operating Income", "Operating Income or Loss"])
            ebitda = _find_row(qfin, ["EBITDA"])

            # Revenue YoY
            rg = _latest_qyoy(rev) if isinstance(rev, pd.Series) else None
            if rg is None:
                qearn = getattr(tk, "quarterly_earnings", None)
                if isinstance(qearn, pd.DataFrame) and "Revenue" in qearn.columns:
                    rg = _latest_qyoy(qearn["Revenue"])
            if rg is not None:
                m["REV_GROWTH_YOY"] = round(rg, 2)

            # Cashflow
            qcf = getattr(tk, "quarterly_cashflow", None)
            fcf = None
            if isinstance(qcf, pd.DataFrame):
                fcf = _find_row(qcf, ["Free Cash Flow"])
                if fcf is None:
                    ocf = _find_row(qcf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
                    capex = _find_row(qcf, ["Capital Expenditures", "Capital Expenditure"])
                    if isinstance(ocf, pd.Series) and isinstance(capex, pd.Series):
                        try:
                            fcf = (ocf - capex)
                        except Exception:
                            fcf = None

            # Balance
            qbs = getattr(tk, "quarterly_balance_sheet", None)
            cash = _find_row(qbs, ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"])
            debt = None
            if isinstance(qbs, pd.DataFrame):
                td = _find_row(qbs, ["Total Debt"])
                ltd = _find_row(qbs, ["Long Term Debt"])
                std = _find_row(qbs, ["Short Long Term Debt"])
                # sum parts if needed
                def _v(s):
                    if isinstance(s, pd.Series):
                        s2 = s.sort_index().dropna()
                        return float(s2.iloc[-1]) if not s2.empty else None
                    return None
                debt_vals = [x for x in [_v(td), _v(ltd), _v(std)] if x is not None]
                if debt_vals:
                    # reconstruct a pseudo-series with only latest value (for uniformity down below)
                    last_val = debt_vals[0] if len(debt_vals) == 1 else sum(debt_vals)
                    debt = pd.Series([last_val], index=[pd.Timestamp("today")])

            # TTM aggregates
            rev_ttm    = _ttm_sum(rev)    if isinstance(rev, pd.Series) else None
            gp_ttm     = _ttm_sum(gp)     if isinstance(gp, pd.Series) else None
            opi_ttm    = _ttm_sum(opi)    if isinstance(opi, pd.Series) else None
            fcf_ttm    = _ttm_sum(fcf)    if isinstance(fcf, pd.Series) else None
            ebitda_ttm = _ttm_sum(ebitda) if isinstance(ebitda, pd.Series) else None

            gm = _pct(gp_ttm,  rev_ttm)
            om = _pct(opi_ttm, rev_ttm)
            fm = _pct(fcf_ttm, rev_ttm)
            m["GROSS_MARGIN"] = round(gm, 2) if gm is not None else None
            m["OPER_MARGIN"]  = round(om, 2) if om is not None else None
            m["FCF_MARGIN"]   = round(fm, 2) if fm is not None else None

            # OP_EFF_TREND (QoQ change in operating margin)
            if isinstance(opi, pd.Series) and isinstance(rev, pd.Series):
                try:
                    om_q = (opi / rev).dropna().astype(float).sort_index()
                    if om_q.shape[0] >= 2:
                        delta = float(om_q.iloc[-1] - om_q.iloc[-2])
                        m["OP_EFF_TREND"] = "Up" if delta > 0 else "Down" if delta < 0 else "Flat"
                except Exception:
                    pass

            # Debt/EBITDA and Net cash
            td_latest = None
            if isinstance(debt, pd.Series):
                d2 = debt.sort_index().dropna()
                if not d2.empty:
                    td_latest = float(d2.iloc[-1])
            cash_latest = None
            if isinstance(cash, pd.Series):
                c2 = cash.sort_index().dropna()
                if not c2.empty:
                    cash_latest = float(c2.iloc[-1])
            if td_latest is not None and ebitda_ttm and ebitda_ttm != 0:
                m["DEBT_TO_EBITDA"] = round(td_latest / ebitda_ttm, 2)
            if td_latest is not None and cash_latest is not None:
                m["NET_CASH"] = bool(cash_latest - td_latest > 0)

            # Valuation
            finfo = {}
            try: finfo = dict(getattr(tk, "fast_info", {})) or {}
            except Exception: pass
            info = {}
            try: info = dict(getattr(tk, "info", {})) or {}
            except Exception: pass

            def g(d, *keys):
                for k in keys:
                    if k in d and d[k] is not None:
                        return d[k]
                return None

            market_cap = g(finfo, "market_cap") or g(info, "marketCap")
            pe  = g(finfo, "trailing_pe") or g(info, "trailingPE")
            peg = g(info, "pegRatio")
            ev_ebitda = g(info, "enterpriseToEbitda")
            ps  = g(info, "priceToSalesTrailing12Months") or g(info, "priceToSales")
            free_cf_abs = g(info, "freeCashflow")
            if free_cf_abs is None and fcf_ttm is not None:
                free_cf_abs = fcf_ttm

            m["PE"] = round(float(pe), 2) if pe is not None else None
            m["PEG"] = round(float(peg), 2) if peg is not None else None
            m["EV_EBITDA"] = round(float(ev_ebitda), 2) if ev_ebitda is not None else None
            m["PS"] = round(float(ps), 2) if ps is not None else None
            if market_cap and free_cf_abs:
                try:
                    m["FCF_YIELD"] = round(float(free_cf_abs)/float(market_cap)*100.0, 2)
                except Exception:
                    pass

        except Exception:
            pass

        out[t] = m
    return out

def catalyst_severity_from_feats(feats: dict) -> dict:
    vs50 = feats.get("vsSMA50") or 0.0
    vs200 = feats.get("vsSMA200") or 0.0
    rsi = feats.get("RSI14") or 50
    macd = feats.get("MACD_hist") or 0.0
    vv20 = feats.get("vol_vs20")
    is_high = bool(feats.get("is_20d_high"))
    dd = abs(feats.get("drawdown_pct") or 0.0)
    d5 = feats.get("d5") or 0.0

    tech_breakout = 0
    if is_high:
        if vv20 is None: tech_breakout = 1
        elif vv20 > 100: tech_breakout = 5
        elif vv20 > 50:  tech_breakout = 4
        elif vv20 > 20:  tech_breakout = 3
        elif vv20 > 5:   tech_breakout = 2
        else:            tech_breakout = 1

    tech_breakdown = 0
    if vs50 < -3 and macd < 0 and rsi < 45:
        tech_breakdown = -4 if (vs200 < -5 or rsi < 35) else -2

    dip_reversal = 0
    if 5 <= dd <= 12 and (rsi >= 50 or d5 > 0):
        dip_reversal = 3 if dd >= 10 else 2 if dd >= 8 else 1

    return {"TECH_BREAKOUT": tech_breakout, "TECH_BREAKDOWN": tech_breakdown, "DIP_REVERSAL": dip_reversal}

# ------- Wait until 08:00 -------
def seconds_until_target_hour(target_hour=8, target_min=0):
    now = datetime.now(TZ)
    target = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
    if now >= target: return 0
    return max(0, int((target - now).total_seconds()))

def fail(msg: str):
    log(f"[ERROR] {msg}")
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username":"Daily Stock Alert","content":f"⚠️ {msg}"}, timeout=60)
    except Exception:
        pass

def main():
    now = datetime.now(TZ)
    log(f"[INFO] Start {now.isoformat()} Europe/Bucharest. FORCE_RUN={force}")

    # 1) Load ~2000 universe
    universe = load_universe()
    log(f"[INFO] Universe size: {len(universe)}")

    # 2) Compute features for ALL (batched)
    feats_map = build_features(universe, batch_size=150)
    if not feats_map:
        return fail("No features computed (network/data)")

    # 3) Trash filter
    kept = []
    for t, name in universe:
        row = feats_map.get(t)
        if not row:
            continue
        feats = row["features"]
        if not is_garbage(feats):
            kept.append((t, name, feats))
    log(f"[INFO] After trash filter: {len(kept)} remain")
    if not kept:
        return fail("All filtered in trash stage")

    # 4) Daily index filter (placeholder; wire real context if you have it)
    today_context = {"bench_trend": "up", "sector_trend": "up", "breadth50": 55}
    kept2 = [(t,n,f) for (t,n,f) in kept if daily_index_filter(f, today_context)]
    log(f"[INFO] After daily index filter: {len(kept2)} remain")
    if not kept2:
        return fail("All filtered by daily context")

    # 5) Rank to 200 using a robust composite score (trash-aware, small-cap friendly)

    spy_ctx = get_spy_ctx()  # used both for scoring context and for proxy derivation below

    # ranker.score_universe also skips anything that fails its hard filter; kept2 is already trash-filtered, so this is just extra safety
    ranked = [(t, n, f, score) for (t, n, f, score, _parts) in ranked_all]
    top200 = ranked[:200]
    if not top200:
        return fail("No candidates after ranking")
    log(f"[INFO] Reduced to top 200. Example leader: {top200[0][0]}")

    # 6) Prepare TOP 10 blocks for GPT — add fundamentals/valuation for these
    top20 = top200[:10]
    tickers_top20 = [t for t, _, _, _ in top20]

    earn_days_map = fetch_next_earnings_days(tickers_top20)
    funda_map = fetch_funda_valuation_for_top(tickers_top20)

    blocks = []
    debug_inputs: dict[str, dict] = {}  # <-- capture everything we pass per ticker
    baseline_str = "; ".join([f"{k}={v}" for k, v in BASELINE_HINTS.items()])

    for t, name, feats, _ in top20:
        proxies = derive_proxies(feats, spy_ctx)

        # --- Catalysts
        cat = catalyst_severity_from_feats(feats)
        earn_days = earn_days_map.get(t)
        if earn_days is None: earn_sev = 0
        elif earn_days <= 0:  earn_sev = 5
        elif earn_days <= 3:  earn_sev = 4
        elif earn_days <= 7:  earn_sev = 3
        elif earn_days <= 14: earn_sev = 2
        else:                 earn_sev = 0

        def sgn(k:int)->str: return ("+"+str(k)) if k>0 else ("-"+str(abs(k)) if k<0 else "0")
        catalyst_line = "TECH_BREAKOUT={}; TECH_BREAKDOWN={}; DIP_REVERSAL={}; EARNINGS_SOON={}".format(
            sgn(cat["TECH_BREAKOUT"]), sgn(cat["TECH_BREAKDOWN"]), sgn(cat["DIP_REVERSAL"]), sgn(earn_sev)
        )

        # Only keep TECH_BREAKOUT timing hint (no EARNINGS_IN_DAYS)
        timing_tb = "TECH_BREAKOUT={}".format(
            "Today" if (cat["TECH_BREAKOUT"]>0 and feats.get("is_20d_high")) else "None"
        )

        # --- Fundamentals proxies
        fund_proxy = fund_proxies_from_feats(feats)
        fund_line = "GROWTH_TECH={gt}; MARGIN_TREND_TECH={mt}; FCF_TREND_TECH={ft}; OP_EFF_TREND_TECH={ot}".format(
            gt=("+"+str(fund_proxy["GROWTH_TECH"])) if fund_proxy["GROWTH_TECH"]>0 else str(fund_proxy["GROWTH_TECH"]),
            mt=("+"+str(fund_proxy["MARGIN_TREND_TECH"])) if fund_proxy["MARGIN_TREND_TECH"]>0 else str(fund_proxy["MARGIN_TREND_TECH"]),
            ft=("+"+str(fund_proxy["FCF_TREND_TECH"])) if fund_proxy["FCF_TREND_TECH"]>0 else str(fund_proxy["FCF_TREND_TECH"]),
            ot=("+"+str(fund_proxy["OP_EFF_TREND_TECH"])) if fund_proxy["OP_EFF_TREND_TECH"]>0 else str(fund_proxy["OP_EFF_TREND_TECH"]),
        )

        # --- Real Fundamentals/Valuation (partial ok)
        fm = funda_map.get(t, {}) or {}
        has_any_funda = any(fm.get(k) is not None for k in [
            "REV_GROWTH_YOY","EPS_GROWTH_YOY","GROSS_MARGIN","OPER_MARGIN","FCF_MARGIN",
            "DEBT_TO_EBITDA","NET_CASH","OP_EFF_TREND"
        ])
        has_any_val = any(fm.get(k) is not None for k in ["PE","EV_EBITDA","PS","FCF_YIELD","PEG"])
        data_availability = "FUNDAMENTALS=PARTIAL; VALUATION=PARTIAL_HISTORY" if has_any_funda else "FUNDAMENTALS=MISSING; VALUATION=PARTIAL_HISTORY"
        if has_any_val: data_availability = data_availability.replace("VALUATION=PARTIAL_HISTORY", "VALUATION=PARTIAL")

        def fmt_pct(x): return f"{x:.2f}%" if isinstance(x,(int,float)) else "N/A"
        def fmt_num(x): return f"{x:.2f}" if isinstance(x,(int,float)) else "N/A"

        val_fields = (
            "VALUATION_FIELDS: "
            f"PE={fmt_num(fm.get('PE'))}; "
            f"PE_SECTOR=N/A; "
            f"EV_EBITDA={fmt_num(fm.get('EV_EBITDA'))}; "
            f"PS={fmt_num(fm.get('PS'))}; "
            f"FCF_YIELD={fmt_pct(fm.get('FCF_YIELD'))}; "
            f"PEG={fmt_num(fm.get('PEG'))}"
        )

        # ---- coverage counts (for prompt + debug)
        funda_present = [k for k in ["REV_GROWTH_YOY","EPS_GROWTH_YOY","GROSS_MARGIN","OPER_MARGIN","FCF_MARGIN","DEBT_TO_EBITDA","NET_CASH","OP_EFF_TREND"] if fm.get(k) is not None]
        val_present   = [k for k in ["PE","EV_EBITDA","PS","FCF_YIELD","PEG"] if fm.get(k) is not None]
        funda_cov = f"{len(funda_present)}/8"
        val_cov   = f"{len(val_present)}/5"

        # ---- stash full inputs for debug Discord (without EARNINGS_IN_DAYS and without FUNDAMENTALS_FIELDS)
        debug_inputs[t] = {
            "TICKER": t,
            "COMPANY": name,
            "CURRENT_PRICE": feats.get("price"),
            "FEATURES": feats,
            "PROXIES": proxies,
            "FUNDAMENTALS_PROXIES": fund_proxy,
            "CATALYST_PROXIES": cat,
            "DATA_AVAILABILITY": data_availability,
            "COVERAGE": {"fundamentals": funda_cov, "valuation": val_cov},
        }

        blocks.append(
            "TICKER: {t}\n"
            "COMPANY: {name}\n"
            "CURRENT_PRICE: ${price}\n"
            "vsSMA20: {vs20}%\n"
            "vsSMA50: {vs50}%\n"
            "vsSMA200: {vs200}%\n"
            "SMA50: {sma50}\n"
            "AVWAP252: {avwap}\n"
            "RSI14: {rsi}\n"
            "MACD_hist: {macd}\n"
            "ATR%: {atr}\n"
            "Drawdown%: {dd}\n"
            "5d%: {d5}\n"
            "20d%: {d20}\n"
            "60d%: {r60}\n"
            "Vol_vs_20d%: {v20}\n"
            "{val_fields}\n"
            "DATA_AVAILABILITY: {data_av}\n"
            "BASELINE_HINTS: {baselines}\n"
            "PROXIES: MARKET_TREND={mt}; REL_STRENGTH={rs}; BREADTH_VOLUME={bv}; VALUATION_HISTORY={vh}; RISK_VOLATILITY={rv}; RISK_DRAWDOWN={rd}\n"
            "PROXIES_FUNDAMENTALS: {fund_line}\n"
            "PROXIES_CATALYSTS: {cat_line}\n"
            "CATALYST_TIMING_HINTS: {timing_tb}\n"
            "EXPECTED_VOLATILITY_PCT: {ev}\n"
            "FVA_HINT: {fva}\n"
            "FUNDAMENTALS_COVERAGE: {funda_cov}\n"
            "VALUATION_COVERAGE: {val_cov}\n"
            "SUGGESTED_BONUSES: {bon}\n".format(
                t=t, name=name,
                price=feats.get("price"),
                vs20=feats.get("vsSMA20"), vs50=feats.get("vsSMA50"), vs200=feats.get("vsSMA200"),
                sma50=feats.get("SMA50"), avwap=feats.get("AVWAP252"),
                rsi=feats.get("RSI14"), macd=feats.get("MACD_hist"), atr=feats.get("ATRpct"),
                dd=feats.get("drawdown_pct"), d5=feats.get("d5"), d20=feats.get("d20"), r60=feats.get("r60"),
                v20=feats.get("vol_vs20"),
                val_fields=val_fields,
                data_av=data_availability,
                baselines=baseline_str,
                mt=proxies["market_trend"], rs=proxies["relative_strength"],
                bv=proxies["breadth_volume"], vh=proxies["valuation_history"],
                rv=proxies["risk_volatility"], rd=proxies["risk_drawdown"],
                fund_line=fund_line, cat_line=catalyst_line, timing_tb=timing_tb,
                ev=proxies["expected_volatility_pct"],
                fva=proxies["fva_hint"] if proxies["fva_hint"] is not None else "N/A",
                bon=proxies["suggested_bonuses"],
                funda_cov=funda_cov, val_cov=val_cov
            )
        )

    blocks_text = "\n\n".join(blocks)
    user_prompt = USER_PROMPT_TOP20_TEMPLATE.format(today=now.strftime("%b %d"), blocks=blocks_text)

    # 7) GPT-5 adjudication on top-20 only
    try:
        final_text = call_gpt5(SYSTEM_PROMPT_TOP20, user_prompt, max_tokens=13000)
    except Exception as e:
        return fail(f"GPT-5 failed: {repr(e)}")
    # 8) Apply personal bonuses (independent from base)
    final_text = apply_personal_bonuses_to_text(final_text)

    # 8.1) Post debug inputs for the picked ticker(s)
    RE_PICK_TICKER = re.compile(r"(?im)^\s*(?:\d+\)\s*)?(?:\*\*)?([A-Z][A-Z0-9.\-]{1,10})\s+[–-]")
    RE_FORECAST_TICK = re.compile(r"(?im)Forecast\s+image\s+URL:\s*https?://[^/]+/stocks/([A-Z0-9.\-]+)/forecast\b")

    picked = RE_PICK_TICKER.findall(final_text)
    picked += RE_FORECAST_TICK.findall(final_text)

    # De-dup while preserving order
    seen = set(); picked_unique = []
    for x in picked:
        if x not in seen:
            seen.add(x); picked_unique.append(x)

    # Fallbacks
    if not picked_unique:
        if len(debug_inputs) == 1:
            picked_unique = list(debug_inputs.keys())
        elif debug_inputs:
            first_top20 = next(iter(debug_inputs.keys()))
            picked_unique = [first_top20]

    if picked_unique:
        post_debug_inputs_to_discord(picked_unique, debug_inputs)
    else:
        log("[WARN] Could not parse any selected ticker from GPT output; skipping debug post.")

    # 9) Save & (optionally) wait until 08:00
    with open("daily_pick.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    log("[INFO] Draft saved to daily_pick.txt")

    if not force:
        wait_s = seconds_until_target_hour(8, 0)
        log(f"[INFO] Waiting {wait_s} seconds until 08:00 Europe/Bucharest…")
        if wait_s > 0:
            time.sleep(wait_s)

    # 10) Send to Discord
    embed = {
        "title": f"Daily Stock Pick — {datetime.now(TZ).strftime('%Y-%m-%d')}",
        "description": final_text
    }
    payload = {"username": "Daily Stock Alert", "embeds": [embed]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=60).raise_for_status()
        log("[INFO] Posted alert to Discord ✅")
    except Exception as e:
        log(f"[ERROR] Discord webhook error: {repr(e)}")

if __name__ == "__main__":
    main()