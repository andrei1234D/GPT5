# scripts/fundamentals.py
from datetime import datetime
import pandas as pd
import pytz
import yfinance as yf

TZ = pytz.timezone("Europe/Bucharest")

def fetch_next_earnings_days(tickers: list[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    today = datetime.now(TZ).date()
    for t in tickers:
        hy = t.replace(".", "-")
        days: int | None = None
        try:
            tk = yf.Ticker(hy)
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

# ---- helpers
import re as _re
def _find_row(df: pd.DataFrame, patterns: list[str]) -> pd.Series | None:
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
            "REV_GROWTH_YOY": None, "EPS_GROWTH_YOY": None,
            "GROSS_MARGIN": None, "OPER_MARGIN": None, "FCF_MARGIN": None,
            "DEBT_TO_EBITDA": None, "NET_CASH": None, "GUIDANCE_CHANGE": None, "OP_EFF_TREND": None,
            "PE": None, "PE_SECTOR": None, "EV_EBITDA": None, "PS": None, "FCF_YIELD": None, "PEG": None,
        }
        try:
            tk = yf.Ticker(hy)
            qfin = getattr(tk, "quarterly_financials", None)
            rev = _find_row(qfin, ["Total Revenue", "Revenue"])
            gp  = _find_row(qfin, ["Gross Profit"])
            opi = _find_row(qfin, ["Operating Income", "Operating Income or Loss"])
            ebitda = _find_row(qfin, ["EBITDA"])

            rg = _latest_qyoy(rev) if isinstance(rev, pd.Series) else None
            if rg is None:
                qearn = getattr(tk, "quarterly_earnings", None)
                if isinstance(qearn, pd.DataFrame) and "Revenue" in qearn.columns:
                    rg = _latest_qyoy(qearn["Revenue"])
            if rg is not None:
                m["REV_GROWTH_YOY"] = round(rg, 2)

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

            qbs = getattr(tk, "quarterly_balance_sheet", None)
            cash = _find_row(qbs, ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"])
            debt = None
            if isinstance(qbs, pd.DataFrame):
                td = _find_row(qbs, ["Total Debt"])
                ltd = _find_row(qbs, ["Long Term Debt"])
                std = _find_row(qbs, ["Short Long Term Debt"])
                def _v(s):
                    if isinstance(s, pd.Series):
                        s2 = s.sort_index().dropna()
                        return float(s2.iloc[-1]) if not s2.empty else None
                    return None
                debt_vals = [x for x in [_v(td), _v(ltd), _v(std)] if x is not None]
                if debt_vals:
                    last_val = debt_vals[0] if len(debt_vals) == 1 else sum(debt_vals)
                    debt = pd.Series([last_val], index=[pd.Timestamp("today")])

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

            if isinstance(opi, pd.Series) and isinstance(rev, pd.Series):
                try:
                    om_q = (opi / rev).dropna().astype(float).sort_index()
                    if om_q.shape[0] >= 2:
                        delta = float(om_q.iloc[-1] - om_q.iloc[-2])
                        m["OP_EFF_TREND"] = "Up" if delta > 0 else "Down" if delta < 0 else "Flat"
                except Exception:
                    pass

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
