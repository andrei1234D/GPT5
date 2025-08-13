# scripts/universe_from_trading_212.py
from __future__ import annotations
import os, sys, json, csv, re, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests

API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com").rstrip("/")
API_BASE_DEMO = os.getenv("T212_API_BASE_DEMO", "https://demo.trading212.com").rstrip("/")
API_KEY = os.getenv("T212_API_KEY")

OUT_PATH = Path("data/universe.csv")
REJECTS_PATH = Path("data/universe_rejects.csv")

ALLOW_STALE = os.getenv("ALLOW_STALE", "1").lower() in {"1","true","yes"}
TIMEOUT = float(os.getenv("T212_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("T212_MAX_RETRIES", "3"))

def log(msg): print(msg, flush=True)

# ----- Junk filter -----
ALLOWED = re.compile(r"^[A-Z0-9.\-]+$")
def is_junk_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    if not re.search(r"[A-Z]", s):  # must contain a letter
        return True
    if not ALLOWED.match(s):        # only A-Z/0-9/./-
        return True
    if s.endswith("D") and re.search(r"\d", s):  # vendor codes like 0Q0D, 3AG1D, 71YD
        return True
    if len(s) > 15:
        return True
    return False

# ----- Yahoo suffix maps -----
SUFFIX_BY_MIC = {
    "XNAS":"", "XNYS":"", "ARCX":"", "BATS":"", "IEXG":"", "CBOE":"", "XNGS":"", "XNCM":"",
    "XLON":"L", "XETR":"DE", "XFRA":"F", "XSWX":"SW", "XVTX":"SW",
    "XPAR":"PA", "XAMS":"AS", "XBRU":"BR", "XLIS":"LS",
    "XMAD":"MC", "XMIL":"MI", "XWBO":"VI",
    "XSTO":"ST", "XHEL":"HE", "XCSE":"CO", "XOSL":"OL", "XWAR":"WA", "XPRA":"PR",
    "XTSE":"TO", "XTSX":"V",
    "XASX":"AX", "XHKG":"HK", "XTKS":"T", "XSES":"SI",
    "XKRX":"KS", "XKOS":"KQ",
    "XNSE":"NS", "XBOM":"BO",
    "BVMF":"SA", "B3SA":"SA", "B3":"SA", "XMEX":"MX", "BMV":"MX",
    "XJSE":"JO", "XDFM":"DU", "XADS":"AD", "XSAU":"SR", "XTAE":"TA",
    "XSHG":"SS", "XSHE":"SZ", "XTAI":"TW", "ROCO":"TWO"
}
SUFFIX_BY_EXCHANGE = {
    "NASDAQ":"", "NYSE":"", "ARCA":"", "AMEX":"", "NYSE MKT":"",
    "LSE":"L", "LON":"L", "LSE_INTL":"L",
    "XETRA":"DE", "FRANKFURT":"F", "FRA":"F",
    "SIX":"SW", "SWX":"SW",
    "PARIS":"PA", "EURONEXT PARIS":"PA",
    "AMSTERDAM":"AS", "EURONEXT AMSTERDAM":"AS",
    "BRUSSELS":"BR", "EURONEXT BRUSSELS":"BR",
    "LISBON":"LS", "EURONEXT LISBON":"LS",
    "MADRID":"MC", "BME":"MC",
    "MILAN":"MI", "BORSA ITALIANA":"MI",
    "STOCKHOLM":"ST", "HELSINKI":"HE", "COPENHAGEN":"CO", "OSLO":"OL",
    "WARSAW":"WA", "PRAGUE":"PR", "VIENNA":"VI",
    "TSX":"TO", "TORONTO":"TO", "TSX VENTURE":"V", "TSXV":"V",
    "ASX":"AX", "HKEX":"HK", "HONG KONG":"HK", "TOKYO":"T", "JPX":"T", "SGX":"SI",
    "KRX":"KS", "KOSDAQ":"KQ",
    "NSE":"NS", "BSE":"BO",
    "B3":"SA", "SAO PAULO":"SA", "BMV":"MX", "MEXICO":"MX",
    "JSE":"JO", "DUBAI":"DU", "ABU DHABI":"AD", "SAUDI":"SR", "TASE":"TA",
    "SHANGHAI":"SS", "SHENZHEN":"SZ", "TAIWAN":"TW"
}
SUFFIX_BY_ISIN_COUNTRY = {
    "US":"", "GB":"L", "DE":"DE", "FR":"PA", "NL":"AS", "BE":"BR", "PT":"LS",
    "ES":"MC", "IT":"MI", "CH":"SW", "CA":"TO", "AU":"AX", "JP":"T", "HK":"HK",
    "SG":"SI", "SE":"ST", "FI":"HE", "DK":"CO", "NO":"OL", "PL":"WA", "CZ":"PR",
    "AT":"VI", "BR":"SA", "MX":"MX", "ZA":"JO", "IE":"IR", "SA":"SR", "AE":"DU",
    "QA":"QA", "IL":"TA", "KR":"KS", "IN":"NS", "TW":"TW", "CN":"SS"
}
KNOWN_YH_SUFFIXES = set(SUFFIX_BY_MIC.values()) | {"HK","TWO","IR","TW"}

def simplify_symbol(t212_ticker: str) -> str:
    return (t212_ticker or "").split("_", 1)[0].strip().replace(" ", "-")

def _isin_cc(isin: str) -> Optional[str]:
    if not isin or len(isin) < 2: return None
    cc = isin[:2].upper()
    return cc if re.fullmatch(r"[A-Z]{2}", cc) else None

def _yahoo_base_with_padding(base: str, suffix: str) -> str:
    if base.isdigit() and suffix in {"T", "HK", "TW", "SS", "SZ"}:
        return base.zfill(4)
    return base

def map_to_yahoo(symbol: str, exchange: Optional[str], mic: Optional[str], isin: Optional[str]) -> str:
    base = (symbol or "").upper()
    ex = (exchange or "").upper().strip()
    mc = (mic or "").upper().strip()
    if "." in base:
        head, tail = base.rsplit(".", 1)
        if tail in KNOWN_YH_SUFFIXES:
            return f"{_yahoo_base_with_padding(head, tail)}.{tail}"
    suf = SUFFIX_BY_MIC.get(mc)
    if suf is None: suf = SUFFIX_BY_EXCHANGE.get(ex)
    if suf is None:
        cc = _isin_cc(isin or ""); suf = SUFFIX_BY_ISIN_COUNTRY.get(cc) if cc else None
    suf = "" if suf is None else suf
    base = _yahoo_base_with_padding(base, suf)
    if suf == "" and "." in base:
        base = base.replace(".", "-")
    return f"{base}.{suf}" if suf else base

def _req_with_retry(url: str, headers: dict) -> requests.Response:
    last = None
    for i in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            if r.status_code in {429,500,502,503,504} and i < MAX_RETRIES:
                wait = min(60, 1.5 ** (i+1))
                log(f"[WARN] {r.status_code} from {url} — retrying in {wait:.1f}s ({i+1}/{MAX_RETRIES})")
                time.sleep(wait); continue
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last = e
            if i < MAX_RETRIES:
                wait = min(60, 1.5 ** (i+1))
                log(f"[WARN] {type(e).__name__} — retrying in {wait:.1f}s ({i+1}/{MAX_RETRIES})")
                time.sleep(wait); continue
            break
    if last: raise last
    raise RuntimeError("request failed")

def fetch_instruments():
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY environment variable"); sys.exit(1)

    # Try Live
    try:
        r = _req_with_retry(f"{API_BASE_LIVE}/api/v0/equity/metadata/instruments", {"Authorization": API_KEY})
        if r.status_code == 200:
            return r.json(), API_BASE_LIVE
    except requests.HTTPError as e:
        sc = getattr(e.response, "status_code", None)
        if sc not in (401,403,429):
            log(f"[WARN] Live HTTP {sc}; will try Demo…")
    except Exception as e:
        log(f"[WARN] Live error: {repr(e)}; will try Demo…")

    # Try Demo (only if you actually have practice access)
    try:
        r2 = _req_with_retry(f"{API_BASE_DEMO}/api/v0/equity/metadata/instruments", {"Authorization": API_KEY})
        if r2.status_code == 200:
            log("[WARN] Demo accepted. Using demo base.")
            return r2.json(), API_BASE_DEMO
        r2.raise_for_status()
    except Exception as e:
        raise e

def main():
    log("[INFO] Fetching instrument metadata from Trading 212…")
    try:
        data, base_used = fetch_instruments()
    except Exception as e:
        log(f"[ERROR] T212 fetch failed: {repr(e)}")
        if OUT_PATH.exists() and ALLOW_STALE:
            log(f"[INFO] Using existing {OUT_PATH} (stale) and exiting success.")
            sys.exit(0)
        log("[FATAL] No existing universe.csv to fall back to.")
        sys.exit(1)

    total = len(data) if isinstance(data, list) else 0
    log(f"[INFO] Base used: {base_used}")
    log(f"[INFO] Received {total} instruments (all types)")

    rows_raw, rejects = [], []
    for it in (data or []):
        ttype = (it.get("type") or it.get("instrumentType") or "").upper()
        if ttype not in {"EQUITY", "STOCK"}:
            continue
        tick = (it.get("ticker") or it.get("symbol") or "").strip()
        name = (it.get("name") or it.get("shortName") or "").strip()
        if not tick or not name:
            continue

        simple = simplify_symbol(tick)
        if is_junk_symbol(simple):
            rejects.append((simple.upper(), name, "junk-filter")); continue

        exchange = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
        mic = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
        isin = it.get("isin")

        yahoo = map_to_yahoo(simple, exchange, mic, isin)
        rows_raw.append((yahoo, name))

    log(f"[INFO] Filtered equities: {len(rows_raw)} (rejected {len(rejects)} junk-like symbols)")

    seen = set(); rows = []
    for sym, name in rows_raw:
        if sym in seen: 
            continue
        seen.add(sym); rows.append((sym, name))
    rows.sort(key=lambda x: x[0])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "company"])
        w.writerows(rows)

    if rejects:
        with REJECTS_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "company", "reason"])
            w.writerows(rejects)

    log(f"[INFO] Wrote {len(rows)} rows to {OUT_PATH}")
    if rejects:
        log(f"[INFO] Wrote {len(rejects)} rejects to {REJECTS_PATH}")

if __name__ == "__main__":
    main()
