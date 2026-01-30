# scripts/universe_from_trading_212.py
from __future__ import annotations
import os, sys, csv, re, json, time, random, datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import requests

# -------------------- Config --------------------
API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com").rstrip("/")
API_BASE_DEMO = os.getenv("T212_API_BASE_DEMO", "https://demo.trading212.com").rstrip("/")
API_KEY = os.getenv("T212_NEW_API_KEY")

OUT_PATH = Path(os.getenv("UNIVERSE_OUT", "data/universe.csv"))
REJECTS_PATH = Path(os.getenv("UNIVERSE_REJECTS", "data/universe_rejects.csv"))
META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))
ALLOW_STALE = os.getenv("ALLOW_STALE", "0").lower() in {"1", "true", "yes"}

MAX_RETRIES  = int(os.getenv("T212_MAX_RETRIES", "7"))
BACKOFF_BASE = float(os.getenv("T212_BACKOFF_BASE", "1.5"))
BACKOFF_CAP  = float(os.getenv("T212_BACKOFF_CAP", "60"))
TIMEOUT      = float(os.getenv("T212_TIMEOUT", "30"))
USER_AGENT   = os.getenv("HTTP_USER_AGENT", "universe-builder/1.0 (+github actions)")

SESSION = requests.Session()
RETRY_STATUS = {429, 500, 502, 503, 504}
ALLOWED = re.compile(r"^[A-Z0-9.\-]+$")

def log(msg: str): print(msg, flush=True)

# -------------------- Retry helpers --------------------
def _sleep_for_retry(resp: Optional[requests.Response], attempt: int):
    base = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    sleep_s = random.uniform(0.6 * base, 1.4 * base)
    time.sleep(max(0.5, sleep_s))

def request_with_retry(method: str, url: str, *, headers=None, timeout=TIMEOUT) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers: hdrs.update(headers)
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.request(method, url, headers=hdrs, timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                log(f"[WARN] HTTP {resp.status_code} from {url} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(resp, attempt + 1); continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                log(f"[WARN] {type(e).__name__} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(getattr(e, "response", None), attempt + 1); continue
            break
    if last_exc: raise last_exc
    raise RuntimeError("request failed after retries")

# -------------------- Junk filter --------------------
def is_junk_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    if not re.search(r"[A-Z]", s): return True
    if not ALLOWED.match(s): return True
    if s.endswith("D") and re.search(r"\d", s): return True
    if len(s) > 15: return True
    return False

# -------------------- Yahoo mapping --------------------
SUFFIX_BY_MIC = {
    "XNAS":"", "XNYS":"", "ARCX":"", "BATS":"", "IEXG":"", "CBOE":"", "XNGS":"", "XNCM":"",
    "XLON":"L", "XETR":"DE", "XFRA":"F", "XSWX":"SW", "XVTX":"SW",
    "XPAR":"PA", "XAMS":"AS", "XBRU":"BR", "XLIS":"LS",
    "XMAD":"MC", "XMIL":"MI", "XWBO":"VI",
    "XSTO":"ST", "XHEL":"HE", "XCSE":"CO", "XOSL":"OL",
    "XWAR":"WA", "XPRA":"PR",
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
    "LSE":"L", "LON":"L", "XETRA":"DE", "FRANKFURT":"F", "FRA":"F",
    "SIX":"SW", "SWX":"SW", "PARIS":"PA", "EURONEXT PARIS":"PA",
    "AMSTERDAM":"AS", "EURONEXT AMSTERDAM":"AS",
    "BRUSSELS":"BR", "EURONEXT BRUSSELS":"BR",
    "LISBON":"LS", "EURONEXT LISBON":"LS",
    "MADRID":"MC", "MILAN":"MI", "STOCKHOLM":"ST",
    "HELSINKI":"HE", "COPENHAGEN":"CO", "OSLO":"OL",
    "TORONTO":"TO", "TSXV":"V", "ASX":"AX", "HKEX":"HK", "TOKYO":"T", "SGX":"SI",
    "KRX":"KS", "KOSDAQ":"KQ", "NSE":"NS", "BSE":"BO", "B3":"SA",
    "BMV":"MX", "JSE":"JO", "SAUDI":"SR", "TASE":"TA",
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
    return isin[:2].upper()

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

    suf = SUFFIX_BY_MIC.get(mc) or SUFFIX_BY_EXCHANGE.get(ex)
    if not suf and isin: suf = SUFFIX_BY_ISIN_COUNTRY.get(_isin_cc(isin), "")
    suf = "" if suf is None else suf

    base = _yahoo_base_with_padding(base, suf)
    if suf == "" and "." in base: base = base.replace(".", "-")
    return f"{base}.{suf}" if suf else base

# -------------------- Fetch instruments --------------------
def _do_get(base: str, path: str) -> requests.Response:
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY environment variable"); sys.exit(1)
    url = f"{base}{path}"
    headers = {"Authorization": API_KEY}
    return request_with_retry("GET", url, headers=headers)

def fetch_instruments() -> List[Dict[str, Any]]:
    try:
        r = _do_get(API_BASE_LIVE, "/api/v0/equity/metadata/instruments")
        return r.json()
    except Exception as e:
        log(f"[WARN] Live failed: {e}, trying Demo…")
        r2 = _do_get(API_BASE_DEMO, "/api/v0/equity/metadata/instruments")
        return r2.json()

# -------------------- Main --------------------
def main():
    try:
        data = fetch_instruments()
        total = len(data)
        log(f"[INFO] Received {total} instruments (all types)")
    except Exception as e:
        if ALLOW_STALE and OUT_PATH.exists():
            log(f"[WARN] {e} — using stale universe")
            sys.exit(0)
        log(f"[FATAL] {e}")
        sys.exit(1)

    rows_raw = []
    for it in data:
        ttype = (it.get("type") or it.get("instrumentType") or "").upper()
        if ttype not in {"EQUITY", "STOCK"}:
            continue

        tick = (it.get("ticker") or it.get("symbol") or "").strip()
        name = (it.get("name") or it.get("shortName") or "").strip()
        if not tick or not name:
            continue

        simple = simplify_symbol(tick)
        if is_junk_symbol(simple):
            # Instead of writing rejects here, just skip
            continue

        yh = map_to_yahoo(simple, it.get("exchange"), it.get("mic"), it.get("isin"))
        rows_raw.append((yh.upper(), name))

    seen, rows = set(), []
    for sym, name in rows_raw:
        if sym in seen:
            continue
        seen.add(sym)
        rows.append((sym, name))
    rows.sort(key=lambda x: x[0])

    # --- Write outputs ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("ticker", "company"))
        w.writerows(rows)

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    log(f"[INFO] Wrote {len(rows)} rows to {OUT_PATH}")
    log(f"[INFO] Saved raw metadata to {META_PATH}")

if __name__ == "__main__":
    main()
