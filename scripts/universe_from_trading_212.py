# scripts/universe_from_trading_212.py
from __future__ import annotations

import os, sys, csv, re, json, time, random
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import requests

# -------------------- Config --------------------
API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com").rstrip("/")
API_BASE_DEMO = os.getenv("T212_API_BASE_DEMO", "https://demo.trading212.com").rstrip("/")

# Prefer the "new" var if present, otherwise fall back to the old one.
API_KEY = os.getenv("T212_NEW_API_KEY") or os.getenv("T212_API_KEY")

OUT_PATH = Path(os.getenv("UNIVERSE_OUT", "data/universe.csv"))
META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))
# Optional: emit what got skipped by hygiene for debugging
SKIPPED_PATH = Path(os.getenv("UNIVERSE_SKIPPED", "data/universe_skipped.csv"))
WRITE_SKIPPED = os.getenv("UNIVERSE_WRITE_SKIPPED", "0").lower() in {"1", "true", "yes"}

ALLOW_STALE = os.getenv("ALLOW_STALE", "0").lower() in {"1", "true", "yes"}

MAX_RETRIES  = int(os.getenv("T212_MAX_RETRIES", "7"))
BACKOFF_BASE = float(os.getenv("T212_BACKOFF_BASE", "1.5"))
BACKOFF_CAP  = float(os.getenv("T212_BACKOFF_CAP", "60"))
TIMEOUT      = float(os.getenv("T212_TIMEOUT", "30"))
USER_AGENT   = os.getenv("HTTP_USER_AGENT", "universe-builder/1.4 (+github actions)")

SESSION = requests.Session()
RETRY_STATUS = {429, 500, 502, 503, 504}

# Allow Yahoo-ish symbols after normalization (A-Z, digits, dot, dash).
ALLOWED = re.compile(r"^[A-Z0-9.\-]+$")

def log(msg: str) -> None:
    print(msg, flush=True)

# -------------------- Retry helpers --------------------
def _sleep_for_retry(attempt: int) -> None:
    base = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    sleep_s = random.uniform(0.6 * base, 1.4 * base)
    time.sleep(max(0.5, sleep_s))

def request_with_retry(method: str, url: str, *, headers=None, timeout=TIMEOUT) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.request(method, url, headers=hdrs, timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                log(f"[WARN] HTTP {resp.status_code} from {url} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(attempt + 1)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                log(f"[WARN] {type(e).__name__} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(attempt + 1)
                continue
            break

    if last_exc:
        raise last_exc
    raise RuntimeError("request failed after retries")

# -------------------- Symbol normalization & hygiene --------------------
def simplify_symbol(t212_ticker: str) -> str:
    """
    Trading212 tickers often look like 'UUUU_US_EQ'. For Yahoo mapping we want the base symbol.
    """
    return (t212_ticker or "").split("_", 1)[0].strip().replace(" ", "-")

def hygiene_reason(sym: str) -> Optional[str]:
    """
    Return a string reason if symbol is considered junk, else None.
    Designed to keep legitimate T212 base symbols like 'ZGYd', while filtering synthetic IDs like '013CD'.
    """
    s = (sym or "").upper().strip()
    if not s:
        return "empty"
    if not re.search(r"[A-Z]", s):
        return "no-letters"
    if len(s) > 40:
        return "too-long"
    if not ALLOWED.match(s):
        return "bad-chars"
    # Synthetic IDs typically: short, ends with D, includes digits, no dot.
    # Example: 013CD, 2QKD, 1SXP1D.
    if "." not in s and s.endswith("D") and any(ch.isdigit() for ch in s) and len(s) <= 10:
        return "synthetic-D"
    return None

def is_junk_symbol(sym: str) -> bool:
    return hygiene_reason(sym) is not None

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

def _isin_cc(isin: str) -> Optional[str]:
    if not isin or len(isin) < 2:
        return None
    return isin[:2].upper()

def _yahoo_base_with_padding(base: str, suffix: str) -> str:
    if base.isdigit() and suffix in {"T", "HK", "TW", "SS", "SZ"}:
        return base.zfill(4)
    return base

def map_to_yahoo(symbol: str, exchange: Optional[str], mic: Optional[str], isin: Optional[str]) -> str:
    base = (symbol or "").upper()
    ex = (exchange or "").upper().strip()
    mc = (mic or "").upper().strip()

    # If it already looks like a Yahoo symbol with a known suffix, respect it.
    if "." in base:
        head, tail = base.rsplit(".", 1)
        if tail in KNOWN_YH_SUFFIXES:
            return f"{_yahoo_base_with_padding(head, tail)}.{tail}"

    suf = SUFFIX_BY_MIC.get(mc) or SUFFIX_BY_EXCHANGE.get(ex)
    if not suf and isin:
        suf = SUFFIX_BY_ISIN_COUNTRY.get(_isin_cc(isin), "")
    suf = "" if suf is None else suf

    base = _yahoo_base_with_padding(base, suf)
    if suf == "" and "." in base:
        base = base.replace(".", "-")
    return f"{base}.{suf}" if suf else base

# -------------------- Fetch instruments --------------------
def _do_get(base: str, path: str) -> requests.Response:
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY/T212_NEW_API_KEY environment variable")
        sys.exit(1)
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
def main() -> None:
    try:
        data = fetch_instruments()
        log(f"[INFO] Received {len(data)} instruments (all types)")
    except Exception as e:
        if ALLOW_STALE and OUT_PATH.exists():
            log(f"[WARN] {e} — using stale universe")
            sys.exit(0)
        log(f"[FATAL] {e}")
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    skipped_rows: List[Dict[str, str]] = []
    seen_t212: set[str] = set()
    skipped = 0

    for it in data:
        ttype = (it.get("type") or it.get("instrumentType") or "").upper()
        if ttype not in {"EQUITY", "STOCK"}:
            continue

        t212_ticker = (it.get("ticker") or it.get("symbol") or "").strip()
        company = (it.get("name") or it.get("shortName") or "").strip()
        if not t212_ticker or not company:
            continue

        if t212_ticker in seen_t212:
            continue
        seen_t212.add(t212_ticker)

        simple = simplify_symbol(t212_ticker)
        reason = hygiene_reason(simple)
        if reason:
            skipped += 1
            if WRITE_SKIPPED:
                skipped_rows.append({
                    "t212_ticker": t212_ticker,
                    "base_symbol": simple,
                    "company": company,
                    "reason": reason,
                })
            continue

        exchange = (it.get("exchange") or "").strip()
        mic = (it.get("mic") or "").strip()
        isin = (it.get("isin") or "").strip()
        currency = (it.get("currencyCode") or it.get("currency") or "").strip()

        yahoo = map_to_yahoo(simple, exchange, mic, isin).upper()

        rows.append({
            "ticker": yahoo,               # Yahoo ticker (yfinance-compatible)
            "company": company,
            "t212_ticker": t212_ticker,    # Raw Trading212 identifier
            "shortName": (it.get("shortName") or "").strip(),
            "isin": isin,
            "exchange": exchange,
            "mic": mic,
            "currency": currency,
        })

    rows.sort(key=lambda r: (r["ticker"], r["t212_ticker"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=("ticker", "company", "t212_ticker", "shortName", "isin", "exchange", "mic", "currency"),
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    if WRITE_SKIPPED:
        SKIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SKIPPED_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=("t212_ticker", "base_symbol", "company", "reason"))
            w.writeheader()
            w.writerows(skipped_rows)
        log(f"[INFO] Saved skipped symbols to {SKIPPED_PATH}")

    log(f"[INFO] Wrote {len(rows)} rows to {OUT_PATH} (unique Trading212 tickers)")
    if skipped:
        log(f"[INFO] Skipped {skipped} rows due to symbol hygiene")
    log(f"[INFO] Saved raw metadata to {META_PATH}")

if __name__ == "__main__":
    main()
