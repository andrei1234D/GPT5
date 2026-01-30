# scripts/universe_from_trading_212.py
from __future__ import annotations

import os, sys, csv, re, json, time, random
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import requests

# -------------------- Config --------------------
API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com").rstrip("/")
API_BASE_DEMO = os.getenv("T212_API_BASE_DEMO", "https://demo.trading212.com").rstrip("/")
API_KEY       = os.getenv("T212_API_KEY") or os.getenv("T212_NEW_API_KEY")

OUT_PATH  = Path(os.getenv("UNIVERSE_OUT", "data/universe.csv"))
META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))
ALLOW_STALE = os.getenv("ALLOW_STALE", "0").lower() in {"1", "true", "yes"}

MAX_RETRIES  = int(os.getenv("T212_MAX_RETRIES", "7"))
BACKOFF_BASE = float(os.getenv("T212_BACKOFF_BASE", "1.5"))
BACKOFF_CAP  = float(os.getenv("T212_BACKOFF_CAP", "60"))
TIMEOUT      = float(os.getenv("T212_TIMEOUT", "30"))
USER_AGENT   = os.getenv("HTTP_USER_AGENT", "universe-builder/1.2 (+github actions)")

SESSION = requests.Session()
RETRY_STATUS = {429, 500, 502, 503, 504}
ALLOWED = re.compile(r"^[A-Z0-9.\-]+$")

def log(msg: str) -> None:
    print(msg, flush=True)

# -------------------- Retry helpers --------------------
def _sleep_for_retry(_resp: Optional[requests.Response], attempt: int) -> None:
    base = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    sleep_s = random.uniform(0.6 * base, 1.4 * base)
    time.sleep(max(0.5, sleep_s))

def request_with_retry(method: str, url: str, *, headers=None, timeout=TIMEOUT) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    last_exc: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.request(method, url, headers=hdrs, timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                log(f"[WARN] HTTP {resp.status_code} from {url} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(resp, attempt + 1)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                log(f"[WARN] {type(e).__name__} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(getattr(e, "response", None), attempt + 1)
                continue
            break
    if last_exc:
        raise last_exc
    raise RuntimeError("request failed after retries")

# -------------------- Symbol parsing --------------------
def parse_t212_ticker(t212_ticker: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Trading212 uses e.g. UUUU_US_EQ, IREN_US_EQ.
    Returns: (base_symbol, market_code, asset_code)
    """
    raw = (t212_ticker or "").strip()
    parts = raw.split("_")
    base = (parts[0] if parts else raw).strip()
    market = parts[1].strip().upper() if len(parts) >= 2 and parts[1].strip() else None
    asset  = parts[2].strip().upper() if len(parts) >= 3 and parts[2].strip() else None
    return base, market, asset

def sanitize_symbol(sym: str) -> str:
    return (sym or "").strip().upper().replace(" ", "-")

def is_junk_symbol(sym: str) -> bool:
    s = sanitize_symbol(sym)
    if not re.search(r"[A-Z]", s):
        return True
    if not ALLOWED.match(s):
        return True
    if len(s) > 20:
        return True
    # common garbage tokens like "013CD" etc
    if s.endswith("D") and re.search(r"\d", s):
        return True
    return False

# -------------------- Yahoo mapping --------------------
# Prefer MIC/exchange when present, but if T212 market code is known, force suffix.
MARKET_SUFFIX = {
    "US": "",
    "UK": "L", "GB": "L",
    "DE": "DE",
    "FR": "PA",
    "NL": "AS",
    "BE": "BR",
    "PT": "LS",
    "ES": "MC",
    "IT": "MI",
    "CH": "SW",
    "AT": "VI",
    "PL": "WA",
    "CZ": "PR",
    "SE": "ST",
    "FI": "HE",
    "DK": "CO",
    "NO": "OL",
    "IE": "IR",
    "CA": "TO",
    "AU": "AX",
    "JP": "T",
    "HK": "HK",
    "SG": "SI",
    "KR": "KS",
    "IN": "NS",
    "BR": "SA",
    "MX": "MX",
    "ZA": "JO",
    "TW": "TW",
    "CN": "SS",  # last resort
}

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

KNOWN_YH_SUFFIXES = set(SUFFIX_BY_MIC.values()) | {"HK","TWO","IR","TW","SS","SZ","SA","AX","TO","V"}

def _isin_cc(isin: str) -> Optional[str]:
    if not isin or len(isin) < 2:
        return None
    return isin[:2].upper()

def _yahoo_base_with_padding(base: str, suffix: str) -> str:
    if base.isdigit() and suffix in {"T", "HK", "TW", "SS", "SZ"}:
        return base.zfill(4)
    return base

def map_to_yahoo(symbol: str,
                 exchange: Optional[str],
                 mic: Optional[str],
                 isin: Optional[str],
                 market_hint: Optional[str]) -> str:
    """
    Map to a Yahoo-style ticker.
    Priority:
      1) If market_hint from T212 ticker is known -> force suffix for that market (fixes US listings w/ non-US ISIN).
      2) Otherwise, use MIC or exchange.
      3) Last resort: ISIN country (only if nothing else).
    """
    base = sanitize_symbol(symbol)
    ex = (exchange or "").upper().strip()
    mc = (mic or "").upper().strip()
    mh = (market_hint or "").upper().strip() or None

    # If already Yahoo suffix, keep it.
    if "." in base:
        head, tail = base.rsplit(".", 1)
        if tail in KNOWN_YH_SUFFIXES:
            return f"{_yahoo_base_with_padding(head, tail)}.{tail}"

    # 1) market hint override (e.g., UUUU_US_EQ should be UUUU)
    if mh in MARKET_SUFFIX:
        suf = MARKET_SUFFIX[mh]
        base2 = _yahoo_base_with_padding(base, suf)
        return f"{base2}.{suf}" if suf else base2

    # 2) MIC/exchange mapping
    suf = SUFFIX_BY_MIC.get(mc) or SUFFIX_BY_EXCHANGE.get(ex)

    # 3) ISIN fallback (only if no MIC/exchange)
    if not suf:
        cc = _isin_cc(isin or "")
        suf = MARKET_SUFFIX.get(cc, "") if cc else ""

    suf = "" if suf is None else suf
    base2 = _yahoo_base_with_padding(base, suf)

    # Yahoo uses '-' for some symbols when no suffix and there's a dot
    if suf == "" and "." in base2:
        base2 = base2.replace(".", "-")

    return f"{base2}.{suf}" if suf else base2

# -------------------- Fetch instruments --------------------
def _do_get(base: str, path: str) -> requests.Response:
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY / T212_NEW_API_KEY environment variable")
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

def extract_row(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ttype = (it.get("type") or it.get("instrumentType") or "").upper()
    if ttype not in {"EQUITY", "STOCK"}:
        return None

    t212_tick = (it.get("ticker") or it.get("symbol") or "").strip()
    short = (it.get("shortName") or "").strip()
    name = (it.get("name") or short or "").strip()
    if not t212_tick or not name:
        return None

    base_from_t212, market_hint, asset_code = parse_t212_ticker(t212_tick)

    # Prefer shortName when available (T212 ticker often has synthetic postfixes)
    base = sanitize_symbol(short or base_from_t212)

    if is_junk_symbol(base):
        return None

    yh = map_to_yahoo(
        base,
        it.get("exchange"),
        it.get("mic"),
        it.get("isin"),
        market_hint
    ).upper()

    return {
        "ticker": yh,
        "company": name,
        "t212_ticker": t212_tick,
        "shortName": short,
        "market": market_hint or "",
        "asset": asset_code or "",
        "isin": (it.get("isin") or ""),
        "exchange": (it.get("exchange") or ""),
        "mic": (it.get("mic") or ""),
        "currency": (it.get("currencyCode") or it.get("currency") or ""),
        "type": ttype,
    }

# -------------------- Main --------------------
def main() -> None:
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

    rows: List[Dict[str, Any]] = []
    skipped_hygiene = 0
    for it in data:
        row = extract_row(it)
        if row is None:
            # this includes hygiene rejects
            # (cannot easily distinguish without duplicating logic)
            continue
        rows.append(row)

    # De-duplicate on Yahoo ticker (keep first, stable)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        t = r["ticker"]
        if t in seen:
            continue
        seen.add(t)
        uniq.append(r)
    uniq.sort(key=lambda x: x["ticker"])

    # --- Write outputs ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = ["ticker","company","t212_ticker","shortName","market","asset","isin","exchange","mic","currency","type"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(uniq)

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    log(f"[INFO] Wrote {len(uniq)} rows to {OUT_PATH} (Yahoo-mapped tickers)")
    log(f"[INFO] Saved raw metadata to {META_PATH}")

if __name__ == "__main__":
    main()
