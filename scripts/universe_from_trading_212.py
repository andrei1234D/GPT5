# scripts/universe_from_trading_212.py
from __future__ import annotations
import os, sys, csv, re, json, time, random, datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import requests

# -------------------- CLI overrides (key=value) --------------------
def apply_cli_overrides():
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k in {"APIKEY", "T212_API_KEY"}:
                os.environ["T212_API_KEY"] = v
            elif k in {"BASE", "T212_API_BASE"}:
                os.environ["T212_API_BASE"] = v.rstrip("/")
            elif k in {"ALLOW_STALE", "ALLOWSTALE"}:
                os.environ["ALLOW_STALE"] = v
            elif k == "OUT":
                os.environ["UNIVERSE_OUT"] = v
            elif k == "REJECTS":
                os.environ["UNIVERSE_REJECTS"] = v
            elif k == "MAX_RETRIES":
                os.environ["T212_MAX_RETRIES"] = v
            elif k == "BACKOFF_BASE":
                os.environ["T212_BACKOFF_BASE"] = v
            elif k == "BACKOFF_CAP":
                os.environ["T212_BACKOFF_CAP"] = v
apply_cli_overrides()

# -------------------- Config --------------------
API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com").rstrip("/")
API_BASE_DEMO = os.getenv("T212_API_BASE_DEMO", "https://demo.trading212.com").rstrip("/")
API_KEY = os.getenv("T212_API_KEY")

OUT_PATH = Path(os.getenv("UNIVERSE_OUT", "data/universe.csv"))
REJECTS_PATH = Path(os.getenv("UNIVERSE_REJECTS", "data/universe_rejects.csv"))
ALLOW_STALE = os.getenv("ALLOW_STALE", "0").lower() in {"1", "true", "yes"}

MAX_RETRIES  = int(os.getenv("T212_MAX_RETRIES", "7"))
BACKOFF_BASE = float(os.getenv("T212_BACKOFF_BASE", "1.5"))
BACKOFF_CAP  = float(os.getenv("T212_BACKOFF_CAP", "60"))
TIMEOUT      = float(os.getenv("T212_TIMEOUT", "30"))

SESSION = requests.Session()
USER_AGENT = os.getenv("HTTP_USER_AGENT", "universe-builder/1.0 (+github actions)")

def log(msg: str): print(msg, flush=True)

# -------------------- Retry helper --------------------
RETRY_STATUS = {429, 500, 502, 503, 504}

def _sleep_for_retry(resp: Optional[requests.Response], attempt: int):
    # 1) Honor Retry-After if present
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            # seconds?
            try:
                sec = int(ra)
                time.sleep(min(sec, BACKOFF_CAP))
                return
            except Exception:
                pass
            # HTTP-date?
            try:
                when = dt.datetime.strptime(ra, "%a, %d %b %Y %H:%M:%S %Z")
                delta = (when - dt.datetime.utcnow()).total_seconds()
                if delta > 0:
                    time.sleep(min(delta, BACKOFF_CAP))
                    return
            except Exception:
                pass
    # 2) Exponential backoff with jitter
    base = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    sleep_s = random.uniform(0.6 * base, 1.4 * base)
    time.sleep(max(0.5, sleep_s))

def request_with_retry(method: str, url: str, *, headers=None, timeout=TIMEOUT) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT}
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
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if isinstance(e, requests.HTTPError) and status not in RETRY_STATUS:
                break  # non-retryable HTTP
            if attempt < MAX_RETRIES:
                log(f"[WARN] {type(e).__name__} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(getattr(e, "response", None), attempt + 1); continue
            break
    if last_exc: raise last_exc
    raise RuntimeError("request failed after retries")

# -------------------- Junk filter --------------------
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

# -------------------- Yahoo suffix maps --------------------
SUFFIX_BY_MIC = {
    # US
    "XNAS":"", "XNYS":"", "ARCX":"", "BATS":"", "IEXG":"", "CBOE":"", "XNGS":"", "XNCM":"",
    # UK & Europe
    "XLON":"L", "XETR":"DE", "XFRA":"F", "XSWX":"SW", "XVTX":"SW",
    "XPAR":"PA", "XAMS":"AS", "XBRU":"BR", "XLIS":"LS",
    "XMAD":"MC", "XMIL":"MI", "XWBO":"VI",
    "XSTO":"ST", "XHEL":"HE", "XCSE":"CO", "XOSL":"OL",
    "XWAR":"WA", "XPRA":"PR",
    # Canada
    "XTSE":"TO", "XTSX":"V",
    # APAC
    "XASX":"AX", "XHKG":"HK", "XTKS":"T", "XSES":"SI",
    "XKRX":"KS", "XKOS":"KQ",
    # India
    "XNSE":"NS", "XBOM":"BO",
    # LatAm
    "BVMF":"SA", "B3SA":"SA", "B3":"SA", "XMEX":"MX", "BMV":"MX",
    # MEA
    "XJSE":"JO", "XDFM":"DU", "XADS":"AD", "XSAU":"SR", "XTAE":"TA",
    # China/Taiwan
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
KNOWN_YH_SUFFIXES = set(SUFFIX_BY_MIC.values()) | {"HK", "TWO", "IR", "TW"}

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
        cc = _isin_cc(isin or "")
        suf = SUFFIX_BY_ISIN_COUNTRY.get(cc) if cc else None
    suf = "" if suf is None else suf

    base = _yahoo_base_with_padding(base, suf)
    if suf == "" and "." in base:
        base = base.replace(".", "-")

    return f"{base}.{suf}" if suf else base

# -------------------- HTTP to T212 --------------------
def _do_get(base: str, path: str) -> requests.Response:
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY environment variable"); sys.exit(1)
    url = f"{base}{path}"
    headers = {"Authorization": API_KEY, "Accept": "application/json"}
    return request_with_retry("GET", url, headers=headers)

def fetch_instruments() -> Tuple[List[Dict[str, Any]], str]:
    # Try LIVE
    try:
        r = _do_get(API_BASE_LIVE, "/api/v0/equity/metadata/instruments")
        if r.status_code == 200:
            return r.json(), API_BASE_LIVE
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            log("[WARN] Live rejected the key; trying Demo…")
        elif status == 429:
            log("[WARN] Live returned 429 (rate-limited); will try Demo after backoff…")
        else:
            log(f"[WARN] Live HTTP {status}; will try Demo…")
    except Exception as e:
        log(f"[WARN] Live error: {repr(e)}; will try Demo…")

    # Try DEMO
    try:
        r2 = _do_get(API_BASE_DEMO, "/api/v0/equity/metadata/instruments")
        if r2.status_code == 200:
            log("[WARN] Demo accepted. Using demo base.")
            return r2.json(), API_BASE_DEMO
        # fall-through -> raise below
        r2.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"T212 fetch failed on both bases: {repr(e)}")

# -------------------- IO helpers --------------------
def atomic_write_csv(path: Path, rows: List[Tuple], header: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    os.replace(tmp, path)

# -------------------- Main --------------------
def main():
    try:
        log("[INFO] Fetching instrument metadata from Trading 212…")
        data, base_used = fetch_instruments()
        total = len(data) if isinstance(data, list) else 0
        log(f"[INFO] Base used: {base_used}")
        log(f"[INFO] Received {total} instruments (all types)")
    except Exception as e:
        msg = f"{e}"
        if ALLOW_STALE and OUT_PATH.exists():
            log(f"[WARN] {msg}")
            log(f"[WARN] Falling back to stale file: {OUT_PATH}")
            sys.exit(0)
        log(f"[FATAL] {msg}")
        sys.exit(1)

    rows_raw: List[Tuple[str, str]] = []
    rejects: List[Tuple[str, str, str]] = []

    for it in (data or []):
        ttype = (it.get("type") or it.get("instrumentType") or "").upper()
        if ttype not in {"EQUITY", "STOCK"}:
            continue

        tick = (it.get("ticker") or it.get("symbol") or "").strip()
        name = (it.get("name") or it.get("shortName") or it.get("description") or "").strip()
        if not tick or not name:
            continue

        simple = simplify_symbol(tick)
        if is_junk_symbol(simple):
            rejects.append((simple.upper(), name, "junk-filter"))
            continue

        exchange = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
        mic = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
        isin = it.get("isin")

        yh = map_to_yahoo(simple, exchange, mic, isin)
        rows_raw.append((yh.upper(), name))

    log(f"[INFO] Filtered equities: {len(rows_raw)} (rejected {len(rejects)} junk-like symbols)")

    # de-dup + sort
    seen = set(); rows: List[Tuple[str, str]] = []
    for sym, name in rows_raw:
        if sym in seen: 
            continue
        seen.add(sym); rows.append((sym, name))
    rows.sort(key=lambda x: x[0])

    if not rows and ALLOW_STALE and OUT_PATH.exists():
        log("[WARN] Zero rows produced; keeping stale file due to ALLOW_STALE=1.")
        sys.exit(0)
    if not rows:
        log("[ERROR] Zero rows produced and no stale fallback available."); sys.exit(1)

    atomic_write_csv(OUT_PATH, rows, ("ticker", "company"))
    if rejects:
        atomic_write_csv(REJECTS_PATH, rejects, ("ticker", "company", "reason"))

    log(f"[INFO] Wrote {len(rows)} rows to {OUT_PATH}")
    if rejects:
        log(f"[INFO] Wrote {len(rejects)} rejects to {REJECTS_PATH}")

if __name__ == "__main__":
    main()
