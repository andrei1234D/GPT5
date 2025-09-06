# scripts/clean_universe.py
from __future__ import annotations
import os, sys, csv, re, time, random, datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import requests

# ---------- CLI overrides (key=value) ----------
def apply_cli_overrides():
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.strip().upper(); v = v.strip()
            os.environ[k] = v
apply_cli_overrides()

def log(msg: str): print(msg, flush=True)

# ---------- Config ----------
API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com").rstrip("/")
API_BASE_DEMO = os.getenv("T212_API_BASE_DEMO", "https://demo.trading212.com").rstrip("/")
API_KEY       = os.getenv("T212_API_KEY")

PRECALL_SLEEP_SECONDS = int(os.getenv("T212_PRECALL_SLEEP", "60")) 


IN_PATH   = Path(os.getenv("CLEAN_INPUT", "data/universe.csv"))
OUT_PATH  = Path(os.getenv("CLEAN_OUT", "data/universe_clean.csv"))
REJ_PATH  = Path(os.getenv("CLEAN_REJECTS", "data/universe_rejects.csv"))

# If we don't even have data/universe.csv today, keep yesterday's clean_universe.csv.
if not IN_PATH.exists():
    if OUT_PATH.exists():
        log(f"[INFO] {IN_PATH} missing. Using existing {OUT_PATH} (stale) and exiting success.")
        sys.exit(0)
    else:
        log(f"[FATAL] {IN_PATH} missing and no previous {OUT_PATH} to fall back to.")
        sys.exit(1)



ALLOW_STALE        = os.getenv("ALLOW_STALE", "1").lower() in {"1","true","yes"}
CLEAN_ALLOW_OFFLINE= os.getenv("CLEAN_ALLOW_OFFLINE", "1").lower() in {"1","true","yes"}

MAX_RETRIES  = int(os.getenv("T212_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("T212_BACKOFF_BASE", "1.5"))
BACKOFF_CAP  = float(os.getenv("T212_BACKOFF_CAP", "60"))
TIMEOUT      = float(os.getenv("T212_TIMEOUT", "30"))
USER_AGENT   = os.getenv("HTTP_USER_AGENT", "universe-clean/1.0")

SESSION = requests.Session()
RETRY_STATUS = {429, 500, 502, 503, 504}

# ---------- small helpers ----------
def atomic_write(path: Path, rows: List[Tuple], header: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    os.replace(tmp, path)

def _sleep_for_retry(resp: Optional[requests.Response], attempt: int):
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                sec = int(ra); time.sleep(min(sec, BACKOFF_CAP)); return
            except Exception: pass
            try:
                when = dt.datetime.strptime(ra, "%a, %d %b %Y %H:%M:%S %Z")
                delta = (when - dt.datetime.utcnow()).total_seconds()
                if delta > 0: time.sleep(min(delta, BACKOFF_CAP)); return
            except Exception: pass
    base = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    time.sleep(max(0.5, random.uniform(0.6*base, 1.4*base)))

def request_with_retry(method: str, url: str, *, headers=None, timeout=TIMEOUT) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT, "Accept":"application/json"}
    if headers: hdrs.update(headers)
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.request(method, url, headers=hdrs, timeout=timeout)
            if r.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                log(f"[WARN] HTTP {r.status_code} from {url} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(r, attempt+1); continue
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if isinstance(e, requests.HTTPError) and status not in RETRY_STATUS:
                break
            if attempt < MAX_RETRIES:
                log(f"[WARN] {type(e).__name__} — retry {attempt+1}/{MAX_RETRIES}")
                _sleep_for_retry(getattr(e, "response", None), attempt+1); continue
            break
    if last_exc: raise last_exc
    raise RuntimeError("request failed after retries")

# ---------- Yahoo mapping + junk/name filters (same as fetcher) ----------
ALLOWED = re.compile(r"^[A-Z0-9.\-]+$")
def is_junk_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    if not re.search(r"[A-Z]", s): return True
    if not ALLOWED.match(s): return True
    if s.endswith("D") and re.search(r"\d", s): return True
    if len(s) > 15: return True
    return False

SUFFIX_BY_MIC = {
    "XNAS":"", "XNYS":"", "ARCX":"", "BATS":"", "IEXG":"", "CBOE":"", "XNGS":"", "XNCM":"",
    "XLON":"L", "XETR":"DE", "XFRA":"F", "XSWX":"SW", "XVTX":"SW",
    "XPAR":"PA", "XAMS":"AS", "XBRU":"BR", "XLIS":"LS", "XMAD":"MC", "XMIL":"MI", "XWBO":"VI",
    "XSTO":"ST", "XHEL":"HE", "XCSE":"CO", "XOSL":"OL", "XWAR":"WA", "XPRA":"PR",
    "XTSE":"TO", "XTSX":"V",
    "XASX":"AX", "XHKG":"HK", "XTKS":"T", "XSES":"SI", "XKRX":"KS", "XKOS":"KQ",
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

# ---------- name nets ----------
NAME_STATUS_BAD = re.compile(r"\b(DELISTED|SUSPENDED|UNLISTED|IN\s+LIQ|INSOLVENC|BANKRUPT|WINDING\s+UP)\b", re.I)
NAME_FUNDS_ETP  = re.compile(r"\b(UCITS|ETF|ETN|ETC|ETP|INDEX\s+FUND|MUTUAL\s+FUND|SICAV|OEIC|BDC|CLOSED[- ]END\s+FUND)\b", re.I)
NAME_DERIV_PACK = re.compile(r"\b(WARRANTS?|RIGHTS?|UNITS?|CERTIFICATES?|TURBO|SPRINTERS?|FACTOR\s+CERTIFICATES|MINI\s+FUTURES)\b", re.I)
NAME_INV_TRUST  = re.compile(r"\bINV(?:ESTMENT)?\s+TRUST\b", re.I)
NAME_REIT       = re.compile(r"\bREIT\b", re.I)

def name_reject(nm: str) -> bool:
    n = nm or ""
    if NAME_STATUS_BAD.search(n): return True
    if NAME_FUNDS_ETP.search(n):  return True
    if NAME_DERIV_PACK.search(n): return True
    if NAME_INV_TRUST.search(n):  return True
    # REITs allowed (unless caught by fund/pack)
    return False

# ---------- IO ----------
def load_in(path: Path) -> List[Tuple[str,str]]:
    if not path.exists():
        sys.exit(f"[ERROR] Missing {path}")
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if not {"ticker","company"}.issubset({(c or "").strip().lower() for c in (rdr.fieldnames or [])}):
            sys.exit(f"[ERROR] {path} must have headers: ticker,company")
        for row in rdr:
            t = (row.get("ticker") or "").strip().upper()
            n = (row.get("company") or "").strip()
            if t and n: out.append((t,n))
    return out

# ---------- T212 fetch ----------
def _do_get(base: str, path: str) -> requests.Response:
    if not API_KEY:
        raise RuntimeError("Missing T212_API_KEY")
    url = f"{base}{path}"
    headers = {"Authorization": API_KEY}
    return request_with_retry("GET", url, headers=headers)

def fetch_t212_instruments() -> Tuple[List[Dict[str,Any]], str]:
    # try live
    try:
        r = _do_get(API_BASE_LIVE, "/api/v0/equity/metadata/instruments")
        if r.status_code == 200:
            return r.json(), API_BASE_LIVE
    except requests.HTTPError as e:
        sc = getattr(e.response, "status_code", None)
        if sc not in (401,403,429):
            log(f"[WARN] Live HTTP {sc}; trying Demo…")
    except Exception as e:
        log(f"[WARN] Live error: {repr(e)}; trying Demo…")
    # try demo
    r2 = _do_get(API_BASE_DEMO, "/api/v0/equity/metadata/instruments")
    if r2.status_code == 200:
        log("[WARN] Demo accepted. Using demo base.")
        return r2.json(), API_BASE_DEMO
    r2.raise_for_status()
    return r2.json(), API_BASE_DEMO  # pragma: no cover

# ---------- main ----------
# ---------- main ----------
def main():
    log("[INFO] Fetching T212 instruments…")

    # Add this block right here, before any API call:
    if PRECALL_SLEEP_SECONDS > 0:
        log(f"[INFO] Sleeping {PRECALL_SLEEP_SECONDS}s before hitting T212 (rate-limit guard)…")
        time.sleep(PRECALL_SLEEP_SECONDS)

    instruments: Optional[List[Dict[str,Any]]] = None
    base_used = None
    try:
        instruments, base_used = fetch_t212_instruments()
        log(f"[INFO] Instruments: {len(instruments or [])}")
    except Exception as e:
        msg = str(e)
        log(f"[WARN] T212 fetch failed: {msg}")
        if CLEAN_ALLOW_OFFLINE:
            log("[WARN] Switching to OFFLINE mode (no T212 cross-check).")
        elif ALLOW_STALE and OUT_PATH.exists():
            log("[WARN] ALLOW_STALE=1 and clean file exists — keeping it as-is.")
            sys.exit(0)
        else:
            log("Error:  T212 returned 429. Check key/base/scope.")
            sys.exit(1)


    # Build whitelist if we have instruments
    whitelist: set[str] = set()
    if instruments:
        for it in instruments:
            ttype = (it.get("type") or it.get("instrumentType") or "").upper()
            if ttype not in {"EQUITY","STOCK"}: continue
            tick = (it.get("ticker") or it.get("symbol") or "").strip()
            name = (it.get("name") or it.get("shortName") or "").strip()
            if not tick or not name: continue
            if name_reject(name): continue
            ex   = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
            mic  = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
            isin = it.get("isin")
            simple = simplify_symbol(tick)
            if is_junk_symbol(simple): continue
            yh = map_to_yahoo(simple, ex, mic, isin)
            s_up = yh.upper()
            if ALLOWED.match(s_up): whitelist.add(s_up)
        log(f"[STAGE] whitelist size: {len(whitelist)}")

    # Load your universe.csv (already from T212 fetcher)
    rows = load_in(IN_PATH)
    kept, rejects = [], []
    seen = set()

    for t, nm in rows:
        if t in seen: continue
        seen.add(t)

        if instruments and whitelist:
            if t not in whitelist:
                rejects.append((t, nm, "not-in-whitelist"))
                continue

        # Name safety net on CSV name too
        if name_reject(nm):
            rejects.append((t, nm, "name-filter"))
            continue

        if is_junk_symbol(t):
            rejects.append((t, nm, "symbol-hygiene"))
            continue

        kept.append((t, nm))

    kept.sort(key=lambda x: x[0])
    atomic_write(OUT_PATH, kept, ("ticker","company"))
    atomic_write(REJ_PATH, rejects, ("ticker","company","reason"))

    log(f"[INFO] Input rows   : {len(rows):,}")
    log(f"[INFO] Kept (clean) : {len(kept):,}")
    log(f"[INFO] Rejected     : {len(rejects):,}")
    if instruments is None:
        log("[INFO] Note: ran in OFFLINE mode (no T212 cross-check).")

if __name__ == "__main__":
    main()
