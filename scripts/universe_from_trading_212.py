import os, csv, sys, requests, re
from pathlib import Path

API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com")
API_BASE_DEMO = "https://live.trading212.com"  # keep as you asked
API_KEY = os.getenv("T212_API_KEY")
OUT_PATH = Path("data/universe.csv")
REJECTS_PATH = Path("data/universe_rejects.csv")

def log(msg): print(msg, flush=True)

# ----- Junk filter -----
ALLOWED = re.compile(r"^[A-Z0-9.\-]+$")
def is_junk_symbol(sym: str) -> bool:
    s = sym.upper()
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
    "XSHG":"SS", "XSHE":"SZ", "XTAI":"TW", "ROCO":"TWO"  # TWO is rare; keep for completeness
}

SUFFIX_BY_EXCHANGE = {
    # US
    "NASDAQ":"", "NYSE":"", "ARCA":"", "AMEX":"", "NYSE MKT":"",
    # UK & Europe
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
    # Canada
    "TSX":"TO", "TORONTO":"TO", "TSX VENTURE":"V", "TSXV":"V",
    # APAC
    "ASX":"AX", "HKEX":"HK", "HONG KONG":"HK", "TOKYO":"T", "JPX":"T", "SGX":"SI",
    "KRX":"KS", "KOSDAQ":"KQ",
    # India
    "NSE":"NS", "BSE":"BO",
    # LatAm
    "B3":"SA", "SAO PAULO":"SA", "BMV":"MX", "MEXICO":"MX",
    # MEA
    "JSE":"JO", "DUBAI":"DU", "ABU DHABI":"AD", "SAUDI":"SR", "TASE":"TA",
    # China/Taiwan
    "SHANGHAI":"SS", "SHENZHEN":"SZ", "TAIWAN":"TW"
}

SUFFIX_BY_ISIN_COUNTRY = {
    "US":"", "GB":"L", "DE":"DE", "FR":"PA", "NL":"AS", "BE":"BR", "PT":"LS",
    "ES":"MC", "IT":"MI", "CH":"SW", "CA":"TO", "AU":"AX", "JP":"T", "HK":"HK",
    "SG":"SI", "SE":"ST", "FI":"HE", "DK":"CO", "NO":"OL", "PL":"WA", "CZ":"PR",
    "AT":"VI", "BR":"SA", "MX":"MX", "ZA":"JO", "IE":"IR", "SA":"SR", "AE":"DU",
    "QA":"QA", "IL":"TA", "KR":"KS", "IN":"NS", "TW":"TW", "CN":"SS"
}

# Known Yahoo suffixes for later normalization
KNOWN_YH_SUFFIXES = set(SUFFIX_BY_MIC.values()) | {"HK","TWO","IR","TW"}

def _do_get(base, path):
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY environment variable"); sys.exit(1)
    url = f"{base}{path}"
    r = requests.get(url, headers={"Authorization": API_KEY}, timeout=120)
    return r

def fetch_instruments():
    r = _do_get(API_BASE_LIVE, "/api/v0/equity/metadata/instruments")
    if r.status_code == 200:
        return r.json(), API_BASE_LIVE
    if r.status_code in (401, 403):
        r2 = _do_get(API_BASE_DEMO, "/api/v0/equity/metadata/instruments")
        if r2.status_code == 200:
            log("[WARN] Live rejected the key; Demo accepted. Using demo base.")
            return r2.json(), API_BASE_DEMO
        if r.status_code == 401: log("[ERROR] 401 Unauthorized from Trading 212.")
        else: log("[ERROR] 403 Forbidden from Trading 212 (missing scope?).")
        log("Hints:")
        log(" • Use the correct environment (Live vs Practice).")
        log(" • Practice keys require https://demo.trading212.com.")
        log(" • Ensure the token has 'metadata' scope.")
        log(" • Header must be: Authorization: <your_key>  (no 'Bearer')")
        log(f"Live={r.status_code}, Demo={r2.status_code}")
        sys.exit(1)
    r.raise_for_status()
    return r.json(), API_BASE_LIVE

def simplify_symbol(t212_ticker: str) -> str:
    # drop suffix after underscore, keep dots/hyphens
    return t212_ticker.split("_", 1)[0].strip().replace(" ", "-")

def _isin_cc(isin: str) -> str | None:
    if not isin or len(isin) < 2: return None
    cc = isin[:2].upper()
    return cc if re.fullmatch(r"[A-Z]{2}", cc) else None

def _yahoo_base_with_padding(base: str, suffix: str) -> str:
    # JP/HK/TW/CN numeric tickers need zero padding to 4 digits
    if base.isdigit() and suffix in {"T", "HK", "TW", "SS", "SZ"}:
        return base.zfill(4)
    return base

def map_to_yahoo(symbol: str, exchange: str | None, mic: str | None, isin: str | None) -> str:
    base = symbol.upper()
    ex = (exchange or "").upper().strip()
    mc = (mic or "").upper().strip()

    # If already ends with a known Yahoo suffix, trust it
    if "." in base:
        head, tail = base.rsplit(".", 1)
        if tail in KNOWN_YH_SUFFIXES:
            return f"{_yahoo_base_with_padding(head, tail)}.{tail}"

    # 1) Try MIC
    suf = SUFFIX_BY_MIC.get(mc)
    # 2) Try exchange name
    if suf is None:
        suf = SUFFIX_BY_EXCHANGE.get(ex)
    # 3) Try ISIN country
    if suf is None:
        cc = _isin_cc(isin or "")
        suf = SUFFIX_BY_ISIN_COUNTRY.get(cc) if cc else None

    # Default: US (no suffix)
    suf = "" if suf is None else suf

    # Format/pad base for certain markets
    base = _yahoo_base_with_padding(base, suf)

    # US class shares: convert inner dot to hyphen (e.g., BRK.B → BRK-B)
    if suf == "" and "." in base:
        base = base.replace(".", "-")

    return f"{base}.{suf}" if suf else base

def main():
    log("[INFO] Fetching instrument metadata from Trading 212…")
    data, base_used = fetch_instruments()
    total = len(data) if isinstance(data, list) else 0
    log(f"[INFO] Base used: {base_used}")
    log(f"[INFO] Received {total} instruments (all types)")

    rows_raw, rejects = [], []
    for it in (data or []):
        ttype = (it.get("type") or "").upper()
        if ttype not in {"EQUITY", "STOCK"}:
            continue
        tick = (it.get("ticker") or "").strip()
        name = (it.get("name") or it.get("shortName") or "").strip()
        if not tick or not name:
            continue

        simple = simplify_symbol(tick)
        if is_junk_symbol(simple):
            rejects.append((simple.upper(), name, "junk-filter"))
            continue

        # Try to pull exchange/mic/isin from the payload (T212 field names vary)
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
