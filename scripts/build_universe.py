# scripts/build_universe.py
import os, sys, csv, time, re, json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import requests

# Optional validation (pip install yfinance)
VALIDATE = os.getenv("UNIVERSE_VALIDATE", "0").lower() in {"1","true","yes"}
TRY_US_ONLY = os.getenv("UNIVERSE_US_ONLY", "0").lower() in {"1","true","yes"}
YF_SLEEP_SEC = float(os.getenv("UNIVERSE_YF_SLEEP", "1.2"))  # throttle between calls
YF_MAX_RETRIES = int(os.getenv("UNIVERSE_YF_RETRIES", "2"))

API_BASE_LIVE = os.getenv("T212_API_BASE", "https://live.trading212.com")
API_BASE_DEMO = "https://demo.trading212.com"
API_KEY = os.getenv("T212_API_KEY")

OUT_DIR = Path("data")
OUT_PATH = OUT_DIR / "universe.csv"
REJECTS_PATH = OUT_DIR / "universe_rejects.csv"
RAW_PATH = OUT_DIR / "universe_raw.csv"

def log(msg): 
    print(msg, flush=True)

def _do_get(base, path):
    if not API_KEY:
        log("[ERROR] Missing T212_API_KEY environment variable"); sys.exit(1)
    url = f"{base}{path}"
    r = requests.get(url, headers={"Authorization": API_KEY}, timeout=120)
    return r

def fetch_instruments() -> Tuple[List[dict], str]:
    r = _do_get(API_BASE_LIVE, "/api/v0/equity/metadata/instruments")
    if r.status_code == 200:
        return r.json(), API_BASE_LIVE
    if r.status_code in (401, 403):
        r2 = _do_get(API_BASE_DEMO, "/api/v0/equity/metadata/instruments")
        if r2.status_code == 200:
            log("[WARN] Live rejected the key; Demo accepted. Using demo base.")
            return r2.json(), API_BASE_DEMO
        log(f"[ERROR] Trading212 auth failed. live={r.status_code}, demo={r2.status_code}")
        log("Hints:\n"
            " • Ensure the key matches the environment (Live vs Practice).\n"
            " • Practice keys require https://demo.trading212.com.\n"
            " • Make sure 'metadata' scope is enabled.\n"
            " • Header must be exactly: Authorization: <token> (no 'Bearer').")
        sys.exit(1)
    r.raise_for_status()
    return r.json(), API_BASE_LIVE

# ---------- Yahoo suffix mapping ----------
# We try to infer Yahoo suffix from exchange / country strings.
YAHOO_SUFFIX_BY_HINT = [
    (("NASDAQ","NSDQ","NMS","NYSE","AMEX","ARCA","BATS","USA","UNITED STATES"), ""),   # US
    (("LSE","LONDON"), ".L"),
    (("TSX","TORONTO"), ".TO"),
    (("TSXV","VENTURE"), ".V"),
    (("XETRA","FRANKFURT","GERMANY","DEUTSCHE BOERSE"), ".DE"),
    (("PARIS","EURONEXT PARIS"), ".PA"),
    (("AMSTERDAM","EURONEXT AMSTERDAM"), ".AS"),
    (("BRUSSELS","EURONEXT BRUSSELS"), ".BR"),
    (("MILAN","BORSA ITALIANA"), ".MI"),
    (("MADRID","BME","BOLSA DE MADRID"), ".MC"),
    (("ZURICH","SIX"), ".SW"),
    (("VIENNA","WIENER BOERSE"), ".VI"),
    (("STOCKHOLM","NASDAQ STOCKHOLM","OMX STOCKHOLM"), ".ST"),
    (("COPENHAGEN","NASDAQ COPENHAGEN"), ".CO"),
    (("OSLO","OSE"), ".OL"),
    (("HELSINKI","NASDAQ HELSINKI"), ".HE"),
    (("LISBON","EURONEXT LISBON"), ".LS"),
    (("HONG KONG","HKEX"), ".HK"),
    (("TOKYO","TSE","JPX"), ".T"),
    (("ASX","AUSTRALIAN SECURITIES"), ".AX"),
    (("SINGAPORE","SGX"), ".SI"),
    (("NSE","NATIONAL STOCK EXCHANGE OF INDIA"), ".NS"),
    (("BSE","BOMBAY STOCK EXCHANGE"), ".BO"),
    (("SAUDI","TADAWUL"), ".SR"),
    (("JSE","JOHANNESBURG"), ".JO"),
]

VENDOR_JUNK = re.compile(r"^\d{1,3}[A-Z]*D$|^[A-Z0-9]{1,3}\d+D$|^[0-9].*D$")  # e.g., 0Q0D, 3AG1D, 1Q5D…

def _guess_suffix(hints: List[str]) -> str:
    txt = " ".join([h for h in hints if h]).upper()
    for keys, suf in YAHOO_SUFFIX_BY_HINT:
        if any(k in txt for k in keys):
            return suf
    return ""  # default to US if unknown

def _simplify_class(tick: str) -> str:
    """
    Normalize class/share notations:
      - 'BRK.B' -> 'BRK-B'
      - 'RDS A' -> 'RDS-A'
      - Remove trailing dot from LSE like 'BP.' -> 'BP'
    """
    t = tick.strip().upper().replace(" ", "-").replace("/", "-")
    t = re.sub(r"\.$", "", t)                 # drop trailing dot
    t = re.sub(r"\.([A-Z])$", r"-\1", t)      # .A -> -A
    return t

def _looks_junky(t: str) -> bool:
    if len(t) > 15: return True
    if VENDOR_JUNK.match(t): return True
    return False

def _to_yahoo(t212: dict) -> Optional[Tuple[str,str,str]]:
    raw = (t212.get("ticker") or "").strip()
    name = (t212.get("name") or t212.get("shortName") or "").strip()
    if not raw or not name:
        return None

    if t212.get("type","").upper() not in {"EQUITY","STOCK"}:
        return None

    # Optional US-only filter
    country = (t212.get("country") or t212.get("countryCode") or t212.get("countryIso2") or "")
    if TRY_US_ONLY and country.upper() not in {"US","USA","UNITED STATES"}:
        return None

    exch = (t212.get("exchange") or t212.get("venue") or t212.get("market") or "")
    mic  = (t212.get("mic") or "")
    currency = (t212.get("currency") or t212.get("currencyCode") or "")
    hints = [exch, mic, country, currency]
    suffix = _guess_suffix(hints)

    base = _simplify_class(raw.split("_",1)[0])
    if _looks_junky(base):
        return None

    # Special LSE quirk: T212 often uses 'BP.'; Yahoo wants BP.L (we already stripped '.')
    yahoo = base + suffix
    return yahoo, name, exch or mic or country or ""

# ---------- Optional Yahoo validation ----------
def _validate_yahoo(symbol: str) -> bool:
    if not VALIDATE:
        return True
    # Lazy import to make script runnable without yfinance when validation off
    import yfinance as yf
    tries = 0
    while True:
        tries += 1
        try:
            df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=False, group_by=False)
            ok = isinstance(df, type(getattr(df, "iloc", None))) and not df.empty
            return bool(ok)
        except Exception as e:
            # Basic backoff on rate limits
            if tries <= YF_MAX_RETRIES:
                time.sleep(YF_SLEEP_SEC * tries)
                continue
            return False
        finally:
            time.sleep(YF_SLEEP_SEC)

def main():
    log("[INFO] Fetching instrument metadata from Trading212…")
    data, base_used = fetch_instruments()
    total = len(data) if isinstance(data, list) else 0
    log(f"[INFO] Base used: {base_used}")
    log(f"[INFO] Received {total} instruments (all types)")

    # Save raw for debugging
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with RAW_PATH.open("w", encoding="utf-8") as f:
            f.write(json.dumps(data[:500], ensure_ascii=False, indent=2))  # sample to keep file small
    except Exception:
        pass

    seen = set()
    kept: List[Tuple[str,str]] = []
    rejects: List[Tuple[str,str]] = []

    for it in (data or []):
        conv = _to_yahoo(it)
        if not conv:
            # Try to preserve why it failed when possible
            raw = (it.get("ticker") or "").strip()
            name = (it.get("name") or it.get("shortName") or "").strip()
            if raw and name:
                rejects.append((raw, name, "not_equity_or_junky_or_nonUS" if TRY_US_ONLY else "not_equity_or_junky"))
            continue

        yahoo, name, _hint = conv
        if yahoo in seen:
            continue
        if _looks_junky(yahoo.replace(".","")):
            rejects.append((yahoo, name, "junk_pattern")); 
            continue

        if _validate_yahoo(yahoo):
            kept.append((yahoo, name))
            seen.add(yahoo)
        else:
            rejects.append((yahoo, name, "yahoo_validation_failed"))

    kept.sort(key=lambda x: x[0])

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker","company"])
        w.writerows(kept)

    with REJECTS_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_or_yahoo","company","reason"])
        w.writerows(rejects)

    log(f"[INFO] Wrote {len(kept)} valid rows to {OUT_PATH}")
    log(f"[INFO] Wrote {len(rejects)} rejects to {REJECTS_PATH}")
    if VALIDATE:
        log("[INFO] Validation was ON (yfinance). Use UNIVERSE_VALIDATE=0 to skip for speed.")
    if TRY_US_ONLY:
        log("[INFO] US-only mode was ON. Use UNIVERSE_US_ONLY=0 to include other markets.")

if __name__ == "__main__":
    main()
