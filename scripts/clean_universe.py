#!/usr/bin/env python3
# Trading212 → Yahoo universe cleaner (robust + precise)
# - Pulls T212 instruments, keeps only investable equities (no OTC)
# - Venue/ccy gate (US + core EU) with substring match; auto-relaxes if empty
# - Refined name filters (ETFs/warrants etc). ADRs KEPT by default.
# - Corporate-action/vendor alias map + LSE ".LL" fixer
# - Builds three whitelists: canonical (one per ISIN), ISIN-equivalents (all listings), and loose (pre-canonical)
# - Intersects against data/universe.csv (ticker,company)
# - Manual overrides: data/overrides_keep.txt / data/overrides_drop.txt
# - Outputs: data/universe_clean.csv, data/universe_rejects.csv

import os, sys, csv, re, requests
from pathlib import Path
from collections import Counter, defaultdict

# ---------------- CLI overrides (so you can pass apikey="..." base="...") ----------------
for _arg in sys.argv[1:]:
    if "=" in _arg:
        _k, _v = _arg.split("=", 1)
        _k = _k.strip().lstrip("-").lower()
        _v = _v.strip().strip('"').strip("'")
        if _k in ("apikey","api_key"):
            os.environ["T212_API_KEY"] = _v
        elif _k in ("base","t212_api_base"):
            os.environ["T212_API_BASE"] = _v

API_BASE = os.getenv("T212_API_BASE", "https://live.trading212.com")
API_KEY  = os.getenv("T212_API_KEY")

IN_PATH  = Path(os.getenv("CLEAN_INPUT",   "data/universe.csv"))
OUT_PATH = Path(os.getenv("CLEAN_OUT",     "data/universe_clean.csv"))
REJ_PATH = Path(os.getenv("CLEAN_REJECTS", "data/universe_rejects.csv"))

# ---------------- Venue / currency knobs ----------------
CORE_MICS = {
    "XNYS","XNAS",                      # US
    "XLON",                             # UK
    "XETR","XPAR","XMIL",               # DE/FR/IT
    "XAMS","XBRU","XLIS","XMAD"         # NL/BE/PT/ES
}
CORE_EXCHANGE_KEYWORDS = {
    "NYSE","NASDAQ",
    "LONDON","LSE",
    "XETRA","FRANKFURT",
    "PARIS","EURONEXT",
    "AMSTERDAM","BRUSSELS","LISBON","MADRID",
    "MILAN","BORSA ITALIANA",
}
_extra = os.getenv("EXCHANGE_KEYWORDS","").strip()
if _extra:
    CORE_EXCHANGE_KEYWORDS |= {s.strip().upper() for s in _extra.split(",") if s.strip()}

ALLOWED_CCYS = set((os.getenv("ALLOWED_CCYS") or "USD,GBP,GBX,GBp,EUR").split(","))

RELAX_IF_ZERO = os.getenv("RELAX_IF_ZERO","1") == "1"
DEBUG = os.getenv("DEBUG","0") == "1"

# ---------------- Instrument toggles ----------------
KEEP_REITS  = os.getenv("KEEP_REITS", "1") == "1"
KILL_TRUSTS = os.getenv("KILL_TRUSTS","1") == "1"
KILL_DR     = os.getenv("KILL_DR",    "0") == "1"  # keep ADR/ADS/GDR by default
KILL_PREFS  = os.getenv("KILL_PREFS", "1") == "1"
KILL_SPAC   = os.getenv("KILL_SPAC",  "1") == "1"

# ---------------- Helpers ----------------
def log(m): print(m, flush=True)
ALLOWED_SYM = re.compile(r"^[A-Z0-9.\-]+$")
OTC_MICS    = {"OTCM","PINX","OTCQ","OOTC","OTCB"}

def simplify_symbol(t212_ticker: str) -> str:
    if not t212_ticker: return ""
    return t212_ticker.split("_", 1)[0].strip().replace(" ", "-")

# Yahoo suffix maps (same spirit as your fetcher)
SUFFIX_BY_MIC = {
    "XNAS":"", "XNYS":"", "ARCX":"", "BATS":"", "IEXG":"", "CBOE":"", "XNGS":"", "XNCM":"",
    "XLON":"L", "XETR":"DE", "XFRA":"F", "XSWX":"SW", "XVTX":"SW", "XPAR":"PA", "XAMS":"AS",
    "XBRU":"BR", "XLIS":"LS", "XMAD":"MC", "XMIL":"MI", "XWBO":"VI", "XSTO":"ST", "XHEL":"HE",
    "XCSE":"CO", "XOSL":"OL", "XWAR":"WA", "XPRA":"PR",
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
}
SUFFIX_BY_ISIN_COUNTRY = {
    "US":"", "GB":"L", "DE":"DE", "FR":"PA", "NL":"AS", "BE":"BR", "PT":"LS",
    "ES":"MC", "IT":"MI",
}
KNOWN_YH_SUFFIXES = set(SUFFIX_BY_MIC.values()) | {"HK","TWO","IR","TW"}

def _isin_cc(isin: str) -> str | None:
    if not isin or len(isin) < 2: return None
    cc = isin[:2].upper()
    return cc if re.fullmatch(r"[A-Z]{2}", cc) else None

def _pad(base: str, suffix: str) -> str:
    if base.isdigit() and suffix in {"T","HK","TW","SS","SZ"}:
        return base.zfill(4)
    return base

def map_to_yahoo(symbol: str, exchange: str | None, mic: str | None, isin: str | None) -> str:
    base = (symbol or "").upper()
    ex = (exchange or "").upper().strip()
    mc = (mic or "").upper().strip()
    if "." in base:
        head, tail = base.rsplit(".", 1)
        if tail in KNOWN_YH_SUFFIXES:
            return f"{_pad(head, tail)}.{tail}"
    suf = SUFFIX_BY_MIC.get(mc)
    if suf is None: suf = SUFFIX_BY_EXCHANGE.get(ex)
    if suf is None:
        cc = _isin_cc(isin or "")
        suf = SUFFIX_BY_ISIN_COUNTRY.get(cc) if cc else None
    suf = "" if suf is None else suf
    base = _pad(base, suf)
    if suf == "" and "." in base:
        base = base.replace(".", "-")
    return f"{base}.{suf}" if suf else base

# ---------------- Name heuristics (conservative; avoid false-positives) ----------------
NAME_FUNDS_ETP  = re.compile(r"\b(UCITS|ETF|ETN|ETC|ETP|EXCHANGE[- ]TRADED(?:\s+(?:FUND|PRODUCT))?|INDEX\s+FUND|MUTUAL\s+FUND|SICAV|OEIC|UNIT\s+TRUST|CLOSED[- ]END\s+FUND|BDC)\b", re.I)
NAME_DERIV_PACK = re.compile(r"\b(WARRANTS?|RIGHTS?|UNITS?|SUBUNITS?|CERTIFICATES?|TURBO|SPRINTERS?|FACTOR\s+CERTIFICATES|MINI\s+FUTURES)\b", re.I)
NAME_INV_TRUST  = re.compile(r"\bINV(?:ESTMENT)?\s+TRUST\b", re.I)
NAME_REIT       = re.compile(r"\bREIT\b", re.I)
NAME_DR         = re.compile(r"\b(ADR|ADS|GDR|SDR|DEPOSITARY)\b", re.I)  # kept unless KILL_DR=1
PREF_CUES       = re.compile(r"\b(Preferred\s+(Stock|Shares?)|Depositary\s+Shares?|Series\s+[A-Z0-9]+|Non[- ]?Cumulative|Perpetual|Preference\s+Shares?)\b", re.I)
def is_preferred_ticker(yh: str) -> bool:
    s = (yh or "").upper()
    return bool(re.search(r"-(PR|P)[A-Z]$", s) or re.search(r"\.P[A-Z]$", s))
NAME_SPAC_STRICT = re.compile(r"\b(Acquisition\s+(?:Corp(?:oration)?|Company|Holdings?)\b).*?\b(Class\s+[A-Z]\b|Units?|Warrants?)", re.I)

def name_reject(nm: str, yh_ticker: str) -> tuple[bool,str]:
    n = nm or ""
    if NAME_FUNDS_ETP.search(n): return True, "fund/etp"
    if NAME_DERIV_PACK.search(n): return True, "warrant/right/unit/cert"
    if KILL_TRUSTS and NAME_INV_TRUST.search(n): return True, "investment-trust"
    if KILL_SPAC and NAME_SPAC_STRICT.search(n): return True, "spac"
    if KILL_DR and NAME_DR.search(n): return True, "depositary"
    if KILL_PREFS and (PREF_CUES.search(n) or is_preferred_ticker(yh_ticker)):
        return True, "preferred"
    if KEEP_REITS and NAME_REIT.search(n): return False, ""
    return False, ""

# ---------------- Aliases & normalizer ----------------
ALIAS_TICKERS = {
    # Corporate actions / vendor codes → investable Yahoo tickers
    "AAXN":   "AXON",  # Axon Enterprise
    "ABC":    "COR",   # AmerisourceBergen → Cencora
    "ADGI":   "IVVD",  # Adagio → Invivyd
    "ACIC":   "ACHR",  # Atlas Crest → Archer
    "ACEV":   "TMPO",  # Tempo Automation
    "ACAC":   "MYPS",  # Playstudios
    "AACQ":   "ORGN",  # Artius → Origin Materials
    "ADF":    "HGTY",  # Aldel → Hagerty
    "ABEAD":  "GOOGL", # Alphabet A
    "ABECD":  "GOOG",  # Alphabet C
    "ABFL.L": "ABF.L", # Associated British Foods
}
def normalize_csv_ticker(t: str) -> str:
    s = (t or "").strip().upper()
    # Fix LSE double "L" base: "ABDXL.L" -> "ABDX.L"
    if s.endswith(".L") and s[:-2].endswith("L"):
        s = s[:-3] + ".L"
    return ALIAS_TICKERS.get(s, s)

# ---------------- Overrides ----------------
def read_overrides(path: Path) -> set[str]:
    if not path.exists(): return set()
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                out.add(t.upper())
    return out
OVR_KEEP = read_overrides(Path("data/overrides_keep.txt"))
OVR_DROP = read_overrides(Path("data/overrides_drop.txt"))

# ---------------- API & IO ----------------
def _get(path: str):
    if not API_KEY:
        print("[ERROR] Missing T212_API_KEY"); sys.exit(1)
    return requests.get(f"{API_BASE}{path}", headers={"Authorization": API_KEY}, timeout=120)

def fetch_instruments():
    r = _get("/api/v0/equity/metadata/instruments")
    if r.status_code != 200:
        print(f"[ERROR] T212 returned {r.status_code}. Check key/base/scope."); sys.exit(1)
    return r.json() or []

def load_universe_csv(path: Path):
    if not path.exists(): sys.exit(f"[ERROR] Missing {path}.")
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        need = {"ticker","company"}
        have = {c.strip().lower() for c in (rdr.fieldnames or [])}
        if not need.issubset(have): sys.exit(f"[ERROR] {path} must have headers: ticker,company")
        for row in rdr:
            t = (row.get("ticker") or "").strip()
            n = (row.get("company") or "").strip()
            if t and n: out.append((t,n))
    return out

def atomic_write(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    os.replace(tmp, path)

# ---------------- Gates ----------------
def passes_core_venue_and_ccy(it: dict) -> bool:
    mic = (it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode") or "")
    mic = mic.upper().strip()
    exc_vals = [str(x).upper().strip() for x in [it.get("exchange"), it.get("exchangeCode"), it.get("venue"), it.get("market")] if x]
    ccy = (it.get("currencyCode") or it.get("currency") or "").upper().strip()

    if mic in OTC_MICS: return False
    pass_mic = mic in CORE_MICS if mic else False
    pass_exc = any(kw in v for v in exc_vals for kw in CORE_EXCHANGE_KEYWORDS)
    if not (pass_mic or pass_exc): return False
    if ccy and (ccy not in ALLOWED_CCYS): return False
    return True

def looks_investable(it: dict) -> bool:
    ttype = (it.get("type") or it.get("instrumentType") or "").upper()
    if ttype not in {"EQUITY","STOCK"}: return False
    for k in ("isSuspended","suspended","isDelisted","delisted","isRestricted","investRestricted","tradingDisabled"):
        v = it.get(k)
        if isinstance(v,bool) and v: return False
    pos = [bool(it.get(k)) for k in ("tradable","isTradable","isInvestAllowed","investAllowed","investAvailable","tradingAllowed","isEnabled","isActive") if k in it]
    return True if not pos else any(pos)

# ---------------- Main ----------------
def main():
    log("[INFO] Fetching T212 instruments…")
    inst = fetch_instruments()
    log(f"[INFO] Instruments: {len(inst):,}")

    # Stage 1: investable + venue/ccy
    pool, drops = [], Counter()
    for it in inst:
        if not looks_investable(it):
            drops["not-investable"] += 1; continue
        if not passes_core_venue_and_ccy(it):
            drops["venue/ccy"] += 1; continue
        pool.append(it)
    log(f"[STAGE] after investable+venue/ccy: {len(pool):,}")

    # Auto-relax venue if zero
    if len(pool) == 0 and RELAX_IF_ZERO:
        log("[WARN] Venue/ccy gate yielded 0 — relaxing venue filter.")
        for it in inst:
            if looks_investable(it):
                pool.append(it)
        log(f"[STAGE] after relax (investable only): {len(pool):,}")
    elif len(pool) == 0 and not RELAX_IF_ZERO:
        log("[FATAL] Pool is 0 and RELAX_IF_ZERO=0. Broaden EXCHANGE_KEYWORDS/ALLOWED_CCYS or enable relax.")
        sys.exit(1)

    # Stage 1.5: refined name nets AFTER relax
    pool2 = []
    for it in pool:
        name = (it.get("name") or it.get("shortName") or "").strip()
        tmp_tick = simplify_symbol(it.get("ticker") or "")
        exchange = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
        mic = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
        yh_tmp = map_to_yahoo(tmp_tick, exchange, mic, it.get("isin"))
        bad, why = name_reject(name, (yh_tmp or "").upper())
        if bad:
            drops[f"name-{why}"] += 1; continue
        pool2.append(it)
    pool = pool2
    log(f"[STAGE] after refined name nets: {len(pool):,}")

    # Loose whitelist (pre-canonical): catch majors like NVDA even if canonical misses
    loose_wl = set()
    for it in pool:
        tick = simplify_symbol(it.get("ticker") or "")
        exchange = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
        mic = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
        isin = it.get("isin")
        yh = map_to_yahoo(tick, exchange, mic, isin)
        if not yh: continue
        s = yh.upper().strip()
        if not ALLOWED_SYM.match(s) or (s.endswith("D") and re.search(r"\d", s)) or len(s) > 15:
            continue
        loose_wl.add(s)
    log(f"[STAGE] loose whitelist (pre-canonical): {len(loose_wl):,}")

    # Canonicalize per ISIN (fallback to ticker@exchange if ISIN missing)
    by_key = defaultdict(list)
    for it in pool:
        isin = (it.get("isin") or "").strip().upper()
        if isin and re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", isin):
            key = ("ISIN", isin)
        else:
            tick = simplify_symbol(it.get("ticker") or "").upper()
            exc  = (it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market") or "").upper()
            key = ("SYM", f"{tick}@{exc}")
        by_key[key].append(it)

    # Build ISIN-equivalent whitelist (all listings per group) + choose one canonical per group
    eq_wl = set()
    exc_pref = ["NYSE","NASDAQ","LONDON","LSE","XETRA","FRANKFURT","PARIS","AMSTERDAM","BRUSSELS","LISBON","MADRID","MILAN"]
    exc_rank = {e:i for i,e in enumerate(exc_pref)}
    chosen = []
    for key, lst in by_key.items():
        # Add all equivalents
        for it in lst:
            tick = simplify_symbol(it.get("ticker") or "")
            exchange = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
            mic = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
            yh = map_to_yahoo(tick, exchange, mic, it.get("isin"))
            if not yh: continue
            s = yh.upper().strip()
            if ALLOWED_SYM.match(s) and not (s.endswith("D") and re.search(r"\d", s)) and len(s) <= 15:
                eq_wl.add(s)
        # Pick one canonical
        lst.sort(key=lambda it: exc_rank.get((it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market") or "").upper(), 9999))
        chosen.append(lst[0])
    log(f"[STAGE] ISIN-equivalent whitelist: {len(eq_wl):,}")
    log(f"[STAGE] canonical listings: {len(chosen):,}")

    # Build canonical whitelist
    wl = set()
    for it in chosen:
        tick = simplify_symbol(it.get("ticker") or "")
        exchange = it.get("exchange") or it.get("exchangeCode") or it.get("venue") or it.get("market")
        mic = it.get("mic") or it.get("primaryMic") or it.get("marketIdentifierCode")
        yh = map_to_yahoo(tick, exchange, mic, it.get("isin"))
        if not yh: continue
        s = yh.upper().strip()
        if not ALLOWED_SYM.match(s) or (s.endswith("D") and re.search(r"\d", s)) or len(s) > 15:
            continue
        wl.add(s)
    log(f"[STAGE] whitelist size (canonical): {len(wl):,}")
    if not wl and not loose_wl and not eq_wl:
        log("[FATAL] All whitelists empty. Set DEBUG=1 to inspect.")
        sys.exit(1)

    # Intersect with your CSV (normalize + overrides + refined name nets on CSV name)
    rows = load_universe_csv(IN_PATH)
    kept, rejects = [], []
    reasons = Counter()
    seen = set()
    for yh, nm in rows:
        k_raw = (yh or "").upper().strip()
        if not k_raw:
            rejects.append((k_raw, nm, "empty-ticker")); reasons["empty-ticker"] += 1; continue
        k = normalize_csv_ticker(k_raw)
        if k in seen: 
            continue
        seen.add(k)

        # Overrides
        if k in OVR_DROP:
            rejects.append((k, nm, "override-drop")); reasons["override-drop"] += 1; continue
        if k in OVR_KEEP:
            kept.append((k, nm)); continue

        # Membership: canonical → ISIN equivalents → loose; also allow dot/hyphen alt
        k_alt = k.replace("-", ".")
        in_wl = (
            (k in wl) or (k_alt in wl) or
            (k in eq_wl) or (k_alt in eq_wl) or
            (k in loose_wl) or (k_alt in loose_wl)
        )
        if not in_wl:
            rejects.append((k, nm, "not-in-whitelist")); reasons["not-in-whitelist"] += 1
            continue

        bad, why = name_reject(nm, k)
        if bad:
            rejects.append((k, nm, f"csvname-{why}")); reasons[f"csvname-{why}"] += 1
            continue

        kept.append((k, nm))

    kept.sort(key=lambda x: x[0])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(OUT_PATH, kept, ("ticker","company"))
    atomic_write(REJ_PATH, rejects, ("ticker","company","reason"))

    log(f"[INFO] Input rows   : {len(rows):,}")
    log(f"[INFO] Kept (clean) : {len(kept):,}")
    log(f"[INFO] Rejected     : {len(rejects):,}")
    if rejects: log("[INFO] CSV reject breakdown: " + ", ".join(f"{k}:{v}" for k,v in reasons.most_common()))
    if DEBUG and drops: log("[INFO] Drops (API stage): " + ", ".join(f"{k}:{v}" for k,v in drops.most_common()))

if __name__ == "__main__":
    main()
