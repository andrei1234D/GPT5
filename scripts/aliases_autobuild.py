# aliases_autobuild.py (core logic snippet)

import os, re, csv, time
import yfinance as yf

YF_TEST_PERIOD   = os.getenv("YF_TEST_PERIOD", "60d")
YF_TEST_INTERVAL = os.getenv("YF_TEST_INTERVAL", "1d")
MIN_OK_ROWS      = int(os.getenv("ALIASES_MIN_ROWS", "20"))
MAX_PER_RUN      = int(os.getenv("ALIASES_MAX_PER_RUN", "250"))


# Map Trading212-style hyphen suffixes to Yahoo’s dot suffixes
SUFFIX_MAP = {
    "L": "L",   # London
    "T": "TO",  # Toronto
    "V": "V",   # TSXV
    "SI": "SI", "KS": "KS", "CO": "CO", "HE": "HE", "OL": "OL", "WA": "WA",
    "PR": "PR", "AX": "AX", "HK": "HK", "MI": "MI", "MC": "MC", "ST": "ST",
    "PA": "PA", "AS": "AS", "F": "F", "SW": "SW", "DE": "DE"
}

def _has_data(sym: str) -> bool:
    try:
        hist = yf.Ticker(sym).history(period=YF_TEST_PERIOD, interval=YF_TEST_INTERVAL, auto_adjust=True)
        return isinstance(hist, type(hist)) and len(hist.index) >= MIN_OK_ROWS
    except Exception:
        return False

def _generate_candidates(bad: str) -> list[str]:
    s = bad.strip().upper()

    # If it already has a dot suffix and fails, also try the plain base (for US tickers mis-suffixed)
    plain_from_dot = []
    if "." in s:
        base = s.split(".", 1)[0]
        if base and base.isalnum():
            plain_from_dot.append(base)

    # Hyphen → dot (T212 quirk): e.g., "BBDC-L", "SSONL-L", "ANIIL-L"
    cand = []
    m = re.match(r"^([A-Z0-9\.]+)-([A-Z]{1,2})$", s)
    if m:
        base, suff = m.group(1), m.group(2)
        if suff in SUFFIX_MAP:
            yy = SUFFIX_MAP[suff]
            # 1) direct conversion: base + .yy
            cand.append(f"{base}.{yy}")
            # 2) LSE trust quirk: if base ends with the first letter of the suffix (e.g., 'L'),
            #    also try dropping that final letter: "SSONL" -> "SSON.L", "ANIIL" -> "ANII.L"
            if base.endswith(yy[0]):
                cand.append(f"{base[:-1]}.{yy}")
            # 3) for 1-letter suffixes, also try dropping trailing duplicate letter again
            if len(yy) == 1 and base.endswith(yy):
                cand.append(f"{base[:-1]}.{yy}")

            # 4) if the mapped suffix is UK ".L", also try the plain base (some entries are US)
            if yy == "L":
                cand += plain_from_dot

    # From dot-suffix to plain (US fallback) if not already added
    cand += plain_from_dot

    # Always try the raw base with no suffix if it looks like a US ticker
    if "-" not in s and "." not in s and s.isalnum():
        cand.append(s)

    # De-dup while preserving order
    seen, out = set(), []
    for c in cand:
        if c not in seen:
            seen.add(c); out.append(c)
    return out[:5]  # cap to keep it surgical

def learn_aliases_from_rejects(path_in: str, path_out: str) -> int:
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    learned = []

    # Read rejects (expects a "ticker" column)
    ticks = []
    with open(path_in, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("ticker") or "").strip()
            if t:
                ticks.append(t)
    if not ticks:
        print(f"[aliases_autobuild] No rejected tickers in {path_in}")
        return 0

    print(f"[aliases_autobuild] Trying to auto-alias up to {min(len(ticks), MAX_PER_RUN)} of {len(ticks)} rejected tickers")

    for bad in ticks[:MAX_PER_RUN]:
        for cand in _generate_candidates(bad):
            if _has_data(cand):
                learned.append((bad, cand))
                print(f"[aliases_autobuild] learned {bad} -> {cand}")
                break

    if not learned:
        print("[aliases_autobuild] No aliases learned this run.")
        return 0

    # Append to aliases.csv as "from,to"
    append_header = not os.path.exists(path_out)
    with open(path_out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if append_header:
            w.writerow(["from", "to"])
        for a, b in learned:
            w.writerow([a, b])
    print(f"[aliases_autobuild] Wrote {len(learned)} new aliases -> {path_out}")
    return len(learned)
