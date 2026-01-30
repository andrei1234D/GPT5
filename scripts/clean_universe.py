# scripts/clean_universe.py
from __future__ import annotations

import os, sys, csv, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from universe_from_trading_212 import map_to_yahoo, simplify_symbol

IN_PATH   = Path(os.getenv("CLEAN_INPUT", "data/universe.csv"))
OUT_PATH  = Path(os.getenv("CLEAN_OUT", "data/universe_clean.csv"))
REJ_PATH  = Path(os.getenv("CLEAN_REJECTS", "data/universe_rejects.csv"))
META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))

# If 1, rewrite rejects to only the current run (plus any manual/persisted blocklist rows).
RESET_REJECTS = os.getenv("CLEAN_RESET_REJECTS", "0").lower() in {"1","true","yes"}

DT_REASON = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")

# Only these reasons are "blocking" across runs.
# Anything else in rejects.csv is informational only.
BLOCK_REASONS = {"manual", "persisted"}  # keep this tiny on purpose

def log(msg: str) -> None:
    print(msg, flush=True)

# -------------------- Helpers -------------------- #
def atomic_write_dicts(path: Path, rows: List[Dict[str, str]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def load_in(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        sys.exit(f"[ERROR] Missing {path}")
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t = (row.get("ticker") or "").strip().upper()
            n = (row.get("company") or "").strip()
            if not (t and n):
                continue
            out.append({
                "ticker": t,
                "company": n,
                "t212_ticker": (row.get("t212_ticker") or "").strip(),
                "shortName": (row.get("shortName") or "").strip(),
                "isin": (row.get("isin") or "").strip(),
                "exchange": (row.get("exchange") or "").strip(),
                "mic": (row.get("mic") or "").strip(),
                "currency": (row.get("currency") or "").strip(),
            })
    return out

def load_instruments() -> Optional[List[Dict[str,Any]]]:
    if META_PATH.exists():
        try:
            with META_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[WARN] Failed to read {META_PATH}: {e}")
    return None

def load_blocklist() -> Tuple[Dict[str, Tuple[str, str]], int, int]:
    """
    Returns:
      - block_map: {ticker: (company, reason)} for blocking reasons only
      - ignored_dt: count of timestamp-reason rows
      - ignored_other: count of non-blocking reason rows
    """
    block_map: Dict[str, Tuple[str, str]] = {}
    ignored_dt = 0
    ignored_other = 0
    if REJ_PATH.exists():
        try:
            with REJ_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    n = (row.get("company") or "").strip()
                    reason_raw = (row.get("reason") or "").strip() or "persisted"
                    if not t or not n:
                        continue
                    if DT_REASON.match(reason_raw):
                        ignored_dt += 1
                        continue
                    reason = reason_raw.lower()
                    if reason in BLOCK_REASONS:
                        block_map[t] = (n, reason_raw)
                    else:
                        ignored_other += 1
        except Exception as e:
            log(f"[WARN] Failed to read rejects {REJ_PATH}: {e}")
    return block_map, ignored_dt, ignored_other

def is_junk_symbol(sym: str) -> bool:
    # extremely conservative at this stage
    if not sym:
        return True
    return len(sym) > 60

def name_reject(nm: str) -> bool:
    # Keep conservative; you can loosen further if you want ETFs too.
    bad = ["DELISTED", "ETF", "ETN", "FUND", "CERTIFICATE", "WARRANT"]
    u = (nm or "").upper()
    return any(x in u for x in bad)

def choose_preferred(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """
    If multiple Trading212 instruments map to the same Yahoo ticker, pick one deterministically.
    Preference:
      1) currency == USD
      2) t212_ticker endswith '_US_EQ'
      3) otherwise shortest t212_ticker (stable)
    """
    def score(r: Dict[str, str]) -> Tuple[int, int, int]:
        cur = (r.get("currency") or "").upper()
        t212 = r.get("t212_ticker") or ""
        s1 = 1 if cur == "USD" else 0
        s2 = 1 if t212.endswith("_US_EQ") else 0
        s3 = -len(t212)  # shorter is better
        return (s1, s2, s3)

    best = max(rows, key=score)
    return best

# -------------------- Main -------------------- #
def main() -> None:
    block_map, ignored_dt, ignored_other = load_blocklist()
    log(f"[INFO] Blocklist loaded: {len(block_map)} (ignored timestamp rows: {ignored_dt}, ignored non-blocking rows: {ignored_other})")

    if not IN_PATH.exists():
        if OUT_PATH.exists():
            log(f"[INFO] {IN_PATH} missing. Using stale {OUT_PATH}")
            sys.exit(0)
        else:
            sys.exit(f"[FATAL] Missing {IN_PATH} and no fallback {OUT_PATH}")

    instruments = load_instruments()
    whitelist: set[str] = set()
    if instruments:
        for it in instruments:
            if (it.get("type") or it.get("instrumentType") or "").upper() not in {"EQUITY","STOCK"}:
                continue
            tick = (it.get("ticker") or it.get("symbol") or "").strip()
            mic  = it.get("mic"); exch = it.get("exchange"); isin = it.get("isin")
            if tick:
                simple = simplify_symbol(tick)
                yh = map_to_yahoo(simple, exch, mic, isin)
                whitelist.add(yh.upper())
        log(f"[INFO] Whitelist size (normalized to Yahoo): {len(whitelist)}")
    else:
        log("[WARN] No instrument metadata found, skipping whitelist")

    rows = load_in(IN_PATH)

    # Group by yahoo ticker to resolve mapping collisions deterministically
    by_ticker: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    kept: List[Dict[str, str]] = []
    run_rejects: List[Dict[str, str]] = []  # for universe_rejects.csv output
    reason_counts: Dict[str, int] = {}

    for ticker, group in by_ticker.items():
        chosen = choose_preferred(group)
        nm = chosen["company"]

        reason: Optional[str] = None
        if ticker in block_map:
            reason = block_map[ticker][1] or "persisted"
        elif whitelist and ticker not in whitelist:
            reason = "not-in-whitelist"
        elif name_reject(nm):
            reason = "name-filter"
        elif is_junk_symbol(ticker):
            reason = "symbol-hygiene"

        if reason:
            # record for this run (informational)
            run_rejects.append({"ticker": ticker, "company": nm, "reason": reason})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        kept.append(chosen)

    kept.sort(key=lambda r: (r["ticker"], r.get("t212_ticker","")))
    atomic_write_dicts(
        OUT_PATH,
        kept,
        ("ticker", "company", "t212_ticker", "shortName", "isin", "exchange", "mic", "currency"),
    )

    # Write universe_rejects.csv as a *report*, not a denylist.
    # Keep manual/persisted rows unless RESET_REJECTS=1.
    preserved: Dict[str, Tuple[str, str]] = {}
    if not RESET_REJECTS and REJ_PATH.exists():
        try:
            with REJ_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    n = (row.get("company") or "").strip()
                    r = (row.get("reason") or "").strip() or "persisted"
                    if not t or not n:
                        continue
                    if DT_REASON.match(r):
                        continue  # drop junk timestamps
                    if r.lower() in BLOCK_REASONS:
                        preserved[t] = (n, r)
        except Exception as e:
            log(f"[WARN] Failed to preserve blocklist rows: {e}")

    merged: Dict[str, Tuple[str, str]] = dict(preserved)
    for row in run_rejects:
        merged[row["ticker"]] = (row["company"], row["reason"])

    rej_rows = [{"ticker": t, "company": nm, "reason": reason} for t, (nm, reason) in merged.items()]
    rej_rows.sort(key=lambda r: r["ticker"])
    atomic_write_dicts(REJ_PATH, rej_rows, ("ticker","company","reason"))

    log(f"[INFO] Input rows (raw)       : {len(rows)}")
    log(f"[INFO] Unique tickers (raw)   : {len(by_ticker)}")
    log(f"[INFO] Kept (clean)           : {len(kept)}")
    log(f"[INFO] Rejected (this run)    : {len(run_rejects)}")
    log(f"[INFO] Rejects written (final): {len(rej_rows)}  (reset={RESET_REJECTS})")

    if reason_counts:
        top = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        log("[INFO] Reject breakdown (this run): " + ", ".join([f"{k}={v}" for k, v in top[:10]]))

    log(f"[DEBUG] Sample kept    : {[(r['ticker'], r['company']) for r in kept[:10]]}")
    log(f"[DEBUG] Sample rejects : {[(r['ticker'], r['company'], r['reason']) for r in rej_rows[:10]]}")

if __name__ == "__main__":
    main()
