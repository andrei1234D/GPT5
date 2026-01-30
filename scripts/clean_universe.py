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

# reject-list hygiene (matches junk you showed: 013CD, 2QKD, etc.)
SYNTHETIC_D = re.compile(r"^[0-9A-Z]{1,6}D$")

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
    """
    Load universe rows. Supports old format (ticker, company) and new extended format.
    """
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

def load_rejects() -> Dict[str, Tuple[str, str]]:
    """
    Load persistent rejects as {ticker: (company, reason)}.

    Hygiene:
      - ignore rows with empty company (your file contains many)
      - ignore obvious synthetic IDs like 013CD/2QKD/etc.
    """
    rejects: Dict[str, Tuple[str, str]] = {}
    if REJ_PATH.exists():
        try:
            with REJ_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    n = (row.get("company") or "").strip()
                    r = (row.get("reason") or "").strip() or "persisted"
                    if not t:
                        continue
                    if not n:
                        # drop blank-company garbage
                        continue
                    if SYNTHETIC_D.match(t) and "." not in t:
                        continue
                    rejects[t] = (n, r)
        except Exception as e:
            log(f"[WARN] Failed to read rejects {REJ_PATH}: {e}")
    return rejects

def is_junk_symbol(sym: str) -> bool:
    if not sym:
        return True
    return len(sym) > 25

def name_reject(nm: str) -> bool:
    # NOTE: removed "TRUST" because it rejects legitimate REITs and operating companies.
    bad = ["DELISTED", "ETF", "ETN", "FUND", "CERTIFICATE", "WARRANT"]
    u = (nm or "").upper()
    return any(x in u for x in bad)

# -------------------- Main -------------------- #
def main() -> None:
    reject_map = load_rejects()
    log(f"[INFO] Persistent rejects loaded (sanitized): {len(reject_map)}")

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
    kept: List[Dict[str, str]] = []
    new_rejects: List[Tuple[str, str, str]] = []  # ticker, company, reason
    seen: set[str] = set()

    # reason breakdown for this run only
    reason_counts: Dict[str, int] = {}

    for row in rows:
        t = row["ticker"]
        nm = row["company"]
        if t in seen:
            continue
        seen.add(t)

        reason: Optional[str] = None
        if t in reject_map:
            reason = reject_map[t][1] or "persisted"
        elif whitelist and t not in whitelist:
            reason = "not-in-whitelist"
        elif name_reject(nm):
            reason = "name-filter"
        elif is_junk_symbol(t):
            reason = "symbol-hygiene"

        if reason:
            new_rejects.append((t, nm, reason))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        kept.append(row)

    kept.sort(key=lambda r: (r["ticker"], r.get("t212_ticker", "")))

    # Write cleaned universe (preserve extra columns)
    atomic_write_dicts(
        OUT_PATH,
        kept,
        ("ticker", "company", "t212_ticker", "shortName", "isin", "exchange", "mic", "currency"),
    )

    # Merge rejects: old + any new from *this run* (new overwrites old reason)
    merged_rejects = dict(reject_map)
    for t, nm, reason in new_rejects:
        merged_rejects[t] = (nm, reason)

    rej_rows = [{"ticker": t, "company": nm, "reason": reason} for t, (nm, reason) in merged_rejects.items()]
    rej_rows.sort(key=lambda r: r["ticker"])
    atomic_write_dicts(REJ_PATH, rej_rows, ("ticker", "company", "reason"))

    log(f"[INFO] Input rows            : {len(rows)}")
    log(f"[INFO] Kept (clean)          : {len(kept)}")
    log(f"[INFO] Rejected (this run)   : {len(new_rejects)}")
    log(f"[INFO] Rejects total (merged): {len(rej_rows)}")

    if reason_counts:
        top = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        log("[INFO] Reject breakdown (this run): " + ", ".join([f"{k}={v}" for k, v in top[:10]]))

    log(f"[DEBUG] Sample kept    : {[(r['ticker'], r['company']) for r in kept[:10]]}")
    log(f"[DEBUG] Sample rejects : {[(r['ticker'], r['company'], r['reason']) for r in rej_rows[:10]]}")

if __name__ == "__main__":
    main()
