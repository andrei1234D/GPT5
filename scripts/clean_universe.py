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

DT_REASON = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")

# Only these reasons are treated as "blocking" persistent rejects.
# Anything else (like timestamps) is treated as non-blocking notes.
BLOCK_REASONS = {
    "persisted",
    "manual",
    "name-filter",
    "symbol-hygiene",
    "not-in-whitelist",
}

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

def load_persistent_rejects() -> Tuple[Dict[str, Tuple[str, str]], int]:
    """
    Returns:
      - reject_map: {ticker: (company, reason)} for BLOCKING reasons only
      - ignored: count of non-blocking rows ignored (e.g. timestamp reasons)
    """
    reject_map: Dict[str, Tuple[str, str]] = {}
    ignored = 0
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

                    reason = reason_raw.lower()

                    # Treat timestamp-like "reasons" as non-blocking notes (ignore for filtering).
                    if DT_REASON.match(reason_raw):
                        ignored += 1
                        continue

                    if reason not in BLOCK_REASONS:
                        ignored += 1
                        continue

                    reject_map[t] = (n, reason_raw)
        except Exception as e:
            log(f"[WARN] Failed to read rejects {REJ_PATH}: {e}")
    return reject_map, ignored

def is_junk_symbol(sym: str) -> bool:
    if not sym:
        return True
    return len(sym) > 40

def name_reject(nm: str) -> bool:
    # NOTE: keep this conservative; your pipeline can decide later what to do with ETFs etc.
    bad = ["DELISTED", "ETF", "ETN", "FUND", "CERTIFICATE", "WARRANT"]
    u = (nm or "").upper()
    return any(x in u for x in bad)

# -------------------- Main -------------------- #
def main() -> None:
    reject_map, ignored = load_persistent_rejects()
    log(f"[INFO] Persistent rejects loaded (blocking only): {len(reject_map)} (ignored non-blocking rows: {ignored})")

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

    atomic_write_dicts(
        OUT_PATH,
        kept,
        ("ticker", "company", "t212_ticker", "shortName", "isin", "exchange", "mic", "currency"),
    )

    # Merge rejects file: keep existing + add any new rejects from this run
    # (but preserve non-blocking timestamp rows already present)
    merged_rows: Dict[str, Tuple[str, str]] = {}

    # Load everything (blocking + non-blocking) to keep file stable
    if REJ_PATH.exists():
        try:
            with REJ_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    n = (row.get("company") or "").strip()
                    r = (row.get("reason") or "").strip() or "persisted"
                    if t and n:
                        merged_rows[t] = (n, r)
        except Exception as e:
            log(f"[WARN] Failed to reload rejects for merge: {e}")

    for t, nm, reason in new_rejects:
        merged_rows[t] = (nm, reason)

    rej_rows = [{"ticker": t, "company": nm, "reason": reason} for t, (nm, reason) in merged_rows.items()]
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
