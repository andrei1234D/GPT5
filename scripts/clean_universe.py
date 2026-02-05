# scripts/clean_universe.py
from __future__ import annotations

import os, sys, csv, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Reuse mapping helpers from the builder
from universe_from_trading_212 import map_to_yahoo, parse_t212_ticker, sanitize_symbol

IN_PATH   = Path(os.getenv("CLEAN_INPUT", "data/universe.csv"))
OUT_PATH  = Path(os.getenv("CLEAN_OUT", "data/universe_clean.csv"))

# This is the "run output" rejects file (used by aliases_autobuild).
REJ_PATH  = Path(os.getenv("CLEAN_REJECTS", "data/universe_rejects.csv"))

# Persistent rejects should NOT share a filename with any yfinance/feature rejection logs.
PERSIST_PATH = Path(os.getenv("CLEAN_PERSIST_REJECTS", "data/universe_rejects_persisted.csv"))

META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))

def log(msg: str) -> None:
    print(msg, flush=True)

# -------------------- Helpers -------------------- #
def atomic_write(path: Path, rows: List[Tuple], header: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)

def load_universe_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        sys.exit(f"[ERROR] Missing {path}")
    out: List[Dict[str,str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t = (row.get("ticker") or "").strip().upper()
            n = (row.get("company") or "").strip()
            if t and n:
                out.append({"ticker": t, "company": n})
    return out

def load_instruments() -> Optional[List[Dict[str, Any]]]:
    if META_PATH.exists():
        try:
            with META_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[WARN] Failed to read {META_PATH}: {e}")
    return None

_ALLOWED_PERSIST_REASONS = {
    "persisted",
    "name-filter",
    "symbol-hygiene",
    "not-in-whitelist",
}
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

def load_persistent_rejects() -> Dict[str, Tuple[str, str]]:
    """
    Load persistent rejects as {ticker: (company, reason)}.
    IMPORTANT: we ignore timestamp-like reasons, because those often come from other scripts
    (e.g., yfinance failures) and would poison the universe.
    """
    rejects: Dict[str, Tuple[str,str]] = {}
    if not PERSIST_PATH.exists():
        return rejects

    try:
        with PERSIST_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                t = (row.get("ticker") or "").strip().upper()
                n = (row.get("company") or "").strip()
                r = (row.get("reason") or "").strip() or "persisted"
                if not t:
                    continue
                if _TS_RE.match(r):
                    # likely polluted rejects; skip
                    continue
                # keep unknown reasons, but normalize empty
                rejects[t] = (n, r)
    except Exception as e:
        log(f"[WARN] Failed to read persistent rejects {PERSIST_PATH}: {e}")
    return rejects

def is_junk_symbol(sym: str) -> bool:
    if not sym:
        return True
    return len(sym) > 20

_NAME_REJECT_RE = re.compile(
    r"\b(DELISTED|ETF|ETN|FUND|TRUST|CERTIFICATE|WARRANT)\b",
    re.IGNORECASE,
)

def name_reject(nm: str) -> bool:
    u = (nm or "").strip()
    return bool(_NAME_REJECT_RE.search(u))

def build_whitelist_from_meta(instruments: List[Dict[str,Any]]) -> set[str]:
    """
    Build whitelist of Yahoo-mapped tickers from T212 meta using the SAME logic as universe builder:
    - prefer shortName
    - use market hint from t212 ticker (e.g., *_US_EQ)
    """
    wl: set[str] = set()
    for it in instruments:
        if (it.get("type") or it.get("instrumentType") or "").upper() not in {"EQUITY","STOCK"}:
            continue
        t212_tick = (it.get("ticker") or it.get("symbol") or "").strip()
        short = (it.get("shortName") or "").strip()
        if not t212_tick:
            continue

        base_from_t212, market_hint, _asset = parse_t212_ticker(t212_tick)
        base = sanitize_symbol(short or base_from_t212)
        yh = map_to_yahoo(base, it.get("exchange"), it.get("mic"), it.get("isin"), market_hint)
        wl.add(yh.upper())
    return wl

# -------------------- Main -------------------- #
def main() -> None:
    persist_map = load_persistent_rejects()
    log(f"[INFO] Persistent rejects loaded: {len(persist_map)} ({PERSIST_PATH})")

    if not IN_PATH.exists():
        if OUT_PATH.exists():
            log(f"[INFO] {IN_PATH} missing. Using stale {OUT_PATH}")
            sys.exit(0)
        sys.exit(f"[FATAL] Missing {IN_PATH} and no fallback {OUT_PATH}")

    instruments = load_instruments()
    whitelist: set[str] = set()
    if instruments:
        whitelist = build_whitelist_from_meta(instruments)
        log(f"[INFO] Whitelist size (Yahoo-normalized): {len(whitelist)}")
    else:
        log("[WARN] No instrument metadata found, skipping whitelist")

    rows = load_universe_csv(IN_PATH)
    kept: List[Tuple[str,str]] = []
    rejects_this_run: List[Tuple[str,str,str]] = []
    seen: set[str] = set()

    for row in rows:
        t = row["ticker"]
        nm = row["company"]

        if t in seen:
            continue
        seen.add(t)

        # persistent rejects
        if t in persist_map:
            reason = (persist_map[t][1] or "persisted")
            rejects_this_run.append((t, nm, reason))
            continue

        # whitelist guard (only if meta exists)
        if whitelist and t not in whitelist:
            rejects_this_run.append((t, nm, "not-in-whitelist"))
            continue

        # name-based filters
        if name_reject(nm):
            rejects_this_run.append((t, nm, "name-filter"))
            continue

        if is_junk_symbol(t):
            rejects_this_run.append((t, nm, "symbol-hygiene"))
            continue

        kept.append((t, nm))

    kept.sort(key=lambda x: x[0])
    atomic_write(OUT_PATH, kept, ("ticker","company"))

    # Merge for "run rejects" output (used by aliases builder)
    merged = dict(persist_map)
    for t, nm, reason in rejects_this_run:
        merged[t] = (nm, reason)

    rej_rows = [(t, nm, reason) for t, (nm, reason) in merged.items()]
    rej_rows.sort(key=lambda x: x[0])
    atomic_write(REJ_PATH, rej_rows, ("ticker","company","reason"))

    # Update persistent rejects (sanitized): we keep ALL reasons except timestamp-like ones.
    # This avoids poisoning from other scripts if they accidentally touch REJ_PATH.
    persist_rows = [(t, nm, reason) for (t, (nm, reason)) in merged.items() if not _TS_RE.match(str(reason or ""))]
    persist_rows.sort(key=lambda x: x[0])
    atomic_write(PERSIST_PATH, persist_rows, ("ticker","company","reason"))

    # stats
    breakdown: Dict[str,int] = {}
    for (_t, _nm, r) in rejects_this_run:
        breakdown[r] = breakdown.get(r, 0) + 1

    log(f"[INFO] Input rows   : {len(rows)}")
    log(f"[INFO] Kept (clean) : {len(kept)} -> {OUT_PATH}")
    log(f"[INFO] Rejected this run: {len(rejects_this_run)}")
    log(f"[INFO] Rejects total (merged): {len(rej_rows)} -> {REJ_PATH}")
    if breakdown:
        top = ", ".join([f"{k}={v}" for k,v in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:10]])
        log(f"[INFO] Reject breakdown (this run): {top}")
    log(f"[DEBUG] Sample kept    : {kept[:10]}")
    log(f"[DEBUG] Sample rejects : {rej_rows[:10]}")

if __name__ == "__main__":
    main()
