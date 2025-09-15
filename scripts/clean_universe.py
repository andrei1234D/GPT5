# scripts/clean_universe.py
from __future__ import annotations
import os, sys, csv, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from universe_from_trading_212 import map_to_yahoo, simplify_symbol

IN_PATH   = Path(os.getenv("CLEAN_INPUT", "data/universe.csv"))
OUT_PATH  = Path(os.getenv("CLEAN_OUT", "data/universe_clean.csv"))
REJ_PATH  = Path(os.getenv("CLEAN_REJECTS", "data/universe_rejects.csv"))
META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))

def log(msg: str):
    print(msg, flush=True)

# -------------------- Helpers -------------------- #
def atomic_write(path: Path, rows: List[Tuple], header: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    os.replace(tmp, path)

def load_in(path: Path) -> List[Tuple[str,str]]:
    if not path.exists(): sys.exit(f"[ERROR] Missing {path}")
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t = (row.get("ticker") or "").strip().upper()
            n = (row.get("company") or "").strip()
            if t and n: out.append((t,n))
    return out

def load_instruments() -> Optional[List[Dict[str,Any]]]:
    if META_PATH.exists():
        try:
            with META_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[WARN] Failed to read {META_PATH}: {e}")
    return None

def load_rejects() -> Dict[str, Tuple[str,str]]:
    """Load persistent rejects as {ticker: (company, reason)}"""
    rejects = {}
    if REJ_PATH.exists():
        try:
            with REJ_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    n = (row.get("company") or "").strip()
                    r = (row.get("reason") or "").strip()
                    if t:
                        rejects[t] = (n, r if r else "persisted")
        except Exception as e:
            log(f"[WARN] Failed to read rejects {REJ_PATH}: {e}")
    return rejects

def is_junk_symbol(sym: str) -> bool:
    if not sym: return True
    return len(sym) > 15

def name_reject(nm: str) -> bool:
    return any(x in nm.upper() for x in ["DELISTED","ETF","ETN","FUND","TRUST","CERTIFICATE","WARRANT"])

# -------------------- Main -------------------- #
def main():
    # Load persistent rejects (full info)
    reject_map = load_rejects()
    if reject_map:
        log(f"[INFO] Persistent rejects loaded: {len(reject_map)}")

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
    kept, new_rejects, seen = [], [], set()

    for t, nm in rows:
        if t in seen: continue
        seen.add(t)

        if t in reject_map:
            new_rejects.append((t, nm, reject_map[t][1] or "persisted"))
            continue
        if whitelist and t not in whitelist:
            new_rejects.append((t, nm, "not-in-whitelist")); continue
        if name_reject(nm):
            new_rejects.append((t, nm, "name-filter")); continue
        if is_junk_symbol(t):
            new_rejects.append((t, nm, "symbol-hygiene")); continue

        kept.append((t, nm))

    kept.sort(key=lambda x: x[0])
    atomic_write(OUT_PATH, kept, ("ticker","company"))

    # Merge all rejects: old + new
    merged_rejects = dict(reject_map)  # start with old
    for t, nm, reason in new_rejects:
        merged_rejects[t] = (nm, reason)  # new overwrite old reason if exists

    rej_rows = [(t, nm, reason) for t,(nm,reason) in merged_rejects.items()]
    rej_rows.sort(key=lambda x: x[0])
    atomic_write(REJ_PATH, rej_rows, ("ticker","company","reason"))

    log(f"[INFO] Input rows   : {len(rows)}")
    log(f"[INFO] Kept (clean) : {len(kept)}")
    log(f"[INFO] Rejected     : {len(rej_rows)} (persisted+new)")
    log(f"[DEBUG] Sample kept: {kept[:10]}")
    log(f"[DEBUG] Sample rejects: {rej_rows[:10]}")

if __name__ == "__main__":
    main()
