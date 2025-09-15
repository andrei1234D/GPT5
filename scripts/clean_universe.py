# scripts/clean_universe.py
from __future__ import annotations
import os, sys, csv, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- imports from universe builder ---
from universe_from_trading_212 import map_to_yahoo, simplify_symbol

IN_PATH   = Path(os.getenv("CLEAN_INPUT", "data/universe.csv"))
OUT_PATH  = Path(os.getenv("CLEAN_OUT", "data/universe_clean.csv"))
REJ_PATH  = Path(os.getenv("CLEAN_REJECTS", "data/universe_rejects.csv"))
META_PATH = Path(os.getenv("UNIVERSE_META", "data/universe_meta.json"))

ALLOW_STALE        = os.getenv("ALLOW_STALE", "1").lower() in {"1","true","yes"}
CLEAN_ALLOW_OFFLINE= os.getenv("CLEAN_ALLOW_OFFLINE", "1").lower() in {"1","true","yes"}

def log(msg: str): 
    print(msg, flush=True)

# -------------------- Helpers --------------------
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

def load_existing_rejects(path: Path) -> set[str]:
    rejects = set()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    if t:
                        rejects.add(t)
        except Exception as e:
            log(f"[WARN] Failed to read existing rejects {path}: {e}")
    return rejects

def main():
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
            if tick: whitelist.add(tick.upper())
        log(f"[INFO] Whitelist size: {len(whitelist)}")
    else:
        log("[WARN] No instrument metadata found, skipping whitelist")

    rows = load_in(IN_PATH)
    kept, rejects, seen = [], [], set()

    # 🔥 Load old rejects to persist them across runs
    old_rejects = load_existing_rejects(REJ_PATH)
    all_rejects = set(old_rejects)

    for t, nm in rows:
        if t in seen: continue
        seen.add(t)

        if t in old_rejects:
            # Already rejected in past → keep rejecting
            rejects.append((t, nm, "persisted-reject"))
            continue
        if whitelist and t not in whitelist:
            all_rejects.add(t)
            rejects.append((t, nm, "not-in-whitelist")); continue
        if name_reject(nm):
            all_rejects.add(t)
            rejects.append((t, nm, "name-filter")); continue
        if is_junk_symbol(t):
            all_rejects.add(t)
            rejects.append((t, nm, "symbol-hygiene")); continue

        kept.append((t, nm))

    kept.sort(key=lambda x: x[0])

    # 🔥 Write clean universe
    atomic_write(OUT_PATH, kept, ("ticker","company"))

    # 🔥 Write combined rejects (old + new)
    rej_rows = [(t, "", "persisted") for t in old_rejects]  # keep history
    rej_rows.extend(rejects)  # add today’s new ones
    rej_rows = list({r[0]: r for r in rej_rows}.values())  # dedup by ticker
    atomic_write(REJ_PATH, rej_rows, ("ticker","company","reason"))

    log(f"[INFO] Input rows   : {len(rows)}")
    log(f"[INFO] Kept (clean) : {len(kept)}")
    log(f"[INFO] Rejected     : {len(rej_rows)} (total, persisted+new)")


def is_junk_symbol(sym: str) -> bool:
    if not sym: return True
    s = sym.upper()
    if len(s) > 15: return True
    return False

def name_reject(nm: str) -> bool:
    return any(x in nm.upper() for x in ["DELISTED","ETF","ETN","FUND","TRUST","CERTIFICATE","WARRANT"])

def load_rejects(path: Path) -> set[str]:
    """Load persistent rejects from universe_rejects.csv (from clean_universe and features.fetch_history)."""
    rejects = set()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    t = (row.get("ticker") or "").strip().upper()
                    if t:
                        rejects.add(t)
        except Exception as e:
            log(f"[WARN] Failed to read rejects {path}: {e}")
    return rejects


# -------------------- Main --------------------
def main():
        # Load persistent rejects
    reject_set = load_rejects()
    if reject_set:
        log(f"[INFO] Persistent rejects loaded: {len(reject_set)}")

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
            mic  = it.get("mic")
            exch = it.get("exchange")
            isin = it.get("isin")
            if tick:
                simple = simplify_symbol(tick)
                yh = map_to_yahoo(simple, exch, mic, isin)
                whitelist.add(yh.upper())
        log(f"[INFO] Whitelist size (normalized to Yahoo): {len(whitelist)}")
    else:
        log("[WARN] No instrument metadata found, skipping whitelist")

    rows = load_in(IN_PATH)
    kept, rejects, seen = [], [], set()

    for t, nm in rows:
        if t in seen:
            continue
        seen.add(t)

        if t in reject_set:
            rejects.append((t, nm, "persistent-reject"))
            continue
        if whitelist and t not in whitelist:
            rejects.append((t, nm, "not-in-whitelist"))
            continue
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

    log(f"[INFO] Input rows   : {len(rows)}")
    log(f"[INFO] Kept (clean) : {len(kept)}")
    log(f"[INFO] Rejected     : {len(rejects)}")

    # Debug peek
    log(f"[DEBUG] Sample kept: {kept[:10]}")
    log(f"[DEBUG] Sample rejects: {rejects[:10]}")

if __name__ == "__main__":
    main()
