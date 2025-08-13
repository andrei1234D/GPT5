import csv, sys
from pathlib import Path

PREFERRED = [Path("data/universe_clean.csv"), Path("data/universe.csv")]

def load_universe():
    path = next((p for p in PREFERRED if p.exists()), None)
    if not path:
        print("[ERROR] Missing data/universe_clean.csv and data/universe.csv", flush=True)
        sys.exit(1)

    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        need = {"ticker","company"}
        have = {c.strip().lower() for c in (rdr.fieldnames or [])}
        if not need.issubset(have):
            print(f"[ERROR] {path} must have headers: ticker,company", flush=True)
            sys.exit(1)
        for row in rdr:
            t = (row.get("ticker") or "").strip().upper()
            n = (row.get("company") or "").strip()
            if t and n:
                out.append((t, n))

    if not out:
        print(f"[ERROR] {path} is empty after parsing.", flush=True)
        sys.exit(1)

    print(f"[INFO] Loaded {len(out)} tickers from {path}", flush=True)
    return out