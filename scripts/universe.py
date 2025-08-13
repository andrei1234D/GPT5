# scripts/universe.py
# Strict loader: require data/universe.csv (ticker,company). No fallback.

import csv
import sys
from pathlib import Path

CSV_PATH = Path("data/universe.csv")

def load_universe():
    if not CSV_PATH.exists():
        print(f"[ERROR] Missing {CSV_PATH}. Build it first (Trading 212 step).", flush=True)
        sys.exit(1)

    out = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Validate required headers (case-insensitive)
        need = {"ticker", "company"}
        have = {c.strip().lower() for c in (reader.fieldnames or [])}
        if not need.issubset(have):
            print(f"[ERROR] {CSV_PATH} must have headers: ticker,company (found: {sorted(have)})", flush=True)
            sys.exit(1)

        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            n = (row.get("company") or "").strip()
            if t and n:
                out.append((t, n))

    if not out:
        print(f"[ERROR] {CSV_PATH} is empty after parsing.", flush=True)
        sys.exit(1)

    print(f"[INFO] Loaded {len(out)} tickers from {CSV_PATH}", flush=True)
    return out
