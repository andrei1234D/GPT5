# scripts/universe.py
from __future__ import annotations
import os, csv
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple

# Allow overrides via env, but keep sane defaults
CLEAN_PATH = Path(os.getenv("UNIVERSE_CLEAN_PATH", "data/universe_clean.csv"))
RAW_PATH   = Path(os.getenv("UNIVERSE_RAW_PATH",   "data/universe.csv"))

def _read_pairs(path: Path) -> List[Tuple[str, str]]:
    """Read (ticker, company) from a CSV with headers 'ticker,company'.
    Returns [] if the file doesn't exist or is empty/invalid."""
    if not path.exists():
        return []
    rows: list[tuple[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            if not rdr.fieldnames:
                return []
            cols = [c.strip().lower() for c in rdr.fieldnames]
            # Expect 'ticker' & 'company'. If not, try the first two columns.
            if "ticker" in cols and "company" in cols:
                for r in rdr:
                    t = (r.get("ticker") or "").strip()
                    n = (r.get("company") or "").strip()
                    if t and n:
                        rows.append((t.upper(), n))
            else:
                # fallback: try first two columns
                idx0, idx1 = 0, 1
                for r in csv.reader(open(path, "r", encoding="utf-8-sig", newline="")):
                    if not r or len(r) < 2:
                        continue
                    if r[0].lower() == "ticker" and r[1].lower() == "company":
                        continue
                    t, n = r[idx0].strip(), r[idx1].strip()
                    if t and n:
                        rows.append((t.upper(), n))
    except Exception:
        return []
    # de-dup & keep stable order
    seen = set()
    out: list[tuple[str, str]] = []
    for t, n in rows:
        if t not in seen:
            seen.add(t); out.append((t, n))
    return out

@lru_cache(maxsize=1)
def load_universe(limit: int | None = None) -> List[Tuple[str, str]]:
    """
    Preferred source: data/universe_clean.csv
    Fallback:         data/universe.csv
    Last resort:      tiny built-in list so the pipeline doesn't crash the first time.
    """
    rows = _read_pairs(CLEAN_PATH)
    if not rows:
        rows = _read_pairs(RAW_PATH)

    if not rows:
        # Last-resort seed (keeps notify.py from dying on first boot/cold cache)
        rows = [
            ("NVDA", "NVIDIA"),
            ("AAPL", "Apple"),
            ("MSFT", "Microsoft"),
            ("AMZN", "Amazon"),
            ("META", "Meta Platforms"),
            ("GOOGL", "Alphabet Class A"),
            ("GOOG", "Alphabet Class C"),
        ]

    if limit and limit > 0:
        rows = rows[:limit]
    return rows
