# scripts/aliases.py
from __future__ import annotations
import csv, os, re, time
from typing import Dict, Iterable, Tuple, List

# A few hard-coded fixes you know are always correct (can keep empty)
STATIC_ALIASES: Dict[str, str] = {
    # "BRK.B": "BRK-B",
    # "RDSA": "SHEL",      # example
}

# Common Yahoo suffixes to try when unknown exchange
COMMON_SUFFIXES = [
    "L","TO","V","PA","AS","F","SW","MC","MI","ST","HE","CO","OL","WA","PR",
    "AX","HK","T","SI","KS","KQ","NS","BO","JO","SA","MX","NZ","DU","AD","SR","TA",
    "SS","SZ","TW","IR"
]

def load_aliases_csv(path: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not os.path.exists(path):
        return m
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = (r.get("original") or "").strip().upper()
            b = (r.get("alias") or "").strip().upper()
            if a and b and a != b:
                m[a] = b
    return m

def save_aliases_csv(path: str, mapping: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Merge with existing file to avoid losing rows
    exists = load_aliases_csv(path)
    exists.update(mapping)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "alias", "added_at"])
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for k, v in sorted(exists.items()):
            w.writerow([k, v, ts])

def apply_alias(sym: str, extra: Dict[str, str] | None = None) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return s
    if extra and s in extra:
        return extra[s]
    if s in STATIC_ALIASES:
        return STATIC_ALIASES[s]
    return s

def generate_alias_candidates(sym: str) -> List[str]:
    """Yield a small set of likely Yahoo tickers for a broken symbol."""
    s = (sym or "").strip().upper().replace(" ", "-")
    cands = []
    def add(x):
        if x and x != sym and x not in cands:
            cands.append(x)

    # class-share dot/hyphen swaps
    if "." in s:
        head, tail = s.rsplit(".", 1)
        add(f"{head}-{tail}")
    if "-" in s:
        head, tail = s.rsplit("-", 1)
        # Only turn last hyphen into dot when both sides look like class/share
        if re.fullmatch(r"[A-Z0-9]+", head) and re.fullmatch(r"[A-Z0-9]+", tail):
            add(f"{head}.{tail}")

    # If no suffix, try adding common suffixes (LSE, TSX, EU, AU, etc.)
    if "." not in s:
        for suf in COMMON_SUFFIXES:
            add(f"{s}.{suf}")

    # Numeric tickers — try JP/HK/TW/CN zero-pads
    m = re.fullmatch(r"(\d{1,5})(?:\.[A-Z]+)?", s)
    if m:
        num = m.group(1)
        # JP: 4 digits + .T
        add(f"{num.zfill(4)}.T")
        # HK: 4 digits + .HK (Yahoo uses 4-digit codes e.g., 0700.HK)
        add(f"{num.zfill(4)}.HK")
        # CN: .SS Shanghai / .SZ Shenzhen (6 digits usually, but try padded)
        add(f"{num.zfill(6)}.SS")
        add(f"{num.zfill(6)}.SZ")
        # TW: 4 digits + .TW
        add(f"{num.zfill(4)}.TW")

    # As a last resort, try raw uppercased
    add(s)

    return cands
