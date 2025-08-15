# scripts/aliases.py
from __future__ import annotations
from typing import Dict, Optional
import os, csv, logging

logger = logging.getLogger("aliases")

# Built-in aliases (input ticker -> Yahoo symbol)
# Add the common offenders here.
ALIASES: Dict[str, str] = {
    "CHFS": "NUWE",   # CHF Solutions -> Nuwellis
    # "FB": "META",   # example of legacy rename (uncomment if you need it)
    # add more static aliases as you discover them
}

def load_aliases_csv(path: str) -> Dict[str, str]:
    """
    Optional: load extra aliases from CSV at `path` with two columns:
      src,dst
    Lines starting with '#' are ignored. Case-insensitive.
    """
    mapping: Dict[str, str] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            src = row[0].strip().upper()
            dst = row[1].strip().upper()
            if src and dst:
                mapping[src] = dst
    logger.info(f"[aliases] loaded {len(mapping)} from {path}")
    return mapping

def apply_alias(sym: str, extra: Optional[Dict[str, str]] = None) -> str:
    """
    Return aliased symbol if present (case-insensitive).
    `extra` overrides built-ins if provided.
    """
    if not sym:
        return sym
    s = sym.strip().upper()
    if extra and s in extra:
        return extra[s]
    return ALIASES.get(s, s)
