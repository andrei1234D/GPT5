# scripts/aliases_autobuild.py
from __future__ import annotations
import os, time, logging
from typing import Dict, List, Tuple
import pandas as pd
import yfinance as yf

from aliases import load_aliases_csv, save_aliases_csv, generate_alias_candidates

logging.basicConfig(level=getattr(logging, os.getenv("ALIAS_LOG_LEVEL", "INFO").upper(), logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("aliases_autobuild")

REJECTS_PATH = os.getenv("ALIASES_FROM", "data/universe_rejects.csv")
ALIASES_OUT  = os.getenv("ALIASES_OUT",  "data/aliases.csv")
MAX_PER_RUN  = int(os.getenv("ALIASES_MAX_PER_RUN", "250"))
TEST_PERIOD  = os.getenv("YF_TEST_PERIOD", "60d")
TEST_INTERVAL= os.getenv("YF_TEST_INTERVAL","1d")
SLEEP_BETWEEN= float(os.getenv("ALIASES_SLEEP_S", "0.10"))

def _is_viable_yahoo(sym: str) -> bool:
    try:
        df = yf.download(sym, period=TEST_PERIOD, interval=TEST_INTERVAL,
                         auto_adjust=True, progress=False, threads=False, group_by="ticker")
        if isinstance(df, pd.DataFrame) and not df.empty and ("Close" in df.columns or "Adj Close" in df.columns):
            return True
    except Exception:
        return False
    return False

def main() -> None:
    if not os.path.exists(REJECTS_PATH):
        log.info("No rejects file found (%s); nothing to do.", REJECTS_PATH)
        return

    # Load newest batch of rejects (dedupe by ticker, keep last reason)
    rej = pd.read_csv(REJECTS_PATH)
    if "ticker" not in rej.columns:
        log.warning("Rejects file has no 'ticker' column; aborting.")
        return

    # Prefer most recent failures first
    if "ts" in rej.columns:
        rej = rej.sort_values("ts", ascending=False)
    tickers: List[str] = list(dict.fromkeys(rej["ticker"].astype(str).str.upper()))

    # Don't try to rebuild aliases we already have
    existing = load_aliases_csv(ALIASES_OUT)
    todo = [t for t in tickers if t not in existing]
    if not todo:
        log.info("No new tickers to alias. Existing=%d", len(existing))
        return

    log.info("Trying to auto-alias up to %d of %d rejected tickers", min(MAX_PER_RUN, len(todo)), len(todo))

    learned: Dict[str, str] = {}
    tries = 0
    for orig in todo:
        if len(learned) >= MAX_PER_RUN:
            break
        tries += 1
        cands = generate_alias_candidates(orig)
        ok = None
        for cand in cands[:20]:  # keep it tight
            if _is_viable_yahoo(cand):
                ok = cand
                break
            time.sleep(SLEEP_BETWEEN)
        if ok:
            learned[orig] = ok
            log.info("[learned] %s -> %s", orig, ok)
        else:
            log.debug("[miss] %s (no viable candidate)", orig)

    if learned:
        save_aliases_csv(ALIASES_OUT, learned)
        log.info("Saved %d new aliases to %s", len(learned), ALIASES_OUT)
    else:
        log.info("No aliases learned this run.")

if __name__ == "__main__":
    main()
