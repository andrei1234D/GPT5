from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/daily_scored.parquet")
    parser.add_argument("--out", default="data/daily_keep_7/keep.parquet")
    parser.add_argument("--keep-pct", type=float, default=0.07)
    parser.add_argument("--direction", choices=["asc", "desc"], default="asc")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    for c in ("QMI", "RLI"):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    df["composite"] = 0.5 * df["QMI"] + 0.5 * df["RLI"]
    ascending = args.direction == "asc"
    df = df.sort_values("composite", ascending=ascending).reset_index(drop=True)

    keep_n = max(1, int(np.ceil(len(df) * float(args.keep_pct))))
    kept = df.head(keep_n).copy()
    kept["rank"] = np.arange(1, len(kept) + 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["date", "ticker", "company", "QMI", "RLI", "composite", "rank"] if c in kept.columns]
    kept[cols].to_parquet(out_path, index=False)
    print(f"Wrote {out_path} rows={len(kept)} keep_pct={args.keep_pct:.4f}")


if __name__ == "__main__":
    main()
