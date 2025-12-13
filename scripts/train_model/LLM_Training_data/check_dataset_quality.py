# check_dataset_quality.py
# -----------------------------------------------------------
# Quick validation script for LLM_Training_data_with_response.parquet
# Writes all console output into checked_data.txt
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from contextlib import redirect_stdout

DATA_PATH = "LLM_Training_data_with_response.parquet"
OUTPUT_LOG = "checked_data.txt"


def main():
    print(f"[INFO] Loading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"[INFO] Loaded {len(df):,} rows and {len(df.columns)} columns.\n")

    # ==============================================================
    # 1) BASIC STRUCTURE
    # ==============================================================
    print("=== BASIC INFO ===")
    print(df.head(5))
    print("\nDtypes:")
    print(df.dtypes)

    # ==============================================================
    # 2) NaN ratios
    # ==============================================================
    print("\n=== NaN RATIO PER COLUMN ===")
    nan_ratio = df.isna().mean().sort_values(ascending=False)
    print(nan_ratio)

    # ==============================================================
    # 3) Numeric column stats
    # ==============================================================
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n[INFO] Numeric columns: {num_cols}\n")

    print("=== BASIC STATS FOR NUMERIC COLUMNS ===")
    with pd.option_context("display.max_rows", 200,
                           "display.float_format", "{:,.6g}".format):
        print(df[num_cols].describe().T)

    # ==============================================================
    # 4) Detect all-zero or constant columns
    # ==============================================================
    zero_cols = []
    const_cols = []

    for col in num_cols:
        non_na = df[col].dropna()
        if len(non_na) == 0:
            continue
        if (non_na == 0).all():
            zero_cols.append(col)
        if df[col].nunique(dropna=False) == 1:
            const_cols.append(col)

    print("\n=== ALL-ZERO COLUMNS ===")
    print(zero_cols if zero_cols else "None")

    print("\n=== CONSTANT COLUMNS (only one unique value) ===")
    print(const_cols if const_cols else "None")

    # ==============================================================
    # 5) Check for suspiciously large values
    # ==============================================================
    THRESH = 1_000_000  # flag values above 1 million

    suspect = {}

    for col in num_cols:
        col_min = df[col].min(skipna=True)
        col_max = df[col].max(skipna=True)

        if abs(col_min) > THRESH or abs(col_max) > THRESH:
            suspect[col] = (col_min, col_max)

    print(f"\n=== COLUMNS WITH VALUES ABOVE ±{THRESH:,} ===")
    if suspect:
        for c, (mn, mx) in suspect.items():
            print(f"{c}: min={mn:,}, max={mx:,}")
    else:
        print("None")

    # ==============================================================
    # 6) Show outliers for suspicious columns
    # ==============================================================
    def show_outliers(col, threshold=THRESH):
        print("\n---------------------------------------")
        print(f"OUTLIERS for {col} (> {threshold:,})")
        print("---------------------------------------")

        mask = df[col].abs() > threshold
        if not mask.any():
            print("No outliers.")
            return

        cols_to_show = [col]
        for extra in ["Ticker", "Date"]:
            if extra in df.columns:
                cols_to_show.insert(0, extra)

        print(df.loc[mask, cols_to_show].head(10))

    if suspect:
        print("\n=== SHOWING SAMPLE OUTLIERS ===")
        for col in suspect:
            show_outliers(col)
    else:
        print("\n[INFO] No suspicious values detected.")

    print("\n[COMPLETE] Dataset quality check finished.")


if __name__ == "__main__":
    # Everything printed in main() will go into checked_data.txt
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            main()
