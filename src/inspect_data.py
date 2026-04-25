"""
inspect_data.py

This file inspects the raw Online Retail Excel dataset before cleaning.
It helps us understand the columns, data types, missing values, and sample rows.

Input:
    data/raw/Online Retail.xlsx
"""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/Online Retail.xlsx")


def main() -> None:
    """Inspect the raw dataset."""

    print("Loading raw Excel file...")
    df = pd.read_excel(RAW_DATA_PATH)

    print("\n==============================")
    print("BASIC INFO")
    print("==============================")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n==============================")
    print("COLUMN NAMES")
    print("==============================")
    print(df.columns.tolist())

    print("\n==============================")
    print("DATA TYPES")
    print("==============================")
    print(df.dtypes)

    print("\n==============================")
    print("FIRST 10 ROWS")
    print("==============================")
    print(df.head(10))

    print("\n==============================")
    print("MISSING VALUES")
    print("==============================")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n==============================")
    print("NUMERIC SUMMARY")
    print("==============================")
    print(df.describe())

    print("\n==============================")
    print("UNIQUE VALUES SAMPLE")
    print("==============================")
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        print(f"{column}: {unique_count:,} unique values")

    print("\n==============================")
    print("CANCELLED INVOICE CHECK")
    print("==============================")
    cancelled_count = df["InvoiceNo"].astype(str).str.startswith("C").sum()
    print(f"Cancelled invoices/rows: {cancelled_count:,}")

    print("\n==============================")
    print("NEGATIVE OR ZERO VALUES")
    print("==============================")
    print(f"Quantity <= 0 rows: {(df['Quantity'] <= 0).sum():,}")
    print(f"UnitPrice <= 0 rows: {(df['UnitPrice'] <= 0).sum():,}")


if __name__ == "__main__":
    main()