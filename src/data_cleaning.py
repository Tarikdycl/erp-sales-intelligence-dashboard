"""
data_cleaning.py

Loads the raw Online Retail Excel dataset, cleans invalid rows,
creates useful sales features, and saves the cleaned dataset.

Input:
    data/raw/Online Retail.xlsx

Output:
    data/processed/cleaned_sales.csv
"""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/Online Retail.xlsx")
PROCESSED_DATA_DIR = Path("data/processed")
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_sales.csv"


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load the raw Excel dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            "Make sure the file is inside data/raw/ and named exactly 'Online Retail.xlsx'."
        )

    return pd.read_excel(file_path)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the sales dataset and create useful columns."""

    df = df.copy()

    # Standardize column names for easier coding.
    df = df.rename(
        columns={
            "InvoiceNo": "invoice_no",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "UnitPrice": "unit_price",
            "CustomerID": "customer_id",
            "Country": "country",
        }
    )

    # Remove rows that cannot be used for customer analysis.
    df = df.dropna(subset=["customer_id", "description"])

    # Remove cancelled invoices.
    df = df[~df["invoice_no"].astype(str).str.startswith("C")]

    # Remove returns, invalid quantities, and invalid prices.
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]

    # Convert data types.
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype(int).astype(str)

    # Create revenue.
    df["revenue"] = df["quantity"] * df["unit_price"]

    # Create date features.
    df["year"] = df["invoice_date"].dt.year
    df["month"] = df["invoice_date"].dt.month
    df["day"] = df["invoice_date"].dt.day
    df["day_of_week"] = df["invoice_date"].dt.day_name()
    df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)

    # Reorder columns.
    columns = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
        "revenue",
        "year",
        "month",
        "day",
        "day_of_week",
        "year_month",
    ]

    df = df[columns]

    return df


def save_cleaned_data(df: pd.DataFrame, file_path: Path) -> None:
    """Save cleaned data to CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def main() -> None:
    """Run the full cleaning pipeline."""
    print("Loading raw data...")
    raw_df = load_raw_data(RAW_DATA_PATH)

    print("Cleaning data...")
    cleaned_df = clean_sales_data(raw_df)

    print("Saving cleaned data...")
    save_cleaned_data(cleaned_df, CLEANED_DATA_PATH)

    print("\nCleaning complete.")
    print(f"Raw rows: {len(raw_df):,}")
    print(f"Cleaned rows: {len(cleaned_df):,}")
    print(f"Rows removed: {len(raw_df) - len(cleaned_df):,}")
    print(f"Saved to: {CLEANED_DATA_PATH}")

    print("\nCleaned columns:")
    print(cleaned_df.columns.tolist())

    print("\nCleaned data preview:")
    print(cleaned_df.head())


if __name__ == "__main__":
    main()