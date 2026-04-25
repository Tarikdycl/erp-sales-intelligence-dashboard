"""
features.py

Creates ML-ready feature tables from the cleaned sales dataset.

Input:
    data/processed/cleaned_sales.csv

Outputs:
    data/processed/monthly_sales.csv
    data/processed/customer_rfm.csv
"""

from pathlib import Path

import pandas as pd


CLEANED_DATA_PATH = Path("data/processed/cleaned_sales.csv")
MONTHLY_SALES_PATH = Path("data/processed/monthly_sales.csv")
CUSTOMER_RFM_PATH = Path("data/processed/customer_rfm.csv")


def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    """Load cleaned sales data."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Run src/data_cleaning.py first."
        )

    df = pd.read_csv(file_path)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    return df


def create_monthly_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly sales features for forecasting."""

    monthly = (
        df.set_index("invoice_date")
        .resample("M")
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            orders=("invoice_no", "nunique"),
            unique_customers=("customer_id", "nunique"),
            unique_products=("stock_code", "nunique"),
        )
        .reset_index()
    )

    monthly = monthly.rename(columns={"invoice_date": "month"})
    # Remove the final month if the raw data does not contain a full month.
    last_date = df["invoice_date"].max()
    if last_date.day < 28:
        monthly = monthly[monthly["month"] < last_date.to_period("M").to_timestamp("M")]
    
    monthly["month_index"] = range(len(monthly))
    monthly["average_order_value"] = monthly["revenue"] / monthly["orders"]

    # Lag and rolling features
    monthly["prev_month_revenue"] = monthly["revenue"].shift(1)
    monthly["rolling_3_month_revenue"] = monthly["revenue"].rolling(3).mean()
    monthly["rolling_3_month_quantity"] = monthly["quantity"].rolling(3).mean()

    # Drop first rows where lag/rolling values are missing
    monthly = monthly.dropna().reset_index(drop=True)

    return monthly


def create_customer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create RFM customer features for segmentation."""

    snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=("invoice_date", lambda dates: (snapshot_date - dates.max()).days),
            frequency=("invoice_no", "nunique"),
            monetary=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            unique_products=("stock_code", "nunique"),
        )
        .reset_index()
    )

    rfm["average_order_value"] = rfm["monetary"] / rfm["frequency"]

    return rfm


def save_features(monthly: pd.DataFrame, rfm: pd.DataFrame) -> None:
    """Save feature tables."""
    monthly.to_csv(MONTHLY_SALES_PATH, index=False)
    rfm.to_csv(CUSTOMER_RFM_PATH, index=False)


def main() -> None:
    """Run feature engineering pipeline."""
    print("Loading cleaned sales data...")
    df = load_cleaned_data(CLEANED_DATA_PATH)

    print("Creating monthly sales features...")
    monthly = create_monthly_sales_features(df)

    print("Creating customer RFM features...")
    rfm = create_customer_rfm_features(df)

    print("Saving feature tables...")
    save_features(monthly, rfm)

    print("\nFeature engineering complete.")
    print(f"Monthly sales rows: {len(monthly):,}")
    print(f"Customer RFM rows: {len(rfm):,}")
    print(f"Saved monthly sales to: {MONTHLY_SALES_PATH}")
    print(f"Saved customer RFM to: {CUSTOMER_RFM_PATH}")

    print("\nMonthly sales preview:")
    print(monthly.head())

    print("\nCustomer RFM preview:")
    print(rfm.head())


if __name__ == "__main__":
    main()