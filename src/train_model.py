"""
train_model.py

Trains simple forecasting models on monthly sales features.

Input:
    data/processed/monthly_sales.csv

Output:
    data/processed/forecast_results.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


MONTHLY_SALES_PATH = Path("data/processed/monthly_sales.csv")
FORECAST_RESULTS_PATH = Path("data/processed/forecast_results.csv")


FEATURE_COLUMNS = [
    "month_index",
    "quantity",
    "orders",
    "unique_customers",
    "unique_products",
    "average_order_value",
    "prev_month_revenue",
    "rolling_3_month_revenue",
    "rolling_3_month_quantity",
]

TARGET_COLUMN = "revenue"


def load_monthly_sales(file_path: Path) -> pd.DataFrame:
    """Load monthly sales feature table."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Run src/features.py first."
        )

    df = pd.read_csv(file_path)
    df["month"] = pd.to_datetime(df["month"])

    return df


def train_and_evaluate_models(monthly: pd.DataFrame) -> tuple:
    """Train Linear Regression and Random Forest models, then compare MAE."""

    X = monthly[FEATURE_COLUMNS]
    y = monthly[TARGET_COLUMN]

    # Time-based split: train on earlier months, test on later months.
    split_index = int(len(monthly) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    random_forest_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    random_forest_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_test)
    rf_predictions = random_forest_model.predict(X_test)

    linear_mae = mean_absolute_error(y_test, linear_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)

    print("\nModel Performance")
    print("-----------------")
    print(f"Linear Regression MAE: ${linear_mae:,.2f}")
    print(f"Random Forest MAE:     ${rf_mae:,.2f}")

    if rf_mae <= linear_mae:
        best_model_name = "Random Forest"
        best_model = random_forest_model
        best_mae = rf_mae
    else:
        best_model_name = "Linear Regression"
        best_model = linear_model
        best_mae = linear_mae

    return best_model, best_model_name, best_mae, X_test, y_test


def create_next_month_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Create a one-row feature table for next month prediction."""

    last_row = monthly.iloc[-1]

    next_month_features = pd.DataFrame(
        [
            {
                "month_index": last_row["month_index"] + 1,
                "quantity": last_row["quantity"],
                "orders": last_row["orders"],
                "unique_customers": last_row["unique_customers"],
                "unique_products": last_row["unique_products"],
                "average_order_value": last_row["average_order_value"],
                "prev_month_revenue": last_row["revenue"],
                "rolling_3_month_revenue": monthly["revenue"].tail(3).mean(),
                "rolling_3_month_quantity": monthly["quantity"].tail(3).mean(),
            }
        ]
    )

    return next_month_features


def save_forecast_result(
    monthly: pd.DataFrame,
    model_name: str,
    model_mae: float,
    prediction: float,
) -> None:
    """Save next month forecast results to CSV."""

    last_month = monthly["month"].max()
    next_month = last_month + pd.offsets.MonthEnd(1)

    results = pd.DataFrame(
        [
            {
                "last_month_in_data": last_month.date(),
                "forecast_month": next_month.date(),
                "best_model": model_name,
                "model_mae": model_mae,
                "predicted_revenue": prediction,
            }
        ]
    )

    results.to_csv(FORECAST_RESULTS_PATH, index=False)


def main() -> None:
    """Run model training and prediction."""
    print("Loading monthly sales data...")
    monthly = load_monthly_sales(MONTHLY_SALES_PATH)

    print(f"Monthly rows available: {len(monthly)}")

    print("Training models...")
    best_model, best_model_name, best_mae, _, _ = train_and_evaluate_models(monthly)

    print(f"\nBest model: {best_model_name}")
    print(f"Best model MAE: ${best_mae:,.2f}")

    print("\nCreating next month forecast...")
    next_month_features = create_next_month_features(monthly)
    next_month_prediction = best_model.predict(next_month_features)[0]

    print(f"Predicted next month revenue: ${next_month_prediction:,.2f}")

    save_forecast_result(
        monthly=monthly,
        model_name=best_model_name,
        model_mae=best_mae,
        prediction=next_month_prediction,
    )

    print(f"\nForecast saved to: {FORECAST_RESULTS_PATH}")


if __name__ == "__main__":
    main()