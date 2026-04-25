"""
segmentation.py

Uses RFM features to segment customers into business-friendly groups.

Input:
    data/processed/customer_rfm.csv

Output:
    data/processed/segmented_customers.csv

segmentation.py

Uses RFM features to segment customers into business-friendly groups.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


CUSTOMER_RFM_PATH = Path("data/processed/customer_rfm.csv")
SEGMENTED_CUSTOMERS_PATH = Path("data/processed/segmented_customers.csv")


FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "unique_products",
    "average_order_value",
]


def load_customer_rfm(file_path: Path) -> pd.DataFrame:
    """Load customer RFM feature table."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Run src/features.py first."
        )

    return pd.read_csv(file_path)


def segment_customers(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster customers using K-Means."""

    rfm = rfm.copy()

    X = rfm[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    return rfm


def create_cluster_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    """Create cluster-level summary statistics."""

    summary = (
        segmented.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            avg_unique_products=("unique_products", "mean"),
            avg_order_value=("average_order_value", "mean"),
        )
        .reset_index()
    )

    return summary


def assign_business_labels(segmented: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable labels to clusters.

    Logic:
    - Best customers: high frequency + high monetary + low recency
    - At-risk valuable customers: high monetary + high recency
    - Recent/active customers: low recency but lower monetary
    - Occasional/low-value customers: lower frequency/monetary
    """

    segmented = segmented.copy()
    summary = summary.copy()

    labels = {}

    # Lower recency is better. Higher frequency and monetary are better.
    summary["value_score"] = (
        summary["avg_frequency"].rank(ascending=True)
        + summary["avg_monetary"].rank(ascending=True)
        + summary["avg_recency"].rank(ascending=False)
    )

    best_cluster = summary.sort_values("value_score", ascending=False).iloc[0]["cluster"]
    labels[best_cluster] = "High-Value Loyal Customers"

    remaining = summary[~summary["cluster"].isin(labels.keys())].copy()

    at_risk_cluster = remaining.sort_values(
        ["avg_monetary", "avg_recency"],
        ascending=[False, False],
    ).iloc[0]["cluster"]
    labels[at_risk_cluster] = "At-Risk Valuable Customers"

    remaining = summary[~summary["cluster"].isin(labels.keys())].copy()

    recent_cluster = remaining.sort_values(
        "avg_recency",
        ascending=True,
    ).iloc[0]["cluster"]
    labels[recent_cluster] = "Recent Active Customers"

    remaining = summary[~summary["cluster"].isin(labels.keys())].copy()

    for cluster in remaining["cluster"]:
        labels[cluster] = "Occasional Low-Value Customers"

    segmented["segment"] = segmented["cluster"].map(labels)

    return segmented


def main() -> None:
    """Run customer segmentation pipeline."""
    print("Loading customer RFM data...")
    rfm = load_customer_rfm(CUSTOMER_RFM_PATH)

    print("Segmenting customers...")
    segmented = segment_customers(rfm)

    print("Creating cluster summary...")
    summary = create_cluster_summary(segmented)

    print("Assigning business labels...")
    segmented = assign_business_labels(segmented, summary)

    print("Saving segmented customers...")
    segmented.to_csv(SEGMENTED_CUSTOMERS_PATH, index=False)

    print("\nSegmentation complete.")
    print(f"Customers segmented: {len(segmented):,}")
    print(f"Saved to: {SEGMENTED_CUSTOMERS_PATH}")

    print("\nCluster summary:")
    final_summary = (
        segmented.groupby(["cluster", "segment"])
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    print(final_summary)


if __name__ == "__main__":
    main()