from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="ERP Sales Intelligence",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    [data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    [data-testid="stMetricValue"] {
        color: #f9fafb;
        font-size: 1.7rem;
        font-weight: 700;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #0f172a 100%);
        border: 1px solid #1f2937;
        padding: 28px 32px;
        border-radius: 22px;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3);
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.6;
    }

    .section-card {
        background-color: #0f172a;
        border: 1px solid #1f2937;
        padding: 20px;
        border-radius: 18px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #9ca3af;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# File Paths
# -----------------------------
CLEANED_SALES_PATH = Path("data/processed/cleaned_sales.csv")
MONTHLY_SALES_PATH = Path("data/processed/monthly_sales.csv")
FORECAST_RESULTS_PATH = Path("data/processed/forecast_results.csv")
SEGMENTED_CUSTOMERS_PATH = Path("data/processed/segmented_customers.csv")


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_data():
    sales = pd.read_csv(CLEANED_SALES_PATH)
    monthly = pd.read_csv(MONTHLY_SALES_PATH)
    forecast = pd.read_csv(FORECAST_RESULTS_PATH)
    segmented = pd.read_csv(SEGMENTED_CUSTOMERS_PATH)

    sales["invoice_date"] = pd.to_datetime(sales["invoice_date"])
    monthly["month"] = pd.to_datetime(monthly["month"])

    return sales, monthly, forecast, segmented


sales, monthly, forecast, segmented = load_data()


# -----------------------------
# Main Metrics
# -----------------------------
total_revenue = sales["revenue"].sum()
total_orders = sales["invoice_no"].nunique()
total_customers = sales["customer_id"].nunique()
predicted_revenue = forecast["predicted_revenue"].iloc[0]


# -----------------------------
# Hero Section
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1>ERP Sales Intelligence Dashboard</h1>
        <p class="hero-subtitle">
            An AI/ML-powered business intelligence prototype that turns raw transaction data
            into revenue forecasts, customer segments, and automated business recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Metric Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Revenue Analyzed", f"${total_revenue:,.2f}")
col2.metric("Orders Processed", f"{total_orders:,}")
col3.metric("Customers Segmented", f"{total_customers:,}")
col4.metric("Forecasted Revenue", f"${predicted_revenue:,.2f}")


st.markdown(
    """
    <div class="section-card">
        <h3>What this dashboard does</h3>
        <p class="small-muted">
            This project simulates an ERP analytics layer. It cleans retail transaction data,
            engineers monthly sales and customer RFM features, trains a forecasting model,
            segments customers with K-Means, and generates business recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Sales Forecast",
        "Customer Segments",
        "AI Business Report",
    ]
)


# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.subheader("Business Overview")

    col_left, col_right = st.columns(2)

    with col_left:
        monthly_fig = px.line(
            monthly,
            x="month",
            y="revenue",
            markers=True,
            title="Monthly Revenue Trend",
        )
        monthly_fig.update_layout(
            height=420,
            margin=dict(l=30, r=30, t=80, b=40),
            xaxis_title="Month",
            yaxis_title="Revenue",
        )
        st.plotly_chart(monthly_fig, use_container_width=True)

    with col_right:
        country_revenue = (
            sales.groupby("country")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        country_fig = px.bar(
            country_revenue,
            x="country",
            y="revenue",
            title="Top 10 Countries by Revenue",
        )
        country_fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title="Country",
            yaxis_title="Revenue",
        )
        st.plotly_chart(country_fig, use_container_width=True)

    st.subheader("Top Products by Revenue")

    top_products = (
        sales.groupby("description")
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            orders=("invoice_no", "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
        .reset_index()
    )

    top_products_fig = px.bar(
        top_products.sort_values("revenue", ascending=True),
        x="revenue",
        y="description",
        orientation="h",
        title="Top 10 Products by Revenue",
    )
    top_products_fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="",
        xaxis_title="Revenue",
    )
    st.plotly_chart(top_products_fig, use_container_width=True)

    with st.expander("View product table"):
        top_products_display = top_products.copy()

        top_products_display["revenue"] = top_products_display["revenue"].round(2)
        top_products_display["quantity"] = top_products_display["quantity"].astype(int)
        top_products_display["orders"] = top_products_display["orders"].astype(int)

        top_products_display = top_products_display.rename(
            columns={
                "description": "Product",
                "revenue": "Revenue",
                "quantity": "Quantity Sold",
                "orders": "Orders",
            }
        )

        st.dataframe(top_products_display, use_container_width=True)


# -----------------------------
# Tab 2: Sales Forecast
# -----------------------------
with tab2:
    st.subheader("Sales Forecasting Model")

    best_model = forecast["best_model"].iloc[0]
    model_mae = forecast["model_mae"].iloc[0]
    forecast_month = forecast["forecast_month"].iloc[0]

    col_a, col_b, col_c = st.columns(3)

    col_a.metric("Best Model", best_model)
    col_b.metric("Model MAE", f"${model_mae:,.2f}")
    col_c.metric("Forecast Month", forecast_month)

    forecast_fig = px.line(
        monthly,
        x="month",
        y="revenue",
        markers=True,
        title="Historical Monthly Revenue Used for Forecasting",
    )
    forecast_fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Month",
        yaxis_title="Revenue",
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    st.info(
        "Note: This dataset contains roughly one year of sales data. "
        "The forecasting model is a prototype demonstration of an ML pipeline, "
        "not a production-grade forecast."
    )

    st.markdown(
        f"""
        <div class="section-card">
            <h3>Forecast Result</h3>
            <p class="small-muted">
                The selected model predicts next-month revenue of approximately
                <strong>${predicted_revenue:,.2f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Tab 3: Customer Segments
# -----------------------------
with tab3:
    st.subheader("Customer Segmentation")

    segment_summary = (
        segmented.groupby("segment")
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

    segment_fig = px.bar(
        segment_summary,
        x="segment",
        y="total_revenue",
        title="Total Revenue by Customer Segment",
    )
    segment_fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Customer Segment",
        yaxis_title="Total Revenue",
    )
    st.plotly_chart(segment_fig, use_container_width=True)

    segment_summary_display = segment_summary.copy()
    segment_summary_display["avg_recency"] = segment_summary_display["avg_recency"].round(2)
    segment_summary_display["avg_frequency"] = segment_summary_display["avg_frequency"].round(2)
    segment_summary_display["avg_monetary"] = segment_summary_display["avg_monetary"].round(2)
    segment_summary_display["total_revenue"] = segment_summary_display["total_revenue"].round(2)

    segment_summary_display = segment_summary_display.rename(
        columns={
            "segment": "Segment",
            "customers": "Customers",
            "avg_recency": "Avg Recency",
            "avg_frequency": "Avg Frequency",
            "avg_monetary": "Avg Monetary",
            "total_revenue": "Total Revenue",
        }
    )

    st.dataframe(segment_summary_display, use_container_width=True)

    st.markdown(
        """
        <div class="section-card">
            <h3>Segment interpretation</h3>
            <p class="small-muted">
                <strong>High-Value Loyal Customers:</strong> small group, very frequent, very high revenue.<br>
                <strong>Recent Active Customers:</strong> currently engaged customers with recent purchases.<br>
                <strong>Occasional Low-Value Customers:</strong> lower purchase frequency and lower spending.<br>
                <strong>At-Risk Valuable Customers:</strong> valuable customers who have not purchased recently.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Tab 4: AI Business Report
# -----------------------------
with tab4:
    st.subheader("AI-Style Business Report")

    latest_revenue = monthly["revenue"].iloc[-1]
    previous_revenue = monthly["revenue"].iloc[-2]
    growth_rate = ((latest_revenue - previous_revenue) / previous_revenue) * 100

    top_product = (
        sales.groupby("description")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    top_country = (
        sales.groupby("country")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    highest_value_segment = segment_summary.iloc[0]["segment"]

    if growth_rate > 0:
        trend_sentence = (
            f"Revenue increased by {growth_rate:.2f}% compared to the previous month."
        )
    else:
        trend_sentence = (
            f"Revenue decreased by {abs(growth_rate):.2f}% compared to the previous month."
        )

    st.markdown(
        f"""
        <div class="section-card">
            <h3>Monthly Business Summary</h3>
            <p class="small-muted">
                {trend_sentence}
                <br><br>
                The top-selling product by revenue is <strong>{top_product}</strong>.
                <br><br>
                The strongest market by revenue is <strong>{top_country}</strong>.
                <br><br>
                The highest revenue customer segment is <strong>{highest_value_segment}</strong>.
                <br><br>
                The forecasting model predicts next-month revenue of approximately
                <strong>${predicted_revenue:,.2f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <h3>Recommended Actions</h3>
            <p class="small-muted">
                1. Protect high-value loyal customers with VIP offers or personalized outreach.<br>
                2. Re-engage at-risk valuable customers before they churn.<br>
                3. Prepare inventory around top-selling products before peak demand.<br>
                4. Monitor monthly revenue trends because the model is trained on limited historical data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Project limitations"):
    st.write(
        """
        This project uses approximately one year of transaction data, so the forecasting model
        is intended as a prototype demonstration rather than a production-ready forecasting system.
        With more historical data, the model could be improved using stronger time-series validation,
        seasonality features, and more advanced forecasting models.
        """
    )