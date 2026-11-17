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
            A machine-learning business analytics prototype that turns raw transaction data
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
        "Business Insights & Rules",
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

    with st.expander("Forecast methodology and limitations"):
        st.write(
            "The model was evaluated on only ten monthly feature rows. "
            "Some inputs (such as same-month orders and average order value) are not "
            "known at the time a genuine next-month forecast would be made. "
            "The displayed figure is therefore a pipeline demonstration, not a "
            "validated forward-looking business forecast."
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

    st.subheader("How to read these segments")
    st.caption(
        "K-Means groups customers using standardized purchasing features. "
        "The four business labels are assigned after clustering using summary statistics; "
        "they are interpretations, not verified churn or loyalty predictions."
    )
    segment_explanations = {
        "High-Value Loyal Customers": "Higher historical purchase frequency and spending, with recent purchases.",
        "Recent Active Customers": "Relatively recent purchasers with more typical spending patterns.",
        "Occasional Low-Value Customers": "Lower frequency and historical spending within this dataset.",
        "At-Risk Valuable Customers": "High historical value but longer time since the last purchase; not a validated churn prediction.",
    }
    for _, segment_row in segment_summary.iterrows():
        st.markdown(f"**{segment_row['segment']}**")
        st.write(
            f"{segment_explanations.get(segment_row['segment'], 'Inspect this group using its summary statistics.')} "
            f"**{int(segment_row['customers']):,} customers**, "
            f"**${segment_row['total_revenue']:,.0f}** historical revenue, "
            f"average recency **{segment_row['avg_recency']:.0f} days**."
        )


# -----------------------------
# Tab 4: Business Insights & Rules
# -----------------------------
with tab4:
    st.subheader("Business Insights & Decision Rules")
    st.caption(
        "Transparent, deterministic analytics based on historical transactions and K-Means segments. "
        "No language model or generative AI is used in this report."
    )

    latest_revenue = monthly["revenue"].iloc[-1]
    previous_revenue = monthly["revenue"].iloc[-2]
    growth_rate = (
        (latest_revenue - previous_revenue) / previous_revenue * 100
        if previous_revenue != 0
        else None
    )

    top_product_revenue = (
        sales.groupby("description")["revenue"].sum().sort_values(ascending=False)
    )
    top_country_revenue = (
        sales.groupby("country")["revenue"].sum().sort_values(ascending=False)
    )
    top_product = top_product_revenue.index[0]
    top_country = top_country_revenue.index[0]
    top_product_amount = top_product_revenue.iloc[0]
    top_country_amount = top_country_revenue.iloc[0]

    if growth_rate is not None:
        st.metric("Latest complete month vs prior month", f"{growth_rate:+.1f}%")
    else:
        st.write("Month-over-month change is unavailable because the prior month's revenue is zero.")

    c1, c2 = st.columns(2)
    c1.metric("Top product by historical revenue", f"${top_product_amount:,.0f}")
    c1.caption(top_product)
    c2.metric("Top country by historical revenue", f"${top_country_amount:,.0f}")
    c2.caption(top_country)

    st.markdown("#### Evidence-linked action ideas")
    st.caption(
        "These are illustrative business rules derived from the displayed data, "
        "not tested recommendations or causal claims."
    )

    high_value = segment_summary.loc[
        segment_summary["segment"] == "High-Value Loyal Customers"
    ]
    if not high_value.empty:
        group = high_value.iloc[0]
        st.markdown("**Consider a high-value customer outreach review**")
        st.write(
            f"Evidence: {int(group['customers']):,} customers account for "
            f"${group['total_revenue']:,.0f} of historical revenue. "
            "Possible action: review VIP service or retention outreach; test impact before scaling."
        )

    at_risk = segment_summary.loc[
        segment_summary["segment"] == "At-Risk Valuable Customers"
    ]
    if not at_risk.empty:
        group = at_risk.iloc[0]
        st.markdown("**Review historically valuable but less recent customers**")
        st.write(
            f"Evidence: {int(group['customers']):,} customers, "
            f"${group['total_revenue']:,.0f} historical revenue, "
            f"average recency {group['avg_recency']:.0f} days. "
            "Possible action: investigate whether outreach is warranted. "
            "This segment does not establish churn probability."
        )

    st.markdown("**Investigate the highest-revenue product**")
    st.write(
        f"Evidence: {top_product} generated ${top_product_amount:,.0f} in historical sales. "
        "Possible action: inspect its demand and inventory history; revenue rank alone "
        "does not predict future demand."
    )

    st.markdown("**Validate forecasting before operational use**")
    st.write(
        f"Evidence: the selected model's holdout MAE is ${forecast['model_mae'].iloc[0]:,.0f}. "
        "Only ten monthly feature rows are available, and some same-month inputs "
        "would not be known in advance. Use a true forward-looking feature set "
        "and more out-of-time tests before treating the forecast as actionable."
    )

    with st.expander("How these rules are produced"):
        st.markdown(
            """
            - **Revenue and top product/country:** sums of `revenue = quantity × unit_price`
              from cleaned transactions.
            - **Customer groups:** K-Means on standardized RFM and purchasing features;
              names are assigned heuristically after clustering.
            - **Action ideas:** fixed decision rules attached to the observed metrics above.
              No LLM, uplift model, or automated optimization is involved.
            - **Forecast:** experimental regression comparison, not a validated production forecast.
            """
        )

with st.expander("Project limitations"):
    st.write(
        "This is a prototype using about one year of transactions and ten monthly "
        "modeling rows. The forecasting pipeline uses same-month features that would "
        "not be available at a real forecast date; it needs a properly lagged feature "
        "set, out-of-time validation, and more history. K-Means segment names are "
        "heuristic interpretations rather than verified customer churn labels. "
        "Recommendations are transparent rules, not AI-generated or impact-tested."
    )
