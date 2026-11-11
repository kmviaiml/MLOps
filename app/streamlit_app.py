import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import streamlit as st

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.db_manager import DatabaseManager
from src.models.predict import ChurnPredictor
from src.monitoring.drift_monitor import DriftMonitor

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Sales Analytics & Churn MLOps Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode / Glassmorphism Theme)
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Data Loader
@st.cache_resource
def get_db_manager():
    return DatabaseManager(db_path="data/warehouse/ecommerce_dw.db")

@st.cache_data(ttl=600)
def load_analytics_data():
    db = get_db_manager()
    try:
        df_sales = db.execute_query("SELECT * FROM fact_sales")
        df_retention = db.execute_query("SELECT * FROM fact_customer_retention")
        df_products = db.execute_query("SELECT * FROM dim_product")
        df_audit = db.execute_query("SELECT * FROM etl_extraction_log")
        df_rejections = db.execute_query("SELECT * FROM etl_rejected_records")
        return df_sales, df_retention, df_products, df_audit, df_rejections
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_sales, df_retention, df_products, df_audit, df_rejections = load_analytics_data()

# Header Section
st.markdown("""
<div class="header-box">
    <div class="header-title">E-Commerce Sales Analytics & Customer Churn MLOps</div>
    <div class="header-subtitle">Enterprise Data Warehouse, Star Schema Analytics, and Machine Learning Inference Console</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filter
st.sidebar.title("🎛️ Analytics Controls")
if not df_sales.empty:
    df_sales["invoice_date"] = pd.to_datetime(df_sales["invoice_date"])
    countries = ["All Countries"] + sorted(list(df_sales["country"].unique()))
    selected_country = st.sidebar.selectbox("Filter Country", countries)
    
    if selected_country != "All Countries":
        filtered_sales = df_sales[df_sales["country"] == selected_country]
        filtered_retention = df_retention[df_retention["country"] == selected_country] if (not df_retention.empty and "country" in df_retention.columns) else df_retention
    else:
        filtered_sales = df_sales
        filtered_retention = df_retention
else:
    filtered_sales = df_sales
    filtered_retention = df_retention
    selected_country = "All Countries"

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Pipeline Status**: Active ✅  
**Database**: SQLite Star Schema  
**ML Model**: XGBoost / RandomForest  
**API Status**: Healthy  
""")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Revenue Trends", 
    "📦 Product Analytics", 
    "👥 Customer RFM", 
    "🌍 Country Insights", 
    "🔮 Churn Prediction & MLOps"
])

# -----------------------------------------------------------------------------
# TAB 1: REVENUE TRENDS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📊 Executive Revenue & Sales Volume Overview")
    if filtered_sales.empty:
        st.warning("No sales data available. Run `python scripts/run_pipeline.py` first.")
    else:
        total_rev = filtered_sales["total_revenue"].sum()
        total_orders = filtered_sales["invoice_no"].nunique()
        total_qty = filtered_sales["quantity"].sum()
        avg_order_val = total_rev / total_orders if total_orders > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-val">£{total_rev:,.2f}</div><div class="metric-lbl">Total Gross Revenue</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val">{total_orders:,}</div><div class="metric-lbl">Total Orders</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-val">£{avg_order_val:,.2f}</div><div class="metric-lbl">Average Order Value</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-val">{total_qty:,}</div><div class="metric-lbl">Units Sold</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Monthly Revenue Trend Chart
        filtered_sales["month_year"] = filtered_sales["invoice_date"].dt.to_period("M").astype(str)
        monthly_df = filtered_sales.groupby("month_year").agg(
            revenue=("total_revenue", "sum"),
            orders=("invoice_no", "nunique")
        ).reset_index()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_rev = px.area(
                monthly_df, x="month_year", y="revenue",
                title="Monthly Revenue Trajectory (£)",
                color_discrete_sequence=["#38bdf8"],
                template="plotly_dark"
            )
            fig_rev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rev, use_container_width=True)

        with col_b:
            fig_orders = px.bar(
                monthly_df, x="month_year", y="orders",
                title="Monthly Order Count",
                color_discrete_sequence=["#818cf8"],
                template="plotly_dark"
            )
            fig_orders.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_orders, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: PRODUCT ANALYTICS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📦 Top Performing Products & Categories")
    if filtered_sales.empty:
        st.warning("No sales data available.")
    else:
        prod_perf = filtered_sales.groupby("stock_code").agg(
            revenue=("total_revenue", "sum"),
            units_sold=("quantity", "sum")
        ).reset_index()
        
        if not df_products.empty:
            prod_perf = pd.merge(prod_perf, df_products, on="stock_code", how="left")
        else:
            prod_perf["description"] = prod_perf["stock_code"]
            prod_perf["category"] = "General"

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            top_rev_products = prod_perf.sort_values("revenue", ascending=False).head(10)
            fig_top_rev = px.bar(
                top_rev_products, x="revenue", y="description", orientation="h",
                title="Top 10 Products by Gross Revenue (£)",
                color="revenue", color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig_top_rev.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_top_rev, use_container_width=True)

        with col_p2:
            cat_summary = prod_perf.groupby("category")["revenue"].sum().reset_index()
            fig_cat = px.pie(
                cat_summary, values="revenue", names="category",
                title="Revenue Share by Product Category",
                hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_dark"
            )
            fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cat, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: CUSTOMER RFM SEGMENTATION
# -----------------------------------------------------------------------------
with tab3:
    st.subheader(f"👥 Customer RFM (Recency, Frequency, Monetary) Analysis — {selected_country}")
    if filtered_retention.empty:
        st.warning(f"No customer retention mart data available for {selected_country}.")
    else:
        st.markdown(f"Customer segmentation matrix based on historical transaction frequency and recency for **{selected_country}**.")
        
        # Ensure avg_order_value column exists cleanly
        if "avg_order_value" not in filtered_retention.columns:
            if "avg_order_value_x" in filtered_retention.columns:
                filtered_retention["avg_order_value"] = filtered_retention["avg_order_value_x"]
            elif "avg_order_value_y" in filtered_retention.columns:
                filtered_retention["avg_order_value"] = filtered_retention["avg_order_value_y"]
            else:
                filtered_retention["avg_order_value"] = 0.0

        rfm_counts = filtered_retention["rfm_segment"].value_counts().reset_index()
        rfm_counts.columns = ["rfm_segment", "customer_count"]

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            fig_rfm_pie = px.pie(
                rfm_counts, values="customer_count", names="rfm_segment",
                title=f"Customer Segment Distribution ({selected_country})",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                template="plotly_dark"
            )
            fig_rfm_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rfm_pie, use_container_width=True)

        with col_r2:
            fig_rfm_scatter = px.scatter(
                filtered_retention, x="recency_days", y="monetary_value",
                color="rfm_segment", size="frequency_orders",
                title=f"Recency vs Monetary Value ({selected_country}) — Bubble size = Order Frequency",
                labels={"recency_days": "Recency (Days)", "monetary_value": "Monetary Value (£)"},
                hover_data=["customer_id", "avg_order_value"],
                template="plotly_dark"
            )
            fig_rfm_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rfm_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: COUNTRY INSIGHTS
# -----------------------------------------------------------------------------
with tab4:
    st.subheader(f"🌍 Country Sales Insights — {selected_country}")
    if df_sales.empty:
        st.warning("No sales data available.")
    else:
        global_country_df = df_sales.groupby("country").agg(
            revenue=("total_revenue", "sum"),
            orders=("invoice_no", "nunique"),
            customers=("customer_id", "nunique")
        ).reset_index().sort_values("revenue", ascending=False)

        if selected_country != "All Countries":
            c_sales = df_sales[df_sales["country"] == selected_country]
            c_rev = c_sales["total_revenue"].sum()
            c_orders = c_sales["invoice_no"].nunique()
            c_cust = c_sales["customer_id"].nunique()
            c_aov = c_rev / c_orders if c_orders > 0 else 0.0

            # Single Country KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="metric-card"><div class="metric-val">£{c_rev:,.2f}</div><div class="metric-lbl">{selected_country} Revenue</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-card"><div class="metric-val">{c_orders:,}</div><div class="metric-lbl">Total Orders</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-card"><div class="metric-val">{c_cust:,}</div><div class="metric-lbl">Unique Customers</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-card"><div class="metric-val">£{c_aov:,.2f}</div><div class="metric-lbl">Avg Order Value</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Monthly Trend for Selected Country
            c_sales["month_year"] = c_sales["invoice_date"].dt.to_period("M").astype(str)
            c_monthly = c_sales.groupby("month_year").agg(revenue=("total_revenue", "sum")).reset_index()

            fig_c_trend = px.line(
                c_monthly, x="month_year", y="revenue",
                title=f"Monthly Revenue Growth in {selected_country} (£)",
                markers=True, color_discrete_sequence=["#38bdf8"],
                template="plotly_dark"
            )
            fig_c_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_c_trend, use_container_width=True)

        # Global Top 15 Comparison Chart with Highlight
        global_country_df["highlight"] = global_country_df["country"].apply(
            lambda x: "Selected Country" if x == selected_country else "Other Markets"
        )
        fig_country = px.bar(
            global_country_df.head(15), x="country", y="revenue",
            title=f"Global Country Revenue Comparison (Highlighted: {selected_country})",
            color="highlight" if selected_country != "All Countries" else "revenue",
            color_discrete_map={"Selected Country": "#38bdf8", "Other Markets": "#475569"},
            color_continuous_scale="Magma",
            template="plotly_dark"
        )
        fig_country.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_country, use_container_width=True)
        
        st.dataframe(global_country_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: CHURN PREDICTION & MLOPS CONSOLE
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🔮 Predictive Churn Simulator & MLOps Console")
    
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.markdown("### 🧪 Live Churn Risk Simulator")
        with st.form("churn_sim_form"):
            sim_recency = st.slider("Recency (Days since last purchase)", 1, 365, 45)
            sim_frequency = st.slider("Frequency (Total orders)", 1, 100, 12)
            sim_monetary = st.number_input("Monetary Value (£)", 10.0, 50000.0, 1250.0)
            sim_aov = sim_monetary / sim_frequency
            sim_tenure = st.slider("Customer Tenure (Months)", 1, 120, 24)
            sim_tickets = st.slider("Support Tickets Count", 0, 50, 3)
            sim_sub = st.selectbox("Subscription Tier", ["Basic", "Silver", "Gold", "Platinum"])
            sim_gender = st.selectbox("Gender", ["Female", "Male"])
            sim_city = st.selectbox("City", ["London", "Manchester", "Birmingham", "Delhi", "Mumbai", "New York"])
            
            submit_sim = st.form_submit_button("⚡ Predict Churn Risk")

        if submit_sim:
            predictor = ChurnPredictor(artifacts_dir="models")
            sim_input = {
                "customer_id": 99999,
                "recency_days": sim_recency,
                "frequency_orders": sim_frequency,
                "monetary_value": sim_monetary,
                "avg_order_value": sim_aov,
                "total_items_bought": sim_frequency * 5,
                "cancellation_count": 0,
                "tenure_months": sim_tenure,
                "age": 35,
                "support_tickets": sim_tickets,
                "gender": sim_gender,
                "city": sim_city,
                "subscription_type": sim_sub,
                "rfm_segment": "Potential Loyalists"
            }
            res = predictor.predict_single(sim_input)
            
            prob_pct = res["churn_probability"] * 100
            risk = res["risk_level"]
            color = "#ef4444" if risk == "High Risk" else "#f59e0b" if risk == "Medium Risk" else "#10b981"
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); padding: 20px; border-radius: 12px; border: 2px solid {color}; text-align: center;">
                <h2 style="color: {color}; margin: 0;">Predicted Churn Probability: {prob_pct:.1f}%</h2>
                <h3 style="color: #f8fafc; margin-top: 8px;">Risk Level: {risk}</h3>
                <p style="color: #94a3b8;">Model Version: {res['model_version']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("### 📈 MLOps Model Registry & Performance Metrics")
        meta_path = "models/model_metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            best_name = meta.get("best_model_name", "RandomForest")
            all_metrics = meta.get("all_metrics", {})
            best_metrics = all_metrics.get(best_name, {})

            # Key Metric Cards for Active Model
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="metric-val">{best_name}</div><div class="metric-lbl">Active Model</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-val">{best_metrics.get("f1_score", 1.0):.2f}</div><div class="metric-lbl">F1-Score</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-val">{best_metrics.get("roc_auc", 1.0):.2f}</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Formatted Table of All Evaluated Models
            if all_metrics:
                df_metrics = pd.DataFrame(all_metrics).T.reset_index()
                df_metrics.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
                st.dataframe(df_metrics, use_container_width=True)
        else:
            st.info("No model metadata found. Execute `python scripts/run_pipeline.py` to train models.")

        st.markdown("---")
        st.markdown("### 🛡️ Feature Drift & Data Quality Status")
        if not df_retention.empty:
            monitor = DriftMonitor()
            half = len(df_retention) // 2
            cols = ["recency_days", "frequency_orders", "monetary_value", "avg_order_value", "tenure_months", "support_tickets"]
            drift_report = monitor.analyze_dataset_drift(df_retention.iloc[:half], df_retention.iloc[half:], cols)
            
            st.success(f"Overall Drift Status: **{drift_report['overall_drift_status']}**")
            st.write(f"Monitored Features: {drift_report['total_features_monitored']} | Drifted: {drift_report['drifted_features_count']}")
