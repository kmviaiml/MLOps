from datetime import datetime
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("etl_transformer")

class StarSchemaTransformer:
    """Transforms validated raw data into dimensional model tables and analytical data marts."""
    
    def __init__(self, inactivity_threshold_days: int = 90):
        self.inactivity_threshold_days = inactivity_threshold_days

    def categorize_product(self, description: str) -> str:
        """Infer high-level product category based on description keywords."""
        if not isinstance(description, str):
            return "General Retail"
        desc = description.upper()
        if any(w in desc for w in ["MUG", "BOWL", "PLATE", "CUP", "GLASS", "KITCHEN", "TEASPOON", "BOTTLE"]):
            return "Kitchenware & Dining"
        elif any(w in desc for w in ["HEART", "DECORATION", "LIGHT", "LANTERN", "FRAME", "HANGING", "WALL"]):
            return "Home Decor & Lighting"
        elif any(w in desc for w in ["BAG", "WALLET", "PURSE", "CASES", "BOX", "HANGER", "TOTE"]):
            return "Bags & Storage"
        elif any(w in desc for w in ["TOY", "DOLL", "GAME", "JIGSAW", "PLAYHOUSE", "BLOCK", "CRAFT"]):
            return "Toys & Crafts"
        elif any(w in desc for w in ["WARMER", "KNITTED", "FLAG", "CHRISTMAS", "VINTAGE", "POLKA"]):
            return "Apparel & Seasonal"
        else:
            return "Gifts & Novelties"

    def assign_rfm_segment(self, r: int, f: int, m: float) -> str:
        """Categorizes customers into RFM segments based on Recency, Frequency, Monetary metrics."""
        if r <= 30 and f >= 5 and m >= 500:
            return "Champions"
        elif r <= 60 and f >= 3 and m >= 250:
            return "Loyal Customers"
        elif r <= 60 and f >= 1:
            return "Potential Loyalists"
        elif r <= 90 and f >= 2:
            return "At Risk"
        elif r <= 120:
            return "About To Sleep"
        elif r <= 180:
            return "Hibernating"
        else:
            return "Lost Customers"

    def transform(self, df_retail_valid: pd.DataFrame, df_churn_master: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Runs transformation pipeline producing all dimension and fact tables."""
        logger.info("Starting Star Schema transformation pipeline...")
        df = df_retail_valid.copy()

        # 1. Date Dimension (dim_date)
        df["full_date"] = df["InvoiceDate"].dt.date
        df["date_key"] = df["InvoiceDate"].dt.strftime("%Y%m%d").astype(int)
        
        unique_dates = df[["full_date", "date_key", "InvoiceDate"]].drop_duplicates("date_key").copy()
        dim_date = pd.DataFrame({
            "date_key": unique_dates["date_key"],
            "full_date": unique_dates["full_date"],
            "day": unique_dates["InvoiceDate"].dt.day,
            "month": unique_dates["InvoiceDate"].dt.month,
            "month_name": unique_dates["InvoiceDate"].dt.strftime("%B"),
            "quarter": unique_dates["InvoiceDate"].dt.quarter,
            "year": unique_dates["InvoiceDate"].dt.year,
            "day_of_week": unique_dates["InvoiceDate"].dt.strftime("%A"),
            "is_weekend": unique_dates["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)
        }).sort_values("date_key").reset_index(drop=True)
        
        logger.info(f"Generated dim_date table with {len(dim_date)} rows.")

        # 2. Product Dimension (dim_product)
        products = df.groupby("StockCode").agg({
            "Description": "first",
            "UnitPrice": "median"
        }).reset_index()
        
        products["Description"] = products["Description"].fillna("UNKNOWN ITEM").astype(str).str.strip()
        products["category"] = products["Description"].apply(self.categorize_product)
        products["is_active"] = 1
        
        dim_product = pd.DataFrame({
            "stock_code": products["StockCode"],
            "description": products["Description"],
            "category": products["category"],
            "unit_price": products["UnitPrice"],
            "is_active": products["is_active"]
        }).drop_duplicates("stock_code").reset_index(drop=True)
        
        logger.info(f"Generated dim_product table with {len(dim_product)} rows.")

        # 3. Customer Dimension (dim_customer)
        retail_customers = df.dropna(subset=["CustomerID"])[["CustomerID", "Country"]].drop_duplicates("CustomerID")
        retail_customers["customer_id"] = retail_customers["CustomerID"].astype(float)
        
        if not df_churn_master.empty:
            churn_cust = df_churn_master.copy()
            churn_cust["customer_id"] = churn_cust["customer_id"].astype(float)
            
            # Merge retail customers with churn master details
            dim_customer = pd.merge(retail_customers, churn_cust, on="customer_id", how="outer")
        else:
            dim_customer = retail_customers.copy()
            dim_customer["age"] = np.nan
            dim_customer["gender"] = "Unknown"
            dim_customer["city"] = "Unknown"
            dim_customer["tenure_months"] = np.nan
            dim_customer["subscription_type"] = "Standard"
            dim_customer["support_tickets"] = 0
            
        if "country" not in dim_customer.columns:
            if "Country" in dim_customer.columns:
                dim_customer["country"] = dim_customer["Country"]
            else:
                dim_customer["country"] = "United Kingdom"
        else:
            if "Country" in dim_customer.columns:
                dim_customer["country"] = dim_customer["country"].fillna(dim_customer["Country"])
                
        dim_customer["country"] = dim_customer["country"].fillna("United Kingdom")
        dim_customer = dim_customer.drop(columns=["Country", "CustomerID"], errors="ignore")
        dim_customer = dim_customer.drop_duplicates("customer_id").dropna(subset=["customer_id"]).reset_index(drop=True)
        logger.info(f"Generated dim_customer table with {len(dim_customer)} rows.")

        # 4. Geography Dimension (dim_geography)
        countries = df["Country"].dropna().unique()
        dim_geography = pd.DataFrame({
            "geography_id": range(1, len(countries) + 1),
            "country": countries,
            "city": "Capital/Major City",
            "region": "International"
        })
        dim_geography.loc[dim_geography["country"] == "United Kingdom", "region"] = "Domestic"
        logger.info(f"Generated dim_geography table with {len(dim_geography)} rows.")

        # 5. Sales Fact Table (fact_sales)
        df["total_revenue"] = df["Quantity"] * df["UnitPrice"]
        
        fact_sales = pd.DataFrame({
            "invoice_no": df["InvoiceNo"],
            "stock_code": df["StockCode"],
            "customer_id": df["CustomerID"],
            "date_key": df["date_key"],
            "invoice_date": df["InvoiceDate"],
            "quantity": df["Quantity"],
            "unit_price": df["UnitPrice"],
            "total_revenue": df["total_revenue"],
            "country": df["Country"],
            "is_cancelled": df["is_cancelled"]
        }).reset_index(drop=True)
        
        logger.info(f"Generated fact_sales table with {len(fact_sales)} rows.")

        # 6. Customer Retention & RFM Analytical Fact Table (fact_customer_retention)
        df_valid_cust = df.dropna(subset=["CustomerID"]).copy()
        
        # Reference date for Recency calculation (max dataset date + 1 day)
        max_date = df_valid_cust["InvoiceDate"].max()
        
        rfm = df_valid_cust.groupby("CustomerID").agg(
            first_purchase_date=("InvoiceDate", "min"),
            last_purchase_date=("InvoiceDate", "max"),
            frequency_orders=("InvoiceNo", "nunique"),
            total_items_bought=("Quantity", lambda x: x[x > 0].sum()),
            cancellation_count=("is_cancelled", "sum"),
            monetary_value=("total_revenue", lambda x: x[x > 0].sum())
        ).reset_index()

        rfm["customer_id"] = rfm["CustomerID"].astype(float)
        rfm["recency_days"] = (max_date - rfm["last_purchase_date"]).dt.days
        rfm["avg_order_value"] = np.where(rfm["frequency_orders"] > 0, rfm["monetary_value"] / rfm["frequency_orders"], 0.0)
        
        # Assign RFM segment
        rfm["rfm_segment"] = rfm.apply(
            lambda r: self.assign_rfm_segment(r["recency_days"], r["frequency_orders"], r["monetary_value"]), axis=1
        )
        
        # Merge with dim_customer for demographic & subscription features
        fact_retention = pd.merge(rfm, dim_customer, on="customer_id", how="left")
        
        # Deduplicate duplicate columns resulting from merge
        if "avg_order_value_x" in fact_retention.columns:
            fact_retention["avg_order_value"] = fact_retention["avg_order_value_x"].fillna(fact_retention.get("avg_order_value_y", 0.0))
            fact_retention = fact_retention.drop(columns=["avg_order_value_x", "avg_order_value_y"], errors="ignore")

        # Determine churn label (1 if recency > threshold OR churn column in churn master)
        if "churn" in fact_retention.columns:
            fact_retention["churn_label"] = np.where(
                (fact_retention["recency_days"] > self.inactivity_threshold_days) | (fact_retention["churn"] == 1), 1, 0
            )
        else:
            fact_retention["churn_label"] = np.where(fact_retention["recency_days"] > self.inactivity_threshold_days, 1, 0)
            
        fact_retention = fact_retention.drop(columns=["CustomerID", "Country"], errors="ignore")
        logger.info(f"Generated fact_customer_retention mart with {len(fact_retention)} rows.")

        return {
            "dim_date": dim_date,
            "dim_product": dim_product,
            "dim_customer": dim_customer,
            "dim_geography": dim_geography,
            "fact_sales": fact_sales,
            "fact_customer_retention": fact_retention
        }
