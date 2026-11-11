import json
from datetime import datetime
from typing import Tuple, List, Dict, Any
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("etl_validator")

class DataValidator:
    """Applies strict data quality rules and captures rejected records."""
    
    def __init__(self, min_quantity: int = 1, min_unit_price: float = 0.001):
        self.min_quantity = min_quantity
        self.min_unit_price = min_unit_price

    def validate_retail_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Validates raw online retail transactions.
        Returns:
            df_valid: Cleaned DataFrame ready for transformation
            df_rejected: DataFrame of rejected records with reason
            rejection_logs: List of dictionary records for DB rejected_records table
        """
        if df_raw.empty:
            logger.warning("Empty raw DataFrame passed to validator.")
            return pd.DataFrame(), pd.DataFrame(), []

        logger.info(f"Starting data validation on {len(df_raw)} records...")
        df = df_raw.copy()

        # Parse InvoiceDate
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

        rejection_reasons = []
        rejection_logs = []

        # Rule 1: Null InvoiceDate
        mask_invalid_date = df["InvoiceDate"].isna()
        
        # Rule 2: Missing StockCode or Description
        mask_missing_stock = df["StockCode"].isna() | (df["StockCode"].astype(str).str.strip() == "")
        
        # Rule 3: Invalid Price (<= 0)
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
        mask_invalid_price = df["UnitPrice"] <= self.min_unit_price
        
        # Rule 4: Cancelled orders vs Invalid Quantity
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
        df["is_cancelled"] = df["InvoiceNo"].str.startswith("C", na=False).astype(int)
        
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        # Quantity <= 0 when not marked cancelled is invalid
        mask_invalid_quantity = (df["Quantity"] <= 0) & (df["is_cancelled"] == 0)

        # Combine rejection masks
        mask_rejected = mask_invalid_date | mask_missing_stock | mask_invalid_price | mask_invalid_quantity
        
        df_valid = df[~mask_rejected].copy()
        df_rejected = df[mask_rejected].copy()

        # Assign rejection reasons
        df_rejected["rejection_reason"] = "UNKNOWN"
        df_rejected.loc[df_rejected["InvoiceDate"].isna(), "rejection_reason"] = "Invalid Date Format"
        df_rejected.loc[df_rejected["StockCode"].isna(), "rejection_reason"] = "Missing StockCode"
        df_rejected.loc[df_rejected["UnitPrice"] <= self.min_unit_price, "rejection_reason"] = "Invalid Unit Price (<=0)"
        df_rejected.loc[(df_rejected["Quantity"] <= 0) & (df_rejected["is_cancelled"] == 0), "rejection_reason"] = "Non-positive Quantity for non-cancelled order"

        logger.info(f"Validation finished: {len(df_valid)} valid records, {len(df_rejected)} rejected records.")

        # Create structured rejection logs
        for idx, row in df_rejected.head(1000).iterrows(): # Sample top 1000 for audit table
            rejection_logs.append({
                "source_file": "online_retail.csv",
                "invoice_no": str(row.get("InvoiceNo", "")),
                "stock_code": str(row.get("StockCode", "")),
                "customer_id": float(row.get("CustomerID")) if pd.notna(row.get("CustomerID")) else None,
                "raw_record": json.dumps(row.to_dict(), default=str),
                "rejection_reason": str(row.get("rejection_reason", "Invalid Data")),
                "rejected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        return df_valid, df_rejected, rejection_logs

    def validate_customer_master(self, df_churn: pd.DataFrame) -> pd.DataFrame:
        """Validates supplementary customer master dataset."""
        if df_churn.empty:
            return df_churn

        logger.info(f"Validating customer master data ({len(df_churn)} records)...")
        df = df_churn.copy()
        df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
        df = df.dropna(subset=["customer_id"])
        df["customer_id"] = df["customer_id"].astype(float)
        return df
