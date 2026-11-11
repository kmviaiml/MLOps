from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from src.database.db_manager import DatabaseManager
from src.utils.logger import setup_logger

logger = setup_logger("etl_loader")

class StarSchemaLoader:
    """Handles loading transformed datasets into Data Warehouse with audit logging."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def load_warehouse(self, transformed_tables: Dict[str, pd.DataFrame], rejection_logs: List[Dict[str, Any]], extraction_metrics: Dict[str, Any]):
        """Loads dimension tables, fact tables, and audit logs into SQL Data Warehouse."""
        logger.info("Starting Data Warehouse loading process...")
        
        # 1. Load Dimensions first to satisfy foreign key relationships
        dimension_tables = ["dim_date", "dim_geography", "dim_product", "dim_customer"]
        for tbl in dimension_tables:
            if tbl in transformed_tables and not transformed_tables[tbl].empty:
                self.db.insert_dataframe(transformed_tables[tbl], table_name=tbl, if_exists="replace")
                
        # 2. Load Fact Tables
        fact_tables = ["fact_sales", "fact_customer_retention"]
        for tbl in fact_tables:
            if tbl in transformed_tables and not transformed_tables[tbl].empty:
                self.db.insert_dataframe(transformed_tables[tbl], table_name=tbl, if_exists="replace")

        # 3. Load Audit Logs (etl_extraction_log & etl_rejected_records)
        valid_sales_cnt = len(transformed_tables.get("fact_sales", []))
        rejected_cnt = len(rejection_logs)
        
        audit_log = pd.DataFrame([{
            "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": "online_retail.csv",
            "status": extraction_metrics.get("status", "SUCCESS"),
            "total_raw_rows": extraction_metrics.get("retail_raw_rows", 0),
            "valid_rows": valid_sales_cnt,
            "rejected_rows": rejected_cnt,
            "execution_duration_sec": extraction_metrics.get("duration_sec", 0.0)
        }])
        self.db.insert_dataframe(audit_log, table_name="etl_extraction_log", if_exists="append")

        if rejection_logs:
            df_rejections = pd.DataFrame(rejection_logs)
            self.db.insert_dataframe(df_rejections, table_name="etl_rejected_records", if_exists="append")

        logger.info("Data Warehouse loading complete!")
