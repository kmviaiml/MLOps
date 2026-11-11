import os
import shutil
import time
from datetime import datetime
from typing import Tuple, Dict, Any
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("etl_extractor")

class RawDataExtractor:
    """Handles raw data ingestion, file copy to landing area, and audit logging."""
    
    def __init__(self, raw_retail_path: str = "online_retail.csv", 
                 raw_churn_path: str = "ecommerce_customer_churn_large.csv",
                 landing_dir: str = "data/raw"):
        self.raw_retail_path = raw_retail_path
        self.raw_churn_path = raw_churn_path
        self.landing_dir = landing_dir
        os.makedirs(self.landing_dir, exist_ok=True)

    def copy_to_landing(self) -> Tuple[str, str]:
        """Preserves immutable raw copies in landing area."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        landing_retail = os.path.join(self.landing_dir, f"raw_online_retail_{timestamp}.csv")
        landing_churn = os.path.join(self.landing_dir, f"raw_customer_churn_{timestamp}.csv")
        
        if os.path.exists(self.raw_retail_path):
            shutil.copy(self.raw_retail_path, landing_retail)
            logger.info(f"Copied raw online retail dataset to landing: {landing_retail}")
        else:
            logger.error(f"Source file {self.raw_retail_path} not found!")

        if os.path.exists(self.raw_churn_path):
            shutil.copy(self.raw_churn_path, landing_churn)
            logger.info(f"Copied raw customer churn dataset to landing: {landing_churn}")
        else:
            logger.error(f"Source file {self.raw_churn_path} not found!")
            
        return landing_retail, landing_churn

    def extract(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Reads source datasets into DataFrames and records extraction metrics."""
        start_time = time.time()
        logger.info("Starting dataset extraction...")
        
        # Load Online Retail Data
        if os.path.exists(self.raw_retail_path):
            df_retail = pd.read_csv(self.raw_retail_path, encoding='utf-8', on_bad_lines='skip')
        else:
            logger.warning(f"File {self.raw_retail_path} missing. Returning empty DataFrame.")
            df_retail = pd.DataFrame()

        # Load Customer Churn Master Data
        if os.path.exists(self.raw_churn_path):
            df_churn = pd.read_csv(self.raw_churn_path, encoding='utf-8', on_bad_lines='skip')
        else:
            logger.warning(f"File {self.raw_churn_path} missing. Returning empty DataFrame.")
            df_churn = pd.DataFrame()

        duration = time.time() - start_time
        metrics = {
            "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "retail_raw_rows": len(df_retail),
            "churn_raw_rows": len(df_churn),
            "duration_sec": round(duration, 4),
            "status": "SUCCESS" if len(df_retail) > 0 else "FAILED"
        }
        
        logger.info(f"Extraction complete in {metrics['duration_sec']}s. Retail rows: {metrics['retail_raw_rows']}, Churn rows: {metrics['churn_raw_rows']}")
        return df_retail, df_churn, metrics
