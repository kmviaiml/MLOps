import sys
import os
from datetime import datetime

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.database.db_manager import DatabaseManager
from src.etl.extractor import RawDataExtractor
from src.etl.validator import DataValidator
from src.etl.transformer import StarSchemaTransformer
from src.etl.loader import StarSchemaLoader
from src.models.feature_engineering import FeatureEngineeringPipeline
from src.models.train import ChurnModelTrainer
from src.monitoring.drift_monitor import DriftMonitor

logger = setup_logger("master_pipeline")

def run_full_pipeline():
    """Executes complete end-to-end Data Engineering and MLOps pipeline."""
    logger.info("==================================================================")
    logger.info(" STARTING E-COMMERCE SALES & CHURN MLOPS PIPELINE")
    logger.info("==================================================================")
    
    # 1. Database Initialization
    db = DatabaseManager(db_path="data/warehouse/ecommerce_dw.db")
    db.initialize_warehouse()

    # 2. Raw Data Extraction & Ingestion
    extractor = RawDataExtractor(
        raw_retail_path="online_retail.csv",
        raw_churn_path="ecommerce_customer_churn_large.csv",
        landing_dir="data/raw"
    )
    extractor.copy_to_landing()
    df_retail_raw, df_churn_raw, extraction_metrics = extractor.extract()

    # 3. Data Validation & Quality Audit
    validator = DataValidator(min_quantity=1, min_unit_price=0.001)
    df_retail_valid, df_retail_rejected, rejection_logs = validator.validate_retail_data(df_retail_raw)
    df_churn_valid = validator.validate_customer_master(df_churn_raw)

    # 4. Star Schema Transformation
    transformer = StarSchemaTransformer(inactivity_threshold_days=90)
    transformed_tables = transformer.transform(df_retail_valid, df_churn_valid)

    # 5. Load to Data Warehouse
    loader = StarSchemaLoader(db_manager=db)
    loader.load_warehouse(transformed_tables, rejection_logs, extraction_metrics)

    # 6. Machine Learning Model Training (Part 2)
    df_retention = transformed_tables["fact_customer_retention"]
    fe_pipeline = FeatureEngineeringPipeline(artifacts_dir="models")
    X_train, X_test, y_train, y_test, feature_names = fe_pipeline.prepare_data(df_retention, test_size=0.2)

    trainer = ChurnModelTrainer(artifacts_dir="models")
    training_summary = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test, feature_names)

    # 7. Drift Monitoring & Liveness Check
    monitor = DriftMonitor(psi_threshold=0.2)
    # Check drift using half of retention set as baseline and half as live
    half_idx = len(df_retention) // 2
    drift_summary = monitor.analyze_dataset_drift(
        df_baseline=df_retention.iloc[:half_idx],
        df_current=df_retention.iloc[half_idx:],
        num_features=fe_pipeline.NUMERIC_FEATURES
    )

    logger.info("==================================================================")
    logger.info("PIPELINE EXECUTED SUCCESSFULLY! ALL DATA WAREHOUSE & MODEL ARTIFACTS CREATED.")
    logger.info(f"   - Total Sales Transactions Loaded: {len(transformed_tables['fact_sales'])}")
    logger.info(f"   - Total Customers Analyzed: {len(transformed_tables['fact_customer_retention'])}")
    logger.info(f"   - Best ML Model Selected: {training_summary['best_model_name']} (F1: {training_summary['best_f1_score']})")
    logger.info("==================================================================")

if __name__ == "__main__":
    run_full_pipeline()
