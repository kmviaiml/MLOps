"""
Apache Airflow DAG Definition for E-Commerce Data Engineering & MLOps Pipeline
"""

from datetime import datetime, timedelta
import sys
import os

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

# Fallback definitions for non-airflow execution environments
if HAS_AIRFLOW:
    default_args = {
        'owner': 'data_engineering_team',
        'depends_on_past': False,
        'start_date': datetime(2026, 9, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        'ecommerce_sales_and_churn_pipeline',
        default_args=default_args,
        description='Daily ETL pipeline for E-Commerce Data Warehouse and Churn Model Training',
        schedule_interval='@daily',
        catchup=False
    )

    def task_extract_raw_data(**kwargs):
        from src.etl.extractor import RawDataExtractor
        extractor = RawDataExtractor()
        extractor.copy_to_landing()
        _, _, metrics = extractor.extract()
        return metrics

    def task_validate_data(**kwargs):
        from src.etl.extractor import RawDataExtractor
        from src.etl.validator import DataValidator
        extractor = RawDataExtractor()
        df_retail, df_churn, _ = extractor.extract()
        validator = DataValidator()
        df_retail_valid, _, rejection_logs = validator.validate_retail_data(df_retail)
        return len(df_retail_valid)

    def task_transform_and_load(**kwargs):
        from scripts.run_pipeline import run_full_pipeline
        run_full_pipeline()
        return "SUCCESS"

    extract_task = PythonOperator(
        task_id='extract_raw_data',
        python_callable=task_extract_raw_data,
        dag=dag,
    )

    validate_task = PythonOperator(
        task_id='validate_data_quality',
        python_callable=task_validate_data,
        dag=dag,
    )

    transform_load_task = PythonOperator(
        task_id='transform_and_load_dw',
        python_callable=task_transform_and_load,
        dag=dag,
    )

    extract_task >> validate_task >> transform_load_task
