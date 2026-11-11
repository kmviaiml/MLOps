import pytest
import pandas as pd
from src.etl.validator import DataValidator
from src.etl.transformer import StarSchemaTransformer

def test_data_validator():
    validator = DataValidator(min_quantity=1, min_unit_price=0.01)
    sample_raw = pd.DataFrame([
        {"InvoiceNo": "536365", "StockCode": "85123A", "Quantity": 6, "UnitPrice": 2.55, "InvoiceDate": "2010-12-01 08:26:00", "CustomerID": 17850},
        {"InvoiceNo": "C536365", "StockCode": "85123A", "Quantity": -6, "UnitPrice": 2.55, "InvoiceDate": "2010-12-01 08:26:00", "CustomerID": 17850}, # Cancelled
        {"InvoiceNo": "536366", "StockCode": "22633", "Quantity": -1, "UnitPrice": 0.0, "InvoiceDate": "InvalidDate", "CustomerID": 17850} # Invalid
    ])
    
    df_valid, df_rejected, logs = validator.validate_retail_data(sample_raw)
    assert len(df_valid) == 2
    assert len(df_rejected) == 1
    assert len(logs) == 1

def test_star_schema_transformer():
    transformer = StarSchemaTransformer()
    sample_valid = pd.DataFrame([
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 6, "UnitPrice": 2.55, "InvoiceDate": pd.to_datetime("2010-12-01 08:26:00"), "CustomerID": 17850.0, "Country": "United Kingdom", "is_cancelled": 0}
    ])
    churn_master = pd.DataFrame()
    
    tables = transformer.transform(sample_valid, churn_master)
    assert "dim_date" in tables
    assert "dim_product" in tables
    assert "dim_customer" in tables
    assert "fact_sales" in tables
    assert "fact_customer_retention" in tables
    assert len(tables["fact_sales"]) == 1
