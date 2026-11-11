import os
import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import ChurnPredictor
from src.database.db_manager import DatabaseManager
from src.monitoring.drift_monitor import DriftMonitor

app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description="REST API service for online retail customer churn risk scoring and model monitoring.",
    version="1.0.0"
)

# Initialize predictor and database manager
predictor = ChurnPredictor(artifacts_dir="models")
db_manager = DatabaseManager(db_path="data/warehouse/ecommerce_dw.db")

class SingleCustomerInput(BaseModel):
    customer_id: Optional[float] = Field(default=99999.0, description="Customer Identifier")
    recency_days: int = Field(default=30, description="Days since last order")
    frequency_orders: int = Field(default=5, description="Total order count")
    monetary_value: float = Field(default=450.0, description="Total monetary spend")
    avg_order_value: float = Field(default=90.0, description="Average order value")
    total_items_bought: int = Field(default=25, description="Total quantity of items purchased")
    cancellation_count: int = Field(default=0, description="Number of cancelled orders")
    tenure_months: int = Field(default=24, description="Customer tenure in months")
    age: int = Field(default=35, description="Customer age")
    support_tickets: int = Field(default=2, description="Number of customer support tickets")
    gender: str = Field(default="Female", description="Gender (Male/Female/Unknown)")
    city: str = Field(default="London", description="City location")
    subscription_type: str = Field(default="Gold", description="Subscription tier (Basic/Silver/Gold/Platinum)")
    rfm_segment: str = Field(default="Loyal Customers", description="RFM Segment")

class PredictionResponse(BaseModel):
    customer_id: Any
    churn_probability: float
    predicted_churn: int
    risk_level: str
    model_version: str

@app.get("/", tags=["General"])
def root():
    return {
        "service": "E-Commerce Customer Churn Prediction API",
        "status": "ONLINE",
        "documentation": "/docs",
        "endpoints": ["/health", "/predict", "/batch-predict", "/model-info", "/drift-report"]
    }

@app.get("/health", tags=["Health & Status"])
def health_check():
    is_model_ready = predictor.model is not None
    return {
        "status": "HEALTHY" if is_model_ready else "MODEL_UNINITIALIZED",
        "model_loaded": is_model_ready,
        "database_connected": os.path.exists("data/warehouse/ecommerce_dw.db")
    }

@app.get("/model-info", tags=["Model Registry"])
def get_model_info():
    if not predictor.metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found. Train model first.")
    return predictor.metadata

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: SingleCustomerInput):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run pipeline first.")
    
    data_dict = customer.dict()
    result = predictor.predict_single(data_dict)
    return result

@app.post("/batch-predict", tags=["Inference"])
def predict_churn_batch(customers: List[SingleCustomerInput]):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    
    input_list = [c.dict() for c in customers]
    df_input = pd.DataFrame(input_list)
    df_res = predictor.predict_dataframe(df_input)
    
    return df_res[["customer_id", "churn_probability", "predicted_churn", "risk_level"]].to_dict(orient="records")

@app.get("/drift-report", tags=["MLOps Monitoring"])
def get_drift_report():
    if not os.path.exists("data/warehouse/ecommerce_dw.db"):
        raise HTTPException(status_code=404, detail="Data Warehouse DB missing.")
        
    df_retention = db_manager.execute_query("SELECT * FROM fact_customer_retention")
    if df_retention.empty:
        raise HTTPException(status_code=404, detail="Retention table is empty.")

    monitor = DriftMonitor(psi_threshold=0.2)
    half = len(df_retention) // 2
    numeric_cols = [
        "recency_days", "frequency_orders", "monetary_value", 
        "avg_order_value", "total_items_bought", "cancellation_count",
        "tenure_months", "age", "support_tickets"
    ]
    report = monitor.analyze_dataset_drift(df_retention.iloc[:half], df_retention.iloc[half:], numeric_cols)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
