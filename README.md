# E-Commerce Sales Analytics & Customer Churn Prediction (Data Engineering & MLOps Platform)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)

An enterprise-grade, end-to-end Data Engineering ETL Warehouse and MLOps Churn Prediction platform built for online retail analytics.

---

## 🌟 Key Features

### Part 1: Data Engineering & Star Schema Warehouse
- **Automated Ingestion Pipeline**: Ingests UCI Online Retail dataset (`online_retail.csv`) and Customer Master records (`ecommerce_customer_churn_large.csv`).
- **Data Quality & Rejection Audit**: Validates transaction records against strict pricing, quantity, and date formatting rules. Failed rows are logged to `etl_rejected_records`.
- **SQL Star Schema Warehouse**:
  - `dim_customer`, `dim_product`, `dim_geography`, `dim_date`
  - `fact_sales` (transaction details & revenue)
  - `fact_customer_retention` (RFM analytics mart)
- **Airflow DAG Orchestration**: Scheduled execution workflow in `dags/ecommerce_etl_dag.py`.
- **Interactive Streamlit Dashboard**: 5 executive views covering Revenue Trends, Top Products, RFM Customer Segments, Country Sales, and Live Predictions.

### Part 2: MLOps Pipeline Extension
- **Feature Engineering**: Recency, Frequency, Monetary (RFM), Tenure, Support Tickets, Subscription Level features.
- **Model Training & Experiment Tracking**: Trains Logistic Regression, Random Forest, and XGBoost models tracked with **MLflow**.
- **FastAPI REST API**: High-performance model inference service (`/predict`, `/batch-predict`, `/health`, `/drift-report`).
- **Data Drift Monitoring**: Calculates Population Stability Index (PSI) to detect feature distribution shifts and trigger automated retraining.
- **Docker Containerization**: Full multi-container setup via `docker-compose.yml`.

---

## 📁 Repository Structure

```
d:\churn mlops\
├── .gitignore                          # Git ignore rules for data, models & environment
├── README.md                           # Master README documentation
├── PROJECT_REPORT.md                   # Full 8-10 page academic & technical report
├── docker-compose.yml                  # Multi-container service orchestrator
├── Dockerfile.api                      # Dockerfile for FastAPI model service
├── Dockerfile.dashboard                # Dockerfile for Streamlit web app
├── requirements.txt                    # Project dependencies
│
├── config/
│   └── config.yaml                     # Pipeline parameters & DB settings
├── dags/
│   └── ecommerce_etl_dag.py            # Apache Airflow DAG
├── data/
│   ├── raw/                            # Immutable raw dataset copies
│   ├── warehouse/                      # SQLite Star Schema DB (ecommerce_dw.db)
│   └── logs/                           # ETL & execution audit logs
├── models/                             # Saved ML models, scalers & metadata
├── src/                                # Core python package
│   ├── etl/                            # Extractor, Validator, Transformer, Loader
│   ├── database/                       # Schema DDL and DB manager
│   ├── models/                         # Feature engineering, training, predictor
│   ├── monitoring/                     # Population Stability Index (PSI) drift monitor
│   └── utils/                          # Structured logger
├── app/
│   ├── api.py                          # FastAPI REST API app
│   └── streamlit_app.py                # Streamlit UI dashboard
├── scripts/
│   └── run_pipeline.py                 # Master pipeline runner
└── tests/
    └── test_etl.py                     # Unit & integration tests
```

---

## 🚀 Quick Start Guide

### 1. Git Setup & Version Control

Initialize your Git repository and make your first commit (datasets, SQLite database, and large model artifacts are safely excluded via `.gitignore`):

```bash
git init
git add .
git commit -m "feat: initial commit for e-commerce sales analytics and churn mlops platform"
```

To connect to a remote GitHub / GitLab repository:
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### 2. Installation

Clone or locate the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Execute Master Pipeline (ETL + Model Training)

Run the end-to-end data pipeline to populate the Star Schema database and train machine learning models:
```bash
python scripts/run_pipeline.py
```

*Outputs:*
- Database populated at `data/warehouse/ecommerce_dw.db`
- Best ML model saved at `models/best_churn_model.joblib`
- Execution logs stored at `data/logs/`

---

## 🗄️ Inspecting with DB Browser for SQLite

The Data Warehouse is built as a standard, file-based SQLite database located at:  
`d:\churn mlops\data\warehouse\ecommerce_dw.db`

### How to open in DB Browser for SQLite:
1. Launch **DB Browser for SQLite** on your computer.
2. Click **Open Database** (or press `Ctrl+O`).
3. Open/Select file: `data/warehouse/ecommerce_dw.db`
4. Inspect tables (`dim_customer`, `dim_product`, `dim_geography`, `dim_date`, `fact_sales`, `fact_customer_retention`, `etl_rejected_records`), run SQL queries, or export datasets!

### Quick Inspector Tool:
```bash
python scripts/inspect_sqlite.py
```

---

## 💻 Launching Applications

### Launch Streamlit Executive Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open browser at `http://localhost:8501` to view interactive analytics and the live churn simulator.

### Launch FastAPI REST API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

To launch the complete stack (FastAPI + Streamlit Dashboard + Database) via Docker:

```bash
docker-compose up --build
```

- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8000/docs`

---

## 🧪 Running Tests

Execute pytest suite to verify ETL pipeline integrity:
```bash
pytest tests/
```

---

## 📊 Sample API Usage

### Single Customer Churn Prediction (`POST /predict`)

```json
POST http://localhost:8000/predict
Content-Type: application/json

{
  "customer_id": 12345,
  "recency_days": 120,
  "frequency_orders": 2,
  "monetary_value": 150.0,
  "avg_order_value": 75.0,
  "total_items_bought": 10,
  "cancellation_count": 1,
  "tenure_months": 6,
  "age": 28,
  "support_tickets": 8,
  "gender": "Male",
  "city": "London",
  "subscription_type": "Basic",
  "rfm_segment": "At Risk"
}
```

**Response:**
```json
{
  "customer_id": 12345,
  "churn_probability": 0.8425,
  "predicted_churn": 1,
  "risk_level": "High Risk",
  "model_version": "XGBoost"
}
```
