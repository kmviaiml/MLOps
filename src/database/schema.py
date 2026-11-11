"""
SQL DDL Schemas for E-Commerce Sales Data Warehouse (Star Schema)
Supports both SQLite and PostgreSQL syntax.
"""

CREATE_TABLES_SQL = """
-- Operational Audit Log Table
CREATE TABLE IF NOT EXISTS etl_extraction_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_file TEXT NOT NULL,
    status TEXT NOT NULL,
    total_raw_rows INTEGER NOT NULL,
    valid_rows INTEGER NOT NULL,
    rejected_rows INTEGER NOT NULL,
    execution_duration_sec REAL
);

-- Data Quality Rejection Log Table
CREATE TABLE IF NOT EXISTS etl_rejected_records (
    rejection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT,
    invoice_no TEXT,
    stock_code TEXT,
    customer_id REAL,
    raw_record TEXT,
    rejection_reason TEXT NOT NULL,
    rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Customer
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id REAL PRIMARY KEY,
    country TEXT,
    age INTEGER,
    gender TEXT,
    city TEXT,
    tenure_months INTEGER,
    subscription_type TEXT,
    support_tickets INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Product
CREATE TABLE IF NOT EXISTS dim_product (
    stock_code TEXT PRIMARY KEY,
    description TEXT,
    category TEXT,
    unit_price REAL,
    is_active INTEGER DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Geography
CREATE TABLE IF NOT EXISTS dim_geography (
    geography_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country TEXT NOT NULL,
    city TEXT,
    region TEXT
);

-- Dimension: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL,
    day INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    quarter INTEGER NOT NULL,
    year INTEGER NOT NULL,
    day_of_week TEXT NOT NULL,
    is_weekend INTEGER NOT NULL
);

-- Fact: Sales Transactions
CREATE TABLE IF NOT EXISTS fact_sales (
    sales_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    customer_id REAL,
    date_key INTEGER NOT NULL,
    invoice_date TIMESTAMP NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    total_revenue REAL NOT NULL,
    country TEXT NOT NULL,
    is_cancelled INTEGER DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
    FOREIGN KEY (stock_code) REFERENCES dim_product(stock_code),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

-- Fact / Data Mart: Customer Retention & Churn Features
CREATE TABLE IF NOT EXISTS fact_customer_retention (
    customer_id REAL PRIMARY KEY,
    recency_days INTEGER,
    frequency_orders INTEGER,
    monetary_value REAL,
    avg_order_value REAL,
    total_items_bought INTEGER,
    cancellation_count INTEGER,
    first_purchase_date TIMESTAMP,
    last_purchase_date TIMESTAMP,
    rfm_segment TEXT,
    tenure_months INTEGER,
    age INTEGER,
    gender TEXT,
    city TEXT,
    subscription_type TEXT,
    support_tickets INTEGER,
    churn_label INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Query Performance
CREATE INDEX IF NOT EXISTS idx_sales_customer ON fact_sales(customer_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON fact_sales(date_key);
CREATE INDEX IF NOT EXISTS idx_sales_invoice ON fact_sales(invoice_no);
CREATE INDEX IF NOT EXISTS idx_sales_stock ON fact_sales(stock_code);
CREATE INDEX IF NOT EXISTS idx_retention_rfm ON fact_customer_retention(rfm_segment);
"""
