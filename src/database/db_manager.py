import os
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from src.database.schema import CREATE_TABLES_SQL
from src.utils.logger import setup_logger

logger = setup_logger("database_manager")

class DatabaseManager:
    """
    Handles connection, initialization, and data querying for the SQLite Data Warehouse.
    Produces a standard .db / .sqlite file fully compatible with DB Browser for SQLite.
    """
    
    def __init__(self, db_path: Optional[str] = None, postgres_url: Optional[str] = None):
        if db_path is None:
            config_file = "config/config.yaml"
            if HAS_YAML and os.path.exists(config_file):
                with open(config_file, "r") as f:
                    cfg = yaml.safe_load(f)
                    db_path = cfg.get("database", {}).get("sqlite_path", "data/warehouse/ecommerce_dw.db")
            else:
                db_path = "data/warehouse/ecommerce_dw.db"

        self.db_path = os.path.abspath(db_path)
        self.postgres_url = postgres_url
        
        if postgres_url:
            self.engine = create_engine(postgres_url)
            self.db_type = "postgres"
        else:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.engine = create_engine(f"sqlite:///{self.db_path}")
            self.db_type = "sqlite"
            
        logger.info(f"Connected to {self.db_type.upper()} Data Warehouse: '{self.db_path}'")

    def initialize_warehouse(self):
        """Executes DDL statements to create Star Schema tables and indexes."""
        logger.info(f"Initializing Star Schema tables in SQLite database at '{self.db_path}'...")
        with self.engine.connect() as conn:
            statements = [s.strip() for s in CREATE_TABLES_SQL.split(";") if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()
        logger.info(f"Star Schema initialized! Open '{self.db_path}' in DB Browser for SQLite to inspect.")

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Executes a SELECT query and returns a pandas DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(query), conn, params=params)

    def execute_non_query(self, sql_statement: str, params: Optional[Dict[str, Any]] = None):
        """Executes INSERT, UPDATE, DELETE statements."""
        with self.engine.connect() as conn:
            conn.execute(text(sql_statement), params or {})
            conn.commit()

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append", index: bool = False):
        """Inserts a Pandas DataFrame directly into a target SQL table."""
        if df.empty:
            logger.warning(f"Attempted to insert empty DataFrame into table '{table_name}'. Skipping.")
            return
        logger.info(f"Loading {len(df)} rows into SQLite table '{table_name}'...")
        df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=index, chunksize=10000)
        logger.info(f"Successfully loaded {len(df)} rows into '{table_name}'.")

    def get_table_row_count(self, table_name: str) -> int:
        """Returns total row count for a given table."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        res = self.execute_query(query)
        return int(res["count"].iloc[0])
