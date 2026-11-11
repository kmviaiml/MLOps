import os
import sys
import sqlite3
import pandas as pd

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.db_manager import DatabaseManager

def inspect_database():
    db_path = os.path.abspath("data/warehouse/ecommerce_dw.db")
    print("==================================================================")
    print(" 🗄️ LOCAL SQLITE DATABASE INSPECTOR FOR DB BROWSER FOR SQLITE")
    print("==================================================================")
    print(f" Database File Location: {db_path}")
    print(" Status: Ready to open in DB Browser for SQLite")
    print("------------------------------------------------------------------")
    
    if not os.path.exists(db_path):
        print(" [!] Database file not found yet. Run `python scripts/run_pipeline.py` to create and populate it.")
        return

    db = DatabaseManager(db_path=db_path)
    tables = [
        "dim_customer", "dim_product", "dim_geography", "dim_date",
        "fact_sales", "fact_customer_retention",
        "etl_extraction_log", "etl_rejected_records"
    ]
    
    print("\n📊 Tables in Star Schema Data Warehouse:")
    for tbl in tables:
        try:
            cnt = db.get_table_row_count(tbl)
            print(f"  - {tbl:<25}: {cnt:>10,} rows")
        except Exception:
            print(f"  - {tbl:<25}: [Table Not Created Yet]")

    print("\n------------------------------------------------------------------")
    print(" 💡 How to open in DB Browser for SQLite:")
    print("  1. Launch 'DB Browser for SQLite' on your computer.")
    print("  2. Click 'Open Database' (Ctrl+O).")
    print(f"  3. Paste/Select path: {db_path}")
    print("  4. Browse tables, execute custom SQL queries, or export schemas!")
    print("==================================================================")

if __name__ == "__main__":
    inspect_database()
