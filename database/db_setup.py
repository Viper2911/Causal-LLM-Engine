import pandas as pd
import os
from sqlalchemy import create_engine

db_path="sqlite:///database/olist.db"

def build_db():
    print("Initializing SQLite database engine...")
    engine=create_engine(db_path)
    data_dir="data/"

    files_to_load = {
        "orders": "olist_orders_dataset.csv",
        "order_reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv"
    }

    for table_name,file_name in files_to_load.items():
        file_path=os.path.join(data_dir,file_name)
        if os.path.exists(file_path):
            df=pd.read_csv(file_path)
            df.dropna(how="all",inplace=True)
            df.to_sql(table_name,engine,index=False,if_exists="replace")
            print(f"Loaded '{table_name}' table successfully.")
        else:
            print(f"Skipped '{file_name}': File not found in data/ directory.")
    print("Database build complete: database/olist.db is live.")

def execute_query(sql_string: str)->pd.DataFrame:
    engine=create_engine(db_path)
    try:
        return pd.read_sql_query(sql_string,engine)
    except Exception as e:
        print(f"SQL Execution Error: {e}")
        return pd.DataFrame()

if __name__=="__main__":
    build_db()