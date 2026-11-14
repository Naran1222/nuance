from astroquery.ipac.irsa import Irsa
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# 1. Перевіряємо спеціалізовані таблиці SPHEREx у схемі IRSA
query_tables = """
SELECT schema_name, table_name, description
FROM TAP_SCHEMA.tables
WHERE LOWER(table_name) LIKE '%spherex%'
   OR LOWER(description) LIKE '%spherex%'
"""
tables_df = Irsa.query_tap(query_tables).to_table().to_pandas()
print("=== Спеціалізовані таблиці SPHEREx ===")
print(tables_df)

# 2. Перевіряємо стандартизовану таблицю спостережень ivoa.obscore
query_collections = """
SELECT obs_collection, calib_level, dataproduct_type, COUNT(*) AS total_frames
FROM ivoa.obscore
WHERE LOWER(obs_collection) LIKE '%spherex%'
GROUP BY obs_collection, calib_level, dataproduct_type
"""
collections_df = Irsa.query_tap(query_collections).to_table().to_pandas()
print("\n=== Доступні колекції в ivoa.obscore ===")
print(collections_df)