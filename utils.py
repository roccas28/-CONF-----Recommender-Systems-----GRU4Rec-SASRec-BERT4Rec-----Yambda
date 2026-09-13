# Roccas Raveendran - 500853856
# Sequential Recommender System - Utility & Out-of-Core ETL Functions (500M Scale)

import pandas as pd
import numpy as np
import duckdb
import os
import pyarrow.parquet as pq

DATA_PATH = './Data/'

def extract_eda_sample(file_name='multi_event.parquet', sample_percent=10): 
    print(f"Extracting {sample_percent}% random sample from 500M dataset for EDA plotting...")
    file_path = os.path.join(DATA_PATH, file_name) 
    
    query = f"""
        SELECT played_ratio_pct, item_id 
        FROM '{file_path}' 
        USING SAMPLE {sample_percent}%
    """
    return duckdb.query(query).df() 

def duckdb_group_and_export(file_name='multi_event.parquet', out_file='./output/raw_grouped.parquet'): 
    print("Using DuckDB to group 500M rows out-of-core and export directly to disk...")
    file_path = os.path.join(DATA_PATH, file_name)
    
    query = f"""
        COPY (
            SELECT 
                uid,
                list(item_id ORDER BY timestamp ASC) as item_sequence
            FROM '{file_path}'
            WHERE uid IS NOT NULL 
              AND item_id IS NOT NULL 
              AND timestamp IS NOT NULL
            GROUP BY uid
            HAVING count(item_id) >= 5 
        ) TO '{out_file}' (FORMAT PARQUET);
    """
    duckdb.execute(query) 
    print("DuckDB 500M aggregation successfully saved to disk!")

def get_global_item_mapping(file_name='multi_event.parquet'): 
    print("Extracting global unique items (approx. 3 Million) for Label Encoding...")
    file_path = os.path.join(DATA_PATH, file_name)
    
    unique_items_df = duckdb.query(f"SELECT DISTINCT item_id FROM '{file_path}' WHERE item_id IS NOT NULL").df()
    item_ids = unique_items_df['item_id'].values 
    
    item2idx = {int(item): idx + 1 for idx, item in enumerate(item_ids)}
    return item2idx 

def extract_full_eda_stats(file_name='multi_event.parquet'): 
    print("Extracting full sequence lengths and event distributions via DuckDB (500M Scale)...")
    file_path = os.path.join(DATA_PATH, file_name)
    
    query_lengths = f"""
        SELECT count(item_id) as seq_length 
        FROM '{file_path}' 
        GROUP BY uid 
        HAVING count(item_id) >= 5
    """
    df_lengths = duckdb.query(query_lengths).df() 
    
    query_events = f"""
        SELECT event_type, count(*) as total_count 
        FROM '{file_path}' 
        GROUP BY event_type
    """
    df_events = duckdb.query(query_events).df() 
    
    return df_lengths, df_events 

def export_processed_preview(file_path='./output/processed_sequences.parquet', out_path='./output/processed_preview.csv'): 
    print(f"Exporting a 15-row preview of the final arrays to {out_path}...")
    
    pf = pq.ParquetFile(file_path) 
    preview_df = next(pf.iter_batches(batch_size=15)).to_pandas() 
    preview_df.to_csv(out_path, index=False)
