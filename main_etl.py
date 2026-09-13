# Roccas Raveendran - 500853856
# Sequential Recommender System - Main ETL Pipeline Orchestrator (500M Scale)

import pandas as pd
import numpy as np
import time
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Import Custom Modules
import utils as ut
import plots as pt

def run():
    print("Extended Convergence Analysis: 500M Sequential Recommender ETL Pipeline")
    
    if not os.path.exists('./output'):
        os.makedirs('./output')

    start_time = time.time() 
    
    # --- PHASE 1: Exploratory Data Analysis ---
    print("\n--- Running Exploratory Data Analysis ---")
    eda_df = ut.extract_eda_sample('multi_event.parquet', sample_percent=10) 
    pt.plot_played_ratio_distribution(eda_df) 
    pt.plot_long_tail_popularity(eda_df) 
    del eda_df 

    df_lengths, df_events = ut.extract_full_eda_stats('multi_event.parquet') 
    pt.plot_sequence_length_distribution(df_lengths) 
    pt.plot_event_type_distribution(df_events) 
    del df_lengths, df_events 
    
    # --- PHASE 2: Heavy ETL Grouping ---
    print("\n--- Running Data Preprocessing ---")
    raw_grouped_path = './output/raw_grouped.parquet' 
    ut.duckdb_group_and_export('multi_event.parquet', raw_grouped_path) 
    
    item2idx = ut.get_global_item_mapping('multi_event.parquet') 
    print(f"Total Unique Items Found: {len(item2idx)}")
    
    # --- PHASE 3: Streaming, Encoding & LOO Split ---
    print("\n--- Executing Label Encoding and LOO Temporal Split ---")
    print("Note: BERT4Rec [MASK] tokens will be injected dynamically by the DataLoader during training.")
    
    writer = None 
    uid_counter = 1 
    
    # Context manager 'with open' -> Forces Windows to release the file lock after execution completes
    with open(raw_grouped_path, 'rb') as f:
        parquet_file = pq.ParquetFile(f)
        
        for batch in tqdm(parquet_file.iter_batches(batch_size=100000), desc="Processing Batches"):
            df_chunk = batch.to_pandas() 
            
            df_chunk['mapped_uid'] = range(uid_counter, uid_counter + len(df_chunk))
            uid_counter += len(df_chunk) 
            
            train_seqs, val_seqs, test_seqs = [], [], [] 
            
            for seq in df_chunk['item_sequence']: 
                mapped_seq = [item2idx[x] for x in seq] 
                
                # Execute Leave-One-Out (LOO) Temporal Split
                train_seqs.append(mapped_seq[:-2]) 
                val_seqs.append(mapped_seq[-2])    
                test_seqs.append(mapped_seq[-1])   
                
            final_chunk = pd.DataFrame({
                'uid': df_chunk['mapped_uid'].astype(np.int32),
                'train_sequence': train_seqs,
                'val_item': np.array(val_seqs, dtype=np.int32),
                'test_item': np.array(test_seqs, dtype=np.int32)
            })
            
            table = pa.Table.from_pandas(final_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter('./output/processed_sequences.parquet', table.schema)
            writer.write_table(table) 
            
    if writer:
        writer.close() 
        
    os.remove(raw_grouped_path) 

    ut.export_processed_preview()
    
    print(f"Successfully exported final padded sequences to ./output/processed_sequences.parquet")
    print(f"\nETL Pipeline Complete! Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    run()
