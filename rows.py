import pyarrow.parquet as pq

# Read only the metadata, not the data itself
metadata = pq.read_metadata('./output/processed_sequences.parquet')
print(f"Total sequences (users) ready for ML: {metadata.num_rows}")
