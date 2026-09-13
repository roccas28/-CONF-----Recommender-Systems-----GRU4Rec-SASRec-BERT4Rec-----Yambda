# Roccas Raveendran - 500853856
# Sequential Recommender System - TensorFlow Dataset & Dynamic Masking

import pandas as pd
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq

class SequentialDataLoader:
    
    def __init__(self, file_path='./output/processed_sequences.parquet', max_len=70):
        print("Initializing Strict Sequential DataLoader...")
        self.file_path = file_path
        self.max_len = max_len 
        
        df = pd.read_parquet(self.file_path, engine='pyarrow')
        
        self.uids = df['uid'].values
        self.train_seqs = df['train_sequence'].tolist()
        self.val_items = df['val_item'].values
        self.test_items = df['test_item'].values
        
        flat_train = np.concatenate(self.train_seqs) 
        self.total_items = int(max(flat_train.max(), self.val_items.max(), self.test_items.max()))
        
        # Dynamically assign the [MASK] token to the integer right above the max item ID
        self.mask_token = self.total_items + 1
        print(f"Dataset Cardinality: {self.total_items} | [MASK] Token ID: {self.mask_token}")
        
    def _pad_and_truncate(self, seq):
        seq = [int(x) for x in seq]
        if len(seq) > self.max_len:
            return seq[-self.max_len:]
        else:
            pad_length = self.max_len - len(seq)
            return [0] * pad_length + seq

    def _data_generator(self, is_training, model_type='causal'): 
        indices = np.arange(len(self.train_seqs))
        
        if is_training:
            np.random.shuffle(indices)
            
        for idx in indices:
            seq = self.train_seqs[idx]
            pos_target = self.val_items[idx] if is_training else self.test_items[idx]
            
            # --- STRICT NEGATIVE SAMPLING (Used for BPR) ---
            invalid_items = set(seq)
            invalid_items.add(self.val_items[idx])
            invalid_items.add(self.test_items[idx])
            
            neg_target = np.random.randint(1, self.total_items + 1)
            while neg_target in invalid_items:
                neg_target = np.random.randint(1, self.total_items + 1)
                
            padded_seq = self._pad_and_truncate(seq)
            
            # --- BERT4Rec DYNAMIC CLOZE MASKING ---
            if model_type == 'bert4rec':
                padded_seq = np.array(padded_seq)
                if is_training:
                    # Randomly mask 20% of the sequence items (ignoring 0 padding)
                    mask_indices = (np.random.rand(len(padded_seq)) < 0.20) & (padded_seq != 0)
                    padded_seq[mask_indices] = self.mask_token
                else:
                    # For evaluation, append [MASK] at the very end to predict the next item
                    padded_seq = np.append(padded_seq[1:], self.mask_token)
                padded_seq = padded_seq.tolist()
            
            yield (
                {"input_sequence": np.array(padded_seq, dtype=np.int32)},
                {"positive_target": np.int32(pos_target), "negative_target": np.int32(neg_target)}
            )

    def generate_tf_dataset(self, batch_size=256, is_training=True, model_type='causal'):
        output_signature = (
            {"input_sequence": tf.TensorSpec(shape=(self.max_len,), dtype=tf.int32)},
            {
                "positive_target": tf.TensorSpec(shape=(), dtype=tf.int32),
                "negative_target": tf.TensorSpec(shape=(), dtype=tf.int32)
            }
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(is_training=is_training, model_type=model_type),
            output_signature=output_signature
        )
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
