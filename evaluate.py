# Roccas Raveendran - 500853856
# Sequential Recommender System - Evaluation Metrics (HR@10 & NDCG@10)

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import dataset as ds
import model as mdl
import plots as pt

class RecommenderEvaluator: 
    
    def __init__(self, model, dataset_loader, total_items, model_type='causal'):
        print(f"Initializing Evaluator for {model.__class__.__name__}...")
        self.model = model
        self.loader = dataset_loader
        self.total_items = total_items
        self.model_type = model_type
        
    def _calculate_ndcg(self, rank): 
        return 1.0 / np.log2(rank + 1)

    def evaluate_model(self, top_k=10, num_negatives=99): 
        hits = 0.0
        ndcgs = 0.0
        total_users = 0
        
        # Ensures the correct mask token logic is applied during testing
        test_ds = self.loader.generate_tf_dataset(batch_size=256, is_training=False, model_type=self.model_type)
        
        for inputs, targets in tqdm(test_ds, desc="Evaluating Batches"):
            user_states = self.model(inputs['input_sequence'], training=False)
            
            for i in range(len(user_states)):
                current_user_state = user_states[i] 
                true_target = targets['positive_target'][i]
                
                test_negatives = np.random.randint(1, self.total_items + 1, size=num_negatives)
                items_to_rank = np.append(true_target, test_negatives)
                
                item_embs = self.model.item_embedding(items_to_rank) 
                scores = tf.reduce_sum(current_user_state * item_embs, axis=-1).numpy()
                
                rank = (scores > scores[0]).sum() + 1
                
                if rank <= top_k: 
                    hits += 1.0
                    ndcgs += self._calculate_ndcg(rank)
                    
                total_users += 1
                
        hr_at_10 = hits / total_users
        ndcg_at_10 = ndcgs / total_users
        
        print(f"\n{self.model.__class__.__name__} Results | HR@{top_k}: {hr_at_10:.4f} | NDCG@{top_k}: {ndcg_at_10:.4f}")
        return hr_at_10, ndcg_at_10

def evaluate_popularity_baseline(parquet_path, top_k=10): # Statistical baseline that ignores deep learning
    print("\nEvaluating Global Popularity Baseline...")
    
    # 1. Read data -> We need to find what the most globally popular tracks are
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    # Flatten the training sequences to count how many times every song was played globally
    all_train_items = np.concatenate(df['train_sequence'].tolist())
    
    # Find the top 10 most popular songs in the entire training set
    print("Calculating global item frequencies...")
    popular_items = pd.Series(all_train_items).value_counts().head(top_k).index.values
    
    hits = 0.0
    total_users = len(df)
    
    # 2. Evaluation -> For every user, ignore their history and just recommend the global Top 10
    print("Scoring Popularity Baseline...")
    for test_item in df['test_item']:
        if test_item in popular_items:
            hits += 1.0 # The user actually listened to one of the global Top 10
            
    hr_at_10 = hits / total_users
    
    # Note: NDCG is technically computable here, but HR is the standard benchmark for the Pop baseline
    print(f"Popularity Baseline Results | HR@{top_k}: {hr_at_10:.4f}")
    return hr_at_10

def run():
    print("MRP Project: Sequential Recommender Evaluation Phase")
    
    loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=70)
    dynamic_vocab_size = loader.total_items + 2 # +1 for padding, +1 for BERT [MASK]
    
    # --- 1. Evaluate SASRec ---
    sasrec_model = mdl.SASRec(vocab_size=dynamic_vocab_size, max_len=70, embed_dim=32, num_heads=2, num_blocks=2)
    sasrec_model.build(input_shape=(None, 70)) 
    sasrec_model.load_weights('./saved_models/sasrec_final.weights.h5')
    sasrec_hr, sasrec_ndcg = RecommenderEvaluator(sasrec_model, loader, loader.total_items, 'causal').evaluate_model()

    # --- 2. Evaluate BERT4Rec ---
    bert_model = mdl.BERT4Rec(vocab_size=dynamic_vocab_size, max_len=70, embed_dim=32, num_heads=2, num_blocks=2)
    bert_model.build(input_shape=(None, 70)) 
    bert_model.load_weights('./saved_models/bert4rec_final.weights.h5')
    bert_hr, bert_ndcg = RecommenderEvaluator(bert_model, loader, loader.total_items, 'bert4rec').evaluate_model()
    
    # --- 3. Evaluate GRU4Rec ---
    gru_model = mdl.GRU4Rec(vocab_size=dynamic_vocab_size, max_len=70, embed_dim=256, gru_units=256)
    gru_model.build(input_shape=(None, 70))
    gru_model.load_weights('./saved_models/gru4rec_final.weights.h5')
    gru_hr, gru_ndcg = RecommenderEvaluator(gru_model, loader, loader.total_items, 'causal').evaluate_model()
    
    print("\n==========================================")
    print("FINAL 500M COMPARATIVE RESULTS (HR@10 & NDCG@10)")
    print("==========================================")
    print(f"1. SASRec (Transformer) : HR = {sasrec_hr:.4f} | NDCG = {sasrec_ndcg:.4f}")
    print(f"2. BERT4Rec (Bi-Trans)  : HR = {bert_hr:.4f} | NDCG = {bert_ndcg:.4f}")
    print(f"3. GRU4Rec (RNN)        : HR = {gru_hr:.4f} | NDCG = {gru_ndcg:.4f}")
    print("==========================================")

if __name__ == "__main__":
    run()
