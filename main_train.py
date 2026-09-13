# Roccas Raveendran - 500853856
# Sequential Recommender System - Machine Learning Orchestrator (Extended Convergence)

import os
import time
import tensorflow as tf

# Custom Modules
import dataset as ds
import model as mdl
import train as tr
import plots as pt

def run():
    print("MRP Project: Sequential Recommender ML Training Phase (500M Scale)")
    
    # Ensure checkpoint directory exists -> Prevents crash during final weight saving[cite: 11]
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    start_time = time.time()
    
    # --- PHASE 1: Data Ingestion ---
    print("\n--- Initializing Data Pipeline ---")
    # Hyperparameters: max_len=70 restricts context to last 70 interactions[cite: 11]
    loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=70)
    
    # Generate identical pipelines -> Shuffling enabled for training, disabled for testing[cite: 11]
    # Batch size locked to 256 for static ablation stability
    train_dataset = loader.generate_tf_dataset(batch_size=256, is_training=True)
    test_dataset = loader.generate_tf_dataset(batch_size=256, is_training=False)
    
    # Dynamically extract the vocab_size from the loader[cite: 11]
    # The +1 is mathematically mandatory to account for the '0' padding token![cite: 11]
    dynamic_vocab_size = loader.total_items + 1

    # Define Optimizers for Static Ablation 
    # Standard Adam for baseline, AdamW (Decoupled Weight Decay) for deep Transformers
    optimal_optimizer_gru = tf.keras.optimizers.Adam(learning_rate=0.005)
    optimal_optimizer_sasrec = tf.keras.optimizers.AdamW(learning_rate=0.005, weight_decay=1e-4)
    optimal_optimizer_bert4rec = tf.keras.optimizers.AdamW(learning_rate=0.005, weight_decay=1e-4)
    
    # --- PHASE 2: Train SASRec (Causal Transformer) ---
    print("\n--- Building & Training SASRec ---")
    sasrec_model = mdl.SASRec(
        vocab_size=dynamic_vocab_size, 
        max_len=70, 
        embed_dim=32, 
        num_heads=2, 
        num_blocks=2, 
        dropout_rate=0.2
    )
    
    sasrec_trainer = tr.RecommenderTrainer(model=sasrec_model, train_dataset=train_dataset, val_dataset=test_dataset, optimizer=optimal_optimizer_sasrec)
    
    # Extended Training Horizon: 50 Epochs for terminal generalization testing
    sasrec_history = sasrec_trainer.train(epochs=100)
    
    pt.plot_training_loss(sasrec_history, model_name="SASRec")
    sasrec_model.save_weights('./saved_models/sasrec_final.weights.h5')

    # --- PHASE 3: Train BERT4Rec (Bidirectional Transformer) ---
    print("\n--- Building & Training BERT4Rec ---")
    # Instantiate the BERT4Rec Transformer -> Requires bidirectional Cloze context
    bert4rec_model = mdl.BERT4Rec(
        vocab_size=dynamic_vocab_size, 
        max_len=70, 
        embed_dim=32, 
        num_heads=2, 
        num_blocks=2, 
        dropout_rate=0.2
    )
    
    # Note: Ensure RecommenderTrainer is equipped to handle InfoNCE loss mapping for BERT4Rec
    bert4rec_trainer = tr.RecommenderTrainer(model=bert4rec_model, train_dataset=train_dataset, val_dataset=test_dataset, optimizer=optimal_optimizer_bert4rec)
    
    bert4rec_history = bert4rec_trainer.train(epochs=100)
    
    pt.plot_training_loss(bert4rec_history, model_name="BERT4Rec")
    bert4rec_model.save_weights('./saved_models/bert4rec_final.weights.h5')
    
    # --- PHASE 4: Train GRU4Rec (Recurrent Baseline) ---
    print("\n--- Building & Training GRU4Rec ---")
    # Uses high-capacity embed_dim=256 as proven optimal for RNNs during tuning[cite: 11]
    gru_model = mdl.GRU4Rec(
        vocab_size=dynamic_vocab_size, 
        max_len=70, 
        embed_dim=256, 
        gru_units=256, 
        dropout_rate=0.2
    )
    
    gru_trainer = tr.RecommenderTrainer(model=gru_model, train_dataset=train_dataset, val_dataset=test_dataset, optimizer=optimal_optimizer_gru)
    
    gru_history = gru_trainer.train(epochs=100)
    
    pt.plot_training_loss(gru_history, model_name="GRU4Rec")
    gru_model.save_weights('./saved_models/gru4rec_final.weights.h5')
    
    # --- PHASE 5: Comparative Plotting ---
    print("\n--- Generating Comparative Visualizations ---")
    # Execute comparative plot to show which architecture converged best over 50 epochs[cite: 11]
    pt.plot_comparative_loss(sasrec_history, gru_history, bert4rec_history=bert4rec_history)
    
    print(f"\nML Pipeline Complete! Total Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    run() # Execute main block
