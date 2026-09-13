# Roccas Raveendran - 500853856
# Sequential Recommender System - Targeted Static Ablation Study

import pandas as pd
import tensorflow as tf
import time
import os

# Custom Modules
import dataset as ds
import model as mdl
import train as tr
import evaluate as ev
import plots as pt

def run_static_ablation():
    print("Initiating Targeted Static Ablation Study (SASRec vs GRU4Rec vs BERT4Rec)...")
    
    # Locked constraints to protect VRAM during the extended runs
    LOCKED_SEQ_LEN = 70
    LOCKED_BATCH_SIZE = 256
    
    # Discrete Ablation Parameters
    dropout_rates = [0.1, 0.2, 0.3]
    learning_rates = [0.001, 0.005]
    models_to_test = ['GRU4Rec', 'SASRec', 'BERT4Rec']
    loss_types = ['bpr', 'infonce']
    optimizers_to_test = ['Adam', 'AdamW']
    
    results = [] 
    loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=LOCKED_SEQ_LEN)
    dynamic_vocab_size = loader.total_items + 2 # +1 for padding, +1 for BERT [MASK]
    
    for arch in models_to_test:
        model_type_flag = 'bert4rec' if arch == 'BERT4Rec' else 'causal'
        train_dataset = loader.generate_tf_dataset(batch_size=LOCKED_BATCH_SIZE, is_training=True, model_type=model_type_flag)
        
        for lr in learning_rates:
            for drop in dropout_rates:
                for loss_fn in loss_types:
                    # Dynamically assign optimizer -> AdamW for Transformers, Adam for RNN
                    opt_name = 'AdamW' if 'Rec' in arch and arch != 'GRU4Rec' else 'Adam'
                    
                    print(f"\n=== Evaluating | {arch} | LR: {lr} | Drop: {drop} | Opt: {opt_name} | Loss: {loss_fn.upper()} ===")
                    
                    # Instantiate chosen architecture
                    if arch == 'SASRec':
                        model = mdl.SASRec(vocab_size=dynamic_vocab_size, max_len=LOCKED_SEQ_LEN, embed_dim=32, num_heads=2, num_blocks=2, dropout_rate=drop)
                    elif arch == 'BERT4Rec':
                        model = mdl.BERT4Rec(vocab_size=dynamic_vocab_size, max_len=LOCKED_SEQ_LEN, embed_dim=32, num_heads=2, num_blocks=2, dropout_rate=drop)
                    else:
                        model = mdl.GRU4Rec(vocab_size=dynamic_vocab_size, max_len=LOCKED_SEQ_LEN, embed_dim=256, gru_units=256, dropout_rate=drop)
                    
                    if opt_name == 'AdamW':
                        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
                    else:
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        
                    trainer = tr.RecommenderTrainer(model=model, train_dataset=train_dataset, val_dataset=None, optimizer=optimizer, loss_type=loss_fn)
                    trainer.train(epochs=10) # 10 epochs for rapid ablation testing
                    
                    # Evaluate model
                    model.build(input_shape=(None, LOCKED_SEQ_LEN))
                    evaluator = ev.RecommenderEvaluator(model=model, dataset_loader=loader, total_items=loader.total_items, model_type=model_type_flag)
                    hr, ndcg = evaluator.evaluate_model(top_k=10)
                    
                    results.append({
                        'Model': arch,
                        'Dropout': drop,
                        'Learning_Rate': lr,
                        'Optimizer': opt_name,
                        'Loss': loss_fn.upper(),
                        'HR@10': float(hr),
                        'NDCG@10': float(ndcg)
                    })
                    
                    tf.keras.backend.clear_session()
            
    # Export Results
    print("\n--- Ablation Study Complete. Generating Artifacts ---")
    results_df = pd.DataFrame(results)
    results_df.to_csv('./output/static_ablation_results.csv', index=False)

    pt.plot_ablation_results(results_df, metric='HR@10')
    pt.plot_ablation_line_graphs(results_df, metric='HR@10')

if __name__ == "__main__":
    run_static_ablation()
