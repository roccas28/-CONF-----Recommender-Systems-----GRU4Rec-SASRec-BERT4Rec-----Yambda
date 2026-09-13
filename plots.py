# Roccas Raveendran - 500853856
# Sequential Recommender System - Exploratory Data Analysis (EDA) $ Evaluation Plots

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global aesthetics -> Matches formatting from previous clinical/deep learning projects
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def plot_played_ratio_distribution(df): # Plots implicit feedback completion rates
    print("Generating played ratio histogram...")
    plt.figure(figsize=(10, 6)) # Initialize figure size -> Standard widescreen aspect
    
    # Plot histogram -> Hyperparameter: bins=50 provides enough granularity to see the bimodal peaks
    sns.histplot(data=df, x="played_ratio_pct", bins=50, color="b", kde=False)
    
    # Draw reference lines for feedback thresholds -> Visually justifies the Hard Filter bounds
    plt.axvline(x=10, color='r', linestyle='--', label="Skip Threshold (<10%)")
    plt.axvline(x=70, color='g', linestyle='--', label="Listen Threshold (>70%)")
    
    plt.title("Distribution of Track Completion Rates (Implicit Feedback)")
    plt.ylabel("Frequency")
    plt.xlabel("Played Ratio (%)")
    plt.legend(loc='upper center')
    plt.tight_layout() # Compress margins -> Prevents label cutoff
    plt.savefig("./Output/eda_played_ratio.png", dpi=300) # Export high-res PNG
    print("Saved eda_played_ratio.png")

def plot_long_tail_popularity(df): # Plots item popularity to justify negative sampling
    print("Generating long-tail popularity curve...")
    
    item_counts = df['item_id'].value_counts().values # Count frequency of each item -> Arrays automatically sorted descending
    
    plt.figure(figsize=(10, 6)) # Initialize figure size
    plt.plot(item_counts, color="purple", linewidth=2) # Line plot showcases the steep drop-off
    
    plt.title("Long-Tail Item Popularity Distribution")
    plt.ylabel("Number of Interactions")
    plt.xlabel("Items (Sorted by Popularity Rank)")
    plt.yscale('log') # Hyperparameter/Design: Log scale is mandatory -> Condenses 900k items into readable variance
    plt.tight_layout()
    plt.savefig("./Output/eda_long_tail.png", dpi=300)
    print("Saved eda_long_tail.png")

def plot_sequence_length_distribution(df_lengths): # Plots distribution of user history lengths
    print("Generating sequence length distribution...")
    plt.figure(figsize=(10, 6))
    
    # Plot histogram -> Hyperparameter: bins=100 provides fine detail on the extreme right-skewed outliers
    sns.histplot(data=df_lengths, x="seq_length", bins=100, color="teal", kde=False)
    
    plt.title("Distribution of User History Lengths")
    plt.ylabel("Number of Users (Log Scale)")
    plt.xlabel("Sequence Length (Total Interactions)")
    plt.yscale('log') # Log scale -> Highlights extreme power users (ex: users with 10k+ interactions)
    plt.tight_layout()
    plt.savefig("./output/eda_sequence_lengths.png", dpi=300)
    print("Saved eda_sequence_lengths.png")

def plot_event_type_distribution(df_events): # Plots interaction composition breakdown
    print("Generating event type distribution...")
    plt.figure(figsize=(10, 6))
    
    df_events = df_events.sort_values('total_count', ascending=False) # Sort descending -> Creates clean waterfall aesthetic
    
    # Plot bar chart -> Visualizes ratio of passive vs active events
    sns.barplot(data=df_events, x="event_type", y="total_count", palette="viridis") 
    
    plt.title("Total Interactions by Event Type")
    plt.ylabel("Total Count (Tens of Millions)")
    plt.xlabel("Event Type")
    plt.tight_layout()
    plt.savefig("./output/eda_event_types.png", dpi=300)
    print("Saved eda_event_types.png")

def plot_training_loss(history, model_name="SASRec"): # Visualizes the descent of the loss function
    print(f"Generating Training Loss Curve for {model_name}...")
    plt.figure(figsize=(10, 6))
    
    # Plot the BPR loss over time -> Visually confirms if the network is actively learning the gradient
    sns.lineplot(x=range(1, len(history['loss']) + 1), y=history['loss'], color="red", marker='o')
    
    plt.title(f"{model_name} Training Curve (BPR Loss)")
    plt.ylabel("Bayesian Personalized Ranking (BPR) Loss")
    plt.xlabel("Training Epoch")
    plt.tight_layout()
    plt.savefig(f"./output/{model_name}_training_loss.png", dpi=300) # Save artifact for MRP report
    print(f"Saved {model_name}_training_loss.png")

def plot_comparative_loss(sasrec_history, gru_history, bert4rec_history=None):
    print("Generating Extended Comparative Training Loss Curve...")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x=range(1, len(sasrec_history['loss']) + 1), y=sasrec_history['loss'], 
                 color="blue", marker='o', label="SASRec (Causal Transformer)")
                 
    sns.lineplot(x=range(1, len(gru_history['loss']) + 1), y=gru_history['loss'], 
                 color="red", marker='s', label="GRU4Rec (RNN)")
                 
    if bert4rec_history:
        sns.lineplot(x=range(1, len(bert4rec_history['loss']) + 1), y=bert4rec_history['loss'], 
                     color="green", marker='^', label="BERT4Rec (Bidirectional Transformer)")
    
    plt.title("Terminal Convergence Comparison (50 Epochs)")
    plt.ylabel("Training Loss")
    plt.xlabel("Training Epoch")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("./output/comparative_training_loss.png", dpi=300) 
    print("Saved comparative_training_loss.png")

def plot_ablation_results(df, metric='HR@10'):
    print(f"Generating Static Ablation Bar Charts for {metric} (Split by Loss)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    models = ['SASRec', 'BERT4Rec', 'GRU4Rec']
    loss_palette = {'BPR': '#1f77b4', 'INFONCE': '#ff7f0e'} # Blue vs Orange

    for idx, arch in enumerate(models):
        arch_df = df[df['Model'] == arch]
        sns.barplot(
            data=arch_df, 
            x='Dropout', 
            y=metric, 
            hue='Loss', 
            palette=loss_palette, 
            ax=axes[idx]
        )
        axes[idx].set_title(f"{arch} Sensitivity")
        axes[idx].set_xlabel("Dropout Rate")
        axes[idx].set_ylabel(f"Achieved {metric}" if idx == 0 else "")
        
    plt.tight_layout()
    plt.savefig(f"./output/static_ablation_bars_{metric.replace('@', '_')}.png", dpi=300)
    print("Saved plot.")

def plot_ablation_line_graphs(df, metric='HR@10'): 
    print(f"Generating isolated line graphs for {metric} ablation trends...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color_map = {'SASRec': '#1f77b4', 'BERT4Rec': '#2ca02c', 'GRU4Rec': '#d62728'} 
    
    # --- GRAPH 1: Dropout Scaling ---
    drop_df = df.groupby(['Model', 'Dropout'])[metric].max().reset_index()
    sns.lineplot(data=drop_df, x='Dropout', y=metric, hue='Model', marker='o', palette=color_map, ax=axes[0])
    axes[0].set_title("Sensitivity to Dropout Rate")
    axes[0].set_xlabel("Dropout Rate")
    axes[0].set_ylabel(f"Maximum Achieved {metric}")
    
    # --- GRAPH 2: Learning Rate Scaling ---
    lr_df = df.groupby(['Model', 'Learning_Rate'])[metric].max().reset_index()
    lr_df['Learning_Rate'] = lr_df['Learning_Rate'].astype(str) 
    sns.lineplot(data=lr_df, x='Learning_Rate', y=metric, hue='Model', marker='^', palette=color_map, ax=axes[1])
    axes[1].set_title("Sensitivity to Learning Rate")
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    filename = metric.replace('@', '_')
    plt.savefig(f"./output/ablation_scaling_lines_{filename}.png", dpi=300)
    print(f"Saved ablation_scaling_lines_{filename}.png")
