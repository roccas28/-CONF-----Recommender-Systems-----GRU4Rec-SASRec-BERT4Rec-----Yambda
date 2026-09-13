# Benchmarking Bidirectional and Causal Transformers at Scale: An Extended Convergence Analysis
**Course:**  Conference Submission Ext.
**Student:** Roccas Raveendran  
**Student ID:** 500853856  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview
This project performs a detailed empirical analysis of Transformer-based architectures versus Recurrent Neural Networks (RNNs) for sequential next-item prediction. Utilizing an expanded 500-million interaction subset of the "Yambda" dataset (Yandex), an out-of-core ETL pipeline was engineered to process massive flat logs into 100,000 chronological user sequences. The project strictly evaluates **SASRec (Causal Transformer)** and **BERT4Rec (Bidirectional Transformer)** against **GRU4Rec** using Bayesian Personalized Ranking (BPR) and InfoNCE contrastive loss functions. To eliminate data leakage, models are trained using a Leave-One-Out (LOO) temporal split[cite: 15]. A targeted static ablation study maps architectural sensitivities over extended training horizons (50-100 epochs) utilizing decoupled weight decay (AdamW). Models are graded on retrieval accuracy (Hit Rate @ 10) and ranking quality (NDCG @ 10) against a competitive statistical Popularity baseline[cite: 15].

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
The codebase is modularized into nine Python scripts and three designated directories to separate data engineering, static ablation, model architecture, and final evaluation:

### 1. `utils.py`
Handles memory-optimized data ingestion and mapping out-of-core[cite: 15].
* Extracts statistical EDA samples without overloading RAM[cite: 15].
* Executes DuckDB SQL aggregations to group 500M flat rows into user arrays directly on the SSD.
* Generates the global label-encoding dictionary to dynamically map sparse items to contiguous integers[cite: 15].

### 2. `plots.py`
Generates all visualization artifacts for the final report[cite: 15].
* **EDA Plots:** Played ratio distributions, long-tail item popularity, event types, and sequence length variance[cite: 15].
* **Ablation Plots:** Generates isolated line graphs capturing metric variances across dropout and learning rates.
* **Training Plots:** Comparative loss curves across extended 50-100 epoch horizons for SASRec, BERT4Rec, and GRU4Rec.
* **Evaluation Plots:** Grouped bar charts visualizing final HR@10 and NDCG@10 metrics[cite: 15].

### 3. `main_etl.py`
The primary Data Engineering orchestrator[cite: 15].
* Triggers EDA metric extraction[cite: 15].
* Streams the DuckDB grouped data in discrete PyArrow batches[cite: 15].
* Applies the **Leave-One-Out (LOO) Temporal Split** (Train / Validation / Test)[cite: 15].
* Exports the final padded, neural-network-ready arrays to `processed_sequences.parquet` using an optimized context window of 70[cite: 15].

### 4. `dataset.py`
Manages the TensorFlow `tf.data.Dataset` pipeline for GPU streaming[cite: 15].
* **Padding & Masking:** Enforces variable context windows, dynamically injecting the `[MASK]` token during batch generation to facilitate BERT4Rec's bidirectional Cloze objective.
* **Strict Negative Sampling:** Utilizes a custom Python generator and Hash Sets to perform collision checks for pairwise loss mapping[cite: 15].

### 5. `model.py`
Defines the deep learning architectures using the `tf.keras.Model` API[cite: 15].
* **`SASRec`**: A Transformer architecture utilizing Multi-Head Attention and a **Causal Mask** to strictly enforce causal learning[cite: 15].
* **`BERT4Rec`**: A bidirectional Transformer architecture utilizing unmasked self-attention to predict masked central sequence items.
* **`GRU4Rec`**: The recurrent baseline utilizing update and reset gates to process sequential arrays left-to-right[cite: 15].

### 6. `train.py`
Contains the custom training loops and mathematical gradient descent logic[cite: 15].
* Computes **Bayesian Personalized Ranking (BPR)** and **InfoNCE Contrastive Loss**.
* Integrates both standard `Adam` and decoupled `AdamW` optimization steps via `tf.GradientTape`[cite: 15].

### 7. `tune.py`
The Static Ablation Orchestrator[cite: 15].
* Evaluates static permutations across Spatial Dropout Rates (0.1, 0.2, 0.3) and Learning Rates (0.001, 0.005) against a maximum locked batch size of 256.
* Leverages explicit calls to `tf.keras.backend.clear_session()` between runs to clear Video RAM (VRAM)[cite: 15].

### 8. `main_train.py`
The orchestrator for the Machine Learning phase[cite: 15].
* Configures SASRec/BERT4Rec for low-capacity builds (d=32) and GRU4Rec for high-capacity builds (d=256)[cite: 15].
* Builds and trains the models sequentially across extended 50-epoch stress tests to evaluate terminal generalization.
* Saves the final `.h5` model weights to disk and generates comparative training visual summaries[cite: 15].

### 9. `evaluate.py`
Executes the strict 100-item ranking simulation for model grading[cite: 15].
* Rebuilds the models according to the optimized structural configurations and loads the trained weights[cite: 15].
* Calculates **Hit Rate (HR@10)**, **Normalized Discounted Cumulative Gain (NDCG@10)**, and compares them against a purely statistical **Popularity Baseline**[cite: 15].

---

### Directories

* **`Data/`**: **[IMPORTANT]** This folder is deliberately left empty in the submission zip file due to size constraints. The `flat_multievent_500m.parquet` file must be downloaded and extracted here before running the code.
* **`output/`**: Destination for the processed training/validation/testing matrices and all generated `.png` plots[cite: 15].
* **`saved_models/`**: Checkpoint directory where the `.h5` tensor weights are saved post-training[cite: 15].

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Installation & Requirements
This project requires **Python 3.10+**.

1. **Install Dependencies:**
   Ensure you have the necessary data engineering and deep learning libraries installed by running[cite: 15]:
   ```bash
   pip install -r requirements.txt

## Execution Workflow
   To reproduce the entire pipeline from scratch, execute the following order:
   ```bash
   python main_etl.py     # Step 1: Run Data Engineering & Preprocessing
   python tune.py         # Step 2: Run Static Ablation Study
   python main_train.py   # Step 3: Train Optimized Models for 100 Epochs
   python evaluate.py     # Step 4: Run Ranking Simulations & Final Evaluation