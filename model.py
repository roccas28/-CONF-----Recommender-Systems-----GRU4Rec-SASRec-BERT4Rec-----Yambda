# Roccas Raveendran - 500853856
# Sequential Recommender System - Core Neural Architectures (SASRec, BERT4Rec, & GRU4Rec)

import tensorflow as tf
from tensorflow.keras import layers, Model

# ==========================================
# TRANSFORMER COMPONENTS
# ==========================================

class TransformerBlock(layers.Layer): 
    def __init__(self, embed_dim, num_heads, dropout_rate=0.2, use_causal_mask=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.use_causal_mask = use_causal_mask
        
        # Multi-Head Attention Layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
        
        # Feed-Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Applies causal mask for SASRec, leaves unmasked for BERT4Rec
        attn_output = self.attention(inputs, inputs, use_causal_mask=self.use_causal_mask, training=training)
        
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class SASRec(Model): 
    def __init__(self, vocab_size, max_len=70, embed_dim=32, num_heads=2, num_blocks=2, dropout_rate=0.2, **kwargs):
        super(SASRec, self).__init__(**kwargs)
        
        self.max_len = max_len
        self.vocab_size = vocab_size 
        
        self.item_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.emb_dropout = layers.Dropout(dropout_rate)
        
        # Stack blocks WITH causal masking
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, dropout_rate, use_causal_mask=True) for _ in range(num_blocks)
        ]

    def call(self, inputs, training=False):
        seq_embeddings = self.item_embedding(inputs) 
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        pos_embeddings = self.pos_embedding(positions) 
        
        x = seq_embeddings + pos_embeddings
        x = self.emb_dropout(x, training=training)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
            
        return x[:, -1, :] # Extract strictly the final state for next-item prediction


class BERT4Rec(Model): 
    def __init__(self, vocab_size, max_len=70, embed_dim=32, num_heads=2, num_blocks=2, dropout_rate=0.2, **kwargs):
        super(BERT4Rec, self).__init__(**kwargs)
        
        self.max_len = max_len
        self.vocab_size = vocab_size 
        
        self.item_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.emb_dropout = layers.Dropout(dropout_rate)
        
        # Stack blocks WITHOUT causal masking -> Allows bidirectional context
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, dropout_rate, use_causal_mask=False) for _ in range(num_blocks)
        ]

    def call(self, inputs, training=False):
        seq_embeddings = self.item_embedding(inputs) 
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        pos_embeddings = self.pos_embedding(positions) 
        
        x = seq_embeddings + pos_embeddings
        x = self.emb_dropout(x, training=training)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
            
        return x[:, -1, :] # Extracts the final state where the [MASK] token is placed during evaluation

# ==========================================
# RECURRENT COMPONENTS (GRU4Rec)
# ==========================================

class GRU4Rec(Model): 
    def __init__(self, vocab_size, max_len=70, embed_dim=256, gru_units=256, dropout_rate=0.2, **kwargs):
        super(GRU4Rec, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.item_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.gru = layers.GRU(units=gru_units, return_sequences=False, dropout=dropout_rate)

    def call(self, inputs, training=False):
        x = self.item_embedding(inputs)
        return self.gru(x, training=training)
