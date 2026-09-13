# Roccas Raveendran - 500853856
# Sequential Recommender System - Custom Training Loop & Loss Functions

import tensorflow as tf
import numpy as np

class RecommenderTrainer: 
    
    def __init__(self, model, train_dataset, val_dataset, optimizer=None, loss_type='bpr', temperature=0.1):
        print(f"Initializing Custom Training Loop with {loss_type.upper()} Loss...")
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_type = loss_type
        self.temperature = temperature
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function 
    def _train_step(self, inputs, targets):
        
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            user_state = self.model(inputs['input_sequence'], training=True) 
            
            if self.loss_type == 'bpr':
                # Bayesian Personalized Ranking (1-to-1 Pairwise)
                pos_emb = self.model.item_embedding(targets['positive_target'])
                neg_emb = self.model.item_embedding(targets['negative_target'])
                
                pos_scores = tf.reduce_sum(user_state * pos_emb, axis=-1)
                neg_scores = tf.reduce_sum(user_state * neg_emb, axis=-1)
                
                loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(pos_scores - neg_scores) + 1e-9))
                
            elif self.loss_type == 'infonce':
                # InfoNCE Contrastive Loss (In-Batch Negatives)
                batch_pos_emb = self.model.item_embedding(targets['positive_target'])
                
                # Matrix multiplication creates a (batch, batch) grid of scores. 
                # Diagonals are true pairs, off-diagonals serve as negative contrastive examples.
                logits = tf.matmul(user_state, batch_pos_emb, transpose_b=True) / self.temperature
                labels = tf.range(tf.shape(user_state)[0])
                
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
            
        # Backpropagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def train(self, epochs=50): 
        print(f"Beginning Training for {epochs} Epochs...")
        history = {'loss': []} 
        
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            
            for inputs, targets in self.train_dataset:
                loss = self._train_step(inputs, targets) 
                total_loss += float(loss) 
                steps += 1
                
            avg_loss = total_loss / steps 
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} | {self.loss_type.upper()} Loss: {avg_loss:.4f}")
            
        return history
