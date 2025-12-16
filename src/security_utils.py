# src/security_utils.py
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression

def apply_label_flipping(y, poison_rate, target_class=1):
    """
    ATTACK: Flips labels of the target class (Fraud) to the opposite class.
    target_class=1 means we turn 'Fraud' into 'Normal' to make the model blind.
    """
    y_poisoned = y.copy()
    target_indices = np.where(y == target_class)[0]
    n_poison = int(len(target_indices) * poison_rate)
    
    if n_poison > 0:
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        y_poisoned[poison_indices] = 1 - target_class # Flip 1 to 0
    return y_poisoned

def apply_roni(X_train, y_train, X_val, y_val, threshold=0.01):
    """
    DEFENSE: Reject on Negative Impact (RONI).
    Checks if a batch of data significantly hurts model performance.
    """
    # 1. Train a baseline model on a small clean validation set
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_val, y_val)
    baseline_recall = recall_score(y_val, baseline_model.predict(X_val))
    
    # 2. Simplified RONI: We check samples in small chunks
    # For this educational demo, we'll simulate 'Scan and Reject'
    keep_indices = []
    chunk_size = 100 
    
    for i in range(0, len(X_train), chunk_size):
        X_chunk = X_train[i:i+chunk_size]
        y_chunk = y_train[i:i+chunk_size]
        
        # Test performance with this chunk added
        test_model = LogisticRegression(max_iter=1000)
        X_combined = np.vstack([X_val, X_chunk])
        y_combined = np.append(y_val, y_chunk)
        test_model.fit(X_combined, y_combined)
        
        new_recall = recall_score(y_val, test_model.predict(X_val))
        
        # If recall drops significantly, reject the chunk
        if (baseline_recall - new_recall) < threshold:
            keep_indices.extend(range(i, min(i + chunk_size, len(X_train))))
            
    return X_train[keep_indices], y_train[keep_indices], len(X_train) - len(keep_indices)