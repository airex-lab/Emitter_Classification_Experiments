#!/usr/bin/env python3
"""
Evaluation utilities for emitter classification experiments.
"""

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm.
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster assignments
        
    Returns:
        Clustering accuracy
    """
    cm = pd.crosstab(y_pred, y_true)
    r, c = linear_sum_assignment(-cm.values)
    return cm.values[r, c].sum() / len(y_true)

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, x_test: np.ndarray, 
                  y_test: np.ndarray, device: str = 'cuda') -> float:
    """
    Evaluate model using clustering accuracy.
    
    Args:
        model: Trained model
        x_test: Test features
        y_test: Test labels
        device: Device to run evaluation on
        
    Returns:
        Clustering accuracy
    """
    model.eval()
    x_tensor = torch.tensor(x_test, device=device)
    embeddings = model(x_tensor).cpu().numpy()
    
    # Perform clustering
    k = len(np.unique(y_test))
    cluster_ids = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)
    
    return clustering_accuracy(y_test, cluster_ids)

def save_results(results: Dict[str, Any], filepath: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save results
    """
    import json
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

def compute_embeddings(model: torch.nn.Module, x: np.ndarray, 
                      device: str = 'cuda') -> np.ndarray:
    """
    Compute embeddings for given data.
    
    Args:
        model: Trained model
        x: Input features
        device: Device to run computation on
        
    Returns:
        Embeddings
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, device=device)
        embeddings = model(x_tensor).cpu().numpy()
    return embeddings

def evaluate_multiple_dims(model_class, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray,
                          embedding_dims: list, **model_kwargs) -> Dict[int, float]:
    """
    Evaluate model with multiple embedding dimensions.
    
    Args:
        model_class: Model class to instantiate
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        embedding_dims: List of embedding dimensions to test
        **model_kwargs: Additional model parameters
        
    Returns:
        Dictionary mapping embedding dimension to accuracy
    """
    results = {}
    
    for dim in embedding_dims:
        model = model_class(embedding_dim=dim, **model_kwargs)
        acc = evaluate_model(model, x_test, y_test)
        results[dim] = acc
        
    return results 