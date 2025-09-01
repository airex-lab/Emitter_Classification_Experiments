#!/usr/bin/env python3
"""
Dataset classes for different training approaches in emitter classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Tuple, List, Optional

class TripletPDWDataset(Dataset):
    """Dataset that returns (anchor, positive, negative) triplets for triplet loss training."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: Feature array
            y: Label array
        """
        self.x = x
        self.y = y
        
        # Build label to index mapping
        self.lbl2idx = defaultdict(list)
        for i, lbl in enumerate(y):
            self.lbl2idx[lbl].append(i)
            
        # Check we have enough classes
        if len(self.lbl2idx) < 2:
            raise ValueError("Need at least 2 classes for triplet learning")
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Anchor
        anchor = torch.from_numpy(self.x[idx])
        anchor_label = self.y[idx]
        
        # Positive (same class as anchor)
        pos_idx = np.random.choice(self.lbl2idx[anchor_label])
        positive = torch.from_numpy(self.x[pos_idx])
        
        # Negative (different class)
        neg_labels = [l for l in self.lbl2idx if l != anchor_label]
        neg_label = np.random.choice(neg_labels)
        neg_idx = np.random.choice(self.lbl2idx[neg_label])
        negative = torch.from_numpy(self.x[neg_idx])
        
        return anchor, positive, negative

class PairPDWDataset(Dataset):
    """Dataset that returns (anchor, positive) pairs for contrastive learning."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: Feature array
            y: Label array
        """
        self.x = x
        self.y = y
        
        # Build label to index mapping
        self.lbl2idx = defaultdict(list)
        for i, lbl in enumerate(y):
            self.lbl2idx[lbl].append(i)
            
        # Check we have enough classes with multiple samples
        enough = [l for l, idxs in self.lbl2idx.items() if len(idxs) >= 2]
        if len(enough) < 2:
            raise ValueError("Need at least 2 labels with 2+ samples each for contrastive learning")
        self.labels_with_pairs = enough
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Anchor
        anchor = torch.from_numpy(self.x[idx])
        anchor_label = self.y[idx]
        
        # Positive (same class as anchor)
        pos_pool = self.lbl2idx[anchor_label]
        if len(pos_pool) == 1:
            pos_idx = pos_pool[0]
        else:
            choices = np.array(pos_pool, dtype=int)
            if choices.size > 1 and (idx in choices):
                choices = choices[choices != idx]
            pos_idx = np.random.choice(choices)
        
        positive = torch.from_numpy(self.x[pos_idx])
        return anchor, positive

class EmitterDataset(Dataset):
    """Simple dataset that returns (feature, label) pairs for supervised contrastive learning."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: Feature array
            y: Label array
        """
        self.x = x
        self.y = y
        
        # Build label to index mapping for validation
        self.lbl2idx = defaultdict(list)
        for i, lbl in enumerate(y):
            self.lbl2idx[lbl].append(i)
            
        # Check we have enough classes with multiple samples
        enough = [l for l, idxs in self.lbl2idx.items() if len(idxs) >= 2]
        if len(enough) < 2:
            raise ValueError("Need at least 2 labels with 2+ samples each for contrastive learning")
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])

class TripletPDWWithIndices(Dataset):
    """Dataset that returns (anchor, positive, negative) + their indices for advanced triplet learning."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: Feature array
            y: Label array
        """
        self.x = x
        self.y = y
        
        # Build label to index mapping
        self.lbl2idx = defaultdict(list)
        for i, lbl in enumerate(y):
            self.lbl2idx[lbl].append(i)
            
        if len(self.lbl2idx) < 2:
            raise ValueError("Need at least 2 classes for triplet learning")
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        # Anchor
        anchor_idx = idx
        anchor = torch.from_numpy(self.x[anchor_idx])
        anchor_label = self.y[anchor_idx]
        
        # Positive
        pos_idx = np.random.choice(self.lbl2idx[anchor_label])
        positive = torch.from_numpy(self.x[pos_idx])
        
        # Negative
        neg_labels = [l for l in self.lbl2idx if l != anchor_label]
        neg_label = np.random.choice(neg_labels)
        neg_idx = np.random.choice(self.lbl2idx[neg_label])
        negative = torch.from_numpy(self.x[neg_idx])
        
        return anchor, positive, negative, anchor_idx, pos_idx, neg_idx

class ClassBalancedSampler:
    """Custom sampler that ensures balanced class representation in batches."""
    
    def __init__(self, labels: np.ndarray, per_class: int = 2, shuffle: bool = True):
        """
        Args:
            labels: Array of labels
            per_class: Number of samples per class in each batch
            shuffle: Whether to shuffle within classes
        """
        self.labels = labels
        self.per_class = per_class
        self.shuffle = shuffle
        
        # Build label to index mapping
        self.lbl2idx = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.lbl2idx[lbl].append(idx)
    
    def get_indices(self) -> List[int]:
        """Get balanced indices for sampling."""
        indices = []
        
        # Get all unique labels
        unique_labels = list(self.lbl2idx.keys())
        
        # For each class, add samples
        for label in unique_labels:
            class_indices = self.lbl2idx[label]
            if self.shuffle:
                np.random.shuffle(class_indices)
            
            # Add samples in groups of per_class
            for i in range(0, len(class_indices), self.per_class):
                batch = class_indices[i:i + self.per_class]
                indices.extend(batch)
        
        return indices 