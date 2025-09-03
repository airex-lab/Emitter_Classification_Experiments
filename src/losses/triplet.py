#!/usr/bin/env python3
"""
Triplet loss functions for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletMarginLoss(nn.Module):
    """Standard triplet margin loss."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [B, D]
            positive: Positive embeddings [B, D] 
            negative: Negative embeddings [B, D]
            
        Returns:
            Triplet loss
        """
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class SemiHardTripletLoss(nn.Module):
    """Semi-hard triplet loss that finds semi-hard negatives."""
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, feat: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Normalized embeddings [B, D]
            lbl: Labels [B]
            
        Returns:
            Semi-hard triplet loss
        """
        # Cosine distance (since features are L2-normalized)
        dist = 1 - feat @ feat.t()
        
        # Positive and negative masks
        pos_mask = (lbl[:, None] == lbl[None, :]).bool()
        neg_mask = ~pos_mask
        
        # Hardest positive distance per anchor
        pos_d = dist.masked_fill(~pos_mask, 1e9).min(1).values
        
        # Semi-hard negative distance per anchor
        # Find negatives that are farther than positives but not too far
        bigger = dist + (pos_d[:, None] - dist) * (~neg_mask)
        neg_d = bigger.masked_fill(~neg_mask, 1e9).min(1).values
        
        # Triplet loss
        loss = F.relu(pos_d - neg_d + self.margin)
        return loss.mean() 