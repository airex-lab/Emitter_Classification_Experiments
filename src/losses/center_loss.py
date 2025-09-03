#!/usr/bin/env python3
"""
Center loss for clustering and feature learning.
"""

import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss for clustering."""
    
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, dim))
    
    def forward(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: Feature embeddings [B, D]
            y: Labels [B]
            
        Returns:
            Center loss
        """
        return (f - self.centers[y]).pow(2).sum(1).mean() 