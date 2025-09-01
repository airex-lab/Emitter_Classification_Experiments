#!/usr/bin/env python3
"""
Residual network model for emitter classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with dropout for regularization."""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class EmitterEncoder(nn.Module):
    """Residual network encoder for emitter classification."""
    
    def __init__(self, in_dim: int, emb_dim: int, hidden_dim: int = 64, 
                 num_blocks: int = 2, dropout: float = 0.3):
        """
        Args:
            in_dim: Input feature dimension
            emb_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = [nn.Linear(in_dim, hidden_dim)]
        
        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
            
        layers.append(nn.Linear(hidden_dim, emb_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1) 