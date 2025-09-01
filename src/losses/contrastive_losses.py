#!/usr/bin/env python3
"""
Contrastive loss functions for supervised and unsupervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_local: torch.Tensor, z_global: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_local: Local embeddings [B, D] (with gradients)
            z_global: Global embeddings [B*world, D] (detached)
            targets: Target indices [B]
            
        Returns:
            InfoNCE loss
        """
        # Compute logits
        logits = (z_local @ z_global.t()) / self.temperature
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_local: torch.Tensor, y_local: torch.Tensor,
                z_global: torch.Tensor, y_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_local: Local embeddings [B, D] (with gradients)
            y_local: Local labels [B]
            z_global: Global embeddings [B*world, D] (detached)
            y_global: Global labels [B*world]
            
        Returns:
            Supervised contrastive loss
        """
        B = z_local.size(0)
        device = z_local.device
        
        # Expand labels for mask computation
        y_local_exp = y_local.view(-1, 1)  # [B, 1]
        y_global_exp = y_global.view(1, -1)  # [1, B*world]
        
        # Positive mask (same label, exclude self)
        mask = (y_local_exp == y_global_exp).float().to(device)  # [B, B*world]
        
        # Self mask to exclude diagonal
        self_mask = torch.eye(B, device=device)
        self_mask = torch.cat([self_mask] * (z_global.size(0) // B), dim=1)  # [B, B*world]
        mask = mask * (1 - self_mask)  # exclude self
        
        # Compute logits
        logits = (z_local @ z_global.t()) / self.temperature  # [B, B*world]
        
        # For numerical stability, subtract max per row
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max
        
        # Positive logits for numerator
        pos_logits = torch.exp(logits) * mask
        pos_sum = pos_logits.sum(1, keepdim=True)
        
        # Denominator: all exp logits
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(1, keepdim=True)
        
        # Log probability, averaged over positives
        log_prob = torch.log(pos_sum / denom + 1e-9)  # avoid division by zero
        mean_log_prob_pos = log_prob.sum(1) / (mask.sum(1) + 1e-9)
        
        # Loss: negative mean log probability
        loss = -mean_log_prob_pos
        valid = (mask.sum(1) > 0)  # ignore anchors with no positives
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        return loss[valid].mean()

class NTXentLoss(nn.Module):
    """Numerically stable NT-Xent loss."""
    
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: L2-normalized embeddings [B, D]
            y: Labels [B]
            
        Returns:
            NT-Xent loss
        """
        # Pairwise cosine similarity
        sim = z @ z.t() / self.temperature  # [B, B]
        
        # Mask self-similarity without inplace operations
        self_mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(self_mask, -float("inf"))
        
        # Log-softmax (row-wise) - numerically stable
        max_row = sim.max(dim=1, keepdim=True).values
        log_soft = (sim - max_row).exp()
        log_soft = log_soft / log_soft.sum(dim=1, keepdim=True)
        log_soft = (log_soft + self.eps).log()
        
        # Positive mask
        pos = (y[:, None] == y[None, :]).float()
        pos = pos.masked_fill(self_mask, 0)  # avoid diagonal write
        
        # Compute loss
        denom = pos.sum(1) + self.eps
        loss = -(log_soft * pos).sum(1) / denom
        return loss.mean() 