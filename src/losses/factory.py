#!/usr/bin/env python3
"""
Loss factory for creating different loss functions.
"""

import torch.nn as nn
from .triplet import TripletMarginLoss, SemiHardTripletLoss
from .contrastive_losses import InfoNCELoss, SupConLoss, NTXentLoss
from .center_loss import CenterLoss

def get_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('triplet', 'semi_hard', 'infonce', 'supcon', 'ntxent', 'center')
        **kwargs: Loss-specific parameters
        
    Returns:
        Loss function instance
    """
    if loss_type == 'triplet':
        return TripletMarginLoss(**kwargs)
    elif loss_type == 'semi_hard':
        return SemiHardTripletLoss(**kwargs)
    elif loss_type == 'infonce':
        return InfoNCELoss(**kwargs)
    elif loss_type == 'supcon':
        return SupConLoss(**kwargs)
    elif loss_type == 'ntxent':
        return NTXentLoss(**kwargs)
    elif loss_type == 'center':
        return CenterLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Available losses
AVAILABLE_LOSSES = {
    'triplet': TripletMarginLoss,
    'semi_hard': SemiHardTripletLoss,
    'infonce': InfoNCELoss,
    'supcon': SupConLoss,
    'ntxent': NTXentLoss,
    'center': CenterLoss
} 