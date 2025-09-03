# Loss functions module

# Import from separate files for backward compatibility
from .triplet import TripletMarginLoss, SemiHardTripletLoss
from .contrastive_losses import InfoNCELoss, SupConLoss, NTXentLoss
from .center_loss import CenterLoss
from .factory import get_loss, AVAILABLE_LOSSES

# Keep the old import path working
__all__ = [
    'TripletMarginLoss',
    'SemiHardTripletLoss',
    'InfoNCELoss',
    'SupConLoss', 
    'NTXentLoss',
    'CenterLoss',
    'get_loss',
    'AVAILABLE_LOSSES'
] 