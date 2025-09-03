# Model architectures module

# Import from separate files for backward compatibility
from .residual import EmitterEncoder, ResidualBlock
from .ft_transformer import FTTransformer
from .deep_ft_transformer import DeepFTTransformer
from .factory import get_model, AVAILABLE_MODELS

# Keep the old import path working
__all__ = [
    'EmitterEncoder',
    'ResidualBlock', 
    'FTTransformer',
    'DeepFTTransformer',
    'get_model',
    'AVAILABLE_MODELS'
] 