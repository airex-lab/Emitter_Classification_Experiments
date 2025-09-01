# Emitter Classification Experiments

A modular and extensible codebase for emitter classification using contrastive learning approaches.

## Installation

1. Clone the repository:
```bash
git clone will add repo link here yoyo
cd Emitter_Classification_Experiments
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your dataset files are in the `dataset/` directory:
   - `set1.xls`
   - `set2.xls` 
   - `set3.xlsx`
   - `set5.xlsx`
   - `set6.xlsx`

## Usage

### Running Experiments

The codebase provides several training scripts for different approaches:
| Training is of 2 types :
1. Using normal ```torchrun --nproc_per_node=[NUMBER OF GPUS] train_[name of architecture].py```
2. Using train_flexible.py : 
```torchrun --nproc_per_node=[NUMBER OF GPUS] train_flexible.py --model ft_transformer --loss triplet```
(in this example, we are using ft_transformer with triplet loss)

These are the available models and losses:

#### 1. Triplet Loss Training
```bash
torchrun --nproc_per_node=NUM_GPUS train_triplet.py
```

#### 2. Dual Encoder with InfoNCE
```bash
torchrun --nproc_per_node=NUM_GPUS train_dual_encoder.py
```

#### 3. Supervised Contrastive Learning
```bash
torchrun --nproc_per_node=NUM_GPUS train_supcon.py
```

#### 4. FT-Transformer with NT-Xent
```bash
torchrun --nproc_per_node=NUM_GPUS train_ft_transformer.py
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

### Configuration

All hyperparameters and settings are managed in `src/config.py`. The module provides:

- **DataConfig**: Data paths, column definitions, and train/test splits
- **ModelConfig**: Model architecture parameters
- **TrainingConfig**: Training hyperparameters and settings
- **ExperimentConfig**: Complete experiment configuration

Predefined configurations are available for different experiments:
- `get_triplet_config()`
- `get_dual_encoder_config()`
- `get_supcon_config()`
- `get_ft_transformer_config()`
- `get_ft_triplet_config()`

### Customizing Experiments

### Using train_flexible.py


```
torchrun --nproc_per_node=1 train_flexible.py \
    --model ft_transformer \
    --loss triplet \
    --embed_dim 192 \
    --layers 6 \
    --heads 8 \
    --margin 0.3
```

### FT-Transformer + Triplet Loss
```bash
torchrun --nproc_per_node=1 train_flexible.py \
    --model ft_transformer \
    --loss triplet \
    --embed_dim 192 \
    --layers 6 \
    --heads 8 \
    --margin 0.3
```

### FT-Transformer + Supervised Contrastive Loss
```bash
torchrun --nproc_per_node=1 train_flexible.py \
    --model ft_transformer \
    --loss supcon \
    --embed_dim 128 \
    --temperature 0.07
```

### FT-Transformer + Semi-Hard Triplet
```bash
torchrun --nproc_per_node=1 train_flexible.py \
    --model ft_transformer \
    --loss semi_hard \
    --embed_dim 256 \
    --margin 0.5
```

## 🔧 Available Combinations

You can now use FTTransformer with:
- `triplet` - Standard triplet loss
- `semi_hard` - Semi-hard triplet loss  
- `infonce` - InfoNCE contrastive loss
- `supcon` - Supervised contrastive loss
- `ntxent` - NT-Xent loss
- `center` - Center loss

## 📖 Full Documentation

Check `FLEXIBLE_TRAINING.md` for:
- Complete argument reference
- Advanced usage examples
- Tips for hyperparameter tuning
- Multi-GPU training setup

The script automatically handles the different data loaders and training loops needed for each loss type, so you can focus on experimenting with different combinations!

To create a custom experiment:

1. Modify the configuration in `src/config.py` or create a new config function
2. Use the modular components to build your training pipeline
3. Create a new training script following the existing patterns

Example:
```python
from src.config import ExperimentConfig
from src.data.processor import DataProcessor
from src.models.factory import get_model
from src.losses.factory import get_loss

# Create custom config
config = ExperimentConfig()
config.training.EPOCHS = 150
config.model.EMBED_DIM = 256

# Use modular components
data_processor = DataProcessor(config.data)
model = get_model('ft_transformer', in_dim=5, emb_dim=256)
criterion = get_loss('ntxent', temperature=0.1)
```

## Key Components

### Data Processing (`src/data/`)

- **DataProcessor**: Handles loading and preprocessing of Excel files
- **Dataset Classes**: 
  - `TripletPDWDataset`: For triplet loss training
  - `PairPDWDataset`: For contrastive learning
  - `EmitterDataset`: For supervised contrastive learning

### Models (`src/models/`)

The models are organized in separate files for better maintainability:

- **`residual.py`**: Residual network with skip connections
  - `EmitterEncoder`: Main residual network encoder
  - `ResidualBlock`: Individual residual blocks
- **`ft_transformer.py`**: Feature Tokenizer Transformer for tabular data
  - `FTTransformer`: Standard FT-Transformer implementation
- **`deep_ft_transformer.py`**: Deeper variant with more layers
  - `DeepFTTransformer`: Deep FT-Transformer with 6+ layers
- **`factory.py`**: Model factory for easy instantiation
  - `get_model()`: Factory function to create models
  - `AVAILABLE_MODELS`: Dictionary of all available models

### Losses (`src/losses/`)

The losses are organized in separate files for better maintainability:

- **`triplet.py`**: Triplet-based loss functions
  - `TripletMarginLoss`: Standard triplet loss
  - `SemiHardTripletLoss`: Semi-hard negative mining
- **`contrastive_losses.py`**: Contrastive learning losses
  - `InfoNCELoss`: InfoNCE for contrastive learning
  - `SupConLoss`: Supervised contrastive loss
  - `NTXentLoss`: Numerically stable NT-Xent
- **`center_loss.py`**: Clustering loss
  - `CenterLoss`: Center loss for clustering
- **`factory.py`**: Loss factory for easy instantiation
  - `get_loss()`: Factory function to create losses
  - `AVAILABLE_LOSSES`: Dictionary of all available losses

### Training (`src/training/`)

- **BaseTrainer**: Base class with common training functionality
- **Specialized Trainers**: Custom trainers for specific approaches
- **Distributed Training**: Multi-GPU support with DDP

### Utilities (`src/utils/`)

- **Distributed**: Distributed training setup and utilities
- **Evaluation**: Clustering accuracy and model evaluation

## Modular Architecture

### Adding New Models

1. Create a new file in `src/models/` (e.g., `custom_model.py`)
2. Define your model class inheriting from `nn.Module`
3. Add it to `src/models/factory.py`
4. Import it in `src/models/__init__.py`

Example:
```python
# src/models/custom_model.py
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        # Your model definition
        
    def forward(self, x):
        # Your forward pass
        return normalized_embeddings

# src/models/factory.py
from .custom_model import CustomModel

def get_model(model_type, **kwargs):
    if model_type == 'custom':
        return CustomModel(**kwargs)
    # ... existing code
```

### Adding New Losses

1. Create a new file in `src/losses/` (e.g., `custom_loss.py`)
2. Define your loss class inheriting from `nn.Module`
3. Add it to `src/losses/factory.py`
4. Import it in `src/losses/__init__.py`

Example:
```python
# src/losses/custom_loss.py
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Your loss definition
        
    def forward(self, *args):
        # Your loss computation
        return loss

# src/losses/factory.py
from .custom_loss import CustomLoss

def get_loss(loss_type, **kwargs):
    if loss_type == 'custom':
        return CustomLoss(**kwargs)
    # ... existing code
```

### Using Individual Components

You can import specific models and losses directly:

```python
# Import specific models
from src.models.residual import EmitterEncoder
from src.models.ft_transformer import FTTransformer
from src.models.deep_ft_transformer import DeepFTTransformer

# Import specific losses
from src.losses.triplet import TripletMarginLoss, SemiHardTripletLoss
from src.losses.contrastive_losses import InfoNCELoss, SupConLoss, NTXentLoss
from src.losses.center_loss import CenterLoss

# Use directly
model = EmitterEncoder(in_dim=5, emb_dim=128)
criterion = TripletMarginLoss(margin=1.0)
```

### Using Factory Functions

The factory functions provide a convenient way to create components:

```python
from src.models.factory import get_model
from src.losses.factory import get_loss

# Create models
residual_model = get_model('residual', in_dim=5, emb_dim=128)
ft_model = get_model('ft_transformer', num_feats=5, dim=192, heads=8, layers=3)
deep_model = get_model('deep_ft', num_feats=5, dim=192, heads=8, layers=6)

# Create losses
triplet_loss = get_loss('triplet', margin=1.0)
semi_hard_loss = get_loss('semi_hard', margin=0.3)
infonce_loss = get_loss('infonce', temperature=0.07)
supcon_loss = get_loss('supcon', temperature=0.07)
ntxent_loss = get_loss('ntxent', temperature=0.1)
center_loss = get_loss('center', num_classes=10, dim=128)
```

## Results

Training results are saved in the `results/` directory as JSON files containing:
- Embedding dimension
- Test clustering accuracy
- Final loss
- Model and loss type
- Hyperparameters

## Documentation

For detailed usage instructions, troubleshooting, and advanced examples, see:
- **`Recipies/COOKBOOK.MD`**: Comprehensive cookbook with all commands and examples
- **`Recipies/Extend_the_exp.md`**: Guide for extending experiments with new models and losses

## Contributing

When adding new features:

1. Follow the modular structure
2. Add appropriate configuration options
3. Create unit tests for new components
4. Update documentation

## License

[Add your license information here] 