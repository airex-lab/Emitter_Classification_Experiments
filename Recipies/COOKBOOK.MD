# Emitter Classification Experiments - Cookbook

A comprehensive guide to running and using the modular emitter classification codebase.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Basic Usage](#basic-usage)
3. [Running Experiments](#running-experiments)
4. [Configuration Management](#configuration-management)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)
7. [Development Guide](#development-guide)
8. [Modular Architecture](#modular-architecture)

## Environment Setup

### 1. Activate Conda Environment

```bash
# Activate the ml conda environment
conda activate ml

# Verify activation
which python
# Should show: /home/vishesh/miniconda3/envs/ml/bin/python
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install torch torchvision
pip install numpy pandas scikit-learn scipy
pip install openpyxl xlrd
```

### 3. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test distributed training
python -c "import torch.distributed; print('Distributed training available')"

# Test imports
python -c "from src.config import get_triplet_config; print('✅ All imports work!')"
```

## Basic Usage

### 1. Check Dataset Files

```bash
# Verify dataset files exist
ls -la dataset/
# Should show: set1.xls, set2.xls, set3.xlsx, set5.xlsx, set6.xlsx
```

### 2. Create Results Directory

```bash
# Create results directory (if not exists)
mkdir -p results
```

## Running Experiments

### 1. Using torchrun (Recommended)

```bash
# Single GPU training
torchrun --nproc_per_node=1 train_triplet.py
torchrun --nproc_per_node=1 train_dual_encoder.py
torchrun --nproc_per_node=1 train_supcon.py
torchrun --nproc_per_node=1 train_ft_transformer.py

# Multi-GPU training
torchrun --nproc_per_node=2 train_triplet.py
torchrun --nproc_per_node=4 train_ft_transformer.py
```

### 2. Using the Launcher Script

```bash
# Run experiments with the launcher
python run_experiment.py triplet --num_gpus 1
python run_experiment.py dual_encoder --num_gpus 2
python run_experiment.py supcon --num_gpus 4
python run_experiment.py ft_transformer --num_gpus 1
```

### 3. Using Python Module (Alternative)

```bash
# If torchrun is not available
python -m torch.distributed.launch --nproc_per_node=1 train_triplet.py
python -m torch.distributed.launch --nproc_per_node=2 train_dual_encoder.py
```

### 4. Single Process Training (Debug Mode)

```bash
# For debugging without distributed training
python train_triplet.py
python train_dual_encoder.py
```

## Configuration Management

### 1. View Available Configurations

```python
# In Python
from src.config import *

# List available configs
configs = [
    get_triplet_config(),
    get_dual_encoder_config(),
    get_supcon_config(),
    get_ft_transformer_config(),
    get_ft_triplet_config()
]

for i, config in enumerate(configs):
    print(f"Config {i}: {config.training.EPOCHS} epochs, {config.model.EMBED_DIM} dim")
```

### 2. Custom Configuration

```python
# Create custom config
from src.config import ExperimentConfig

config = ExperimentConfig()
config.training.EPOCHS = 150
config.training.BATCH_SIZE = 128
config.model.EMBED_DIM = 256
config.training.TEMP = 0.1

# Use in training script
```

### 3. Modify Existing Configurations

```python
# Modify triplet config
config = get_triplet_config()
config.training.MARGIN = 0.5  # Change margin
config.model.EMBEDDING_DIMS_TO_TEST = [16, 32, 64]  # Test specific dimensions
```

## Troubleshooting

### 1. Port Already in Use Error

```bash
# Check what's using port 29500
lsof -i :29500

# Kill processes using the port
kill -9 <PID>

# Or use a different port
export MASTER_PORT=29501
torchrun --nproc_per_node=1 train_triplet.py
```

### 2. CUDA Out of Memory

```bash
# Reduce batch size
# Edit src/config.py or modify config in training script
config.training.BATCH_SIZE = 32  # Reduce from 64

# Or use gradient accumulation
# Add to training loop
```

### 3. Import Errors

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add current directory to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Test imports
python -c "from src.data.processor import DataProcessor; print('✅ Data processor works')"
```

### 4. Dataset Loading Issues

```bash
# Check file permissions
ls -la dataset/

# Test data loading
python -c "
from src.data.processor import DataProcessor
from src.config import DataConfig
processor = DataProcessor(DataConfig())
x_train, y_train, x_test, y_test, label_map = processor.get_processed_data()
print(f'✅ Data loaded: {x_train.shape}, {len(label_map)} classes')
"
```

### 5. Distributed Training Issues

```bash
# Check NCCL installation
python -c "import torch.distributed; print('NCCL available')"

# Test single GPU first
torchrun --nproc_per_node=1 train_triplet.py

# Then try multi-GPU
torchrun --nproc_per_node=2 train_triplet.py
```

## Advanced Usage

### 1. Custom Model Architecture

```python
# Using factory function
from src.models.factory import get_model

# Custom residual network
model = get_model('residual', 
                  in_dim=5, 
                  emb_dim=128, 
                  hidden_dim=256, 
                  num_blocks=4)

# Custom FT-Transformer
model = get_model('ft_transformer',
                  num_feats=5,
                  dim=192,
                  heads=12,
                  layers=8)

# Custom Deep FT-Transformer
model = get_model('deep_ft',
                  num_feats=5,
                  dim=192,
                  heads=8,
                  layers=6)
```

### 2. Custom Loss Function

```python
# Using factory function
from src.losses.factory import get_loss

# Custom triplet loss
criterion = get_loss('triplet', margin=0.5)

# Custom NT-Xent
criterion = get_loss('ntxent', temperature=0.05)

# Custom SupCon
criterion = get_loss('supcon', temperature=0.07)

# Custom Center Loss
criterion = get_loss('center', num_classes=10, dim=128)
```

### 3. Direct Model Imports

```python
# Import specific models directly
from src.models.residual import EmitterEncoder, ResidualBlock
from src.models.ft_transformer import FTTransformer
from src.models.deep_ft_transformer import DeepFTTransformer

# Use directly
model = EmitterEncoder(in_dim=5, emb_dim=128, hidden_dim=256, num_blocks=3)
ft_model = FTTransformer(num_feats=5, dim=192, heads=8, layers=4)
deep_model = DeepFTTransformer(num_feats=5, dim=192, heads=8, layers=6)
```

### 4. Direct Loss Imports

```python
# Import specific losses directly
from src.losses.triplet import TripletMarginLoss, SemiHardTripletLoss
from src.losses.contrastive_losses import InfoNCELoss, SupConLoss, NTXentLoss
from src.losses.center_loss import CenterLoss

# Use directly
triplet_loss = TripletMarginLoss(margin=1.0)
semi_hard_loss = SemiHardTripletLoss(margin=0.3)
infonce_loss = InfoNCELoss(temperature=0.07)
supcon_loss = SupConLoss(temperature=0.07)
ntxent_loss = NTXentLoss(temperature=0.1)
center_loss = CenterLoss(num_classes=10, dim=128)
```

### 5. Custom Dataset

```python
# Create custom dataset
from src.data.datasets import TripletPDWDataset
from torch.utils.data import DataLoader

dataset = TripletPDWDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### 6. Hyperparameter Tuning

```bash
# Test different embedding dimensions
for dim in 8 16 32 64 128; do
    echo "Testing embedding dimension: $dim"
    # Modify config and run
done

# Test different temperatures
for temp in 0.05 0.07 0.1 0.15; do
    echo "Testing temperature: $temp"
    # Modify config and run
done
```

### 7. Monitoring Training

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check log files
tail -f results/training.log
```

## Development Guide

### 1. Adding New Model

```python
# 1. Create new model file: src/models/custom_model.py
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, in_dim, emb_dim, **kwargs):
        super().__init__()
        # Your model definition
        self.fc1 = nn.Linear(in_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x):
        # Your forward pass
        z = self.fc2(F.relu(self.fc1(x)))
        return F.normalize(z, p=2, dim=1)

# 2. Add to factory: src/models/factory.py
from .custom_model import CustomModel

def get_model(model_type, **kwargs):
    if model_type == 'custom':
        return CustomModel(**kwargs)
    # ... existing code

# 3. Add to available models
AVAILABLE_MODELS = {
    'residual': EmitterEncoder,
    'ft_transformer': FTTransformer,
    'deep_ft': DeepFTTransformer,
    'custom': CustomModel  # Add your model
}

# 4. Import in __init__.py: src/models/__init__.py
from .custom_model import CustomModel
__all__ = [
    'EmitterEncoder',
    'FTTransformer', 
    'DeepFTTransformer',
    'CustomModel',  # Add your model
    'get_model',
    'AVAILABLE_MODELS'
]
```

### 2. Adding New Loss

```python
# 1. Create new loss file: src/losses/custom_loss.py
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Your loss definition
        
    def forward(self, *args):
        # Your loss computation
        return loss

# 2. Add to factory: src/losses/factory.py
from .custom_loss import CustomLoss

def get_loss(loss_type, **kwargs):
    if loss_type == 'custom':
        return CustomLoss(**kwargs)
    # ... existing code

# 3. Add to available losses
AVAILABLE_LOSSES = {
    'triplet': TripletMarginLoss,
    'semi_hard': SemiHardTripletLoss,
    'infonce': InfoNCELoss,
    'supcon': SupConLoss,
    'ntxent': NTXentLoss,
    'center': CenterLoss,
    'custom': CustomLoss  # Add your loss
}

# 4. Import in __init__.py: src/losses/__init__.py
from .custom_loss import CustomLoss
__all__ = [
    'TripletMarginLoss',
    'SemiHardTripletLoss',
    'InfoNCELoss',
    'SupConLoss',
    'NTXentLoss',
    'CenterLoss',
    'CustomLoss',  # Add your loss
    'get_loss',
    'AVAILABLE_LOSSES'
]
```

### 3. Adding New Dataset

```python
# 1. Add dataset to src/data/datasets.py
from torch.utils.data import Dataset

class NewDataset(Dataset):
    def __init__(self, x, y):
        # Your dataset initialization
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        # Your data loading
        return self.x[idx], self.y[idx]
```

### 4. Running Tests

```bash
# Test individual components
python -c "
from src.data.processor import DataProcessor
from src.models.factory import get_model
from src.losses.factory import get_loss

# Test data processing
processor = DataProcessor(DataConfig())
x_train, y_train, x_test, y_test, label_map = processor.get_processed_data()
print('✅ Data processing works')

# Test model
model = get_model('residual', in_dim=5, emb_dim=32)
print('✅ Model creation works')

# Test loss
criterion = get_loss('triplet', margin=1.0)
print('✅ Loss creation works')
"
```

## Modular Architecture

### Model Structure

The models are organized in separate files:

```
src/models/
├── residual.py              # Residual network model
├── ft_transformer.py        # FT-Transformer model  
├── deep_ft_transformer.py   # Deep FT-Transformer model
├── factory.py               # Model factory
└── __init__.py              # Backward compatibility
```

### Loss Structure

The losses are organized in separate files:

```
src/losses/
├── triplet.py               # Triplet loss functions
├── contrastive_losses.py    # InfoNCE, SupCon, NT-Xent
├── center_loss.py           # Center loss
├── factory.py               # Loss factory
└── __init__.py              # Backward compatibility
```

### Using Individual Components

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

```python
from src.models.factory import get_model, AVAILABLE_MODELS
from src.losses.factory import get_loss, AVAILABLE_LOSSES

# List available models
print("Available models:", list(AVAILABLE_MODELS.keys()))

# List available losses
print("Available losses:", list(AVAILABLE_LOSSES.keys()))

# Create components
model = get_model('residual', in_dim=5, emb_dim=128)
criterion = get_loss('triplet', margin=1.0)
```

### Backward Compatibility

All existing imports still work:

```python
# These still work for backward compatibility
from src.models.architectures import get_model
from src.losses.contrastive import get_loss

# But now you can also use the new structure
from src.models.factory import get_model
from src.losses.factory import get_loss
```

## Common Commands Reference

### Environment Management
```bash
conda activate ml                    # Activate environment
conda deactivate                     # Deactivate environment
pip list | grep torch               # Check PyTorch installation
nvidia-smi                          # Check GPU status
```

### Training Commands
```bash
torchrun --nproc_per_node=1 train_triplet.py
torchrun --nproc_per_node=2 train_dual_encoder.py
torchrun --nproc_per_node=4 train_supcon.py
python run_experiment.py ft_transformer --num_gpus 1
```

### Debugging Commands
```bash
lsof -i :29500                      # Check port usage
kill -9 <PID>                       # Kill process
ps aux | grep python               # Find Python processes
export MASTER_PORT=29501           # Change port
```

### Data Management
```bash
ls -la dataset/                     # Check dataset files
mkdir -p results                   # Create results directory
ls -la results/                    # Check results
cat results/triplet_dim_32.json    # View results
```

### Development Commands
```bash
find src/ -name "*.py"             # List all Python files
python -c "from src.config import *" # Test imports
git status                         # Check git status
git add . && git commit -m "Update" # Commit changes
```

## Performance Tips

### 1. GPU Memory Optimization
```python
# Use gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(model, x)

# Use mixed precision
scaler = torch.amp.GradScaler()
with torch.amp.autocast():
    loss = model(x)
```

### 2. Data Loading Optimization
```python
# Increase num_workers
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)

# Use prefetch_factor
dataloader = DataLoader(dataset, num_workers=4, prefetch_factor=2)
```

### 3. Distributed Training Optimization
```python
# Use find_unused_parameters=False for better performance
model = DDP(model, find_unused_parameters=False)

# Use gradient_as_bucket_view=True
model = DDP(model, gradient_as_bucket_view=True)
```

## Quick Start Examples

### Example 1: Run Triplet Loss Training
```bash
# 1. Activate environment
conda activate ml

# 2. Check dataset
ls -la dataset/

# 3. Run training
torchrun --nproc_per_node=1 train_triplet.py

# 4. Check results
ls -la results/
cat results/triplet_dim_32.json
```

### Example 2: Run FT-Transformer with 2 GPUs
```bash
# 1. Activate environment
conda activate ml

# 2. Run multi-GPU training
torchrun --nproc_per_node=2 train_ft_transformer.py

# 3. Monitor GPU usage
watch -n 1 nvidia-smi
```

### Example 3: Debug Import Issues
```bash
# 1. Test imports
python -c "from src.config import get_triplet_config; print('Config works')"
python -c "from src.data.processor import DataProcessor; print('Data works')"
python -c "from src.models.factory import get_model; print('Models work')"
python -c "from src.losses.factory import get_loss; print('Losses work')"

# 2. If imports fail, check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Example 4: Custom Configuration
```python
# Create custom_training.py
from src.config import get_triplet_config
from src.data.processor import DataProcessor
from src.models.factory import get_model
from src.losses.factory import get_loss

# Custom config
config = get_triplet_config()
config.training.EPOCHS = 50
config.training.BATCH_SIZE = 32
config.model.EMBEDDING_DIMS_TO_TEST = [16, 32]

# Use in training
processor = DataProcessor(config.data)
model = get_model('residual', in_dim=5, emb_dim=32)
criterion = get_loss('triplet', margin=1.0)
```

### Example 5: Using Individual Components
```python
# Import specific models and losses
from src.models.residual import EmitterEncoder
from src.models.ft_transformer import FTTransformer
from src.losses.triplet import TripletMarginLoss
from src.losses.contrastive_losses import SupConLoss

# Create components directly
residual_model = EmitterEncoder(in_dim=5, emb_dim=128, hidden_dim=256)
ft_model = FTTransformer(num_feats=5, dim=192, heads=8, layers=3)
triplet_criterion = TripletMarginLoss(margin=1.0)
supcon_criterion = SupConLoss(temperature=0.07)
```

## Summary of Key Commands

### Essential Commands for Daily Use:
1. **Activate Environment**: `conda activate ml`
2. **Run Training**: `torchrun --nproc_per_node=1 train_triplet.py`
3. **Check GPU**: `nvidia-smi`
4. **Kill Process**: `kill -9 <PID>`
5. **Check Port**: `lsof -i :29500`
6. **Test Imports**: `python -c "from src.config import *"`

### For Different Training Approaches:
- **Triplet Loss**: `torchrun --nproc_per_node=1 train_triplet.py`
- **Dual Encoder**: `torchrun --nproc_per_node=1 train_dual_encoder.py`
- **SupCon**: `torchrun --nproc_per_node=1 train_supcon.py`
- **FT-Transformer**: `torchrun --nproc_per_node=1 train_ft_transformer.py`

### For Development:
- **Add Model**: Create `src/models/custom_model.py` and update factory
- **Add Loss**: Create `src/losses/custom_loss.py` and update factory
- **Test Components**: Use individual imports for direct access
- **Use Factory**: Use factory functions for easy instantiation

This cookbook should cover all the common scenarios and commands you'll need to work with the emitter classification experiments!