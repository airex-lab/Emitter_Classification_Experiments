## Adding New Models

### 1. Creating a New Model Architecture

```python
# Add to src/models/architectures.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformer(nn.Module):
    """Custom transformer model for emitter classification."""
    
    def __init__(self, num_feats: int, dim: int, heads: int = 8, 
                 layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Feature embedding
        self.feature_embedding = nn.Linear(num_feats, dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_feats, dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_feats]
        B = x.size(0)
        
        # Embed features
        x = self.feature_embedding(x)  # [B, num_feats, dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Pass through transformer
        x = self.transformer(x)  # [B, num_feats, dim]
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, dim]
        
        # Output projection and normalization
        x = self.output_proj(x)
        x = self.layer_norm(x)
        
        return F.normalize(x, p=2, dim=1)

# Add to factory function
def get_model(model_type: str, in_dim: int, emb_dim: int, **kwargs) -> nn.Module:
    if model_type == 'custom_transformer':
        return CustomTransformer(in_dim, emb_dim, **kwargs)
    elif model_type == 'residual':
        return EmitterEncoder(in_dim, emb_dim, **kwargs)
    # ... existing code
```

### 2. Using the New Model

```python
# In your training script
from src.models.architectures import get_model

# Create custom transformer
model = get_model('custom_transformer', 
                  in_dim=5, 
                  emb_dim=128, 
                  heads=8, 
                  layers=4, 
                  dropout=0.1)
```

## Adding New Loss Functions

### 1. Creating a New Loss Function

```python
# Add to src/losses/contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """ArcFace loss for better feature discrimination."""
    
    def __init__(self, num_classes: int, dim: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(features, weight)  # [B, num_classes]
        
        # Apply margin
        cos_theta_m = torch.cos(torch.acos(cos_theta) + self.margin)
        
        # Create mask for positive classes
        one_hot = F.one_hot(labels, self.num_classes).float()
        
        # Apply margin only to positive classes
        output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        
        # Scale
        output = output * self.scale
        
        # Cross entropy loss
        loss = F.cross_entropy(output, labels)
        return loss

class TripletMarginLossWithDistance(nn.Module):
    """Triplet loss with learnable distance metric."""
    
    def __init__(self, margin: float = 1.0, distance_type: str = 'euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
        
        # Learnable distance parameters
        if distance_type == 'learnable':
            self.distance_weight = nn.Parameter(torch.ones(1))
            
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        
        if self.distance_type == 'euclidean':
            pos_dist = (anchor - positive).pow(2).sum(1)
            neg_dist = (anchor - negative).pow(2).sum(1)
        elif self.distance_type == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=1)
        elif self.distance_type == 'learnable':
            pos_dist = self.distance_weight * (anchor - positive).pow(2).sum(1)
            neg_dist = self.distance_weight * (anchor - negative).pow(2).sum(1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Add to factory function
def get_loss(loss_type: str, **kwargs) -> nn.Module:
    if loss_type == 'arcface':
        return ArcFaceLoss(**kwargs)
    elif loss_type == 'triplet_distance':
        return TripletMarginLossWithDistance(**kwargs)
    elif loss_type == 'triplet':
        return TripletMarginLoss(**kwargs)
    # ... existing code
```

### 2. Using the New Loss Function

```python
# In your training script
from src.losses.contrastive import get_loss

# Create ArcFace loss
criterion = get_loss('arcface', 
                    num_classes=10, 
                    dim=128, 
                    margin=0.5, 
                    scale=64.0)

# Create triplet loss with learnable distance
criterion = get_loss('triplet_distance', 
                    margin=1.0, 
                    distance_type='learnable')
```

## Coupling Models with Loss Functions

### 1. Creating Custom Training Scripts

```python
# custom_training.py
#!/usr/bin/env python3
"""
Custom training script combining different models and losses.
"""

import torch
import torch.amp
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.config import get_triplet_config
from src.data.processor import DataProcessor
from src.data.datasets import TripletPDWDataset, EmitterDataset
from src.models.architectures import get_model
from src.losses.contrastive import get_loss
from src.utils.distributed import is_main, get_effective_lr
from src.utils.evaluation import save_results

def train_model_with_loss(model_type: str, loss_type: str, config):
    """Train a model with a specific loss function."""
    
    # Setup distributed training
    dist.init_process_group(backend=config.training.BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    try:
        # Load and preprocess data
        data_processor = DataProcessor(config.data)
        x_train, y_train, x_test, y_test, label_map = data_processor.get_processed_data()
        
        # Choose appropriate dataset based on loss type
        if loss_type in ['triplet', 'triplet_distance']:
            dataset = TripletPDWDataset(x_train, y_train)
        else:
            dataset = EmitterDataset(x_train, y_train)
            
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.BATCH_SIZE,
            sampler=torch.utils.data.DistributedSampler(dataset, shuffle=True),
            num_workers=config.training.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        
        # Get effective learning rate
        lr = get_effective_lr(config.training.BASE_LR, world_size)
        
        if is_main(rank):
            print(f"[Rank0] Model: {model_type}, Loss: {loss_type}")
            print(f"[Rank0] Using {world_size} GPUs, effective LR={lr}")
        
        # Setup mixed precision
        scaler = torch.amp.GradScaler('cuda', enabled=config.training.AMP)
        
        # Test different embedding dimensions
        for dim in config.model.EMBEDDING_DIMS_TO_TEST:
            if is_main(rank):
                print(f"\n===== Embedding {dim} =====")
            
            # Create model
            model = get_model(model_type, 
                            in_dim=data_processor.get_feature_dim(),
                            emb_dim=dim,
                            **get_model_kwargs(model_type, config))
            
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )
            
            # Create loss function
            criterion = get_loss(loss_type, **get_loss_kwargs(loss_type, dim, len(label_map), config))
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Training loop
            for epoch in range(config.training.EPOCHS):
                dataloader.sampler.set_epoch(epoch)
                model.train()
                running_loss = 0.0
                
                for batch in dataloader:
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    with torch.amp.autocast('cuda', enabled=config.training.AMP):
                        loss = forward_step(model, criterion, batch, loss_type, device)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(dataloader)
                
                # Evaluate periodically
                if (epoch + 1) % config.training.CLUSTER_EVERY == 0 and is_main(rank):
                    acc = evaluate_model(model.module, x_test, y_test, device)
                    print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}  "
                          f"test-clust-acc {acc*100:5.2f}%")
            
            # Final evaluation and save results
            if is_main(rank):
                acc = evaluate_model(model.module, x_test, y_test, device)
                results = {
                    "model_type": model_type,
                    "loss_type": loss_type,
                    "embedding_dim": dim,
                    "test_acc": acc,
                    "final_loss": avg_loss,
                    **get_result_kwargs(loss_type, config)
                }
                
                save_results(results, f"{config.training.RESULT_DIR}/{model_type}_{loss_type}_dim_{dim}.json")
                print(f"[Done] {model_type}+{loss_type} dim={dim} acc={acc*100:.2f}%")
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def get_model_kwargs(model_type: str, config):
    """Get model-specific keyword arguments."""
    if model_type == 'residual':
        return {
            'hidden_dim': config.model.HIDDEN_DIM,
            'dropout': config.model.RESIDUAL_DROPOUT
        }
    elif model_type == 'ft_transformer':
        return {
            'heads': config.model.HEADS,
            'layers': config.model.LAYERS,
            'dropout': config.model.DROPOUT
        }
    elif model_type == 'custom_transformer':
        return {
            'heads': 8,
            'layers': 4,
            'dropout': 0.1
        }
    return {}

def get_loss_kwargs(loss_type: str, dim: int, num_classes: int, config):
    """Get loss-specific keyword arguments."""
    if loss_type == 'triplet':
        return {'margin': config.training.MARGIN}
    elif loss_type == 'triplet_distance':
        return {'margin': config.training.MARGIN, 'distance_type': 'learnable'}
    elif loss_type == 'arcface':
        return {
            'num_classes': num_classes,
            'dim': dim,
            'margin': 0.5,
            'scale': 64.0
        }
    elif loss_type == 'supcon':
        return {'temperature': config.training.TEMP}
    elif loss_type == 'ntxent':
        return {'temperature': config.training.TEMP}
    return {}

def forward_step(model, criterion, batch, loss_type: str, device):
    """Forward step for different loss types."""
    if loss_type in ['triplet', 'triplet_distance']:
        anchor, positive, negative = [b.to(device) for b in batch]
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        return criterion(anchor_emb, positive_emb, negative_emb)
    else:
        x, y = [b.to(device) for b in batch]
        embeddings = model(x)
        return criterion(embeddings, y)

def evaluate_model(model, x_test, y_test, device):
    """Evaluate model using clustering accuracy."""
    from src.utils.evaluation import evaluate_model as eval_func
    return eval_func(model, x_test, y_test, device)

def get_result_kwargs(loss_type: str, config):
    """Get result-specific keyword arguments."""
    if loss_type == 'triplet':
        return {'margin': config.training.MARGIN}
    elif loss_type == 'supcon':
        return {'temperature': config.training.TEMP}
    elif loss_type == 'ntxent':
        return {'temperature': config.training.TEMP}
    return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom model-loss training')
    parser.add_argument('--model', type=str, default='residual',
                       choices=['residual', 'ft_transformer', 'custom_transformer'])
    parser.add_argument('--loss', type=str, default='triplet',
                       choices=['triplet', 'triplet_distance', 'arcface', 'supcon', 'ntxent'])
    parser.add_argument('--num_gpus', type=int, default=1)
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_triplet_config()
    
    # Train with specified model and loss
    train_model_with_loss(args.model, args.loss, config)
```

### 2. Running Custom Model-Loss Combinations

```bash
# Run different combinations
torchrun --nproc_per_node=1 custom_training.py --model residual --loss triplet
torchrun --nproc_per_node=1 custom_training.py --model ft_transformer --loss supcon
torchrun --nproc_per_node=2 custom_training.py --model custom_transformer --loss arcface
```

### 3. Creating Experiment Scripts

```python
# run_experiments.py
#!/usr/bin/env python3
"""
Run multiple model-loss combinations.
"""

import subprocess
import sys

def run_experiment(model: str, loss: str, num_gpus: int = 1):
    """Run a single experiment."""
    cmd = [
        'torchrun',
        '--nproc_per_node', str(num_gpus),
        'custom_training.py',
        '--model', model,
        '--loss', loss
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {model}+{loss} completed successfully")
    else:
        print(f"❌ {model}+{loss} failed")
        print(f"Error: {result.stderr}")
    
    return result.returncode == 0

def main():
    # Define experiments
    experiments = [
        ('residual', 'triplet'),
        ('residual', 'supcon'),
        ('ft_transformer', 'ntxent'),
        ('ft_transformer', 'supcon'),
        ('custom_transformer', 'arcface'),
        ('custom_transformer', 'triplet_distance'),
    ]
    
    # Run experiments
    successful = 0
    total = len(experiments)
    
    for model, loss in experiments:
        if run_experiment(model, loss, num_gpus=1):
            successful += 1
    
    print(f"\nResults: {successful}/{total} experiments completed successfully")

if __name__ == "__main__":
    main()
```

## Advanced Model-Loss Combinations

### 1. Multi-Loss Training

```python
# multi_loss_training.py
class MultiLossTrainer:
    """Trainer that combines multiple loss functions."""
    
    def __init__(self, losses: dict):
        self.losses = losses
        
    def compute_loss(self, model, batch, device):
        """Compute combined loss from multiple loss functions."""
        total_loss = 0.0
        
        for loss_name, (criterion, weight) in self.losses.items():
            if loss_name == 'triplet':
                anchor, positive, negative = [b.to(device) for b in batch]
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
            else:
                x, y = [b.to(device) for b in batch]
                embeddings = model(x)
                loss = criterion(embeddings, y)
            
            total_loss += weight * loss
        
        return total_loss

# Usage
losses = {
    'triplet': (get_loss('triplet', margin=1.0), 0.7),
    'center': (get_loss('center', num_classes=10, dim=128), 0.3)
}

trainer = MultiLossTrainer(losses)
```

### 2. Loss Function Ablation Study

```python
# ablation_study.py
def run_ablation_study():
    """Run ablation study with different loss combinations."""
    
    base_config = get_triplet_config()
    model_type = 'residual'
    
    # Define loss combinations
    loss_combinations = [
        ['triplet'],
        ['supcon'],
        ['ntxent'],
        ['triplet', 'center'],
        ['supcon', 'center'],
        ['triplet', 'supcon'],
    ]
    
    results = {}
    
    for i, losses in enumerate(loss_combinations):
        print(f"\n=== Combination {i+1}: {losses} ===")
        
        # Create multi-loss trainer
        loss_dict = {}
        for loss in losses:
            if loss == 'triplet':
                loss_dict[loss] = (get_loss('triplet', margin=1.0), 1.0)
            elif loss == 'supcon':
                loss_dict[loss] = (get_loss('supcon', temperature=0.07), 1.0)
            elif loss == 'ntxent':
                loss_dict[loss] = (get_loss('ntxent', temperature=0.1), 1.0)
            elif loss == 'center':
                loss_dict[loss] = (get_loss('center', num_classes=10, dim=128), 0.1)
        
        trainer = MultiLossTrainer(loss_dict)
        
        # Train and evaluate
        acc = train_and_evaluate(model_type, trainer, base_config)
        results['+'.join(losses)] = acc
    
    # Save results
    save_results(results, 'ablation_study_results.json')
    return results
```

### 3. Hyperparameter Search

```python
# hyperparameter_search.py
def grid_search_hyperparameters():
    """Grid search for optimal hyperparameters."""
    
    # Define search space
    search_space = {
        'margin': [0.5, 1.0, 1.5],
        'temperature': [0.05, 0.07, 0.1, 0.15],
        'embedding_dim': [32, 64, 128],
        'learning_rate': [1e-4, 1e-3, 1e-2]
    }
    
    best_acc = 0.0
    best_params = {}
    
    # Grid search
    for margin in search_space['margin']:
        for temp in search_space['temperature']:
            for dim in search_space['embedding_dim']:
                for lr in search_space['learning_rate']:
                    
                    # Update config
                    config = get_triplet_config()
                    config.training.MARGIN = margin
                    config.training.TEMP = temp
                    config.model.EMBED_DIM = dim
                    config.training.BASE_LR = lr
                    
                    # Train and evaluate
                    acc = train_and_evaluate('residual', 'triplet', config)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            'margin': margin,
                            'temperature': temp,
                            'embedding_dim': dim,
                            'learning_rate': lr,
                            'accuracy': acc
                        }
    
    print(f"Best parameters: {best_params}")
    return best_params
```

## Quick Reference for Model-Loss Combinations

### Common Combinations:

```bash
# Residual Network + Triplet Loss
torchrun --nproc_per_node=1 custom_training.py --model residual --loss triplet

# FT-Transformer + SupCon
torchrun --nproc_per_node=1 custom_training.py --model ft_transformer --loss supcon

# Custom Transformer + ArcFace
torchrun --nproc_per_node=1 custom_training.py --model custom_transformer --loss arcface

# Residual + NT-Xent
torchrun --nproc_per_node=1 custom_training.py --model residual --loss ntxent
```

### Performance Tips for Different Combinations:

```python
# For triplet loss: Use larger batch sizes
config.training.BATCH_SIZE = 128

# For contrastive losses: Use higher temperatures
config.training.TEMP = 0.1

# For ArcFace: Use smaller learning rates
config.training.BASE_LR = 1e-4

# For multi-loss: Balance loss weights
loss_weights = {
    'triplet': 0.7,
    'center': 0.3
}
```
