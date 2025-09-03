# Flexible Training Script

The `train_flexible.py` script allows you to train any model with any loss function using command line arguments. This gives you complete flexibility to experiment with different combinations without modifying code.

## Usage

### Basic Syntax

For **single GPU or CPU training**:
```bash
python train_flexible.py --model MODEL_TYPE --loss LOSS_TYPE [options]
```

For **multi-GPU distributed training**:
```bash
torchrun --nproc_per_node=NUM_GPUS train_flexible.py --model MODEL_TYPE --loss LOSS_TYPE [options]
```

### Available Models
- `residual` - ResNet-style encoder with skip connections  
- `ft_transformer` - Feature Tokenizer Transformer
- `deep_ft` - Deep FT-Transformer with 6+ layers

### Available Losses
- `triplet` - Standard triplet margin loss
- `semi_hard` - Semi-hard triplet loss with negative mining
- `infonce` - InfoNCE contrastive loss
- `supcon` - Supervised contrastive loss
- `ntxent` - NT-Xent contrastive loss  
- `center` - Center loss for clustering

## Examples

### FT-Transformer with Different Losses

#### 1. FT-Transformer + Triplet Loss (Single GPU)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss triplet \
    --embed_dim 192 \
    --layers 6 \
    --heads 8 \
    --margin 0.3 \
    --epochs 80 \
    --batch_size 128
```

#### 2. FT-Transformer + Supervised Contrastive Loss (Single GPU)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss supcon \
    --embed_dim 128 \
    --layers 3 \
    --heads 8 \
    --temperature 0.07 \
    --epochs 60 \
    --batch_size 256
```

#### 3. FT-Transformer + Semi-Hard Triplet Loss (Multi-GPU)
```bash
torchrun --nproc_per_node=2 train_flexible.py \
    --model ft_transformer \
    --loss semi_hard \
    --embed_dim 256 \
    --layers 4 \
    --heads 12 \
    --margin 0.5 \
    --lr 2e-3
```

#### 4. FT-Transformer + InfoNCE Loss (Single GPU)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss infonce \
    --embed_dim 128 \
    --temperature 0.1 \
    --epochs 100
```

#### 5. FT-Transformer + NT-Xent Loss (Standard Configuration)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss ntxent \
    --embed_dim 128 \
    --heads 8 \
    --layers 3 \
    --temperature 0.1 \
    --epochs 60
```

### Other Model Combinations

#### Deep FT-Transformer + Triplet Loss (Single GPU)
```bash
python train_flexible.py \
    --model deep_ft \
    --loss triplet \
    --embed_dim 192 \
    --layers 6 \
    --heads 8 \
    --margin 0.3
```

#### Residual Network + Contrastive Loss (Single GPU)
```bash
python train_flexible.py \
    --model residual \
    --loss supcon \
    --embed_dim 128 \
    --temperature 0.07
```

## Command Line Arguments

### Required Arguments
- `--model` - Model architecture (`residual`, `ft_transformer`, `deep_ft`)
- `--loss` - Loss function (`triplet`, `semi_hard`, `infonce`, `supcon`, `ntxent`, `center`)

### Model Parameters
- `--embed_dim` - Embedding dimension (default: 128)
- `--heads` - Attention heads for transformers (default: 8)  
- `--layers` - Number of transformer layers (default: 3)
- `--dropout` - Dropout rate for transformers (default: 0.2)

### Training Parameters
- `--epochs` - Number of training epochs (default: 100)
- `--batch_size` - Batch size (default: 256)
- `--lr` - Base learning rate (default: 1e-3)
- `--warmup_epochs` - Warmup epochs (default: 2)

### Loss-Specific Parameters
- `--margin` - Margin for triplet losses (default: 1.0)
- `--temperature` - Temperature for contrastive losses (default: 0.07)
- `--center_weight` - Weight for center loss (default: 0.1)

### Training Options
- `--amp` / `--no_amp` - Enable/disable mixed precision (default: enabled)
- `--test_dims` - Multiple embedding dimensions to test (default: [128])

## Advanced Usage

### Testing Multiple Embedding Dimensions (Single GPU)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss triplet \
    --test_dims 64 128 256 \
    --epochs 50
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=4 train_flexible.py \
    --model ft_transformer \
    --loss supcon \
    --embed_dim 256 \
    --batch_size 512 \
    --lr 4e-3
```

### Custom Hyperparameters for Transformers (Single GPU)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss ntxent \
    --embed_dim 384 \
    --heads 12 \
    --layers 8 \
    --dropout 0.1 \
    --temperature 0.05 \
    --lr 5e-4
```

### CPU Training (No CUDA)
```bash
python train_flexible.py \
    --model ft_transformer \
    --loss supcon \
    --embed_dim 64 \
    --batch_size 64 \
    --epochs 20 \
    --no_amp
```

## Tips

1. **Single vs Multi-GPU**: Use `python` for single GPU/CPU, `torchrun` for multi-GPU
2. **Batch Size**: Larger batch sizes generally work better for contrastive losses
3. **Learning Rate**: Scale learning rate with number of GPUs (LR × num_gpus)
4. **Temperature**: Lower temperatures (0.01-0.1) often work better for contrastive losses
5. **Margin**: For triplet losses, try margins between 0.1-1.0
6. **Embedding Dimension**: Larger dimensions (128-512) usually give better results
7. **Transformer Layers**: More layers help with complex patterns but require more memory
8. **Memory Issues**: Reduce batch size or embedding dimension if you get CUDA memory errors

## Troubleshooting

### CUDA Memory Errors
- Reduce `--batch_size` (try 64, 32, or 16)
- Reduce `--embed_dim` (try 64 or 32)
- Use `--no_amp` to disable mixed precision
- Use CPU training if necessary

### Distributed Training Issues
- Make sure you have multiple GPUs available
- Use `python` instead of `torchrun` for single GPU training
- Check CUDA driver compatibility

## Output

Results are automatically saved to the `results/` directory as JSON files containing:
- Model and loss type
- Hyperparameters used  
- Final test accuracy
- Training loss
- Embedding dimension

Use these results to compare different model-loss combinations and find the best configuration for your data! 