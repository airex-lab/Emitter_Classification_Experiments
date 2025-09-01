#!/usr/bin/env python3
"""
Flexible training script for emitter classification with argparse support.
Allows mixing and matching any model with any loss function.
"""

import argparse
import os
import torch
import torch.amp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.config import ExperimentConfig
from src.data.processor import DataProcessor
from src.data.datasets import TripletPDWDataset, PairPDWDataset, EmitterDataset
from src.models.factory import get_model, AVAILABLE_MODELS
from src.losses.factory import get_loss, AVAILABLE_LOSSES
from src.losses.center_loss import CenterLoss
from src.utils.distributed import is_main, get_effective_lr
from src.utils.evaluation import save_results, evaluate_model


class FlexibleTrainer:
    """Flexible trainer that works with any model-loss combination."""
    
    def __init__(self, config):
        self.config = config
        self.rank = None
        self.world_size = None
        self.device = None
        self.use_distributed = False
        
    def setup_distributed(self):
        """Setup distributed training or single GPU training."""
        # Check if we're in a distributed environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Distributed training
            dist.init_process_group(backend=self.config.training.BACKEND)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f'cuda:{self.rank}')
            self.use_distributed = True
        else:
            # Single GPU training
            self.rank = 0
            self.world_size = 1
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                torch.cuda.set_device(0)
            else:
                self.device = torch.device('cpu')
            self.use_distributed = False
        
    def create_model(self, model_type, **kwargs):
        """Create and optionally wrap model with DDP."""
        model = get_model(model_type, **kwargs).to(self.device)
        
        if self.use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank], output_device=self.rank
            )
        return model
    
    def create_dataloader(self, dataset, batch_size):
        """Create dataloader with optional distributed sampling."""
        if self.use_distributed:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),  # Only shuffle if no sampler
            num_workers=self.config.training.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
    
    def get_dataset(self, x_train, y_train, loss_type):
        """Get appropriate dataset based on loss type."""
        if loss_type in ['triplet', 'semi_hard']:
            return TripletPDWDataset(x_train, y_train)
        elif loss_type in ['infonce']:
            return PairPDWDataset(x_train, y_train)
        else:  # supcon, ntxent, center
            return EmitterDataset(x_train, y_train)
    
    def train_epoch_triplet(self, model, dataloader, optimizer, criterion, scaler, epoch):
        """Train epoch for triplet-based losses."""
        model.train()
        if self.use_distributed:
            dataloader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(self.device, non_blocking=True)
            positive = positive.to(self.device, non_blocking=True)
            negative = negative.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.config.training.AMP and self.device.type == 'cuda'):
                # Forward pass
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                # Compute loss
                loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            if torch.isnan(loss):
                raise RuntimeError("Loss became NaN – check data and model.")
            
            if self.device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / num_batches
    
    def train_epoch_contrastive(self, model, dataloader, optimizer, criterion, center_loss, scaler, epoch):
        """Train epoch for contrastive losses."""
        model.train()
        if self.use_distributed:
            dataloader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.config.training.AMP and self.device.type == 'cuda'):
                # Forward pass
                embeddings = model(x)
                
                # Compute main loss based on loss type
                if hasattr(criterion, '__class__') and criterion.__class__.__name__ == 'SupConLoss':
                    # SupConLoss expects (z_local, y_local, z_global, y_global)
                    # For non-distributed training, we use the same embeddings as both local and global
                    if self.use_distributed:
                        # Gather embeddings and labels from all ranks
                        all_embeddings = self._gather_tensor(embeddings)
                        all_labels = self._gather_tensor(y)
                        main_loss = criterion(embeddings, y, all_embeddings.detach(), all_labels)
                    else:
                        # For single GPU, we can't do proper contrastive learning without negatives
                        # So we'll use a simplified version or fall back to NT-Xent style
                        main_loss = self._simplified_supcon_loss(embeddings, y, criterion.temperature)
                        
                elif hasattr(criterion, '__class__') and criterion.__class__.__name__ == 'InfoNCELoss':
                    # InfoNCELoss expects (z_local, z_global, targets)
                    if self.use_distributed:
                        all_embeddings = self._gather_tensor(embeddings)
                        # For InfoNCE, targets are indices into the global batch
                        batch_size = embeddings.size(0)
                        targets = torch.arange(batch_size, device=self.device) + self.rank * batch_size
                        main_loss = criterion(embeddings, all_embeddings.detach(), targets)
                    else:
                        # For single GPU InfoNCE, we need to create artificial negatives
                        # This is a simplified version - ideally you'd use a memory bank
                        main_loss = self._simplified_infonce_loss(embeddings, y, criterion.temperature)
                        
                else:
                    # Standard losses like NTXentLoss or CenterLoss
                    main_loss = criterion(embeddings, y)
                
                # Add center loss if available
                total_loss = main_loss
                if center_loss is not None:
                    total_loss += self.config.training.CENTER_WT * center_loss(embeddings, y)
            
            if torch.isnan(total_loss):
                raise RuntimeError("Loss became NaN – check data and model.")
            
            if self.device.type == 'cuda':
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            running_loss += total_loss.item()
        
        return running_loss / num_batches
    
    def train_epoch_pair(self, model, dataloader, optimizer, criterion, scaler, epoch):
        """Train epoch for pair-based losses like InfoNCE."""
        model.train()
        if self.use_distributed:
            dataloader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for x1, x2, y in dataloader:
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.config.training.AMP and self.device.type == 'cuda'):
                # Forward pass
                emb1 = model(x1)
                emb2 = model(x2)
                
                # Handle InfoNCE loss which expects specific signature
                if hasattr(criterion, '__class__') and criterion.__class__.__name__ == 'InfoNCELoss':
                    if self.use_distributed:
                        # Gather embeddings from all ranks
                        all_emb2 = self._gather_tensor(emb2)
                        # Create targets for InfoNCE (indices into global batch)
                        batch_size = emb1.size(0)
                        targets = torch.arange(batch_size, device=self.device) + self.rank * batch_size
                        loss = criterion(emb1, all_emb2.detach(), targets)
                    else:
                        # For single GPU, use simplified InfoNCE with labels
                        loss = self._simplified_infonce_loss(emb1, y, criterion.temperature)
                else:
                    # For other losses, just pass the embeddings and labels
                    loss = criterion(emb1, emb2, y)
            
            if torch.isnan(loss):
                raise RuntimeError("Loss became NaN – check data and model.")
            
            if self.device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / num_batches
    
    def evaluate(self, model, x_test, y_test):
        """Evaluate model."""
        # Get the actual model (unwrap DDP if needed)
        actual_model = model.module if self.use_distributed else model
        return evaluate_model(actual_model, x_test, y_test, self.device)
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.use_distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _gather_tensor(self, tensor):
        """Gather tensor from all ranks for distributed training."""
        if not self.use_distributed:
            return tensor
            
        world_size = self.world_size
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gather_list, tensor)
        return torch.cat(gather_list, dim=0)
    
    def _simplified_supcon_loss(self, embeddings, labels, temperature):
        """Simplified supervised contrastive loss for single GPU training."""
        # This is essentially NT-Xent but with proper positive/negative pairs
        batch_size = embeddings.size(0)
        
        # Compute pairwise similarity
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
        
        # Create mask for positive pairs (same label, excluding diagonal)
        labels_expanded = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()).float()
        pos_mask.fill_diagonal_(0)  # exclude self-similarity
        
        # Apply log-sum-exp trick for numerical stability
        max_sim = torch.max(sim_matrix, dim=1, keepdim=True)[0]
        sim_matrix = sim_matrix - max_sim
        
        # Compute exp similarities
        exp_sim = torch.exp(sim_matrix)
        
        # Sum of all similarities (denominator)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Sum of positive similarities (numerator)
        pos_sim = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        
        # Compute loss: -log(pos_sim / sum_exp_sim)
        loss = -torch.log(pos_sim / (sum_exp_sim + 1e-8) + 1e-8)
        
        # Only compute loss for samples that have positive pairs
        num_positives = torch.sum(pos_mask, dim=1)
        valid_mask = num_positives > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss[valid_mask].mean()
    
    def _simplified_infonce_loss(self, embeddings, labels, temperature):
        """Simplified InfoNCE loss for single GPU training."""
        # For single GPU, we treat each sample as its own query and use within-batch negatives
        batch_size = embeddings.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # For each sample, find positive samples (same label)
        labels_expanded = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()).float()
        pos_mask.fill_diagonal_(0)  # exclude self
        
        # Apply log-softmax for numerical stability
        log_prob = torch.log_softmax(sim_matrix, dim=1)
        
        # Compute loss: negative log probability of positive samples
        pos_log_prob = log_prob * pos_mask
        num_positives = pos_mask.sum(dim=1)
        
        # Average over positive samples for each anchor
        loss_per_sample = -pos_log_prob.sum(dim=1) / (num_positives + 1e-8)
        
        # Only include samples with positive pairs
        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss_per_sample[valid_mask].mean()


def create_config_from_args(args):
    """Create configuration from command line arguments."""
    config = ExperimentConfig()
    
    # Model configuration
    config.model.EMBED_DIM = args.embed_dim
    if args.model in ['ft_transformer', 'deep_ft']:
        config.model.HEADS = args.heads
        config.model.LAYERS = args.layers
        config.model.DROPOUT = args.dropout
    
    # Training configuration
    config.training.EPOCHS = args.epochs
    config.training.BATCH_SIZE = args.batch_size
    config.training.BASE_LR = args.lr
    config.training.AMP = args.amp
    
    # Loss-specific parameters
    if args.loss in ['triplet', 'semi_hard']:
        config.training.MARGIN = args.margin
    elif args.loss in ['infonce', 'supcon', 'ntxent']:
        config.training.TEMP = args.temperature
    
    config.training.CENTER_WT = args.center_weight
    config.training.WARMUP_EPOCHS = args.warmup_epochs
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Flexible training with any model-loss combination')
    
    # Model and loss selection
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Model architecture to use')
    parser.add_argument('--loss', type=str, required=True,
                       choices=list(AVAILABLE_LOSSES.keys()),
                       help='Loss function to use')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--heads', type=int, default=8,
                       help='Number of attention heads for transformers (default: 8)')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of transformer layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for transformers (default: 0.2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Base learning rate (default: 1e-3)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Number of warmup epochs (default: 2)')
    
    # Loss-specific parameters
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for triplet losses (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive losses (default: 0.07)')
    parser.add_argument('--center_weight', type=float, default=0.1,
                       help='Weight for center loss (default: 0.1)')
    
    # Training options
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use automatic mixed precision (default: True)')
    parser.add_argument('--no_amp', dest='amp', action='store_false',
                       help='Disable automatic mixed precision')
    
    # Testing options  
    parser.add_argument('--test_dims', nargs='+', type=int, default=[128],
                       help='Embedding dimensions to test (default: [128])')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = create_config_from_args(args)
    config.model.EMBEDDING_DIMS_TO_TEST = args.test_dims
    
    # Setup distributed training
    trainer = FlexibleTrainer(config)
    trainer.setup_distributed()
    
    try:
        # Load and preprocess data
        data_processor = DataProcessor(config.data)
        x_train, y_train, x_test, y_test, label_map = data_processor.get_processed_data()
        
        if is_main(trainer.rank):
            print(f"Training {args.model} with {args.loss} loss")
            print(f"Data: {len(x_train)} train, {len(x_test)} test samples")
            print(f"Classes: {len(label_map)}")
            print(f"Device: {trainer.device}")
            print(f"Distributed: {trainer.use_distributed}")
        
        # Test different embedding dimensions
        for embed_dim in config.model.EMBEDDING_DIMS_TO_TEST:
            if is_main(trainer.rank):
                print(f"\n=== Testing embedding dimension: {embed_dim} ===")
            
            # Update embedding dimension
            current_config = config
            current_config.model.EMBED_DIM = embed_dim
            
            # Create dataset
            dataset = trainer.get_dataset(x_train, y_train, args.loss)
            dataloader = trainer.create_dataloader(dataset, config.training.BATCH_SIZE)
            
            # Get effective learning rate
            lr = get_effective_lr(config.training.BASE_LR, trainer.world_size)
            
            # Create model with appropriate parameters
            model_kwargs = {
                'in_dim': data_processor.get_feature_dim(),
                'emb_dim': embed_dim
            }
            
            if args.model in ['ft_transformer', 'deep_ft']:
                model_kwargs.update({
                    'heads': args.heads,
                    'layers': args.layers,
                    'dropout': args.dropout
                })
            
            model = trainer.create_model(args.model, **model_kwargs)
            
            # Create loss function
            loss_kwargs = {}
            if args.loss in ['triplet', 'semi_hard']:
                loss_kwargs['margin'] = args.margin
            elif args.loss in ['infonce', 'supcon', 'ntxent']:
                loss_kwargs['temperature'] = args.temperature
            elif args.loss == 'center':
                loss_kwargs.update({
                    'num_classes': len(label_map),
                    'dim': embed_dim
                })
            
            criterion = get_loss(args.loss, **loss_kwargs)
            
            # Create center loss if needed (for non-triplet losses)
            center_loss = None
            if args.loss not in ['triplet', 'semi_hard', 'center']:
                center_loss = CenterLoss(len(label_map), embed_dim).to(trainer.device)
            
            # Create optimizer
            params = list(model.parameters())
            if center_loss is not None:
                params.extend(center_loss.parameters())
            
            optimizer = torch.optim.AdamW(params, lr=lr)
            
            # Create schedulers
            warmup = LambdaLR(optimizer, lambda e: min(1, (e+1)/config.training.WARMUP_EPOCHS))
            cosine = CosineAnnealingLR(optimizer, config.training.EPOCHS - config.training.WARMUP_EPOCHS)
            
            # Setup mixed precision
            scaler = torch.amp.GradScaler('cuda', enabled=config.training.AMP and trainer.device.type == 'cuda')
            
            if is_main(trainer.rank):
                print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                print(f"Learning rate: {lr}")
            
            # Training loop
            for epoch in range(config.training.EPOCHS):
                # Select training function based on loss type
                if args.loss in ['triplet', 'semi_hard']:
                    avg_loss = trainer.train_epoch_triplet(
                        model, dataloader, optimizer, criterion, scaler, epoch
                    )
                elif args.loss in ['infonce']:
                    avg_loss = trainer.train_epoch_pair(
                        model, dataloader, optimizer, criterion, scaler, epoch
                    )
                else:  # supcon, ntxent, center
                    avg_loss = trainer.train_epoch_contrastive(
                        model, dataloader, optimizer, criterion, center_loss, scaler, epoch
                    )
                
                # Update learning rate
                if epoch < config.training.WARMUP_EPOCHS:
                    warmup.step()
                else:
                    cosine.step()
                
                # Evaluate periodically
                if (epoch + 1) % config.training.CLUSTER_EVERY == 0 and is_main(trainer.rank):
                    accuracy = trainer.evaluate(model, x_test, y_test)
                    print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Final evaluation
            if is_main(trainer.rank):
                final_accuracy = trainer.evaluate(model, x_test, y_test)
                print(f"Final accuracy: {final_accuracy:.4f}")
                
                # Save results
                results = {
                    'model': args.model,
                    'loss': args.loss,
                    'embed_dim': embed_dim,
                    'test_accuracy': final_accuracy,
                    'final_loss': avg_loss,
                    'epochs': config.training.EPOCHS,
                    'batch_size': config.training.BATCH_SIZE,
                    'lr': config.training.BASE_LR,
                }
                
                # Add loss-specific parameters
                if args.loss in ['triplet', 'semi_hard']:
                    results['margin'] = args.margin
                elif args.loss in ['infonce', 'supcon', 'ntxent']:
                    results['temperature'] = args.temperature
                
                save_results(results)
    
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main() 