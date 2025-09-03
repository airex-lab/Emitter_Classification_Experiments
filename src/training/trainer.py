#!/usr/bin/env python3
"""
Base trainer class for emitter classification experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Dict, Any, Optional, Callable

from ..config import ExperimentConfig
from ..utils.distributed import setup_distributed, is_main, suppress_non_main_print, cleanup_distributed
from ..utils.evaluation import evaluate_model, save_results

class BaseTrainer:
    """Base trainer class for emitter classification experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rank = None
        self.world_size = None
        self.device = None
        
    def setup_distributed(self):
        """Setup distributed training."""
        self.rank, self.world_size = setup_distributed(self.config.training.BACKEND)
        self.device = torch.device(f'cuda:{self.rank}')
        suppress_non_main_print(self.rank)
        
    def create_model(self, model_class, **kwargs) -> nn.Module:
        """Create and wrap model with DDP."""
        model = model_class(**kwargs).to(self.device)
        model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        return model
    
    def create_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        """Create optimizer."""
        return optim.Adam(model.parameters(), lr=lr)
    
    def create_scheduler(self, optimizer: optim.Optimizer, 
                        warmup_epochs: int, total_epochs: int) -> Optional[Any]:
        """Create learning rate scheduler."""
        if warmup_epochs > 0:
            warmup = LambdaLR(optimizer, lambda e: min(1, (e+1)/warmup_epochs))
            cosine = CosineAnnealingLR(optimizer, total_epochs - warmup_epochs)
            return warmup, cosine
        else:
            return CosineAnnealingLR(optimizer, total_epochs)
    
    def create_dataloader(self, dataset, batch_size: int, 
                         shuffle: bool = True) -> DataLoader:
        """Create distributed dataloader."""
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.training.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   scaler: Optional[torch.amp.GradScaler] = None) -> float:
        """Train for one epoch."""
        model.train()
        dataloader.sampler.set_epoch(self.current_epoch)
        
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda', enabled=self.config.training.AMP):
                    loss = self.forward_step(model, batch, criterion)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self.forward_step(model, batch, criterion)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / num_batches
    
    def forward_step(self, model: nn.Module, batch, criterion: nn.Module) -> torch.Tensor:
        """Forward step - to be implemented by subclasses."""
        raise NotImplementedError
    
    def evaluate(self, model: nn.Module, x_test: torch.Tensor, 
                y_test: torch.Tensor) -> float:
        """Evaluate model."""
        return evaluate_model(model.module, x_test, y_test, self.device)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, loss: float, filepath: str):
        """Save training checkpoint."""
        if is_main(self.rank):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': self.config
            }
            torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       filepath: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def cleanup(self):
        """Cleanup distributed training."""
        cleanup_distributed()

class TripletTrainer(BaseTrainer):
    """Trainer for triplet loss training."""
    
    def forward_step(self, model: nn.Module, batch, criterion: nn.Module) -> torch.Tensor:
        anchor, positive, negative = [b.to(self.device) for b in batch]
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        return criterion(anchor_emb, positive_emb, negative_emb)

class ContrastiveTrainer(BaseTrainer):
    """Trainer for contrastive learning."""
    
    def forward_step(self, model: nn.Module, batch, criterion: nn.Module) -> torch.Tensor:
        anchor, positive = [b.to(self.device) for b in batch]
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        return criterion(anchor_emb, positive_emb)

class SupConTrainer(BaseTrainer):
    """Trainer for supervised contrastive learning."""
    
    def forward_step(self, model: nn.Module, batch, criterion: nn.Module) -> torch.Tensor:
        x, y = [b.to(self.device) for b in batch]
        z_local = model(x)
        
        # Gather global embeddings for contrastive learning
        z_global = self.gather_embeddings(z_local.detach())
        y_global = self.gather_labels(y)
        
        return criterion(z_local, y, z_global, y_global)
    
    def gather_embeddings(self, z_local: torch.Tensor) -> torch.Tensor:
        """Gather embeddings from all GPUs."""
        from ..utils.distributed import concat_all_gather
        return concat_all_gather(z_local)
    
    def gather_labels(self, y_local: torch.Tensor) -> torch.Tensor:
        """Gather labels from all GPUs."""
        from ..utils.distributed import concat_all_gather
        return concat_all_gather(y_local)

def get_trainer(trainer_type: str, config: ExperimentConfig) -> BaseTrainer:
    """
    Factory function to create trainers.
    
    Args:
        trainer_type: Type of trainer ('triplet', 'contrastive', 'supcon')
        config: Experiment configuration
        
    Returns:
        Trainer instance
    """
    if trainer_type == 'triplet':
        return TripletTrainer(config)
    elif trainer_type == 'contrastive':
        return ContrastiveTrainer(config)
    elif trainer_type == 'supcon':
        return SupConTrainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}") 