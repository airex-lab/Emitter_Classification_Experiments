#!/usr/bin/env python3
"""
Main training script for FT-Transformer with NT-Xent loss experiments.
"""

import torch
import torch.amp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.config import get_ft_transformer_config
from src.data.processor import DataProcessor
from src.data.datasets import EmitterDataset
from src.models.architectures import get_model
from src.losses.contrastive import get_loss, CenterLoss
from src.utils.distributed import is_main, get_effective_lr
from src.utils.evaluation import save_results

class FTTransformerTrainer:
    """Custom trainer for FT-Transformer with NT-Xent and Center loss."""
    
    def __init__(self, config):
        self.config = config
        self.rank = None
        self.world_size = None
        self.device = None
        
    def setup_distributed(self):
        """Setup distributed training."""
        dist.init_process_group(backend=self.config.training.BACKEND)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f'cuda:{self.rank}')
        
    def create_model(self, model_class, **kwargs):
        """Create and wrap model with DDP."""
        model = model_class(**kwargs).to(self.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.rank], output_device=self.rank
        )
        return model
    
    def create_dataloader(self, dataset, batch_size):
        """Create distributed dataloader."""
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.training.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
    
    def train_epoch(self, model, dataloader, optimizer, criterion, center_loss, scaler, epoch):
        """Train for one epoch."""
        model.train()
        dataloader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            f = model(x)
            
            # Compute losses
            loss = criterion(f, y) + self.config.training.CENTER_WT * center_loss(f, y)
            
            if torch.isnan(loss):
                raise RuntimeError("Loss became NaN – check data and model.")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / num_batches
    
    def evaluate(self, model, x_test, y_test):
        """Evaluate model."""
        from src.utils.evaluation import evaluate_model
        return evaluate_model(model.module, x_test, y_test, self.device)
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """Main training function."""
    # Load configuration
    config = get_ft_transformer_config()
    
    # Setup distributed training
    trainer = FTTransformerTrainer(config)
    trainer.setup_distributed()
    
    try:
        # Load and preprocess data
        data_processor = DataProcessor(config.data)
        x_train, y_train, x_test, y_test, label_map = data_processor.get_processed_data()
        
        # Create dataset and dataloader
        dataset = EmitterDataset(x_train, y_train)
        dataloader = trainer.create_dataloader(dataset, config.training.BATCH_SIZE)
        
        # Get effective learning rate for distributed training
        lr = get_effective_lr(config.training.BASE_LR, trainer.world_size)
        
        if is_main(trainer.rank):
            print(f"[Rank0] Using {trainer.world_size} GPUs, effective LR={lr}")
        
        # Create model
        model = trainer.create_model(
            get_model, 
            model_type='ft_transformer',
            num_feats=data_processor.get_feature_dim(),
            dim=config.model.EMBED_DIM,
            heads=config.model.HEADS,
            layers=config.model.LAYERS,
            dropout=config.model.DROPOUT
        )
        
        # Create losses
        criterion = get_loss('ntxent', temperature=config.training.TEMP)
        center_loss = CenterLoss(len(label_map), config.model.EMBED_DIM).to(trainer.device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            [*model.parameters(), *center_loss.parameters()], 
            lr=lr
        )
        
        # Create schedulers
        warmup = LambdaLR(optimizer, lambda e: min(1, (e+1)/config.training.WARMUP_EPOCHS))
        cosine = CosineAnnealingLR(optimizer, config.training.EPOCHS - config.training.WARMUP_EPOCHS)
        
        # Training loop
        for epoch in range(config.training.EPOCHS):
            avg_loss = trainer.train_epoch(
                model, dataloader, optimizer, criterion, center_loss, None, epoch
            )
            
            # Update schedulers
            if epoch < config.training.WARMUP_EPOCHS:
                warmup.step()
            else:
                cosine.step()
            
            # Evaluate periodically
            if (epoch + 1) % config.training.CLUSTER_EVERY == 0 and is_main(trainer.rank):
                acc = trainer.evaluate(model, x_test, y_test)
                print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}  "
                      f"clust-acc {acc*100:5.2f}%")
        
        # Final evaluation and save results
        if is_main(trainer.rank):
            acc = trainer.evaluate(model, x_test, y_test)
            results = {
                "embed_dim": config.model.EMBED_DIM,
                "layers": config.model.LAYERS,
                "heads": config.model.HEADS,
                "epochs": config.training.EPOCHS,
                "clustering_acc": acc,
                "final_loss": avg_loss,
                "temperature": config.training.TEMP,
                "center_weight": config.training.CENTER_WT,
                "model_type": "ft_transformer",
                "loss_type": "ntxent"
            }
            
            save_results(results, f"{config.training.RESULT_DIR}/ft_transformer_{config.model.EMBED_DIM}.json")
            print(f"[Done] Final clustering accuracy {acc*100:.2f}%")
    
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 