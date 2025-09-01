#!/usr/bin/env python3
"""
Main training script for dual encoder with InfoNCE loss experiments.
"""

import torch
import torch.amp
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.config import get_dual_encoder_config
from src.data.processor import DataProcessor
from src.data.datasets import PairPDWDataset
from src.models.architectures import get_model
from src.losses.contrastive import get_loss
from src.training.trainer import get_trainer
from src.utils.distributed import is_main, get_effective_lr, concat_all_gather
from src.utils.evaluation import save_results

class DualEncoderTrainer:
    """Custom trainer for dual encoder with cross-GPU in-batch negatives."""
    
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
    
    def train_epoch(self, model, dataloader, optimizer, criterion, scaler, epoch):
        """Train for one epoch."""
        model.train()
        dataloader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for a, p in dataloader:
            a = a.to(self.device, non_blocking=True)
            p = p.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=self.config.training.AMP):
                z_a_local = model(a)  # [B, D], requires grad
                z_p_local = model(p)  # [B, D], requires grad
            
            # Gather GLOBAL banks for negatives, but DETACHED (no graph)
            z_a_all = concat_all_gather(z_a_local.detach())
            z_p_all = concat_all_gather(z_p_local.detach())
            
            # Build label targets aligned to global diagonal
            B = z_a_local.size(0)
            targets = torch.arange(B, device=a.device) + self.rank * B
            
            # Compute logits and losses where gradients flow from local side only
            with torch.amp.autocast('cuda', enabled=self.config.training.AMP):
                logits_qk = (z_a_local @ z_p_all.t()) / self.config.training.TEMP
                loss_qk = torch.nn.functional.cross_entropy(logits_qk, targets)
                
                if self.config.training.USE_SYMMETRIC:
                    logits_kq = (z_p_local @ z_a_all.t()) / self.config.training.TEMP
                    loss_kq = torch.nn.functional.cross_entropy(logits_kq, targets)
                    loss = 0.5 * (loss_qk + loss_kq)
                else:
                    loss = loss_qk
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
    config = get_dual_encoder_config()
    
    # Setup distributed training
    trainer = DualEncoderTrainer(config)
    trainer.setup_distributed()
    
    try:
        # Load and preprocess data
        data_processor = DataProcessor(config.data)
        x_train, y_train, x_test, y_test, label_map = data_processor.get_processed_data()
        
        # Create dataset and dataloader
        dataset = PairPDWDataset(x_train, y_train)
        dataloader = trainer.create_dataloader(dataset, config.training.BATCH_SIZE)
        
        # Get effective learning rate for distributed training
        lr = get_effective_lr(config.training.BASE_LR, trainer.world_size)
        
        if is_main(trainer.rank):
            print(f"[Rank0] Using {trainer.world_size} GPUs, effective LR={lr}")
        
        # Setup mixed precision
        scaler = torch.amp.GradScaler('cuda', enabled=config.training.AMP)
        
        # Test different embedding dimensions
        for dim in config.model.EMBEDDING_DIMS_TO_TEST:
            if is_main(trainer.rank):
                print(f"\n===== Embedding {dim} =====")
            
            # Create model
            model = trainer.create_model(
                get_model, 
                model_type='residual',
                in_dim=data_processor.get_feature_dim(),
                emb_dim=dim,
                hidden_dim=config.model.HIDDEN_DIM,
                dropout=config.model.RESIDUAL_DROPOUT
            )
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Training loop
            for epoch in range(config.training.EPOCHS):
                avg_loss = trainer.train_epoch(
                    model, dataloader, optimizer, None, scaler, epoch
                )
                
                # Evaluate periodically
                if (epoch + 1) % config.training.CLUSTER_EVERY == 0 and is_main(trainer.rank):
                    acc = trainer.evaluate(model, x_test, y_test)
                    print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}  "
                          f"test-clust-acc {acc*100:5.2f}%")
            
            # Final evaluation and save results
            if is_main(trainer.rank):
                acc = trainer.evaluate(model, x_test, y_test)
                results = {
                    "embedding_dim": dim,
                    "batch_size": config.training.BATCH_SIZE
                    "test_acc": acc,
                    "final_loss": avg_loss,
                    "temperature": config.training.TEMP,
                    "symmetric": config.training.USE_SYMMETRIC,
                    "model_type": "residual",
                    "loss_type": "infonce"
                }
                
                save_results(results, f"{config.training.RESULT_DIR}/dual_encoder_dim_{dim}.json")
                print(f"[Done] dim={dim}  acc={acc*100:.2f}%")
    
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 