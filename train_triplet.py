#!/usr/bin/env python3
"""
Main training script for triplet loss experiments.
"""

import torch
import torch.amp
from torch.utils.data import DataLoader

from src.config import get_triplet_config
from src.data.processor import DataProcessor
from src.data.datasets import TripletPDWDataset
from src.models.architectures import get_model
from src.losses.contrastive import get_loss
from src.training.trainer import get_trainer
from src.utils.distributed import is_main, get_effective_lr
from src.utils.evaluation import save_results

def main():
    """Main training function."""
    # Load configuration
    config = get_triplet_config()
    
    # Setup distributed training
    trainer = get_trainer('triplet', config)
    trainer.setup_distributed()
    
    try:
        # Load and preprocess data
        data_processor = DataProcessor(config.data)
        x_train, y_train, x_test, y_test, label_map = data_processor.get_processed_data()
        
        # Create dataset and dataloader
        dataset = TripletPDWDataset(x_train, y_train)
        dataloader = trainer.create_dataloader(
            dataset, 
            batch_size=config.training.BATCH_SIZE
        )
        
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
            
            # Create loss and optimizer
            criterion = get_loss('triplet', margin=config.training.MARGIN)
            optimizer = trainer.create_optimizer(model, lr)
            
            # Training loop
            for epoch in range(config.training.EPOCHS):
                trainer.current_epoch = epoch
                avg_loss = trainer.train_epoch(
                    model, dataloader, optimizer, criterion, scaler
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
                    "test_acc": acc,
                    "final_loss": avg_loss,
                    "margin": config.training.MARGIN,
                    "model_type": "residual",
                    "loss_type": "triplet"
                }
                
                save_results(results, f"{config.training.RESULT_DIR}/triplet_dim_{dim}.json")
                print(f"[Done] dim={dim}  acc={acc*100:.2f}%")
    
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 