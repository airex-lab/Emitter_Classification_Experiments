#!/usr/bin/env python3
"""
Centralized configuration for emitter classification experiments.
Contains all hyperparameters, data settings, and training configurations.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    """Data processing and dataset configuration."""
    # Column definitions
    COLS = ['Name', 'PW(µs)', 'Azimuth(º)', 'Elevation(º)', 'Power(dBm)', 'Freq(MHz)']
    FEATS = COLS[1:]  # All columns except 'Name'
    LABEL = 'Name'
    
    # Data paths
    DATA_DIR = "dataset"
    SET1_PATH = os.path.join(DATA_DIR, "set1.xls")
    SET2_PATH = os.path.join(DATA_DIR, "set2.xls") 
    SET3_PATH = os.path.join(DATA_DIR, "set3.xlsx")
    SET5_PATH = os.path.join(DATA_DIR, "set5.xlsx")
    SET6_PATH = os.path.join(DATA_DIR, "set6.xlsx")
    
    # Train/test split
    TRAIN_SETS = ['set1', 'set2', 'set5', 'set6']  # sets to use for training
    TEST_SET = 'set3'  # set to use for testing
    
    # Data filtering
    EXCLUDE_STATUS = 'DELETE_EMITTER'

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Embedding dimensions to test
    EMBEDDING_DIMS_TO_TEST: List[int] = None
    
    # FT-Transformer specific
    EMBED_DIM: int = 128
    LAYERS: int = 3
    HEADS: int = 8
    DROPOUT: float = 0.2
    
    # Residual network specific  
    HIDDEN_DIM: int = 64
    RESIDUAL_DROPOUT: float = 0.3
    
    def __post_init__(self):
        if self.EMBEDDING_DIMS_TO_TEST is None:
            self.EMBEDDING_DIMS_TO_TEST = [32, 16, 64, 8, 4, 2]

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Basic training params
    EPOCHS: int = 100
    BASE_LR: float = 1e-3
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 8
    
    # Loss specific params
    MARGIN: float = 1.0  # for triplet loss
    TEMP: float = 0.07   # temperature for contrastive losses
    CENTER_WT: float = 0.1  # weight for center loss
    
    # Training features
    AMP: bool = True  # automatic mixed precision
    USE_SYMMETRIC: bool = True  # for dual encoder
    WARMUP_EPOCHS: int = 2
    
    # Evaluation
    CLUSTER_EVERY: int = 10
    RESULT_DIR: str = "results"
    
    # Distributed training
    BACKEND: str = "nccl"
    
    def __post_init__(self):
        os.makedirs(self.RESULT_DIR, exist_ok=True)

@dataclass 
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()

def get_triplet_config():
    """Configuration for triplet loss training."""
    config = ExperimentConfig()
    config.training.EPOCHS = 100
    config.training.MARGIN = 1.0
    return config

def get_dual_encoder_config():
    """Configuration for dual encoder with InfoNCE."""
    config = ExperimentConfig()
    config.training.EPOCHS = 100
    config.training.TEMP = 0.07
    config.training.USE_SYMMETRIC = True
    return config

def get_supcon_config():
    """Configuration for supervised contrastive learning."""
    config = ExperimentConfig()
    config.training.EPOCHS = 100
    config.training.TEMP = 0.07
    return config

def get_ft_transformer_config():
    """Configuration for FT-Transformer with NT-Xent."""
    config = ExperimentConfig()
    config.model.EMBED_DIM = 128
    config.model.LAYERS = 3
    config.model.HEADS = 8
    config.training.EPOCHS = 60
    config.training.BATCH_SIZE = 256
    config.training.TEMP = 0.1
    config.training.CENTER_WT = 0.1
    return config

def get_ft_triplet_config():
    """Configuration for FT-Transformer with triplet loss."""
    config = ExperimentConfig()
    config.model.EMBED_DIM = 192
    config.model.LAYERS = 6
    config.model.HEADS = 8
    config.training.EPOCHS = 80
    config.training.BATCH_SIZE = 128
    config.training.MARGIN = 0.3
    return config 