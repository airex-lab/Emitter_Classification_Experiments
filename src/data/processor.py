#!/usr/bin/env python3
"""
Data processing utilities for emitter classification experiments.
Handles loading, preprocessing, and dataset creation.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, Any, Optional

from ..config import DataConfig

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class DataProcessor:
    """Handles data loading and preprocessing for emitter classification."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = None
        self.label_map = None
        
    def load_dataset(self, set_name: str) -> pd.DataFrame:
        """Load a specific dataset file."""
        path_map = {
            'set1': self.config.SET1_PATH,
            'set2': self.config.SET2_PATH, 
            'set3': self.config.SET3_PATH,
            'set5': self.config.SET5_PATH,
            'set6': self.config.SET6_PATH
        }
        
        if set_name not in path_map:
            raise ValueError(f"Unknown dataset: {set_name}")
            
        path = path_map[set_name]
        df = pd.read_excel(path)
        
        # Filter out deleted emitters
        if 'Status' in df.columns:
            df = df[df['Status'] != self.config.EXCLUDE_STATUS]
            
        # Select only required columns
        df = df[self.config.COLS]
        
        return df
    
    def load_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test datasets."""
        # Load training sets
        train_dfs = []
        for set_name in self.config.TRAIN_SETS:
            df = self.load_dataset(set_name)
            train_dfs.append(df)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Load test set
        test_df = self.load_dataset(self.config.TEST_SET)
        
        return train_df, test_df
    
    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Preprocess the dataframe into features and labels.
        
        Args:
            df: Input dataframe
            fit_scaler: Whether to fit a new scaler or use existing one
            
        Returns:
            x: Feature array
            y: Label array  
            label_map: Mapping from integer labels to original names
        """
        # Extract features and labels
        x = df[self.config.FEATS].values.astype(np.float32)
        y, uniques = pd.factorize(df[self.config.LABEL])
        
        # Create label mapping
        label_map = {i: name for i, name in enumerate(uniques)}
        
        # Scale features
        if fit_scaler or self.scaler is None:
            self.scaler = RobustScaler()
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.transform(x)
            
        return x, y, label_map
    
    def get_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Get fully processed training and test data.
        
        Returns:
            x_train, y_train, x_test, y_test, label_map
        """
        train_df, test_df = self.load_train_test_data()
        
        # Process training data (fit scaler)
        x_train, y_train, label_map = self.preprocess(train_df, fit_scaler=True)
        
        # Process test data (use fitted scaler)
        x_test, y_test, _ = self.preprocess(test_df, fit_scaler=False)
        
        return x_train, y_train, x_test, y_test, label_map
    
    def get_feature_dim(self) -> int:
        """Get the number of input features."""
        return len(self.config.FEATS)
    
    def get_num_classes(self) -> int:
        """Get the number of unique classes."""
        if self.label_map is None:
            # Load a small sample to get class count
            train_df, _ = self.load_train_test_data()
            _, _, label_map = self.preprocess(train_df, fit_scaler=False)
            self.label_map = label_map
        return len(self.label_map) 