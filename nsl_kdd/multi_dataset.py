"""
Multi-dataset loader for IDS experiments: NSL-KDD, UNSW-NB15, CICIDS2017
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_unsw(project_root: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load UNSW-NB15 dataset (training and testing sets).
    
    Returns:
        (train_df, test_df) - Both with binary labels (0=normal, 1=attack)
    """
    root = Path(project_root)
    train_path = root / "UNSW" / "UNSW_NB15_training-set.csv"
    test_path = root / "UNSW" / "UNSW_NB15_testing-set.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"UNSW dataset not found in {root}/UNSW/")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Ensure binary labels (0=normal, 1=attack)
    if 'label' not in train_df.columns:
        raise ValueError("UNSW dataset must have 'label' column")
    
    # Convert to binary if needed
    if train_df['label'].dtype == object:
        train_df['label'] = (train_df['label'] != 0).astype(int)
    if test_df['label'].dtype == object:
        test_df['label'] = (test_df['label'] != 0).astype(int)
    
    return train_df, test_df


def load_cicids(project_root: str | Path, max_samples_per_file: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CICIDS2017 dataset by concatenating all CSV files.
    
    Splits into 80% training, 20% testing.
    
    Args:
        project_root: Project root directory
        max_samples_per_file: Max samples to load per file (for testing); None=all
        
    Returns:
        (train_df, test_df) - Both with binary labels (0=normal, 1=attack)
    """
    root = Path(project_root)
    cicids_dir = root / "CICIDS2017"
    
    if not cicids_dir.exists():
        raise FileNotFoundError(f"CICIDS dataset not found in {root}/CICIDS2017/")
    
    csv_files = sorted([f for f in os.listdir(cicids_dir) if f.endswith('.csv')])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cicids_dir}")
    
    dfs = []
    for csv_file in csv_files:
        file_path = cicids_dir / csv_file
        print(f"  Loading {csv_file}...")
        
        nrows = max_samples_per_file if max_samples_per_file else None
        df = pd.read_csv(file_path, nrows=nrows)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Get label column (should be ' Label' in CICIDS2017)
    label_col = [c for c in combined_df.columns if 'label' in c.lower()][0]
    
    # Convert to binary: 0=BENIGN, 1=ATTACK
    combined_df['label'] = (combined_df[label_col] != 'BENIGN').astype(int)
    
    # Remove original label column
    combined_df = combined_df.drop(columns=[label_col])
    
    # Split 80/20
    n_train = int(len(combined_df) * 0.8)
    train_df = combined_df.iloc[:n_train].reset_index(drop=True)
    test_df = combined_df.iloc[n_train:].reset_index(drop=True)
    
    return train_df, test_df


def get_dataset(dataset_name: str, project_root: str | Path, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset by name.
    
    Args:
        dataset_name: 'nsl_kdd', 'unsw', or 'cicids'
        project_root: Project root directory
        **kwargs: Additional arguments (e.g., max_samples_per_file for cicids)
        
    Returns:
        (train_df, test_df)
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == 'nsl_kdd':
        from .data import load_nsl_kdd
        root = Path(project_root)
        train_path, test_path = root / "KDDTrain+.txt", root / "KDDTest+.txt"
        train_df = load_nsl_kdd(train_path)
        test_df = load_nsl_kdd(test_path)
        return train_df, test_df
    
    elif dataset_name == 'unsw':
        return load_unsw(project_root)
    
    elif dataset_name == 'cicids':
        return load_cicids(project_root, **kwargs)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: nsl_kdd, unsw, cicids")

