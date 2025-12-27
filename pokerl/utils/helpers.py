"""Utility functions and helpers."""

import yaml
from typing import Dict, Any
import os
import torch


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories for training.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)


def get_device() -> str:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        Device string
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
