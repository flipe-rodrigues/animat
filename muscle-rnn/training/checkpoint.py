"""
Checkpoint management utilities.

Handles saving and loading of training checkpoints across different training methods.
"""

import torch
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ModelConfig


def save_checkpoint(
    path: str,
    model_config: ModelConfig,
    model_state_dict: Dict[str, torch.Tensor],
    training_state: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model_config: Model configuration
        model_state_dict: Controller state dictionary
        training_state: Training algorithm state (optimizer, generation, etc.)
        metadata: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_config': model_config.to_dict(),
        'model_state_dict': model_state_dict,
        'training_state': training_state,
        'metadata': metadata or {},
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def save_config(config: Any, path: str) -> None:
    """
    Save a configuration object to JSON.
    
    Args:
        config: Configuration object with to_dict() method
        path: Path to save configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str, config_class: type) -> Any:
    """
    Load a configuration from JSON.
    
    Args:
        path: Path to configuration file
        config_class: Configuration class to instantiate
        
    Returns:
        Configuration object
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    if hasattr(config_class, 'from_dict'):
        return config_class.from_dict(config_dict)
    else:
        return config_class(**config_dict)


def save_stats(stats: Dict[str, Any], path: str) -> None:
    """
    Save statistics to pickle file.
    
    Args:
        stats: Statistics dictionary
        path: Path to save statistics
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(stats, f)


def load_stats(path: str) -> Dict[str, Any]:
    """
    Load statistics from pickle file.
    
    Args:
        path: Path to statistics file
        
    Returns:
        Statistics dictionary
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_output_directory(base_dir: str, experiment_name: str = None) -> Path:
    """
    Create output directory for experiment.
    
    Args:
        base_dir: Base output directory
        experiment_name: Optional experiment name
        
    Returns:
        Path to created directory
    """
    output_dir = Path(base_dir)
    
    if experiment_name:
        output_dir = output_dir / experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir
