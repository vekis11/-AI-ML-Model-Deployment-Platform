"""
Model configuration for training parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration class for model training parameters."""
    
    # Model architecture
    model_type: str = "tensorflow"
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 10
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Data parameters
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Augmentation
    data_augmentation: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    
    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Azure ML specific
    compute_target: str = "training-cluster"
    experiment_name: str = "model-training"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'train_split': self.train_split,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'data_augmentation': self.data_augmentation,
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'horizontal_flip': self.horizontal_flip,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience,
            'reduce_lr_factor': self.reduce_lr_factor,
            'compute_target': self.compute_target,
            'experiment_name': self.experiment_name
        }
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False) 