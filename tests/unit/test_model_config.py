"""
Unit tests for model configuration.
"""

import pytest
import tempfile
import os
from src.training.model_config import ModelConfig


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.model_type == "tensorflow"
        assert config.input_shape == (224, 224, 3)
        assert config.num_classes == 10
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_type="pytorch",
            input_shape=(128, 128, 1),
            num_classes=5,
            batch_size=16,
            epochs=50,
            learning_rate=0.01,
            optimizer="sgd"
        )
        
        assert config.model_type == "pytorch"
        assert config.input_shape == (128, 128, 1)
        assert config.num_classes == 5
        assert config.batch_size == 16
        assert config.epochs == 50
        assert config.learning_rate == 0.01
        assert config.optimizer == "sgd"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["model_type"] == "tensorflow"
        assert config_dict["input_shape"] == (224, 224, 3)
        assert config_dict["num_classes"] == 10
        assert config_dict["batch_size"] == 32
    
    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        yaml_content = """
model_type: pytorch
input_shape: [128, 128, 1]
num_classes: 5
batch_size: 16
epochs: 50
learning_rate: 0.01
optimizer: sgd
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            config = ModelConfig.from_yaml(yaml_path)
            
            assert config.model_type == "pytorch"
            assert config.input_shape == (128, 128, 1)
            assert config.num_classes == 5
            assert config.batch_size == 16
            assert config.epochs == 50
            assert config.learning_rate == 0.01
            assert config.optimizer == "sgd"
        finally:
            os.unlink(yaml_path)
    
    def test_save_yaml(self):
        """Test saving configuration to YAML file."""
        config = ModelConfig(
            model_type="pytorch",
            input_shape=(128, 128, 1),
            num_classes=5,
            batch_size=16
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            config.save_yaml(yaml_path)
            
            # Load and verify
            loaded_config = ModelConfig.from_yaml(yaml_path)
            
            assert loaded_config.model_type == config.model_type
            assert loaded_config.input_shape == config.input_shape
            assert loaded_config.num_classes == config.num_classes
            assert loaded_config.batch_size == config.batch_size
        finally:
            os.unlink(yaml_path)
    
    def test_data_splits_validation(self):
        """Test data split validation."""
        config = ModelConfig(
            train_split=0.7,
            validation_split=0.2,
            test_split=0.1
        )
        
        total_split = config.train_split + config.validation_split + config.test_split
        assert abs(total_split - 1.0) < 1e-6  # Should sum to 1.0
    
    def test_augmentation_config(self):
        """Test data augmentation configuration."""
        config = ModelConfig(
            data_augmentation=True,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True
        )
        
        assert config.data_augmentation is True
        assert config.rotation_range == 30
        assert config.width_shift_range == 0.3
        assert config.height_shift_range == 0.3
        assert config.horizontal_flip is True
    
    def test_callbacks_config(self):
        """Test callbacks configuration."""
        config = ModelConfig(
            early_stopping_patience=15,
            reduce_lr_patience=8,
            reduce_lr_factor=0.3
        )
        
        assert config.early_stopping_patience == 15
        assert config.reduce_lr_patience == 8
        assert config.reduce_lr_factor == 0.3 