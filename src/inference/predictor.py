"""
Model predictor for inference.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Union
import tensorflow as tf
from tensorflow import keras
from azureml.core.model import Model
from azureml.core import Workspace

from ..training.model_config import ModelConfig


class ModelPredictor:
    """Model predictor for inference."""
    
    def __init__(self, model_path: str, config: ModelConfig = None):
        """Initialize the predictor."""
        self.model_path = model_path
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            self.logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, input_data: Union[np.ndarray, List]) -> np.ndarray:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to numpy array if needed
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        # Ensure correct shape
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Normalize if needed
        if input_data.max() > 1.0:
            input_data = input_data / 255.0
        
        # Make prediction
        predictions = self.model.predict(input_data)
        return predictions
    
    def predict_batch(self, input_data: np.ndarray) -> np.ndarray:
        """Make batch predictions."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Normalize if needed
        if input_data.max() > 1.0:
            input_data = input_data / 255.0
        
        # Make predictions
        predictions = self.model.predict(input_data)
        return predictions
    
    def predict_with_confidence(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make predictions with confidence scores."""
        predictions = self.predict(input_data)
        
        # Get class with highest probability
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]
        top_3_confidences = np.sort(predictions, axis=1)[:, -3:][:, ::-1]
        
        results = []
        for i in range(len(predicted_class)):
            result = {
                'predicted_class': int(predicted_class[i]),
                'confidence': float(confidence[i]),
                'top_3_predictions': [
                    {
                        'class': int(top_3_indices[i][j]),
                        'confidence': float(top_3_confidences[i][j])
                    }
                    for j in range(3)
                ]
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        }
        
        if self.config:
            info['config'] = self.config.to_dict()
        
        return info
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for prediction."""
        # Load and resize image
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.model.input_shape[1:3]  # Exclude batch dimension
        )
        
        # Convert to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalize
        img_array = img_array / 255.0
        
        return img_array
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Predict on a single image."""
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        return self.predict_with_confidence(img_array)
    
    def predict_images_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict on multiple images."""
        # Preprocess all images
        img_arrays = []
        for image_path in image_paths:
            img_array = self.preprocess_image(image_path)
            img_arrays.append(img_array)
        
        # Stack into batch
        batch = np.stack(img_arrays)
        
        # Make predictions
        return self.predict_with_confidence(batch)


class AzureMLPredictor(ModelPredictor):
    """Azure ML model predictor."""
    
    def __init__(self, workspace: Workspace, model_name: str, model_version: str = None):
        """Initialize Azure ML predictor."""
        self.workspace = workspace
        self.model_name = model_name
        self.model_version = model_version
        
        # Download model from Azure ML
        model_path = self._download_model()
        
        super().__init__(model_path)
    
    def _download_model(self) -> str:
        """Download model from Azure ML Model Registry."""
        try:
            # Get model
            if self.model_version:
                model = Model(self.workspace, name=self.model_name, version=self.model_version)
            else:
                model = Model(self.workspace, name=self.model_name)
            
            # Download model
            model_path = model.download(target_dir='./models', exist_ok=True)
            self.logger.info(f"Model downloaded to {model_path}")
            
            return model_path
        except Exception as e:
            self.logger.error(f"Failed to download model: {e}")
            raise 