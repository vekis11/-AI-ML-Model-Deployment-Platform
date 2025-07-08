"""
Model trainer for TensorFlow models with Azure ML integration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import mlflow
import mlflow.tensorflow
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies

from .model_config import ModelConfig
from .data_loader import DataLoader


class ModelTrainer:
    """Trainer class for TensorFlow models with Azure ML integration."""
    
    def __init__(self, config: ModelConfig, workspace: Optional[Workspace] = None):
        """Initialize the trainer."""
        self.config = config
        self.workspace = workspace
        self.model = None
        self.history = None
        self.logger = logging.getLogger(__name__)
        
        # Setup MLflow
        if workspace:
            mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    
    def build_model(self) -> keras.Model:
        """Build the TensorFlow model architecture."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.config.input_shape),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.config.num_classes, activation='softmax')
        ])
        
        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def _get_optimizer(self) -> optimizers.Optimizer:
        """Get optimizer based on configuration."""
        if self.config.optimizer.lower() == 'adam':
            return optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            return optimizers.SGD(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'rmsprop':
            return optimizers.RMSprop(learning_rate=self.config.learning_rate)
        else:
            return optimizers.Adam(learning_rate=self.config.learning_rate)
    
    def _get_callbacks(self) -> list:
        """Get training callbacks."""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7
            ),
            # Model checkpoint
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Add TensorBoard callback if workspace is available
        if self.workspace:
            callbacks_list.append(
                callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            )
        
        return callbacks_list
    
    def train(self, train_data, validation_data, test_data=None) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        # Get callbacks
        callbacks_list = self._get_callbacks()
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config.to_dict())
            
            # Train the model
            self.history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=self.config.epochs,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Evaluate on test data if available
            test_metrics = {}
            if test_data:
                test_loss, test_accuracy, test_top5 = self.model.evaluate(test_data, verbose=0)
                test_metrics = {
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_top5_accuracy': test_top5
                }
                mlflow.log_metrics(test_metrics)
            
            # Log model
            mlflow.tensorflow.log_model(self.model, "model")
            
            # Log training history
            for epoch, (loss, accuracy, top5) in enumerate(zip(
                self.history.history['loss'],
                self.history.history['accuracy'],
                self.history.history['top_5_categorical_accuracy']
            )):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("train_accuracy", accuracy, step=epoch)
                mlflow.log_metric("train_top5_accuracy", top5, step=epoch)
            
            # Save model locally
            self.model.save('final_model.h5')
            mlflow.log_artifact('final_model.h5')
            
            return {
                'history': self.history.history,
                'test_metrics': test_metrics,
                'model_path': 'final_model.h5'
            }
    
    def save_model_to_azure(self, model_name: str, model_description: str = "") -> str:
        """Save model to Azure ML Model Registry."""
        if not self.workspace:
            raise ValueError("Workspace is required to save model to Azure")
        
        if self.model is None:
            raise ValueError("No trained model available")
        
        # Create conda dependencies
        conda_deps = CondaDependencies()
        conda_deps.add_conda_package("tensorflow=2.13.0")
        conda_deps.add_conda_package("numpy=1.24.3")
        conda_deps.add_conda_package("pillow=10.0.0")
        
        # Save model
        model = Model.register(
            workspace=self.workspace,
            model_path='final_model.h5',
            model_name=model_name,
            description=model_description,
            conda_file=conda_deps,
            tags={
                'framework': 'tensorflow',
                'version': '2.13.0',
                'input_shape': str(self.config.input_shape),
                'num_classes': str(self.config.num_classes)
            }
        )
        
        self.logger.info(f"Model registered with ID: {model.id}")
        return model.id
    
    def load_model(self, model_path: str) -> keras.Model:
        """Load a trained model."""
        self.model = keras.models.load_model(model_path)
        return self.model
    
    def evaluate_model(self, test_data) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        loss, accuracy, top5_accuracy = self.model.evaluate(test_data, verbose=0)
        
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_top5_accuracy': top5_accuracy
        }
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "No model available"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def export_model_info(self, output_path: str) -> None:
        """Export model information to JSON file."""
        if self.model is None:
            raise ValueError("No model available")
        
        model_info = {
            'config': self.config.to_dict(),
            'model_summary': self.get_model_summary(),
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        }
        
        if self.history:
            model_info['training_history'] = {
                'final_train_accuracy': self.history.history['accuracy'][-1],
                'final_val_accuracy': self.history.history['val_accuracy'][-1],
                'best_val_accuracy': max(self.history.history['val_accuracy']),
                'epochs_trained': len(self.history.history['accuracy'])
            }
        
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2) 