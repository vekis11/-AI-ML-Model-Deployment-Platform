"""
Data loader for model training.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    """Data loader for handling training data."""
    
    def __init__(self, config: 'ModelConfig'):
        """Initialize data loader with configuration."""
        self.config = config
        self.label_encoder = LabelEncoder()
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
    
    def load_data_from_directory(self, data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load data from directory structure."""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.rotation_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            horizontal_flip=self.config.horizontal_flip,
            validation_split=self.config.validation_split
        )
        
        # Only rescaling for validation/test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        self.validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Load test data (using same directory but different generator)
        self.test_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical'
        )
        
        return self.train_generator, self.validation_generator, self.test_generator
    
    def load_data_from_csv(self, csv_path: str, image_column: str, label_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file with image paths and labels."""
        df = pd.read_csv(csv_path)
        
        # Load images
        images = []
        for img_path in df[image_column]:
            img = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=self.config.input_shape[:2]
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
        
        images = np.array(images)
        images = images / 255.0  # Normalize
        
        # Encode labels
        labels = self.label_encoder.fit_transform(df[label_column])
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.config.num_classes)
        
        return images, labels
    
    def split_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets."""
        # First split: train + temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, 
            test_size=(1 - self.config.train_split), 
            random_state=42, 
            stratify=np.argmax(labels, axis=1)
        )
        
        # Second split: validation + test
        val_size = self.config.validation_split / (self.config.validation_split + self.config.test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_tf_datasets(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create TensorFlow datasets with data augmentation."""
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(1000).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_class_names(self) -> list:
        """Get class names from the data generator."""
        if self.train_generator:
            return list(self.train_generator.class_indices.keys())
        return []
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        info = {
            'input_shape': self.config.input_shape,
            'num_classes': self.config.num_classes,
            'batch_size': self.config.batch_size
        }
        
        if self.train_generator:
            info['train_samples'] = self.train_generator.samples
            info['validation_samples'] = self.validation_generator.samples if self.validation_generator else 0
            info['class_names'] = self.get_class_names()
        
        return info 