"""
Bat Species Classifier Model
A deep learning model for classifying Indian bat species from audio spectrograms
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50
import numpy as np


class BatSpeciesClassifier:
    """
    Deep Learning model for bat species classification
    """
    
    def __init__(self, num_classes=10, input_shape=(128, 128, 3), 
                 model_type='cnn', use_pretrained=False):
        """
        Initialize the bat species classifier
        
        Args:
            num_classes (int): Number of bat species to classify
            input_shape (tuple): Input shape for spectrograms (height, width, channels)
            model_type (str): Type of model ('cnn', 'efficientnet', 'resnet')
            use_pretrained (bool): Whether to use pretrained weights
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_type = model_type
        self.use_pretrained = use_pretrained
        self.model = None
        
    def build_cnn_model(self):
        """
        Build a custom CNN model for bat species classification
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_efficientnet_model(self):
        """
        Build model using EfficientNet as backbone
        """
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet' if self.use_pretrained else None,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        if self.use_pretrained:
            base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet_model(self):
        """
        Build model using ResNet50 as backbone
        """
        base_model = ResNet50(
            include_top=False,
            weights='imagenet' if self.use_pretrained else None,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        if self.use_pretrained:
            base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """
        Build the model based on specified type
        """
        if self.model_type == 'cnn':
            self.model = self.build_cnn_model()
        elif self.model_type == 'efficientnet':
            self.model = self.build_efficientnet_model()
        elif self.model_type == 'resnet':
            self.model = self.build_resnet_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate (float): Learning rate for optimizer
            optimizer (str): Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if self.model is None:
            self.build_model()
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train the model first.")
    
    def load_model(self, filepath):
        """
        Load a saved model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, X):
        """
        Make predictions on input data
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Build/load model first.")
        
        return self.model.predict(X)
    
    def unfreeze_base_model(self, num_layers=None):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            num_layers (int): Number of layers to unfreeze from the end
        """
        if self.model is None:
            raise ValueError("Model not built. Build model first.")
        
        if self.model_type in ['efficientnet', 'resnet']:
            base_model = self.model.layers[0]
            base_model.trainable = True
            
            if num_layers is not None:
                # Freeze all layers except the last num_layers
                for layer in base_model.layers[:-num_layers]:
                    layer.trainable = False
            
            print(f"Base model layers unfrozen for fine-tuning")
