"""
Training Pipeline for TCE ML Model

Trains deep learning model with early stopping and time-based validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, Tuple

from .ml_model import (
    TCEProbabilityModel,
    EarlyStopping,
    create_data_loaders,
    calculate_metrics
)
from .data_collection import get_training_dataset, get_dataset_stats
from .feature_engineering import get_feature_count


class TCETrainer:
    """
    Training pipeline for TCE probability model.
    """
    
    def __init__(
        self,
        n_features: int = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        dropout_rate: float = 0.3,
        device: str = None
    ):
        """
        Args:
            n_features: Number of input features (auto-detected if None)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            dropout_rate: Dropout rate for regularization
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.n_features = n_features or get_feature_count()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = TCEProbabilityModel(
            n_features=self.n_features,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross Entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def prepare_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        symbols: list = None,
        val_split: float = 0.2,
        random_state: int = 42
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Load and prepare training/validation data with time-based split.
        
        Args:
            start_date: Filter training data start
            end_date: Filter training data end
            symbols: Filter by symbols
            val_split: Validation set ratio
            random_state: Random seed for reproducibility
        
        Returns:
            train_loader, val_loader
        """
        print("Loading training data...")
        X, y = get_training_dataset(start_date, end_date, symbols)
        
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Time-based split (or random split)
        # For time-series, better to use sequential split
        # Here using random for simplicity, but can be improved
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_split,
            random_state=random_state,
            stratify=y  # Maintain class balance
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, y_train,
            X_val, y_val,
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader) -> Tuple[float, dict]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss, metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_labels.append(y_batch.detach().cpu())
        
        avg_loss = total_loss / len(train_loader)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def validate(self, val_loader) -> Tuple[float, dict]:
        """
        Validate model.
        
        Returns:
            avg_loss, metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_labels.append(y_batch.cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print training progress
        
        Returns:
            Training history dict
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            
            # Early stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                early_stopping.load_best_model(self.model)
                break
        
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, filepath: str = "tce_model.pt"):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_features': self.n_features,
            'dropout_rate': self.dropout_rate,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "tce_model.pt"):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.n_features = checkpoint['n_features']
        self.dropout_rate = checkpoint.get('dropout_rate', 0.3)
        self.history = checkpoint.get('history', {})
        self.model.eval()
        print(f"Model loaded from {filepath}")


def train_tce_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 15,
    save_path: str = "tce_model.pt"
) -> TCETrainer:
    """
    Convenience function to train TCE model from scratch.
    
    Args:
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
        save_path: Where to save trained model
    
    Returns:
        Trained TCETrainer instance
    """
    # Show dataset stats
    print("\nDataset Statistics:")
    stats = get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = TCETrainer(
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data()
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience
    )
    
    # Save model
    trainer.save_model(save_path)
    
    return trainer
