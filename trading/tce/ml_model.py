"""
PyTorch Deep Neural Network for TCE Trade Probability Prediction

Binary classification model to predict P(success) for valid TCE setups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class TCEDataset(Dataset):
    """
    PyTorch Dataset for TCE training data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,) - 0 or 1
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (n, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TCEProbabilityModel(nn.Module):
    """
    Deep Neural Network for predicting TCE trade success probability.
    
    Architecture:
    - Input layer: n_features
    - Hidden layers: [128, 64, 32] with dropout
    - Output layer: 1 (probability)
    - Activation: ReLU for hidden, Sigmoid for output
    """
    
    def __init__(self, n_features: int, dropout_rate: float = 0.3):
        super(TCEProbabilityModel, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(n_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layers
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        x = torch.sigmoid(x)  # Probability [0, 1]
        
        return x
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for input features.
        
        Args:
            X: Features array (n_samples, n_features)
        
        Returns:
            Probabilities array (n_samples,)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probas = self(X_tensor).squeeze().numpy()
        return probas


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss and stops when it stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
    
    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = TCEDataset(X_train, y_train)
    val_dataset = TCEDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dict with accuracy, precision, recall, f1
    """
    y_pred_binary = (y_pred >= threshold).float()
    
    # True Positives, False Positives, False Negatives
    tp = ((y_pred_binary == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred_binary == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred_binary == 0) & (y_true == 1)).sum().float()
    tn = ((y_pred_binary == 0) & (y_true == 0)).sum().float()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }
