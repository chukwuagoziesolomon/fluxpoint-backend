"""
Automatic Training Problem Fixing

Applies automatic corrections based on detected issues.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AutoTrainingFixer:
    """
    Automatically fixes training problems:
    
    1. Applies gradient clipping
    2. Adjusts learning rate
    3. Implements early stopping
    4. Fixes mode collapse with class weights
    5. Removes outliers
    6. Applies feature scaling
    7. Handles NaN/Inf recovery
    """
    
    def __init__(self):
        self.applied_fixes = []
        self.clip_norm = None
        self.lr_scale_factor = 1.0
    
    def apply_gradient_clipping(
        self,
        model: nn.Module,
        max_norm: float = 1.0
    ) -> None:
        """
        Apply gradient clipping to prevent exploding gradients.
        
        Args:
            model: Neural network model
            max_norm: Maximum gradient norm
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        self.clip_norm = max_norm
        
        fix_msg = f"✅ Applied gradient clipping (max_norm={max_norm})"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
    
    def adjust_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        scale_factor: float
    ) -> float:
        """
        Adjust learning rate by scale factor.
        
        Args:
            optimizer: PyTorch optimizer
            scale_factor: Multiply LR by this (0.5 = halve, 2.0 = double)
        
        Returns:
            new_lr: Updated learning rate
        """
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = old_lr * scale_factor
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.lr_scale_factor *= scale_factor
        
        fix_msg = f"✅ Adjusted learning rate: {old_lr:.6f} → {new_lr:.6f} ({scale_factor}x)"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
        
        return new_lr
    
    def implement_early_stopping(
        self,
        model: nn.Module,
        best_model_state: Dict,
        current_val_loss: float,
        best_val_loss: float
    ) -> Tuple[nn.Module, bool]:
        """
        Implement early stopping - restore best model if stuck.
        
        Args:
            model: Current model
            best_model_state: Saved state dict of best model
            current_val_loss: Current validation loss
            best_val_loss: Best validation loss so far
        
        Returns:
            model: Restored model
            should_stop: Whether to stop training
        """
        if current_val_loss > best_val_loss * 1.1:  # 10% worse
            # Restore best model
            model.load_state_dict(best_model_state)
            
            fix_msg = f"✅ Restored best model (val_loss: {best_val_loss:.4f})"
            self.applied_fixes.append(fix_msg)
            logger.info(fix_msg)
            
            return model, True
        
        return model, False
    
    def fix_mode_collapse(
        self,
        y_train: np.ndarray
    ) -> torch.Tensor:
        """
        Calculate class weights to fix mode collapse.
        
        Args:
            y_train: Training labels
        
        Returns:
            class_weights: Tensor of class weights
        """
        pos_samples = np.sum(y_train == 1)
        neg_samples = np.sum(y_train == 0)
        total = len(y_train)
        
        # Inverse frequency weighting
        pos_weight = total / (2 * pos_samples) if pos_samples > 0 else 1.0
        neg_weight = total / (2 * neg_samples) if neg_samples > 0 else 1.0
        
        class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32)
        
        fix_msg = f"✅ Applied class weights: neg={neg_weight:.2f}, pos={pos_weight:.2f}"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
        
        return class_weights
    
    def remove_outliers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        outlier_indices: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outlier samples from training data.
        
        Args:
            X: Feature matrix
            y: Labels
            outlier_indices: Indices to remove
        
        Returns:
            X_clean: Cleaned features
            y_clean: Cleaned labels
        """
        mask = np.ones(len(X), dtype=bool)
        mask[outlier_indices] = False
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        fix_msg = f"✅ Removed {len(outlier_indices)} outliers ({len(outlier_indices)/len(X)*100:.1f}%)"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
        
        return X_clean, y_clean
    
    def apply_feature_scaling(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Apply standard scaling to features.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
        
        Returns:
            X_train_scaled: Scaled training features
            X_val_scaled: Scaled validation features (if provided)
            scaler_params: Mean and std for each feature
        """
        # Calculate statistics from training data only
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        
        # Avoid division by zero
        stds[stds == 0] = 1.0
        
        # Scale training data
        X_train_scaled = (X_train - means) / stds
        
        # Scale validation data with same parameters
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = (X_val - means) / stds
        
        scaler_params = {
            'means': means.tolist(),
            'stds': stds.tolist()
        }
        
        fix_msg = f"✅ Applied StandardScaler (zero mean, unit variance)"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
        
        return X_train_scaled, X_val_scaled, scaler_params
    
    def handle_nan_inf(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        backup_state: Dict
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Recover from NaN/Inf by restoring backup and reducing LR.
        
        Args:
            model: Current model (with NaN/Inf)
            optimizer: Current optimizer
            backup_state: Last good model state
        
        Returns:
            model: Restored model
            optimizer: Optimizer with reduced LR
        """
        # Restore last good state
        model.load_state_dict(backup_state)
        
        # Drastically reduce learning rate
        new_lr = self.adjust_learning_rate(optimizer, scale_factor=0.1)
        
        # Apply aggressive gradient clipping
        self.apply_gradient_clipping(model, max_norm=0.5)
        
        fix_msg = f"✅ Recovered from NaN/Inf: restored backup, LR={new_lr:.6f}, clip_norm=0.5"
        self.applied_fixes.append(fix_msg)
        logger.warning(fix_msg)
        
        return model, optimizer
    
    def add_batch_normalization(
        self,
        model: nn.Module
    ) -> nn.Module:
        """
        Add batch normalization layers to help with gradient flow.
        
        Note: This creates a new model architecture. Should be done
        before training starts, not during.
        
        Args:
            model: Original model
        
        Returns:
            new_model: Model with batch norm layers
        """
        # This is more complex - would need to recreate model architecture
        # For now, just log recommendation
        
        fix_msg = "⚠️  RECOMMENDATION: Add batch normalization layers to model architecture"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
        
        return model
    
    def apply_learning_rate_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = 'cosine',
        T_max: int = 100
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Apply learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            schedule_type: 'cosine', 'step', or 'plateau'
            T_max: Maximum iterations for cosine schedule
        
        Returns:
            scheduler: LR scheduler object
        """
        if schedule_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=1e-6
            )
        elif schedule_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif schedule_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        fix_msg = f"✅ Applied {schedule_type} learning rate scheduler"
        self.applied_fixes.append(fix_msg)
        logger.info(fix_msg)
        
        return scheduler
    
    def get_applied_fixes_summary(self) -> str:
        """
        Get summary of all fixes applied during training.
        
        Returns:
            summary: Formatted string of all fixes
        """
        if not self.applied_fixes:
            return "No automatic fixes were needed - training was healthy! ✅"
        
        summary = [
            "\n" + "="*80,
            "AUTOMATIC FIXES APPLIED DURING TRAINING",
            "="*80,
            ""
        ]
        
        for i, fix in enumerate(self.applied_fixes, 1):
            summary.append(f"{i}. {fix}")
        
        summary.extend([
            "",
            f"Total fixes applied: {len(self.applied_fixes)}",
            "="*80 + "\n"
        ])
        
        return "\n".join(summary)
    
    def reset(self):
        """Reset fixer state for new training run."""
        self.applied_fixes = []
        self.clip_norm = None
        self.lr_scale_factor = 1.0


def auto_fix_training_iteration(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    diagnostics: Dict,
    fixer: AutoTrainingFixer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    backup_model_state: Dict
) -> Tuple[nn.Module, torch.optim.Optimizer, np.ndarray, np.ndarray]:
    """
    Apply automatic fixes based on diagnostics.
    
    Args:
        model: Current model
        optimizer: Current optimizer
        diagnostics: Dictionary of all diagnostic checks
        fixer: AutoTrainingFixer instance
        X_train: Training features
        y_train: Training labels
        backup_model_state: Last good model state
    
    Returns:
        model: Fixed model
        optimizer: Fixed optimizer
        X_train: Potentially cleaned data
        y_train: Potentially cleaned labels
    """
    # Priority 1: Handle NaN/Inf (critical)
    if diagnostics['nan_check']['should_stop_training']:
        model, optimizer = fixer.handle_nan_inf(model, optimizer, backup_model_state)
        return model, optimizer, X_train, y_train
    
    # Priority 2: Fix gradient issues
    grad_check = diagnostics['gradient_check']
    if grad_check['should_apply_clipping']:
        max_norm = 1.0 if grad_check['severity'] == 'severe' else 5.0
        fixer.apply_gradient_clipping(model, max_norm)
    
    # Priority 3: Adjust learning rate
    lr_check = diagnostics['lr_check']
    if lr_check['severity'] != 'none':
        fixer.adjust_learning_rate(optimizer, lr_check['recommended_lr_adjustment'])
    
    # Priority 4: Fix mode collapse with class weights
    # (This would need to be integrated into loss function - return weights)
    
    # Priority 5: Remove outliers (only once at start)
    outlier_check = diagnostics['outlier_check']
    if outlier_check['outlier_percentage'] > 10:  # Only if severe
        X_train, y_train = fixer.remove_outliers(
            X_train,
            y_train,
            outlier_check['outlier_indices']
        )
    
    return model, optimizer, X_train, y_train
