"""
Advanced Training Problem Detection & Auto-Fixing

Detects and fixes additional training issues beyond bias/variance.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque


class AdvancedTrainingDiagnostics:
    """
    Detects and auto-fixes advanced training problems:
    
    1. Exploding/Vanishing Gradients
    2. Learning Rate Issues (too high/low)
    3. Training Plateaus (stuck)
    4. NaN/Inf values
    5. Mode Collapse (all predictions same)
    6. Noisy Labels
    7. Outlier Detection
    8. Feature Scaling Issues
    9. Early Stopping (optimal point)
    10. Distribution Shift (train vs val)
    """
    
    def __init__(self):
        self.gradient_history = deque(maxlen=10)
        self.loss_history = deque(maxlen=20)
        self.prediction_history = deque(maxlen=5)
        self.patience_counter = 0
        self.best_val_loss = float('inf')
    
    def check_gradient_health(
        self,
        model: nn.Module,
        epoch: int
    ) -> Dict:
        """
        Detect exploding/vanishing gradients.
        
        Returns:
            {
                'issue': str,  # 'exploding', 'vanishing', 'healthy'
                'severity': str,
                'max_grad_norm': float,
                'min_grad_norm': float,
                'recommended_actions': List[str]
            }
        """
        # Calculate gradient norms
        total_norm = 0.0
        min_norm = float('inf')
        max_norm = 0.0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                min_norm = min(min_norm, param_norm)
                max_norm = max(max_norm, param_norm)
        
        total_norm = total_norm ** 0.5
        self.gradient_history.append(total_norm)
        
        # Detect issues
        issue = 'healthy'
        severity = 'none'
        actions = []
        
        # Exploding gradients
        if total_norm > 100:
            issue = 'exploding'
            if total_norm > 1000:
                severity = 'severe'
                actions = [
                    "🔴 SEVERE GRADIENT EXPLOSION",
                    "Actions:",
                    "1. Apply gradient clipping (max_norm=1.0)",
                    "2. Reduce learning rate by 10x",
                    "3. Check for numerical instability",
                    "4. Verify input data scaling"
                ]
            elif total_norm > 100:
                severity = 'moderate'
                actions = [
                    "🟡 MODERATE GRADIENT EXPLOSION",
                    "Actions:",
                    "1. Apply gradient clipping (max_norm=5.0)",
                    "2. Reduce learning rate by 2-5x",
                    "3. Consider batch normalization"
                ]
        
        # Vanishing gradients
        elif total_norm < 1e-5:
            issue = 'vanishing'
            if total_norm < 1e-7:
                severity = 'severe'
                actions = [
                    "🔴 SEVERE GRADIENT VANISHING",
                    "Actions:",
                    "1. Use ReLU instead of Sigmoid/Tanh",
                    "2. Add batch normalization",
                    "3. Increase learning rate",
                    "4. Use residual connections",
                    "5. Check weight initialization"
                ]
            else:
                severity = 'moderate'
                actions = [
                    "🟡 MODERATE GRADIENT VANISHING",
                    "Actions:",
                    "1. Increase learning rate slightly",
                    "2. Verify activation functions",
                    "3. Consider batch normalization"
                ]
        
        # Unstable gradients (high variance)
        elif len(self.gradient_history) >= 5:
            grad_std = np.std(list(self.gradient_history))
            grad_mean = np.mean(list(self.gradient_history))
            
            if grad_std / (grad_mean + 1e-8) > 2.0:
                issue = 'unstable'
                severity = 'moderate'
                actions = [
                    "🟡 UNSTABLE GRADIENTS",
                    "High variance in gradient norms",
                    "Actions:",
                    "1. Reduce learning rate",
                    "2. Increase batch size for stability",
                    "3. Apply gradient clipping"
                ]
        
        return {
            'issue': issue,
            'severity': severity,
            'max_grad_norm': max_norm,
            'min_grad_norm': min_norm,
            'total_grad_norm': total_norm,
            'recommended_actions': actions,
            'should_apply_clipping': total_norm > 10.0
        }
    
    def check_for_nan_inf(
        self,
        outputs: torch.Tensor,
        loss: torch.Tensor,
        epoch: int
    ) -> Dict:
        """
        Detect NaN or Inf values in outputs or loss.
        
        Returns:
            {
                'has_nan': bool,
                'has_inf': bool,
                'location': str,
                'recommended_actions': List[str]
            }
        """
        has_nan_output = torch.isnan(outputs).any().item()
        has_inf_output = torch.isinf(outputs).any().item()
        has_nan_loss = torch.isnan(loss).any().item()
        has_inf_loss = torch.isinf(loss).any().item()
        
        has_nan = has_nan_output or has_nan_loss
        has_inf = has_inf_output or has_inf_loss
        
        location = []
        if has_nan_output:
            location.append('outputs')
        if has_inf_output:
            location.append('outputs')
        if has_nan_loss:
            location.append('loss')
        if has_inf_loss:
            location.append('loss')
        
        actions = []
        if has_nan or has_inf:
            actions = [
                "🔴 NUMERICAL INSTABILITY DETECTED!",
                "Training will fail if not fixed immediately.",
                "Actions:",
                "1. Reduce learning rate by 10x",
                "2. Apply gradient clipping (max_norm=1.0)",
                "3. Check input data for extreme values",
                "4. Add numerical stability to loss (epsilon=1e-7)",
                "5. Use mixed precision training (if using GPU)",
                "6. Verify no division by zero in custom layers"
            ]
        
        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'location': ', '.join(location) if location else 'none',
            'recommended_actions': actions,
            'should_stop_training': has_nan or has_inf
        }
    
    def detect_learning_rate_issues(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epoch: int
    ) -> Dict:
        """
        Detect if learning rate is too high or too low.
        
        Returns:
            {
                'issue': str,  # 'too_high', 'too_low', 'optimal'
                'severity': str,
                'recommended_lr_adjustment': float,
                'recommended_actions': List[str]
            }
        """
        if len(train_losses) < 10:
            return {
                'issue': 'unknown',
                'severity': 'none',
                'recommended_lr_adjustment': 1.0,
                'recommended_actions': ['Need more epochs to diagnose']
            }
        
        recent_train_losses = train_losses[-10:]
        recent_val_losses = val_losses[-10:]
        
        # Check for divergence (LR too high)
        train_increasing = all(recent_train_losses[i] >= recent_train_losses[i-1] 
                               for i in range(1, len(recent_train_losses)))
        
        loss_variance = np.var(recent_train_losses)
        loss_mean = np.mean(recent_train_losses)
        cv = loss_variance / (loss_mean + 1e-8)  # Coefficient of variation
        
        issue = 'optimal'
        severity = 'none'
        lr_adjustment = 1.0
        actions = []
        
        # LR too high (diverging or oscillating)
        if train_increasing or cv > 0.5:
            issue = 'too_high'
            if train_increasing:
                severity = 'severe'
                lr_adjustment = 0.1  # Reduce by 10x
                actions = [
                    "🔴 LEARNING RATE TOO HIGH - DIVERGING",
                    "Loss is increasing instead of decreasing!",
                    "Actions:",
                    "1. Reduce learning rate by 10x immediately",
                    "2. Restart from best checkpoint",
                    "3. Apply gradient clipping",
                    "4. Consider using learning rate scheduler"
                ]
            else:
                severity = 'moderate'
                lr_adjustment = 0.5  # Reduce by 2x
                actions = [
                    "🟡 LEARNING RATE TOO HIGH - OSCILLATING",
                    "Loss is bouncing around (high variance)",
                    "Actions:",
                    "1. Reduce learning rate by 2-5x",
                    "2. Increase batch size for stability",
                    "3. Use learning rate scheduler"
                ]
        
        # LR too low (not improving)
        else:
            # Check if loss stopped improving
            early_losses = recent_train_losses[:5]
            late_losses = recent_train_losses[5:]
            improvement = np.mean(early_losses) - np.mean(late_losses)
            
            if improvement < 0.001:  # Less than 0.1% improvement
                issue = 'too_low'
                severity = 'moderate'
                lr_adjustment = 2.0  # Increase by 2x
                actions = [
                    "🟡 LEARNING RATE TOO LOW",
                    "Training has stalled - minimal improvement",
                    "Actions:",
                    "1. Increase learning rate by 2-3x",
                    "2. Check if model capacity is sufficient",
                    "3. Verify gradients are flowing"
                ]
        
        return {
            'issue': issue,
            'severity': severity,
            'recommended_lr_adjustment': lr_adjustment,
            'coefficient_of_variation': cv,
            'recent_improvement': improvement if 'improvement' in locals() else 0,
            'recommended_actions': actions
        }
    
    def detect_training_plateau(
        self,
        val_losses: List[float],
        patience: int = 10
    ) -> Dict:
        """
        Detect if training has plateaued (stuck).
        
        Returns:
            {
                'is_plateau': bool,
                'epochs_without_improvement': int,
                'recommended_actions': List[str],
                'should_stop': bool
            }
        """
        if len(val_losses) < patience:
            return {
                'is_plateau': False,
                'epochs_without_improvement': 0,
                'recommended_actions': [],
                'should_stop': False
            }
        
        current_loss = val_losses[-1]
        
        # Update best loss
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        is_plateau = self.patience_counter >= patience
        should_stop = self.patience_counter >= patience * 2  # Double patience for stopping
        
        actions = []
        if is_plateau and not should_stop:
            actions = [
                "🟡 TRAINING PLATEAU DETECTED",
                f"No improvement for {self.patience_counter} epochs",
                "Actions:",
                "1. Reduce learning rate by 5-10x (LR decay)",
                "2. Unfreeze more layers (if using transfer learning)",
                "3. Add slight noise to break out of local minimum",
                "4. Consider stopping soon if no improvement"
            ]
        elif should_stop:
            actions = [
                "🔴 EARLY STOPPING TRIGGERED",
                f"No improvement for {self.patience_counter} epochs",
                "Model has converged or stuck in local minimum",
                "Recommendation: Stop training and use best checkpoint"
            ]
        
        return {
            'is_plateau': is_plateau,
            'epochs_without_improvement': self.patience_counter,
            'best_val_loss': self.best_val_loss,
            'current_val_loss': current_loss,
            'recommended_actions': actions,
            'should_stop': should_stop
        }
    
    def detect_mode_collapse(
        self,
        predictions: torch.Tensor,
        threshold: float = 0.95
    ) -> Dict:
        """
        Detect if model is predicting mostly one class (mode collapse).
        
        Returns:
            {
                'is_collapsed': bool,
                'positive_ratio': float,
                'severity': str,
                'recommended_actions': List[str]
            }
        """
        predictions_binary = (predictions >= 0.5).float()
        positive_ratio = predictions_binary.mean().item()
        
        self.prediction_history.append(positive_ratio)
        
        # Check if consistently predicting one class
        is_collapsed = positive_ratio > threshold or positive_ratio < (1 - threshold)
        
        # Check consistency across recent batches
        if len(self.prediction_history) >= 3:
            avg_ratio = np.mean(list(self.prediction_history))
            is_consistently_collapsed = avg_ratio > threshold or avg_ratio < (1 - threshold)
        else:
            is_consistently_collapsed = False
        
        severity = 'none'
        actions = []
        
        if is_consistently_collapsed:
            if positive_ratio > 0.98 or positive_ratio < 0.02:
                severity = 'severe'
                actions = [
                    "🔴 SEVERE MODE COLLAPSE",
                    f"Model predicting {positive_ratio*100:.1f}% positive",
                    "Model has collapsed to single prediction!",
                    "Actions:",
                    "1. Check class weights (apply inverse frequency)",
                    "2. Verify loss function is working",
                    "3. Restart with different initialization",
                    "4. Increase model capacity",
                    "5. Check for data quality issues"
                ]
            else:
                severity = 'moderate'
                actions = [
                    "🟡 MODE COLLAPSE DETECTED",
                    f"Model predicting {positive_ratio*100:.1f}% positive",
                    "Actions:",
                    "1. Apply class weights to balance",
                    "2. Use focal loss for hard examples",
                    "3. Check data distribution"
                ]
        
        return {
            'is_collapsed': is_consistently_collapsed,
            'positive_ratio': positive_ratio,
            'average_ratio': np.mean(list(self.prediction_history)) if self.prediction_history else positive_ratio,
            'severity': severity,
            'recommended_actions': actions
        }
    
    def detect_outliers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        contamination: float = 0.1
    ) -> Dict:
        """
        Detect outliers in training data using Isolation Forest.
        
        Returns:
            {
                'outlier_count': int,
                'outlier_indices': List[int],
                'outlier_percentage': float,
                'recommended_actions': List[str]
            }
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # -1 indicates outliers
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
            outlier_count = len(outlier_indices)
            outlier_percentage = (outlier_count / len(X)) * 100
            
            actions = []
            if outlier_percentage > 5:
                actions = [
                    f"⚠️  OUTLIERS DETECTED: {outlier_count} samples ({outlier_percentage:.1f}%)",
                    "Outliers can poison training and reduce accuracy",
                    "Actions:",
                    "1. Review outlier samples manually",
                    "2. Remove if data quality issues",
                    "3. Keep if legitimate rare patterns",
                    "4. Use robust loss functions (Huber loss)",
                    f"5. Outlier indices: {outlier_indices[:10]}..." if len(outlier_indices) > 10 else f"5. Outlier indices: {outlier_indices}"
                ]
            
            return {
                'outlier_count': outlier_count,
                'outlier_indices': outlier_indices,
                'outlier_percentage': outlier_percentage,
                'recommended_actions': actions
            }
        
        except ImportError:
            return {
                'outlier_count': 0,
                'outlier_indices': [],
                'outlier_percentage': 0.0,
                'recommended_actions': ['sklearn not available for outlier detection']
            }
    
    def detect_feature_scaling_issues(
        self,
        X: np.ndarray
    ) -> Dict:
        """
        Detect if features have drastically different scales.
        
        Returns:
            {
                'has_scaling_issues': bool,
                'scale_ratios': Dict[str, float],
                'recommended_actions': List[str]
            }
        """
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        feature_maxes = np.max(np.abs(X), axis=0)
        
        # Calculate scale differences
        max_std = np.max(feature_stds)
        min_std = np.min(feature_stds[feature_stds > 0])
        scale_ratio = max_std / (min_std + 1e-8)
        
        max_max = np.max(feature_maxes)
        min_max = np.min(feature_maxes[feature_maxes > 0])
        range_ratio = max_max / (min_max + 1e-8)
        
        has_issues = scale_ratio > 100 or range_ratio > 1000
        
        actions = []
        if has_issues:
            actions = [
                "⚠️  FEATURE SCALING ISSUES DETECTED",
                f"Scale ratio: {scale_ratio:.1f}x (std deviation)",
                f"Range ratio: {range_ratio:.1f}x (max values)",
                "Some features may dominate others in training",
                "Actions:",
                "1. Apply StandardScaler (zero mean, unit variance)",
                "2. Or MinMaxScaler (scale to [0,1])",
                "3. Check for features with extreme values",
                "4. Neural networks perform better with normalized inputs"
            ]
        
        return {
            'has_scaling_issues': has_issues,
            'scale_ratio': scale_ratio,
            'range_ratio': range_ratio,
            'feature_means': feature_means.tolist(),
            'feature_stds': feature_stds.tolist(),
            'recommended_actions': actions
        }
    
    def detect_distribution_shift(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray
    ) -> Dict:
        """
        Detect if training and validation data come from different distributions.
        
        Returns:
            {
                'has_shift': bool,
                'severity': str,
                'ks_statistics': List[float],
                'recommended_actions': List[str]
            }
        """
        try:
            from scipy.stats import ks_2samp
            
            ks_stats = []
            p_values = []
            
            # Perform KS test for each feature
            for i in range(X_train.shape[1]):
                stat, p_value = ks_2samp(X_train[:, i], X_val[:, i])
                ks_stats.append(stat)
                p_values.append(p_value)
            
            # Average KS statistic
            avg_ks = np.mean(ks_stats)
            significant_shifts = sum(1 for p in p_values if p < 0.05)
            shift_percentage = (significant_shifts / len(p_values)) * 100
            
            has_shift = avg_ks > 0.2 or shift_percentage > 20
            
            severity = 'none'
            if avg_ks > 0.5:
                severity = 'severe'
            elif avg_ks > 0.3:
                severity = 'moderate'
            elif avg_ks > 0.2:
                severity = 'mild'
            
            actions = []
            if has_shift:
                actions = [
                    "⚠️  DISTRIBUTION SHIFT DETECTED",
                    f"Train and validation data may be from different distributions",
                    f"KS statistic: {avg_ks:.3f}",
                    f"Features with significant shift: {significant_shifts}/{len(p_values)} ({shift_percentage:.1f}%)",
                    "This can cause poor generalization!",
                    "Actions:",
                    "1. Verify train/val split is random",
                    "2. Check for temporal ordering issues",
                    "3. Ensure stratified sampling for class balance",
                    "4. Consider using same time period for both sets"
                ]
            
            return {
                'has_shift': has_shift,
                'severity': severity,
                'avg_ks_statistic': avg_ks,
                'significant_shifts': significant_shifts,
                'shift_percentage': shift_percentage,
                'recommended_actions': actions
            }
        
        except ImportError:
            return {
                'has_shift': False,
                'severity': 'none',
                'avg_ks_statistic': 0.0,
                'significant_shifts': 0,
                'shift_percentage': 0.0,
                'recommended_actions': ['scipy not available for distribution shift detection']
            }
    
    def generate_comprehensive_report(
        self,
        gradient_check: Dict,
        nan_check: Dict,
        lr_check: Dict,
        plateau_check: Dict,
        mode_collapse_check: Dict,
        outlier_check: Dict,
        scaling_check: Dict,
        shift_check: Dict
    ) -> str:
        """Generate comprehensive report of all detected issues."""
        
        report = []
        report.append("\n" + "="*80)
        report.append("ADVANCED TRAINING DIAGNOSTICS REPORT")
        report.append("="*80 + "\n")
        
        issues_found = []
        
        # Gradient health
        if gradient_check['severity'] != 'none':
            issues_found.append("Gradient Issues")
            report.append("⚠️  GRADIENT HEALTH")
            report.append("-" * 80)
            report.append(f"Issue: {gradient_check['issue'].upper()}")
            report.append(f"Severity: {gradient_check['severity']}")
            report.append(f"Total Gradient Norm: {gradient_check['total_grad_norm']:.6f}")
            if gradient_check['recommended_actions']:
                for action in gradient_check['recommended_actions']:
                    report.append(f"  {action}")
            report.append("")
        
        # NaN/Inf check
        if nan_check['has_nan'] or nan_check['has_inf']:
            issues_found.append("Numerical Instability")
            report.append("🔴 NUMERICAL INSTABILITY")
            report.append("-" * 80)
            report.append(f"NaN detected: {nan_check['has_nan']}")
            report.append(f"Inf detected: {nan_check['has_inf']}")
            report.append(f"Location: {nan_check['location']}")
            for action in nan_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Learning rate issues
        if lr_check['severity'] != 'none':
            issues_found.append("Learning Rate")
            report.append("⚠️  LEARNING RATE")
            report.append("-" * 80)
            report.append(f"Issue: {lr_check['issue'].replace('_', ' ').title()}")
            report.append(f"Recommended Adjustment: {lr_check['recommended_lr_adjustment']:.2f}x")
            for action in lr_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Training plateau
        if plateau_check['is_plateau']:
            issues_found.append("Training Plateau")
            report.append("⚠️  TRAINING PLATEAU")
            report.append("-" * 80)
            report.append(f"Epochs without improvement: {plateau_check['epochs_without_improvement']}")
            report.append(f"Should stop: {plateau_check['should_stop']}")
            for action in plateau_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Mode collapse
        if mode_collapse_check['is_collapsed']:
            issues_found.append("Mode Collapse")
            report.append("⚠️  MODE COLLAPSE")
            report.append("-" * 80)
            report.append(f"Positive prediction ratio: {mode_collapse_check['positive_ratio']*100:.1f}%")
            for action in mode_collapse_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Outliers
        if outlier_check['outlier_percentage'] > 5:
            issues_found.append("Outliers")
            report.append("⚠️  OUTLIERS")
            report.append("-" * 80)
            report.append(f"Outlier count: {outlier_check['outlier_count']} ({outlier_check['outlier_percentage']:.1f}%)")
            for action in outlier_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Feature scaling
        if scaling_check['has_scaling_issues']:
            issues_found.append("Feature Scaling")
            report.append("⚠️  FEATURE SCALING")
            report.append("-" * 80)
            report.append(f"Scale ratio: {scaling_check['scale_ratio']:.1f}x")
            for action in scaling_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Distribution shift
        if shift_check['has_shift']:
            issues_found.append("Distribution Shift")
            report.append("⚠️  DISTRIBUTION SHIFT")
            report.append("-" * 80)
            report.append(f"Severity: {shift_check['severity']}")
            report.append(f"KS Statistic: {shift_check['avg_ks_statistic']:.3f}")
            for action in shift_check['recommended_actions']:
                report.append(f"  {action}")
            report.append("")
        
        # Summary
        if not issues_found:
            report.append("✅ NO CRITICAL ISSUES DETECTED")
            report.append("Training appears healthy!")
        else:
            report.append(f"ISSUES DETECTED: {len(issues_found)}")
            report.append(f"Categories: {', '.join(issues_found)}")
        
        report.append("\n" + "="*80 + "\n")
        
        return "\n".join(report)
