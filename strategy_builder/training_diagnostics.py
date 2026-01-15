"""
Training Diagnostics & Data Validation

Ensures sufficient training data and detects bias/variance issues.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import torch

from .models import UserStrategy


class TrainingDiagnostics:
    """
    Comprehensive training diagnostics to ensure high-quality models.
    
    Features:
    1. Data sufficiency validation
    2. Bias/variance detection
    3. Automatic hyperparameter adjustment
    4. Learning curve analysis
    5. Data augmentation suggestions
    """
    
    def __init__(self):
        self.min_samples = {
            'critical': 50,    # Absolute minimum
            'minimum': 100,    # Bare minimum for training
            'recommended': 300,  # Good training
            'optimal': 500     # Excellent training
        }
    
    def validate_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: UserStrategy
    ) -> Dict:
        """
        Validate if training data is sufficient.
        
        Returns:
            {
                'is_sufficient': bool,
                'quality_level': str,  # 'critical', 'poor', 'good', 'excellent'
                'sample_count': int,
                'win_rate': float,
                'class_balance': float,  # Ratio of minority/majority class
                'recommendations': List[str],
                'warnings': List[str]
            }
        """
        sample_count = len(X)
        win_rate = y.mean()
        
        # Calculate class balance
        wins = y.sum()
        losses = len(y) - wins
        class_balance = min(wins, losses) / max(wins, losses) if max(wins, losses) > 0 else 0
        
        # Determine quality level
        if sample_count < self.min_samples['critical']:
            quality_level = 'critical'
            is_sufficient = False
        elif sample_count < self.min_samples['minimum']:
            quality_level = 'poor'
            is_sufficient = True  # Can train but risky
        elif sample_count < self.min_samples['recommended']:
            quality_level = 'good'
            is_sufficient = True
        else:
            quality_level = 'excellent'
            is_sufficient = True
        
        # Generate recommendations
        recommendations = []
        warnings = []
        
        if sample_count < self.min_samples['optimal']:
            shortage = self.min_samples['recommended'] - sample_count
            recommendations.append(
                f"Collect {shortage} more setups by:\n"
                f"  - Extending date range (add {self._estimate_days_needed(shortage, strategy)} more days)\n"
                f"  - Adding more symbols (try correlated pairs)\n"
                f"  - Adding more timeframes (H4, H1 complement each other)"
            )
        
        # Check class imbalance
        if class_balance < 0.3:
            warnings.append(
                f"⚠️  Severe class imbalance detected! "
                f"Minority class is only {class_balance*100:.1f}% of majority class.\n"
                f"  Solution: Use class weights or SMOTE for balancing"
            )
            recommendations.append("Enable automatic class balancing (will be applied)")
        
        elif class_balance < 0.5:
            warnings.append(
                f"⚠️  Moderate class imbalance. Ratio: {class_balance:.2f}\n"
                f"  Consider using class weights"
            )
        
        # Check win rate
        if win_rate < 0.35 or win_rate > 0.75:
            warnings.append(
                f"⚠️  Unusual win rate: {win_rate*100:.1f}%\n"
                f"  Expected range: 35-75%. This may indicate:\n"
                f"  - Unrealistic SL/TP ratios\n"
                f"  - Strategy rules need refinement\n"
                f"  - Data quality issues"
            )
        
        # Check feature dimensionality
        feature_count = X.shape[1]
        samples_per_feature = sample_count / feature_count
        
        if samples_per_feature < 10:
            warnings.append(
                f"⚠️  Low samples-to-features ratio: {samples_per_feature:.1f}\n"
                f"  Risk of overfitting! Recommendations:\n"
                f"  - Collect more data\n"
                f"  - Reduce indicator count\n"
                f"  - Use stronger regularization"
            )
            recommendations.append("Automatically increasing dropout to 0.5")
        
        return {
            'is_sufficient': is_sufficient,
            'quality_level': quality_level,
            'sample_count': sample_count,
            'win_rate': win_rate,
            'class_balance': class_balance,
            'feature_count': feature_count,
            'samples_per_feature': samples_per_feature,
            'recommendations': recommendations,
            'warnings': warnings
        }
    
    def detect_bias_variance_issue(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float]
    ) -> Dict:
        """
        Detect if model has high bias (underfitting) or high variance (overfitting).
        
        Returns:
            {
                'issue': str,  # 'high_bias', 'high_variance', 'good_fit', 'unknown'
                'severity': str,  # 'none', 'mild', 'moderate', 'severe'
                'train_val_gap': float,  # Gap between train and val performance
                'convergence_status': str,  # 'converged', 'still_improving', 'diverging'
                'recommended_actions': List[str]
            }
        """
        if len(train_losses) < 10:
            return {
                'issue': 'unknown',
                'severity': 'none',
                'train_val_gap': 0.0,
                'convergence_status': 'unknown',
                'recommended_actions': ['Train for more epochs to diagnose']
            }
        
        # Calculate metrics
        final_train_loss = np.mean(train_losses[-5:])
        final_val_loss = np.mean(val_losses[-5:])
        final_train_acc = np.mean(train_accuracies[-5:])
        final_val_acc = np.mean(val_accuracies[-5:])
        
        train_val_gap = final_train_acc - final_val_acc
        loss_gap = final_val_loss - final_train_loss
        
        # Check convergence
        recent_val_losses = val_losses[-10:]
        is_improving = np.mean(recent_val_losses[-5:]) < np.mean(recent_val_losses[:5])
        is_diverging = final_val_loss > np.min(val_losses) * 1.2
        
        if is_improving:
            convergence_status = 'still_improving'
        elif is_diverging:
            convergence_status = 'diverging'
        else:
            convergence_status = 'converged'
        
        # Detect issue type
        issue = 'good_fit'
        severity = 'none'
        recommended_actions = []
        
        # High Variance (Overfitting)
        if train_val_gap > 0.15 and loss_gap > 0.3:
            issue = 'high_variance'
            
            if train_val_gap > 0.25:
                severity = 'severe'
                recommended_actions = [
                    "🔴 SEVERE OVERFITTING DETECTED",
                    "Actions to take:",
                    "1. Increase dropout to 0.5-0.6",
                    "2. Add L2 regularization (weight_decay=0.01)",
                    "3. Reduce model complexity (fewer layers/neurons)",
                    "4. Collect more training data",
                    "5. Use data augmentation",
                    "6. Apply early stopping"
                ]
            elif train_val_gap > 0.15:
                severity = 'moderate'
                recommended_actions = [
                    "🟡 MODERATE OVERFITTING",
                    "Actions:",
                    "1. Increase dropout to 0.4-0.5",
                    "2. Add weight decay (0.001-0.01)",
                    "3. Use early stopping",
                    "4. Collect more data if possible"
                ]
            else:
                severity = 'mild'
                recommended_actions = [
                    "🟢 MILD OVERFITTING",
                    "Actions:",
                    "1. Slight dropout increase (0.35)",
                    "2. Monitor for a few more epochs"
                ]
        
        # High Bias (Underfitting)
        elif final_train_acc < 0.60 and final_val_acc < 0.60:
            issue = 'high_bias'
            
            if final_train_acc < 0.55:
                severity = 'severe'
                recommended_actions = [
                    "🔴 SEVERE UNDERFITTING DETECTED",
                    "Actions:",
                    "1. Increase model complexity (more neurons/layers)",
                    "2. Train for more epochs (current performance too low)",
                    "3. Check if features are meaningful",
                    "4. Reduce regularization (lower dropout)",
                    "5. Increase learning rate to 0.001-0.003",
                    "6. Verify data quality and labels"
                ]
            elif final_train_acc < 0.60:
                severity = 'moderate'
                recommended_actions = [
                    "🟡 MODERATE UNDERFITTING",
                    "Actions:",
                    "1. Add more neurons or depth",
                    "2. Train longer (50-100 more epochs)",
                    "3. Reduce dropout to 0.2",
                    "4. Try different learning rate"
                ]
            else:
                severity = 'mild'
                recommended_actions = [
                    "🟢 MILD UNDERFITTING",
                    "Actions:",
                    "1. Train for more epochs",
                    "2. Slight learning rate increase"
                ]
        
        # Good fit but not converged
        elif convergence_status == 'still_improving':
            recommended_actions = [
                "✅ Model is still improving!",
                "Continue training for 50-100 more epochs"
            ]
        
        # Good fit and converged
        else:
            recommended_actions = [
                "✅ Model has good fit!",
                f"Train accuracy: {final_train_acc:.2%}",
                f"Validation accuracy: {final_val_acc:.2%}",
                f"Gap: {train_val_gap:.2%} (acceptable)"
            ]
        
        return {
            'issue': issue,
            'severity': severity,
            'train_val_gap': train_val_gap,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'convergence_status': convergence_status,
            'recommended_actions': recommended_actions
        }
    
    def auto_adjust_hyperparameters(
        self,
        base_hyperparameters: Dict,
        diagnostics: Dict,
        data_validation: Dict
    ) -> Dict:
        """
        Automatically adjust hyperparameters based on diagnostics.
        
        Args:
            base_hyperparameters: Original hyperparameters
            diagnostics: Bias/variance diagnostics
            data_validation: Data quality metrics
            
        Returns:
            Adjusted hyperparameters with explanations
        """
        adjusted = base_hyperparameters.copy()
        adjustments_made = []
        
        # Handle overfitting (high variance)
        if diagnostics['issue'] == 'high_variance':
            if diagnostics['severity'] == 'severe':
                adjusted['dropout'] = 0.5
                adjusted['weight_decay'] = 0.01
                adjustments_made.append("Increased dropout to 0.5 (severe overfitting)")
                adjustments_made.append("Added L2 regularization (weight_decay=0.01)")
            elif diagnostics['severity'] == 'moderate':
                adjusted['dropout'] = 0.4
                adjusted['weight_decay'] = 0.005
                adjustments_made.append("Increased dropout to 0.4 (moderate overfitting)")
                adjustments_made.append("Added weight decay (0.005)")
            else:
                adjusted['dropout'] = 0.35
                adjustments_made.append("Slightly increased dropout to 0.35")
        
        # Handle underfitting (high bias)
        elif diagnostics['issue'] == 'high_bias':
            if diagnostics['severity'] == 'severe':
                adjusted['dropout'] = 0.2
                adjusted['learning_rate'] = 0.002
                adjusted['epochs'] = adjusted.get('epochs', 100) + 100
                adjustments_made.append("Reduced dropout to 0.2 (severe underfitting)")
                adjustments_made.append("Increased learning rate to 0.002")
                adjustments_made.append("Added 100 more epochs")
            elif diagnostics['severity'] == 'moderate':
                adjusted['dropout'] = 0.25
                adjusted['epochs'] = adjusted.get('epochs', 100) + 50
                adjustments_made.append("Reduced dropout to 0.25")
                adjustments_made.append("Added 50 more epochs")
        
        # Handle class imbalance
        if data_validation['class_balance'] < 0.5:
            # Calculate class weights
            wins = data_validation['sample_count'] * data_validation['win_rate']
            losses = data_validation['sample_count'] - wins
            weight_positive = data_validation['sample_count'] / (2 * wins) if wins > 0 else 1.0
            weight_negative = data_validation['sample_count'] / (2 * losses) if losses > 0 else 1.0
            
            adjusted['class_weights'] = {
                0: weight_negative,
                1: weight_positive
            }
            adjustments_made.append(
                f"Added class weights: Loss={weight_negative:.2f}, Win={weight_positive:.2f}"
            )
        
        # Handle low samples-to-features ratio
        if data_validation['samples_per_feature'] < 10:
            adjusted['dropout'] = max(adjusted.get('dropout', 0.3), 0.5)
            adjustments_made.append("Increased dropout to 0.5 (low sample count)")
        
        # Handle small dataset
        if data_validation['sample_count'] < 200:
            adjusted['batch_size'] = min(adjusted.get('batch_size', 32), 16)
            adjustments_made.append("Reduced batch size to 16 (small dataset)")
        
        return {
            'adjusted_hyperparameters': adjusted,
            'adjustments_made': adjustments_made
        }
    
    def suggest_data_augmentation(
        self,
        strategy: UserStrategy,
        current_samples: int
    ) -> List[str]:
        """
        Suggest ways to augment training data.
        """
        suggestions = []
        
        target_samples = self.min_samples['recommended']
        shortage = target_samples - current_samples
        
        if shortage > 0:
            # Suggest more symbols
            if len(strategy.symbols) < 5:
                suggestions.append(
                    f"📊 Add {3 - len(strategy.symbols)} more correlated symbols:\n"
                    f"  Current: {', '.join(strategy.symbols)}\n"
                    f"  Suggested: EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD (major pairs)\n"
                    f"  This could add ~{shortage // 2} setups"
                )
            
            # Suggest more timeframes
            if len(strategy.timeframes) < 2:
                suggestions.append(
                    f"⏰ Add complementary timeframes:\n"
                    f"  Current: {', '.join(strategy.timeframes)}\n"
                    f"  Suggested: H1 + H4 (capture different market dynamics)\n"
                    f"  This could add ~{shortage // 3} setups"
                )
            
            # Suggest longer date range
            suggestions.append(
                f"📅 Extend historical data range:\n"
                f"  Go back {self._estimate_days_needed(shortage, strategy)} more days\n"
                f"  More data = better generalization"
            )
            
            # Suggest synthetic data (careful!)
            if current_samples > 50:
                suggestions.append(
                    f"🔬 Consider synthetic data augmentation (EXPERIMENTAL):\n"
                    f"  - Add Gaussian noise to features (±5%)\n"
                    f"  - Time shifting (±1 candle)\n"
                    f"  - Could generate ~{shortage // 2} synthetic samples\n"
                    f"  ⚠️  Use with caution - may introduce unrealistic scenarios"
                )
        
        return suggestions
    
    def _estimate_days_needed(self, shortage: int, strategy: UserStrategy) -> int:
        """Estimate how many more days of data needed"""
        # Rough estimate: 1 setup per 20 days per symbol/timeframe
        combinations = len(strategy.symbols) * len(strategy.timeframes)
        setups_per_day_per_combo = 1 / 20
        days_needed = shortage / (setups_per_day_per_combo * combinations)
        return max(30, int(days_needed))
    
    def generate_diagnostic_report(
        self,
        data_validation: Dict,
        bias_variance: Dict,
        adjusted_hyperparameters: Dict
    ) -> str:
        """
        Generate comprehensive diagnostic report.
        """
        report = []
        report.append("\n" + "="*80)
        report.append("TRAINING DIAGNOSTICS REPORT")
        report.append("="*80 + "\n")
        
        # Data Quality
        report.append("📊 DATA QUALITY")
        report.append("-" * 80)
        report.append(f"Quality Level: {data_validation['quality_level'].upper()}")
        report.append(f"Sample Count: {data_validation['sample_count']}")
        report.append(f"Win Rate: {data_validation['win_rate']*100:.1f}%")
        report.append(f"Class Balance: {data_validation['class_balance']:.2f}")
        report.append(f"Features: {data_validation['feature_count']}")
        report.append(f"Samples per Feature: {data_validation['samples_per_feature']:.1f}")
        
        if data_validation['warnings']:
            report.append("\n⚠️  WARNINGS:")
            for warning in data_validation['warnings']:
                report.append(f"  {warning}")
        
        if data_validation['recommendations']:
            report.append("\n💡 RECOMMENDATIONS:")
            for rec in data_validation['recommendations']:
                report.append(f"  {rec}")
        
        # Bias/Variance Analysis
        if bias_variance['issue'] != 'unknown':
            report.append("\n🎯 BIAS/VARIANCE ANALYSIS")
            report.append("-" * 80)
            report.append(f"Issue: {bias_variance['issue'].replace('_', ' ').title()}")
            report.append(f"Severity: {bias_variance['severity'].title()}")
            report.append(f"Train-Val Gap: {bias_variance['train_val_gap']*100:.1f}%")
            report.append(f"Convergence: {bias_variance['convergence_status'].replace('_', ' ').title()}")
            
            if bias_variance['recommended_actions']:
                report.append("\n💡 RECOMMENDED ACTIONS:")
                for action in bias_variance['recommended_actions']:
                    report.append(f"  {action}")
        
        # Hyperparameter Adjustments
        if adjusted_hyperparameters['adjustments_made']:
            report.append("\n⚙️  AUTOMATIC ADJUSTMENTS")
            report.append("-" * 80)
            for adjustment in adjusted_hyperparameters['adjustments_made']:
                report.append(f"  ✅ {adjustment}")
        
        report.append("\n" + "="*80 + "\n")
        
        return "\n".join(report)
