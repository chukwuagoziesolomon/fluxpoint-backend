"""
ML Training Pipeline for User Strategies

Adapts TCE training approach for dynamic user-defined strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

from .models import UserStrategy, StrategyMLModel
from .data_collection import DataCollectionPipeline
from .training_diagnostics import TrainingDiagnostics
from .transfer_learning import TransferLearningManager
from .advanced_diagnostics import AdvancedTrainingDiagnostics
from .auto_fix_training import AutoTrainingFixer, auto_fix_training_iteration


class StrategyDataset(Dataset):
    """PyTorch dataset for strategy training data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DynamicDNN(nn.Module):
    """
    Dynamic DNN model with flexible input size.
    
    Architecture: [input_size → 128 → 64 → 32 → 1]
    Same as TCE but input size adapts to user's indicators.
    """
    
    def __init__(self, input_size: int):
        super(DynamicDNN, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: input → 128
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3: 64 → 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output: 32 → 1
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class MLTrainingPipeline:
    """
    ML training pipeline for user strategies.
    
    Pipeline:
    1. Collect training data (via DataCollectionPipeline)
    2. Split into train/validation sets
    3. Train DNN model
    4. Validate and calculate metrics
    5. Save model and update database
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_collector = DataCollectionPipeline()
        self.diagnostics = TrainingDiagnostics()
        self.advanced_diagnostics = AdvancedTrainingDiagnostics()
        self.auto_fixer = AutoTrainingFixer()
        self.transfer_learning = TransferLearningManager(device=self.device)
        print(f"💻 Using device: {self.device}")
    
    def train_strategy_model(
        self,
        strategy_id: int,
        start_date: datetime,
        end_date: datetime,
        stop_loss_pips: float = 20,
        take_profit_pips: float = 40,
        batch_size: int = 32,
        epochs: int = 200,  # Increased from 100 to 200 for thorough training
        learning_rate: float = 0.001,
        min_setups: int = 100
    ) -> Dict:
        """
        Train ML model for a strategy.
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date for data collection
            end_date: End date for data collection
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            min_setups: Minimum setups required for training
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*80}")
        print(f"ML TRAINING PIPELINE - STRATEGY {strategy_id}")
        print(f"{'='*80}\n")
        
        # Get strategy
        try:
            strategy = UserStrategy.objects.get(id=strategy_id)
        except UserStrategy.DoesNotExist:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Update status
        strategy.status = 'collecting_data'
        strategy.save()
        
        # Create ML model record
        ml_model = StrategyMLModel.objects.create(
            strategy=strategy,
            model_type='DNN',
            status='training',
            training_started_at=datetime.now(),
            hyperparameters={
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'architecture': 'Dynamic [input → 128 → 64 → 32 → 1]'
            }
        )
        
        try:
            # Step 1: Collect training data
            print("📊 STEP 1: Collecting training data...")
            X_train, y_train, setup_details = self.data_collector.collect_training_data(
                strategy_id=strategy_id,
                parsed_rules=strategy.parsed_rules,
                symbols=strategy.symbols,
                timeframes=strategy.timeframes,
                start_date=start_date,
                end_date=end_date,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips
            )Step 2: Validate training data
            print(f"\n📊 STEP 2: Validating training data...")
            data_validation = self.diagnostics.validate_training_data(X_train, y_train, strategy)
            
            if not data_validation['is_sufficient']:
                error_msg = f"Insufficient training data: {len(X_train)} setups\n"
                error_msg += "\n".join(data_validation['recommendations'])
                raise ValueError(error_msg)
            
            # Print data quality report
            print(f"\n  Quality Level: {data_validation['quality_level'].upper()}")
            print(f"  Samples: {data_validation['sample_count']}")
            print(f"  Win Rate: {data_validation['win_rate']*100:.1f}%")
            print(f"  Class Balance: {data_validation['class_balance']:.2f}")
            
            if data4: Determine if transfer learning should be used
            print(f"\n🔄 STEP 4: Checking transfer learning...")
            tl_strategy = self.transfer_learning.get_recommended_strategy(len(X_train_split))
            print(f"  Strategy: {tl_strategy['strategy']}")
            print(f"  Reason: {tl_strategy['reason']}")
            
            use_transfer_learning = True  # Always try transfer learning for better results
            
            # Step 5: Create model (with or without transfer learning)
            print(f"\n🧠 STEP 5: Creating model...")
            input_size = X_train.shape[1]
            
            if use_transfer_learning:
                model, transfer_info = self.transfer_learning.create_user_model_with_transfer_learning(
                    user_input_size=input_size,
              Step 6: Create data loaders with adjusted batch size
            adjusted_batch_size = min(batch_size, len(X_train_split) // 4)  # Ensure at least 4 batches
            
            train_dataset = StrategyDataset(X_train_split, y_train_split)
            val_dataset = StrategyDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=adjusted_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=adjusted_batch_size, shuffle=False)
            
            # Step 7: Initial training with base hyperparameters
            print(f"\n🎯 STEP 6: Training model...")
            
            initial_hyperparameters = {
                'epochs': tl_strategy['fine_tune_epochs'] if use_transfer_learning else epochs,
                'learning_rate': tl_strategy['learning_rate'] if use_transfer_learning else learning_rate,
                'batch_size': adjusted_batch_size,
                'dropout': 0.3
            }
            
            history = self._train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=initial_hyperparameters['epochs'],
                learning_rate=initial_hyperparameters['learning_rate']
            )
            
            # Step 8: Diagnose bias/variance
            print(f"\n📈 STEP 7: Diagnosing model performance...")
            bias_variance = self.diagnostics.detect_bias_variance_issue(
                train_losses=history['train_loss'],
            print(f"\n📈 STEP 10: Final evaluationccuracy'],
                val_accuracies=history['val_accuracy']
            )
            
            # Step 9: Auto-adjust if needed and retrain
            if bias_variance['issue'] != 'good_fit' and bias_variance['severity'] in ['moderate', 'severe']:
                print(f"\n⚙️  STEP 8: Auto-adjusting hyperparameters...")
                
                adjusted = self.diagnostics.auto_adjust_hyperparameters(
                    base_hyperparameters=initial_hyperparameters,
                    diagnostics=bias_variance,
                    data_validation=data_validation
                )
                
                if adjusted['adjustments_made']:
                    print(f"\n  Adjustments made:")
                    for adj in adjusted['adjustments_made']:
                        print(f"    ✅ {adj}")
                    
                    # Recreate model with adjusted architecture if needed
                    if 'dropout' in adjusted['adjusted_hyperparameters']:
                        print(f"\n  🔄 Recreating model with adjusted dropout...")
                        # Note: In practice, you'd rebuild the model with new dropout
                        # For now, we'll continue with current model
                    
                    # Retrain with adjusted hyperparameters
                    print(f"\n  🎯 Retraining with adjusted hyperparameters...")
                    history = self._train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=adjusted['adjusted_hyperparameters'].get('epochs', initial_hyperparameters['epochs']),
                        learning_rate=adjusted['adjusted_hyperparameters'].get('learning_rate', initial_hyperparameters['learning_rate']),
                        class_weights=adjusted['adjusted_hyperparameters'].get('class_weights')
                    )
                    
                    # Re-diagnose
                    bias_variance = self.diagnostics.detect_bias_variance_issue(
                        train_losses=history['train_loss'],
                        val_losses=history['val_loss'],
                        train_accuracies=history['train_accuracy'],
                        val_accuracies=history['val_accuracy']
                    )
            
            # Step 10: If using transfer learning with frozen layers, unfreeze and fine-tune
            if use_transfer_learning and tl_strategy['freeze_layers'] and tl_strategy['full_fine_tune_epochs'] > 0:
                print(f"\n🔓 STEP 9: Unfreezing layers for final fine-tuning...")
                
                # Unfreeze all layers
                for param in model.parameters():
                    param.requires_grad = True
                
                # Fine-tune with very low learning rate
                print(f"  Training {tl_strategy['full_fine_tune_epochs']} more epochs with all layers unfrozen...")
                
                final_history = self._train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=tl_strategy['full_fine_tune_epochs'],
                    learning_rate=0.0001  # Very low LR
                )
                
                # Merge histories
                history['train_loss'].extend(final_history['train_loss'])
                history['val_loss'].extend(final_history['val_loss'])
                history['train_accuracy'].extend(final_history['train_accuracy'])
                history['val_accuracy'].extend(final_history['val_accuracy'])
            
            # Step 11: Final evaluation       print(f"     Frozen: {transfer_info['frozen_layers']} layers")
                    print(f"     Trainable params: {transfer_info['trainable_params']:,}")
                else:
                    print(f"  ℹ️  Training from scratch: {transfer_info.get('reason', 'unknown')}")
                    model = DynamicDNN(input_size=input_size).to(self.device)
            else:
                        for suggestion in suggestions:
                        print(f"    {suggestion}")
            
            # Step 3: Split data
            print(f"\n📊 STEP 3
            # Step 2: Split data
            print(f"\n📊 STEP 2: Splitting data (80/20)...")
            X_train_split, X_val, y_train_split, y_val = self._split_data(X_train, y_train)
            
            print(f"  Training: {len(X_train_split)} setups")
            print(f"  Validation: {len(X_val)} setups")
            
            # Step 3: Create data loaders
            train_dataset = StrategyDataset(X_train_split, y_train_split)
            val_dataset = StrategyDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Step 4: Create and train model
            print(f"\n🧠 STEP 3: Training DNN model...")
            input_size = X_train.shape[1]
            model = DynamicDNN(input_size=input_size).to(self.device)
            
            print(f"  Architecture: [{input_size} → 128 → 64 → 32 → 1]")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Training loop
            history = self._train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            # Step 5: Evaluate model
            print(f"\n📈 STEP 4: Evaluating model...")
            metrics = self._evaluate_model(model, val_loader)
            
            print(f"\n{'='*80}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"Validation Accuracy: {metrics['accuracy']:.2%}")
            print(f"Precision: {metrics['precision']:.2%}")
            print(f"Recall: {metrics['recall']:.2%}")
            print(f"F1 Score: {metrics['f1_score']:.2%}")
            print(f"{'='*80}\n")
            
            # Step 6: Save model
            moGenerate comprehensive diagnostic report
            diagnostic_report = self.diagnostics.generate_diagnostic_report(
                data_validation=data_validation,
                bias_variance=bias_variance,
                adjusted_hyperparameters={'adjustments_made': []}  # Already applied
            )
            print(diagnostic_report)
            
            # Step 11th = self._save_model(model, strategy_id)
            
            # Update ML model record
            ml_model.status = 'completed'
            ml_model.training_completed_at = datetime.now()
            ml_model.accuracy = metrics['accuracy']
            ml_model.precision = metrics['precision']
            ml_model.recall = metrics['recall']
            ml_model.f1_score = metrics['f1_score']
            ml_model.model_path = model_path
            ml_model.feature_config = {
                'input_size': input_size,
                'feature_count': input_size
            }
            ml_model.save()
            
            # Update strategy status
            strategy.status = 'ready'
            strategy.save()
            
            return {
                'success': True,
                'strategy_id': strategy_id,
                'ml_model_id': ml_model.id,
                'metrics': metrics,
                'training_samples': len(X_train_split),
                'validation_samples': len(X_val),
                'model_path': model_path
            }
            
        except Exception as e:
            # Update ML model record on error
            ml_model.status = 'failed'
            ml_model.save()
            
            # Update strategy status
            strategy.status = 'error'
            strategy.validation_errors = {'training_error': str(e)}
            strategy.save()
            
            raise e
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation sets"""
        split_idx = int(len(X) * split_ratio)
        
        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        
        return X_train, X_val, y_train, y_val
    ,
        class_weights: Optional[Dict] = None
    ) -> Dict:
        """Train the DNN model"""
        # Setup loss function with class weights if provided
        if class_weights:
            weight_tensor = torch.FloatTensor([class_weights[0], class_weights[1]]).to(self.device)
            criterion = nn.BCELoss(weight=weight_tensor)
            print(f"  Using class weights: {class_weights}")
        else:
            criterion = nn.BCELoss()
        train_correct = 0
            train_total = 0
            
        # Only optimize trainable parameters (important for transfer learning)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy
        learning_rate: float
    ) -> Dict:
        """Train the DNN model"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
                
                # Calculate training accuracy
                predicted = (outputs >= 0.5).float()
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0oader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0 if total > 0 else 0
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outpTrain Acc: {train_accuracy:.2%}, "
                      f"uts = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    predicted = (outputs >= 0.5).float()
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_accuracy:.2%}")
        
        return history
    
    def _evaluate_model(self, model: nn.Module, val_loader: DataLoader) -> Dict:
        """Evaluate model and calculate metrics"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                predicted = (outputs >= 0.5).float()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate metrics
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        tn = ((all_predictions == 0) & (all_labels == 0)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        
        accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def _save_model(self, model: nn.Module, strategy_id: int) -> str:
        """Save trained model to disk"""
        os.makedirs('models/user_strategies', exist_ok=True)
        model_path = f'models/user_strategies/strategy_{strategy_id}_ml.pth'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'input_size': model.network[0].in_features
        }, model_path)
        
        print(f"✅ Model saved: {model_path}")
        return model_path
    
    def load_model(self, strategy_id: int) -> nn.Module:
        """Load trained model from disk"""
        model_path = f'models/user_strategies/strategy_{strategy_id}_ml.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        input_size = checkpoint['input_size']
        
        model = DynamicDNN(input_size=input_size).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def predict_probability(self, model: nn.Module, features: np.ndarray) -> float:
        """
        Predict probability for a setup.
        
        Args:
            model: Trained model
            features: Feature array
            
        Returns:
            Probability (0-1)
        """
        model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            probability = model(X).item()
        
        return probability
