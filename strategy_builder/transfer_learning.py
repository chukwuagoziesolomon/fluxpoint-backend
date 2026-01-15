"""
Transfer Learning for User Strategies

Pre-train base model on TCE data, then fine-tune for user strategies.
This dramatically improves training with limited data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import os
from datetime import datetime

from .ml_training import DynamicDNN


class TransferLearningManager:
    """
    Manages transfer learning from TCE strategy to user strategies.
    
    Approach:
    1. Pre-train base model on TCE data (thousands of setups)
    2. Save base model as foundation
    3. For user strategies:
       - Load base model
       - Replace input/output layers to match user's feature size
       - Freeze early layers (feature extractors)
       - Fine-tune on user data
    
    Benefits:
    - Requires 10x less user data
    - Faster convergence
    - Better generalization
    - Leverages knowledge from proven TCE strategy
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model_path = 'models/transfer_learning/base_model.pth'
    
    def create_base_model_from_tce(
        self,
        tce_data_path: str = 'data/tce_training_data.npz'
    ) -> Dict:
        """
        Pre-train base model on TCE strategy data.
        
        This should be run ONCE to create the foundation model.
        All user strategies will then fine-tune from this.
        
        Args:
            tce_data_path: Path to TCE training data
            
        Returns:
            Training results
        """
        print(f"\n{'='*80}")
        print("CREATING TRANSFER LEARNING BASE MODEL")
        print(f"{'='*80}\n")
        
        # Load TCE data
        print("📊 Loading TCE training data...")
        try:
            data = np.load(tce_data_path)
            X_train = data['X_train']
            y_train = data['y_train']
            print(f"✅ Loaded {len(X_train)} TCE setups with {X_train.shape[1]} features")
        except FileNotFoundError:
            print("⚠️  TCE data not found. You need to run TCE data collection first.")
            print("   Run: python CELL4_COMPLETE_TCE_VALIDATION.py")
            return {'success': False, 'error': 'TCE data not found'}
        
        # Create base model with TCE feature size
        input_size = X_train.shape[1]
        base_model = DynamicDNN(input_size=input_size).to(self.device)
        
        print(f"\n🧠 Training base model...")
        print(f"   Architecture: [{input_size} → 128 → 64 → 32 → 1]")
        print(f"   Samples: {len(X_train)}")
        
        # Train model (simplified - use full ml_training pipeline in practice)
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)
        
        epochs = 100
        for epoch in range(epochs):
            base_model.train()
            total_loss = 0.0
            
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = base_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(loader)
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save base model
        os.makedirs('models/transfer_learning', exist_ok=True)
        
        torch.save({
            'model_state_dict': base_model.state_dict(),
            'input_size': input_size,
            'architecture': str(base_model),
            'trained_on': 'TCE_strategy',
            'samples': len(X_train),
            'timestamp': datetime.now().isoformat()
        }, self.base_model_path)
        
        print(f"\n✅ Base model saved: {self.base_model_path}")
        print(f"{'='*80}\n")
        
        return {
            'success': True,
            'model_path': self.base_model_path,
            'input_size': input_size,
            'training_samples': len(X_train)
        }
    
    def create_user_model_with_transfer_learning(
        self,
        user_input_size: int,
        freeze_layers: bool = True
    ) -> Tuple[nn.Module, Dict]:
        """
        Create model for user strategy using transfer learning.
        
        Args:
            user_input_size: User's feature count
            freeze_layers: Whether to freeze early layers
            
        Returns:
            (model, transfer_info)
        """
        # Check if base model exists
        if not os.path.exists(self.base_model_path):
            print("⚠️  Base model not found. Creating model from scratch...")
            return DynamicDNN(input_size=user_input_size).to(self.device), {
                'transfer_learning': False,
                'reason': 'Base model not available'
            }
        
        # Load base model
        print(f"🔄 Loading base model for transfer learning...")
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        base_input_size = checkpoint['input_size']
        
        # Create new model with user's input size
        user_model = DynamicDNN(input_size=user_input_size).to(self.device)
        
        # Transfer weights from base model
        transfer_info = self._transfer_weights(
            base_checkpoint=checkpoint,
            user_model=user_model,
            base_input_size=base_input_size,
            user_input_size=user_input_size,
            freeze_layers=freeze_layers
        )
        
        return user_model, transfer_info
    
    def _transfer_weights(
        self,
        base_checkpoint: Dict,
        user_model: nn.Module,
        base_input_size: int,
        user_input_size: int,
        freeze_layers: bool
    ) -> Dict:
        """
        Transfer weights from base model to user model.
        
        Strategy:
        - If input sizes match: transfer all layers
        - If different: transfer middle layers (feature extractors), reinit input/output
        - Optionally freeze transferred layers
        """
        base_state_dict = base_checkpoint['model_state_dict']
        user_state_dict = user_model.state_dict()
        
        transferred_layers = []
        reinitialized_layers = []
        frozen_layers = []
        
        # Transfer compatible layers
        for name, param in base_state_dict.items():
            if name in user_state_dict:
                # Check if shapes match
                if param.shape == user_state_dict[name].shape:
                    user_state_dict[name] = param
                    transferred_layers.append(name)
                    
                    # Freeze layer if requested and it's not the output layer
                    if freeze_layers and 'network.0' in name:  # First layer (input layer)
                        # Don't transfer input layer if sizes differ
                        if base_input_size != user_input_size:
                            reinitialized_layers.append(name)
                            continue
                    
                    if freeze_layers and not name.startswith('network.9'):  # Not output layer
                        # Freeze this layer
                        for param_name, param_obj in user_model.named_parameters():
                            if param_name == name:
                                param_obj.requires_grad = False
                                frozen_layers.append(name)
                                break
                else:
                    reinitialized_layers.append(name)
        
        # Load transferred weights
        user_model.load_state_dict(user_state_dict, strict=False)
        
        print(f"✅ Transfer learning applied:")
        print(f"   Transferred: {len(transferred_layers)} layers")
        print(f"   Frozen: {len(frozen_layers)} layers (feature extractors)")
        print(f"   Reinitialized: {len(reinitialized_layers)} layers (input/output)")
        print(f"   Training only: {sum(p.requires_grad for p in user_model.parameters())} parameters")
        
        return {
            'transfer_learning': True,
            'base_model_samples': base_checkpoint.get('samples', 'unknown'),
            'transferred_layers': len(transferred_layers),
            'frozen_layers': len(frozen_layers),
            'reinitialized_layers': len(reinitialized_layers),
            'trainable_params': sum(p.numel() for p in user_model.parameters() if p.requires_grad)
        }
    
    def fine_tune_user_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,  # Fewer epochs needed with transfer learning
        learning_rate: float = 0.0005  # Lower LR for fine-tuning
    ) -> Dict:
        """
        Fine-tune transferred model on user data.
        
        Uses smaller learning rate and fewer epochs since we're fine-tuning.
        
        Args:
            model: Model with transferred weights
            X_train: User's training data
            y_train: User's labels
            epochs: Training epochs (default 50, vs 100 from scratch)
            learning_rate: Learning rate (default 0.0005, vs 0.001 from scratch)
            
        Returns:
            Training results
        """
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        print(f"\n🎯 Fine-tuning on user data...")
        print(f"   Samples: {len(X_train)}")
        print(f"   Epochs: {epochs} (fewer needed with transfer learning)")
        print(f"   Learning Rate: {learning_rate} (lower for fine-tuning)")
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)  # Smaller batch for fine-tuning
        
        criterion = nn.BCELoss()
        
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        history = {'train_loss': [], 'train_accuracy': []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            
            avg_loss = total_loss / len(loader)
            accuracy = correct / total
            
            history['train_loss'].append(avg_loss)
            history['train_accuracy'].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")
        
        return history
    
    def unfreeze_and_fine_tune(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        learning_rate: float = 0.0001  # Very low LR for full fine-tuning
    ) -> Dict:
        """
        Unfreeze all layers and do final fine-tuning.
        
        This is the second stage of transfer learning:
        1. First: Fine-tune with frozen layers (fast, prevents catastrophic forgetting)
        2. Second: Unfreeze all and fine-tune (allows full adaptation)
        
        Args:
            model: Model after initial fine-tuning
            X_train: User's training data
            y_train: User's labels
            epochs: Fine-tuning epochs (default 30)
            learning_rate: Very low LR to prevent forgetting
            
        Returns:
            Training results
        """
        print(f"\n🔓 Unfreezing all layers for final fine-tuning...")
        
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        
        print(f"   All {sum(p.numel() for p in model.parameters())} parameters now trainable")
        
        return self.fine_tune_user_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            learning_rate=learning_rate
        )
    
    def get_recommended_strategy(self, user_sample_count: int) -> Dict:
        """
        Recommend transfer learning strategy based on data availability.
        
        Args:
            user_sample_count: Number of user's training samples
            
        Returns:
            Recommended approach
        """
        if user_sample_count < 50:
            return {
                'strategy': 'aggressive_transfer',
                'freeze_layers': True,
                'freeze_until_epoch': 80,  # Increased from 40 to 80
                'fine_tune_epochs': 120,   # Increased from 50 to 120
                'full_fine_tune_epochs': 40,  # Increased from 20 to 40
                'learning_rate': 0.0003,
                'reason': 'Very limited data - rely heavily on base model with more training'
            }
        
        elif user_sample_count < 150:
            return {
                'strategy': 'moderate_transfer',
                'freeze_layers': True,
                'freeze_until_epoch': 50,  # Increased from 20 to 50
                'fine_tune_epochs': 150,   # Increased from 50 to 150
                'full_fine_tune_epochs': 50,  # Increased from 30 to 50
                'learning_rate': 0.0005,
                'reason': 'Limited data - use transfer learning with thorough training'
            }
        
        elif user_sample_count < 300:
            return {
                'strategy': 'light_transfer',
                'freeze_layers': True,
                'freeze_until_epoch': 30,  # Increased from 10 to 30
                'fine_tune_epochs': 200,   # Increased from 70 to 200
                'full_fine_tune_epochs': 50,  # Increased from 30 to 50
                'learning_rate': 0.0007,
                'reason': 'Decent data - use transfer learning with extensive training'
            }
        
        else:
            return {
                'strategy': 'optional_transfer',
                'freeze_layers': False,  # Don't freeze, just initialize
                'freeze_until_epoch': 0,
                'fine_tune_epochs': 300,  # Increased from 100 to 300
                'full_fine_tune_epochs': 0,  # No need for second stage
                'learning_rate': 0.001,
                'reason': 'Sufficient data - transfer learning with comprehensive training'
            }
