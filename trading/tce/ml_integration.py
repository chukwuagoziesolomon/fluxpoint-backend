"""
ML Integration into TCE Validation

Adds ML probability prediction as an additional filter for TCE trades.
"""

import torch
import numpy as np
from typing import Dict, Optional
import os

from .ml_model import TCEProbabilityModel
from .feature_engineering import extract_features, get_feature_count
from .types import Candle, Indicators, Swing, MarketStructure, HigherTFCandle


class TCEMLPredictor:
    """
    Wrapper for loading and using trained TCE ML model for predictions.
    """
    
    def __init__(self, model_path: str = "tce_model.pt", device: str = None):
        """
        Args:
            model_path: Path to trained model file
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_path = model_path
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.is_loaded = False
        
        # Try to load model
        if os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load trained model from file."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            n_features = checkpoint.get('n_features', get_feature_count())
            dropout_rate = checkpoint.get('dropout_rate', 0.3)
            
            self.model = TCEProbabilityModel(
                n_features=n_features,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_loaded = True
            
            print(f"✅ TCE ML model loaded from {self.model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load ML model: {e}")
            self.is_loaded = False
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict success probability for a TCE setup.
        
        Args:
            features: Feature array (n_features,)
        
        Returns:
            Probability of success [0, 1]
        """
        if not self.is_loaded:
            return 0.5  # Return neutral probability if model not loaded
        
        try:
            with torch.no_grad():
                # Ensure features are 2D: (1, n_features)
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                X_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
                probability = self.model(X_tensor).cpu().numpy()[0][0]
                
                return float(probability)
        except Exception as e:
            print(f"⚠️  ML prediction error: {e}")
            return 0.5


# Global predictor instance (singleton pattern)
_ml_predictor = None


def get_ml_predictor(model_path: str = "tce_model.pt") -> TCEMLPredictor:
    """
    Get or create global ML predictor instance.
    
    Args:
        model_path: Path to model file
    
    Returns:
        TCEMLPredictor instance
    """
    global _ml_predictor
    
    if _ml_predictor is None:
        _ml_predictor = TCEMLPredictor(model_path)
    
    return _ml_predictor


def add_ml_probability_to_validation(
    validation_result: Dict,
    candle: Candle,
    indicators: Indicators,
    swing: Swing,
    structure: MarketStructure,
    higher_tf_candles: list,
    recent_candles: list,
    correlations: Dict,
    direction: str,
    ml_threshold: float = 0.65,
    use_ml: bool = True
) -> Dict:
    """
    Add ML probability prediction to validation result.
    
    This function extends the TCE validation by adding ML-based filtering.
    Only trades with P(success) >= ml_threshold will be taken.
    
    Args:
        validation_result: Result dict from validate_tce()
        ... (other TCE validation inputs)
        ml_threshold: Minimum probability to take trade (default 0.65)
        use_ml: Whether to use ML filter (can be disabled)
    
    Returns:
        Updated validation result with ML probability
    """
    # If validation already failed, skip ML prediction
    if not validation_result.get("is_valid", False):
        validation_result["ml_probability"] = None
        validation_result["ml_filter_ok"] = False
        return validation_result
    
    # If ML is disabled, pass through
    if not use_ml:
        validation_result["ml_probability"] = None
        validation_result["ml_filter_ok"] = True
        validation_result["ml_enabled"] = False
        return validation_result
    
    try:
        # Extract features
        features = extract_features(
            candle=candle,
            indicators=indicators,
            swing=swing,
            structure=structure,
            higher_tf_candles=higher_tf_candles,
            recent_candles=recent_candles,
            correlations=correlations,
            direction=direction
        )
        
        # Get ML prediction
        predictor = get_ml_predictor()
        probability = predictor.predict(features)
        
        # Add to result
        validation_result["ml_probability"] = round(probability, 4)
        validation_result["ml_threshold"] = ml_threshold
        validation_result["ml_enabled"] = True
        
        # Check if probability meets threshold
        if probability >= ml_threshold:
            validation_result["ml_filter_ok"] = True
        else:
            validation_result["ml_filter_ok"] = False
            validation_result["is_valid"] = False
            validation_result["failure_reason"] = f"ML probability {probability:.2%} below threshold {ml_threshold:.2%}"
    
    except Exception as e:
        print(f"⚠️  ML filtering error: {e}")
        validation_result["ml_probability"] = None
        validation_result["ml_filter_ok"] = True  # Don't block trade on ML error
        validation_result["ml_enabled"] = False
    
    return validation_result
