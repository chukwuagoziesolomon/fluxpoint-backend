"""
ML Training Data Collection and Labeling for TCE Strategy

Tracks trade outcomes (TP vs SL hit) to create labeled dataset.
"""

from django.db import models
from trading.models import Trade, Candle
from .feature_engineering import extract_features, get_feature_count
import numpy as np
import json
from typing import Dict, List, Tuple
from datetime import datetime


class TCETrainingData(models.Model):
    """
    Stores feature vectors and labels for ML training.
    Each row represents one valid TCE setup with its outcome.
    """
    trade = models.OneToOneField(Trade, on_delete=models.CASCADE, related_name='ml_training_data')
    
    # Features (stored as JSON for flexibility)
    features = models.JSONField()  # List of normalized feature values
    feature_count = models.IntegerField(default=0)
    
    # Label (binary classification)
    label = models.IntegerField(
        choices=[
            (1, 'TP Hit First'),  # Successful trade
            (0, 'SL Hit First'),  # Failed trade
        ],
        null=True,
        blank=True
    )
    
    # Metadata
    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    entry_timestamp = models.DateTimeField()
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    
    # Trade details for analysis
    direction = models.CharField(max_length=4)
    entry_price = models.FloatField()
    stop_loss = models.FloatField()
    take_profit = models.FloatField()
    exit_price = models.FloatField(null=True, blank=True)
    pnl = models.FloatField(null=True, blank=True)
    
    # R-multiple (profit/loss in terms of risk)
    r_multiple = models.FloatField(null=True, blank=True)
    
    # Training metadata
    is_training_set = models.BooleanField(default=True)  # vs validation/test
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['entry_timestamp']
        indexes = [
            models.Index(fields=['symbol', 'timeframe']),
            models.Index(fields=['entry_timestamp']),
            models.Index(fields=['label']),
            models.Index(fields=['is_training_set']),
        ]
    
    def __str__(self):
        outcome = "TP" if self.label == 1 else "SL" if self.label == 0 else "PENDING"
        return f"{self.symbol} {self.direction} {self.entry_timestamp.date()} - {outcome}"


def create_training_data(
    trade: Trade,
    features: np.ndarray,
    candle: Candle,
    indicators,
    swing,
    structure,
    higher_tf_candles,
    recent_candles,
    correlations,
    direction: str
) -> TCETrainingData:
    """
    Create a training data record for a valid TCE setup.
    Label will be added later when trade closes.
    
    Args:
        trade: Trade model instance
        features: Extracted feature array
        ... other TCE validation inputs
    
    Returns:
        TCETrainingData instance
    """
    training_data = TCETrainingData.objects.create(
        trade=trade,
        features=features.tolist(),
        feature_count=len(features),
        symbol=trade.symbol,
        timeframe=trade.timeframe,
        entry_timestamp=trade.entry_candle.timestamp,
        direction=trade.direction,
        entry_price=trade.entry_price,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        label=None  # Will be set when trade closes
    )
    
    return training_data


def label_trade_outcome(
    training_data: TCETrainingData,
    exit_candles: List[Candle]
) -> bool:
    """
    Monitor trade and label outcome when either TP or SL is hit.
    
    Args:
        training_data: TCETrainingData instance
        exit_candles: Subsequent candles after entry
    
    Returns:
        True if trade is closed and labeled, False if still open
    """
    entry_price = training_data.entry_price
    sl = training_data.stop_loss
    tp = training_data.take_profit
    direction = training_data.direction
    
    for candle in exit_candles:
        if direction == "BUY":
            # Check if TP hit first
            if candle.high >= tp:
                training_data.label = 1
                training_data.exit_price = tp
                training_data.exit_timestamp = candle.timestamp
                training_data.pnl = tp - entry_price
                training_data.r_multiple = (tp - entry_price) / (entry_price - sl)
                training_data.save()
                return True
            
            # Check if SL hit first
            if candle.low <= sl:
                training_data.label = 0
                training_data.exit_price = sl
                training_data.exit_timestamp = candle.timestamp
                training_data.pnl = sl - entry_price
                training_data.r_multiple = (sl - entry_price) / (entry_price - sl)
                training_data.save()
                return True
        
        else:  # SELL
            # Check if TP hit first
            if candle.low <= tp:
                training_data.label = 1
                training_data.exit_price = tp
                training_data.exit_timestamp = candle.timestamp
                training_data.pnl = entry_price - tp
                training_data.r_multiple = (entry_price - tp) / (sl - entry_price)
                training_data.save()
                return True
            
            # Check if SL hit first
            if candle.high >= sl:
                training_data.label = 0
                training_data.exit_price = sl
                training_data.exit_timestamp = candle.timestamp
                training_data.pnl = entry_price - sl
                training_data.r_multiple = (entry_price - sl) / (sl - entry_price)
                training_data.save()
                return True
    
    return False  # Trade still open


def get_training_dataset(
    start_date: datetime = None,
    end_date: datetime = None,
    symbols: List[str] = None,
    min_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve labeled training data as numpy arrays.
    
    Args:
        start_date: Filter by entry date
        end_date: Filter by entry date
        symbols: Filter by symbol list
        min_samples: Minimum samples required
    
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    query = TCETrainingData.objects.filter(
        label__isnull=False,  # Only labeled data
        is_training_set=True
    )
    
    if start_date:
        query = query.filter(entry_timestamp__gte=start_date)
    if end_date:
        query = query.filter(entry_timestamp__lte=end_date)
    if symbols:
        query = query.filter(symbol__in=symbols)
    
    training_data = list(query)
    
    if len(training_data) < min_samples:
        raise ValueError(f"Insufficient training data: {len(training_data)} samples (need {min_samples})")
    
    # Convert to numpy arrays
    X = np.array([td.features for td in training_data], dtype=np.float32)
    y = np.array([td.label for td in training_data], dtype=np.int64)
    
    return X, y


def get_dataset_stats() -> Dict:
    """
    Get statistics about the training dataset.
    """
    total_samples = TCETrainingData.objects.filter(label__isnull=False).count()
    tp_hit = TCETrainingData.objects.filter(label=1).count()
    sl_hit = TCETrainingData.objects.filter(label=0).count()
    pending = TCETrainingData.objects.filter(label__isnull=True).count()
    
    win_rate = (tp_hit / total_samples * 100) if total_samples > 0 else 0
    
    avg_r_multiple = TCETrainingData.objects.filter(
        label__isnull=False
    ).aggregate(models.Avg('r_multiple'))['r_multiple__avg'] or 0
    
    return {
        "total_samples": total_samples,
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "pending": pending,
        "win_rate": round(win_rate, 2),
        "avg_r_multiple": round(avg_r_multiple, 3)
    }
