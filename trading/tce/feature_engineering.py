"""
Feature Engineering for TCE Strategy ML Training

Extracts normalized, pair-agnostic features from valid TCE setups.
These features allow training across multiple currency pairs.
"""

from typing import Dict, List
import numpy as np
from .types import Candle, Indicators, Swing, MarketStructure, HigherTFCandle


def extract_features(
    candle: Candle,
    indicators: Indicators,
    swing: Swing,
    structure: MarketStructure,
    higher_tf_candles: List[HigherTFCandle],
    recent_candles: List[Candle],
    correlations: Dict[str, float],
    direction: str
) -> np.ndarray:
    """
    Extract normalized features from a valid TCE setup.
    
    Features are pair-agnostic (normalized by ATR, price, etc.) to enable
    training on multiple currency pairs simultaneously.
    
    Returns:
        numpy array of features (shape: [n_features,])
    """
    features = []
    
    # === 1. MA Distance Features (normalized by ATR) ===
    price = candle.close
    atr = indicators.atr if indicators.atr > 0 else 1e-6  # Avoid division by zero
    
    features.append((price - indicators.ma6) / atr)      # Distance to MA6
    features.append((price - indicators.ma18) / atr)     # Distance to MA18
    features.append((price - indicators.ma50) / atr)     # Distance to MA50
    features.append((price - indicators.ma200) / atr)    # Distance to MA200
    
    # === 2. MA Slope Features (trend strength) ===
    features.append(indicators.slope6)
    features.append(indicators.slope18)
    features.append(indicators.slope50)
    features.append(indicators.slope200)
    
    # === 3. Fibonacci Retracement Level ===
    fib_encoded = 0.0
    if swing.fib_level == 0.382:
        fib_encoded = 1.0
    elif swing.fib_level == 0.5:
        fib_encoded = 2.0
    elif swing.fib_level == 0.618:
        fib_encoded = 3.0
    features.append(fib_encoded)
    
    # === 4. Market Structure Features ===
    # Trend strength: consistency of higher highs/lows
    if len(structure.highs) >= 2 and len(structure.lows) >= 2:
        # Count consecutive higher highs (uptrend) or lower highs (downtrend)
        hh_count = sum(1 for i in range(1, len(structure.highs)) if structure.highs[i] > structure.highs[i-1])
        hl_count = sum(1 for i in range(1, len(structure.lows)) if structure.lows[i] > structure.lows[i-1])
        
        lh_count = sum(1 for i in range(1, len(structure.highs)) if structure.highs[i] < structure.highs[i-1])
        ll_count = sum(1 for i in range(1, len(structure.lows)) if structure.lows[i] < structure.lows[i-1])
        
        trend_strength = (hh_count + hl_count - lh_count - ll_count) / max(len(structure.highs) + len(structure.lows), 1)
    else:
        trend_strength = 0.0
    
    features.append(trend_strength)
    
    # === 5. ATR-normalized volatility ===
    # How volatile compared to recent average
    if len(recent_candles) >= 5:
        recent_ranges = [c.high - c.low for c in recent_candles[-5:]]
        avg_range = np.mean(recent_ranges)
        current_range = candle.high - candle.low
        volatility_ratio = current_range / (avg_range if avg_range > 0 else 1e-6)
    else:
        volatility_ratio = 1.0
    features.append(volatility_ratio)
    
    # === 6. Candle Position Features ===
    # Where is close relative to high/low (bullish/bearish pressure)
    candle_range = candle.high - candle.low
    if candle_range > 0:
        close_position = (candle.close - candle.low) / candle_range  # 0=low, 1=high
    else:
        close_position = 0.5
    features.append(close_position)
    
    # Body size relative to range (strong vs weak candle)
    body_size = abs(candle.close - candle.open) / (candle_range if candle_range > 0 else 1e-6)
    features.append(body_size)
    
    # === 7. Higher Timeframe Features ===
    if higher_tf_candles:
        htf = higher_tf_candles[0]  # Primary higher TF
        
        # HTF MA alignment strength
        htf_ma_order = 0.0
        if direction == "BUY":
            if htf.indicators.ma6 > htf.indicators.ma18 > htf.indicators.ma50 > htf.indicators.ma200:
                htf_ma_order = 1.0
        else:  # SELL
            if htf.indicators.ma200 > htf.indicators.ma50 > htf.indicators.ma18 > htf.indicators.ma6:
                htf_ma_order = 1.0
        features.append(htf_ma_order)
        
        # HTF slope strength
        features.append(htf.indicators.slope6)
        features.append(htf.indicators.slope18)
    else:
        features.extend([0.0, 0.0, 0.0])  # Placeholder
    
    # === 8. Correlation Strength ===
    if correlations:
        avg_correlation = np.mean(list(correlations.values()))
        min_correlation = np.min(list(correlations.values()))
    else:
        avg_correlation = 0.0
        min_correlation = 0.0
    features.append(avg_correlation)
    features.append(min_correlation)
    
    # === 9. Direction Encoding ===
    direction_encoded = 1.0 if direction == "BUY" else -1.0
    features.append(direction_encoded)
    
    # === 10. Time-based Features (optional - can help with regime changes) ===
    # Number of candles since MA retest (how fresh is the retest?)
    candles_since_retest = count_candles_since_ma_touch(recent_candles, indicators, direction)
    features.append(candles_since_retest)
    
    return np.array(features, dtype=np.float32)


def count_candles_since_ma_touch(
    recent_candles: List[Candle],
    indicators: Indicators,
    direction: str,
    lookback: int = 50
) -> float:
    """
    Count how many candles ago the MA was first touched.
    Normalized by lookback period.
    """
    mas = [indicators.ma6, indicators.ma18, indicators.ma50, indicators.ma200]
    
    for i in range(len(recent_candles) - 2, max(0, len(recent_candles) - lookback - 1), -1):
        candle = recent_candles[i]
        for ma in mas:
            if direction == "BUY":
                if candle.low <= ma * 1.02:  # Within 2%
                    candles_ago = len(recent_candles) - 1 - i
                    return candles_ago / lookback  # Normalize
            else:
                if candle.high >= ma * 0.98:
                    candles_ago = len(recent_candles) - 1 - i
                    return candles_ago / lookback
    
    return 1.0  # Max normalized value if not found


def get_feature_names() -> List[str]:
    """
    Get feature names for interpretability.
    Useful for feature importance analysis.
    """
    return [
        "dist_to_ma6_atr",
        "dist_to_ma18_atr",
        "dist_to_ma50_atr",
        "dist_to_ma200_atr",
        "slope_ma6",
        "slope_ma18",
        "slope_ma50",
        "slope_ma200",
        "fib_level_encoded",
        "trend_strength",
        "volatility_ratio",
        "close_position",
        "body_size_ratio",
        "htf_ma_alignment",
        "htf_slope_ma6",
        "htf_slope_ma18",
        "avg_correlation",
        "min_correlation",
        "direction_encoded",
        "candles_since_retest_norm"
    ]


def get_feature_count() -> int:
    """
    Get total number of features.
    """
    return len(get_feature_names())
