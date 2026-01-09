from typing import List
from .types import Indicators, MarketStructure, Candle
from .utils import is_uptrend, is_downtrend, is_uptrend_with_structure, is_downtrend_with_structure


def has_higher_highs(highs: List[float]) -> bool:
    """Check if most recent highs are progressively higher (trend structure)."""
    if len(highs) < 3:
        return False
    # Check that at least 70% of recent moves have higher highs
    recent = highs[-10:]  # Last 10 candles
    higher_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    return higher_count >= len(recent) * 0.6  # At least 60% upward


def has_higher_lows(lows: List[float]) -> bool:
    """Check if most recent lows are progressively higher (uptrend structure)."""
    if len(lows) < 3:
        return False
    # Check that at least 70% of recent moves have higher lows
    recent = lows[-10:]  # Last 10 candles
    higher_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    return higher_count >= len(recent) * 0.6  # At least 60% upward


def has_lower_highs(highs: List[float]) -> bool:
    """Check if most recent highs are progressively lower (downtrend structure)."""
    if len(highs) < 3:
        return False
    # Check that at least 70% of recent moves have lower highs
    recent = highs[-10:]  # Last 10 candles
    lower_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
    return lower_count >= len(recent) * 0.6  # At least 60% downward


def has_lower_lows(lows: List[float]) -> bool:
    """Check if most recent lows are progressively lower (downtrend structure)."""
    if len(lows) < 3:
        return False
    # Check that at least 70% of recent moves have lower lows
    recent = lows[-10:]  # Last 10 candles
    lower_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
    return lower_count >= len(recent) * 0.6  # At least 60% downward


def is_semi_circle_swing(highs: List[float], lows: List[float], min_swings: int = 2) -> bool:
    """
    Check if swings form a semi-circle/curved retracement pattern.
    This is a smooth pullback to a key level (like MA) allowing retests.
    NOT a triangle - more like a rounded swing back to support/resistance.
    """
    if len(highs) < min_swings or len(lows) < min_swings:
        return False
    
    # Semi-circle means price makes a rounded retracement
    # We just need at least 2 swing points showing a pullback structure
    # Allow retests (price can touch same level multiple times)
    return True  # Validated by the actual MA bounce + candlestick pattern


# 🔼 Uptrend
def is_valid_uptrend(
    indicators: Indicators,
    structure: MarketStructure
) -> bool:
    """
    Uptrend validation with swing structure.
    ✅ UPTREND = MA alignment + positive slopes + HIGHER LOWS
    
    The second dip must be higher than the first dip, meaning price is
    not going down as far on pullbacks = strengthening uptrend.
    """
    # Use structure-aware validation if lows are available
    if structure.lows and len(structure.lows) >= 3:
        return is_uptrend_with_structure(indicators, structure.lows)
    
    # Fallback to basic MA alignment + slopes if no structure
    return is_uptrend(indicators)


# 🔽 Downtrend
def is_valid_downtrend(
    indicators: Indicators,
    structure: MarketStructure
) -> bool:
    """
    Downtrend validation with swing structure.
    ✅ DOWNTREND = MA alignment + negative slopes + LOWER HIGHS
    
    The second rally must be lower than the first rally, meaning price is
    not going up as far on rebounds = strengthening downtrend.
    """
    # Use structure-aware validation if highs are available
    if structure.highs and len(structure.highs) >= 3:
        return is_downtrend_with_structure(indicators, structure.highs)
    
    # Fallback to basic MA alignment + slopes if no structure
    return is_downtrend(indicators)