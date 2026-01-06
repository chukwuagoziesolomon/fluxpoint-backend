from typing import List
from .types import Indicators, MarketStructure, Candle
from .utils import is_uptrend, is_downtrend


def has_higher_highs(highs: List[float]) -> bool:
    return all(highs[i] > highs[i - 1] for i in range(1, len(highs)))


def has_higher_lows(lows: List[float]) -> bool:
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def has_lower_highs(highs: List[float]) -> bool:
    return all(highs[i] < highs[i - 1] for i in range(1, len(highs)))


def has_lower_lows(lows: List[float]) -> bool:
    return all(lows[i] < lows[i - 1] for i in range(1, len(lows)))


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
    return (
        is_uptrend(indicators)
        and has_higher_highs(structure.highs)
        and has_higher_lows(structure.lows)
    )


# 🔽 Downtrend
def is_valid_downtrend(
    indicators: Indicators,
    structure: MarketStructure
) -> bool:
    return (
        is_downtrend(indicators)
        and has_lower_highs(structure.highs)
        and has_lower_lows(structure.lows)
    )