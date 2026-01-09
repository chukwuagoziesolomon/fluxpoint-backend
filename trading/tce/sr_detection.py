"""
Support/Resistance detection for TCE validation.
Identifies actual bounce levels from recent price action.
"""

from typing import List, Tuple
import numpy as np
from .types import Candle


def find_bounce_levels(
    candles: List[Candle],
    lookback: int = 50,
    tolerance_pips: float = 5
) -> List[Tuple[float, int]]:
    """
    Find horizontal S/R levels by detecting price bounces.
    
    A bounce = price touches a level, reverses, then touches it again.
    
    Args:
        candles: Recent candle data
        lookback: How many candles to scan
        tolerance_pips: Tolerance for grouping touches (5 pips = ±0.0005)
    
    Returns:
        List of (level, bounce_count) tuples, sorted by strength
        Example: [(1.0850, 3), (1.0820, 2), (1.0790, 1)]
    """
    if len(candles) < 5:
        return []
    
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    lows = [c.low for c in recent]
    highs = [c.high for c in recent]
    
    # Find potential support levels (price touched and bounced)
    support_touches = {}  # level: count
    resistance_touches = {}  # level: count
    
    for i in range(1, len(recent) - 1):
        # Support: candle low below previous and next low (V-shape bounce)
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            level = round(lows[i] * 10000) / 10000  # Round to 4 decimals
            # Group nearby levels within tolerance
            level_key = round(level, 4)
            support_touches[level_key] = support_touches.get(level_key, 0) + 1
        
        # Resistance: candle high above previous and next high (^-shape bounce)
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            level = round(highs[i] * 10000) / 10000
            level_key = round(level, 4)
            resistance_touches[level_key] = resistance_touches.get(level_key, 0) + 1
    
    # Combine and sort by bounce count
    all_levels = []
    for level, count in support_touches.items():
        all_levels.append((level, count, 'SUPPORT'))
    for level, count in resistance_touches.items():
        all_levels.append((level, count, 'RESISTANCE'))
    
    # Sort by count (descending) - strongest levels first
    all_levels.sort(key=lambda x: x[1], reverse=True)
    
    return all_levels


def is_at_support_resistance(
    entry_price: float,
    sr_levels: List[Tuple[float, int, str]],
    tolerance_pips: float = 5
) -> Tuple[bool, str]:
    """
    Check if entry price is AT or very close to a S/R level.
    
    Args:
        entry_price: Current setup entry price
        sr_levels: List of (level, bounce_count, type) from find_bounce_levels()
        tolerance_pips: How close counts as "at" (default 5 pips = 0.0005)
    
    Returns:
        (is_at_sr: bool, level_info: str)
        - is_at_sr=True: Price is at S/R level (should REJECT)
        - is_at_sr=False: Price is between levels (should ACCEPT)
    """
    tolerance = tolerance_pips * 0.0001  # Convert pips to decimal
    
    for level, count, sr_type in sr_levels:
        distance = abs(entry_price - level)
        
        # If within tolerance of this level
        if distance < tolerance:
            info = f"At {sr_type} (level: {level:.5f}, bounces: {count})"
            return True, info
    
    # Not at any S/R level - safe to trade
    return False, "Between S/R levels - clean entry"


def get_sr_analysis(
    candles: List[Candle],
    entry_price: float,
    direction: str,
    lookback: int = 50,
    tolerance_pips: float = 5
) -> dict:
    """
    Complete S/R analysis for a setup.
    
    Returns:
        {
            'is_valid': bool,  # False if at S/R, True if between
            'reason': str,
            'sr_levels': [...],  # All detected levels
            'nearby_level': float,  # Closest level
            'bounces_at_nearby': int,
        }
    """
    sr_levels = find_bounce_levels(candles, lookback, tolerance_pips)
    is_at_sr, reason = is_at_support_resistance(entry_price, sr_levels, tolerance_pips)
    
    nearest_level = None
    nearest_distance = float('inf')
    nearest_bounces = 0
    
    if sr_levels:
        for level, bounces, sr_type in sr_levels:
            distance = abs(entry_price - level)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_level = level
                nearest_bounces = bounces
    
    return {
        'is_valid': not is_at_sr,
        'reason': reason,
        'sr_levels': sr_levels,
        'nearest_level': nearest_level,
        'nearest_distance_pips': nearest_distance * 10000 if nearest_distance != float('inf') else None,
        'bounces_at_nearest': nearest_bounces,
        'failure_reason': f"Entry AT support/resistance - price may get stuck" if is_at_sr else None
    }
