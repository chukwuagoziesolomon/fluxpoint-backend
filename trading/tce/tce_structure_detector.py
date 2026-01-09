"""
TCE (Trade Confluence Entry) Structure Detector

Detects the complete TCE pattern:
1. Initial bounce from ONE moving average (MA50, MA18, or MA6)
2. Retracement to 38.2%-61.8%
3. Retest at the SAME moving average
4. Entry on candlestick confirmation at retest
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .types import Indicators, MarketStructure


def identify_retest_ma(
    low_idx: int,
    ma6: float,
    ma18: float,
    ma50: float,
    recent_lows: np.ndarray,
    atr: float,
    direction: str
) -> Optional[Tuple[str, float]]:
    """
    Identify which MA price bounced from (initial bounce point).
    This MA becomes the RETEST level.
    
    For UPTREND (BUY):
        - Price bounces UP from an MA (low is ABOVE the MA)
        - Check which MA is closest to the bounce low
        - That MA = retest level
    
    For DOWNTREND (SELL):
        - Price bounces DOWN from an MA (high is BELOW the MA)
        - Check which MA is closest to the bounce high
        - That MA = retest level
    
    Args:
        low_idx: Index of the retracement low
        ma6, ma18, ma50: Current moving average values
        recent_lows: Array of recent lows
        atr: Average true range
        direction: 'BUY' (uptrend) or 'SELL' (downtrend)
    
    Returns:
        Tuple of (ma_name: str, ma_value: float) or None if no valid MA found
    """
    
    if low_idx < 0 or low_idx >= len(recent_lows):
        return None
    
    bounce_low = recent_lows[low_idx]
    
    # For BUY (uptrend): bounce low is ABOVE the MA
    # For SELL (downtrend): bounce high is BELOW the MA
    
    if direction == "BUY":
        # BUY: Price comes down, bounces OFF an MA
        # Find which MA the low is closest to (but above it)
        
        distances = {}
        
        # Check MA50 first (primary level)
        if bounce_low > ma50:  # Bounce must be ABOVE the MA
            distances['MA50'] = (abs(bounce_low - ma50), ma50)
        
        # Check MA18 second
        if bounce_low > ma18:
            distances['MA18'] = (abs(bounce_low - ma18), ma18)
        
        # Check MA6 last
        if bounce_low > ma6:
            distances['MA6'] = (abs(bounce_low - ma6), ma6)
        
        if not distances:
            return None
        
        # Find closest MA (smallest distance)
        closest_ma_name = min(distances.keys(), key=lambda x: distances[x][0])
        closest_ma_value = distances[closest_ma_name][1]
        
        return (closest_ma_name, closest_ma_value)
    
    elif direction == "SELL":
        # SELL: Price goes up, bounces OFF an MA
        # Find which MA the high is closest to (but below it)
        
        bounce_high = recent_lows[low_idx] if low_idx == 0 else max(recent_lows[max(0, low_idx-5):low_idx+1])
        
        distances = {}
        
        # Check MA50 first (primary level)
        if bounce_high < ma50:  # Bounce must be BELOW the MA
            distances['MA50'] = (abs(bounce_high - ma50), ma50)
        
        # Check MA18 second
        if bounce_high < ma18:
            distances['MA18'] = (abs(bounce_high - ma18), ma18)
        
        # Check MA6 last
        if bounce_high < ma6:
            distances['MA6'] = (abs(bounce_high - ma6), ma6)
        
        if not distances:
            return None
        
        # Find closest MA (smallest distance)
        closest_ma_name = min(distances.keys(), key=lambda x: distances[x][0])
        closest_ma_value = distances[closest_ma_name][1]
        
        return (closest_ma_name, closest_ma_value)
    
    return None


def calculate_retracement_depth(
    swing_high: float,
    swing_low: float,
    retracement_low: float
) -> float:
    """
    Calculate Fibonacci retracement depth as percentage (0-1).
    
    For BUY (retracement DOWN from high):
        depth = (swing_high - retracement_low) / (swing_high - swing_low)
    
    Args:
        swing_high: Peak of the initial swing up
        swing_low: Bottom of the initial swing (starting point)
        retracement_low: Lowest point of the retracement
    
    Returns:
        Retracement depth as percentage (0.382, 0.5, 0.618, etc.)
    """
    
    if swing_high == swing_low:
        return 0.0
    
    total_swing = swing_high - swing_low
    retrace_amount = swing_high - retracement_low
    
    depth = retrace_amount / total_swing
    return max(0.0, min(1.0, depth))


def validate_retracement(
    depth: float,
    min_depth: float = 0.382,
    max_depth: float = 0.618
) -> bool:
    """
    Validate that retracement is within valid Fibonacci levels.
    
    Valid range: 38.2% to 61.8%
    
    Args:
        depth: Retracement depth (0-1)
        min_depth: Minimum valid retracement (default 38.2%)
        max_depth: Maximum valid retracement (default 61.8%)
    
    Returns:
        True if within valid range
    """
    return min_depth <= depth <= max_depth


def detect_second_touch(
    recent_lows: np.ndarray,
    retest_ma_value: float,
    initial_bounce_idx: int,
    atr: float,
    direction: str,
    tolerance_pips: float = 3.0
) -> bool:
    """
    Detect if price has retested the SAME MA level (second touch).
    
    Args:
        recent_lows: Array of recent price lows
        retest_ma_value: The MA level price should retest
        initial_bounce_idx: Index where initial bounce occurred
        atr: Average true range for tolerance
        direction: 'BUY' or 'SELL'
        tolerance_pips: Tolerance in pips (default 3)
    
    Returns:
        True if second touch detected at retest MA
    """
    
    if initial_bounce_idx >= len(recent_lows) - 1:
        return False
    
    # Convert ATR to approximate pips (assuming 5-digit pair)
    tolerance = tolerance_pips * 0.0001
    
    # Scan forward from initial bounce for second touch
    for i in range(initial_bounce_idx + 1, len(recent_lows)):
        current_low = recent_lows[i]
        
        if direction == "BUY":
            # Second touch should be NEAR the MA (within tolerance)
            if abs(current_low - retest_ma_value) <= tolerance:
                return True
        
        elif direction == "SELL":
            # For sell, check the high near the MA
            if abs(current_low - retest_ma_value) <= tolerance:
                return True
    
    return False


def detect_ma_slopes(
    ma_values: np.ndarray,
    period: int = 20
) -> float:
    """
    Calculate MA slope using linear regression.
    
    Positive slope = MA trending UP
    Negative slope = MA trending DOWN
    
    Args:
        ma_values: Array of MA values
        period: Number of bars to use for slope (default 20)
    
    Returns:
        Slope value (positive = up, negative = down)
    """
    
    if len(ma_values) < period:
        return 0.0
    
    x = np.arange(period)
    y = np.array(ma_values[-period:])
    
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]


def validate_tce_structure(
    recent_lows: np.ndarray,
    recent_highs: np.ndarray,
    recent_closes: np.ndarray,
    ma6: float,
    ma18: float,
    ma50: float,
    ma200: float,
    slope6: float,
    slope18: float,
    slope50: float,
    slope200: float,
    atr: float,
    direction: str,
    current_idx: int = -1
) -> Dict:
    """
    Complete TCE structure validation:
    
    1. Main trend: MA50 + MA200 sloping UP (for BUY)
    2. Identify initial bounce MA
    3. Verify retracement depth (38.2%-61.8%)
    4. Confirm second touch at same MA
    5. Check that retest MA is sloping UP
    
    Returns:
        {
            'is_valid': bool,
            'retest_ma': str (MA6, MA18, or MA50),
            'retest_ma_value': float,
            'retracement_depth': float,
            'has_main_trend': bool,
            'retest_ma_slope_up': bool,
            'has_second_touch': bool,
            'failure_reason': str
        }
    """
    
    result = {
        'is_valid': False,
        'retest_ma': None,
        'retest_ma_value': None,
        'retracement_depth': None,
        'has_main_trend': False,
        'retest_ma_slope_up': False,
        'has_second_touch': False,
        'failure_reason': 'Unknown'
    }
    
    # 1. CHECK MAIN TREND (MA50 + MA200 sloping up)
    if direction == "BUY":
        if slope50 <= 0 or slope200 <= 0:
            result['failure_reason'] = 'Main trend not up: MA50 or MA200 not sloping up'
            return result
        result['has_main_trend'] = True
    
    elif direction == "SELL":
        if slope50 >= 0 or slope200 >= 0:
            result['failure_reason'] = 'Main trend not down: MA50 or MA200 not sloping down'
            return result
        result['has_main_trend'] = True
    
    else:
        result['failure_reason'] = 'Invalid direction'
        return result
    
    # 2. FIND RETRACEMENT LOW (38.2%-61.8%)
    if len(recent_lows) < 20:
        result['failure_reason'] = 'Not enough data to detect retracement'
        return result
    
    swing_high = np.max(recent_highs)
    swing_low = np.min(recent_lows)
    current_low = recent_lows[-1]
    
    retracement_depth = calculate_retracement_depth(swing_high, swing_low, current_low)
    
    if not validate_retracement(retracement_depth):
        result['failure_reason'] = f'Retracement depth {retracement_depth:.1%} outside 38.2%-61.8%'
        return result
    
    result['retracement_depth'] = retracement_depth
    
    # 3. IDENTIFY WHICH MA WAS BOUNCED FROM
    # Find the low in the last 10 candles (approximate retracement point)
    recent_low_idx = np.argmin(recent_lows[-10:])
    
    retest_ma_info = identify_retest_ma(
        low_idx=recent_low_idx,
        ma6=ma6,
        ma18=ma18,
        ma50=ma50,
        recent_lows=recent_lows,
        atr=atr,
        direction=direction
    )
    
    if not retest_ma_info:
        result['failure_reason'] = f'No MA identified for bounce (direction: {direction})'
        return result
    
    ma_name, ma_value = retest_ma_info
    result['retest_ma'] = ma_name
    result['retest_ma_value'] = ma_value
    
    # 4. CHECK THAT RETEST MA IS SLOPING UP
    if ma_name == 'MA6':
        ma_slope = slope6
    elif ma_name == 'MA18':
        ma_slope = slope18
    elif ma_name == 'MA50':
        ma_slope = slope50
    else:
        result['failure_reason'] = f'Unknown MA: {ma_name}'
        return result
    
    if direction == "BUY" and ma_slope <= 0:
        result['failure_reason'] = f'Retest MA {ma_name} not sloping up'
        return result
    elif direction == "SELL" and ma_slope >= 0:
        result['failure_reason'] = f'Retest MA {ma_name} not sloping down'
        return result
    
    result['retest_ma_slope_up'] = ma_slope > 0 if direction == "BUY" else ma_slope < 0
    
    # 5. CHECK FOR SECOND TOUCH AT RETEST MA
    second_touch = detect_second_touch(
        recent_lows=recent_lows,
        retest_ma_value=ma_value,
        initial_bounce_idx=recent_low_idx,
        atr=atr,
        direction=direction,
        tolerance_pips=3.0
    )
    
    if not second_touch:
        result['failure_reason'] = f'No second touch detected at {ma_name}'
        return result
    
    result['has_second_touch'] = True
    
    # ALL CHECKS PASSED
    result['is_valid'] = True
    result['failure_reason'] = None
    
    return result
