from typing import List
from .types import Indicators, Candle


def is_uptrend(ind: Indicators) -> bool:
    """
    Uptrend: Check MA alignment (3 of 4 MAs in order) + at least 2 of 3 slopes positive.
    This is more realistic for choppy pairs like USDJPY that don't have perfect alignment.
    
    Requires 3 of 4 MAs to show correct ordering (not all 4 strictly):
    - MA6 > MA18 > MA50 (core momentum)
    - OR MA18 > MA50 > MA200 (longer-term trend)
    AND at least 2 of 3 slopes must be positive.
    """
    # Check if fast MAs are above slow MAs (core uptrend)
    ma_check = (
        (ind.ma6 > ind.ma18 and ind.ma18 > ind.ma50) or  # Short-medium alignment
        (ind.ma18 > ind.ma50 and ind.ma50 > ind.ma200)   # Medium-long alignment
    )
    
    # Check if at least 2 of 3 slopes are positive
    slopes_positive = sum([ind.slope6 > 0, ind.slope18 > 0, ind.slope50 > 0]) >= 2
    
    return ma_check and slopes_positive


def is_uptrend_with_structure(ind: Indicators, lows: List[float]) -> bool:
    """
    Uptrend with swing structure validation.
    UPTREND = MA alignment + slopes + HIGHER LOWS.
    The second dip/test of the MA must not go as deep as the first test.
    This confirms the trend is getting stronger (shallower pullbacks).
    """
    # First check basic MA alignment and slopes
    if not is_uptrend(ind):
        return False
    
    # Second check: higher lows (swing structure)
    # For uptrend, recent lows should be progressively higher
    if len(lows) < 3:
        return False
    
    # Check that at least 60% of recent moves have higher lows
    recent = lows[-10:] if len(lows) >= 10 else lows[-3:]
    higher_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    return higher_count >= len(recent) * 0.6  # At least 60% upward


def is_downtrend(ind: Indicators) -> bool:
    """
    Downtrend: Check MA alignment (3 of 4 MAs in reverse) + at least 2 of 3 slopes negative.
    This is more realistic for choppy pairs that don't have perfect alignment.
    
    Requires 3 of 4 MAs to show correct ordering (not all 4 strictly):
    - MA50 > MA18 > MA6 (core momentum)
    - OR MA200 > MA50 > MA18 (longer-term trend)
    AND at least 2 of 3 slopes must be negative.
    """
    # Check if slow MAs are above fast MAs (core downtrend)
    ma_check = (
        (ind.ma50 > ind.ma18 and ind.ma18 > ind.ma6) or      # Short-medium alignment
        (ind.ma200 > ind.ma50 and ind.ma50 > ind.ma18)       # Medium-long alignment
    )
    
    # Check if at least 2 of 3 slopes are negative
    slopes_negative = sum([ind.slope6 < 0, ind.slope18 < 0, ind.slope50 < 0]) >= 2
    
    return ma_check and slopes_negative


def is_downtrend_with_structure(ind: Indicators, highs: List[float]) -> bool:
    """
    Downtrend with swing structure validation.
    DOWNTREND = MA alignment + slopes + LOWER HIGHS.
    The second rally/test of the MA must not go as high as the first test.
    This confirms the trend is getting stronger (lower retracements).
    """
    # First check basic MA alignment and slopes
    if not is_downtrend(ind):
        return False
    
    # Second check: lower highs (swing structure)
    # For downtrend, recent highs should be progressively lower
    if len(highs) < 3:
        return False
    
    # Check that at least 60% of recent moves have lower highs
    recent = highs[-10:] if len(highs) >= 10 else highs[-3:]
    lower_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
    return lower_count >= len(recent) * 0.6  # At least 60% downward


def valid_fib(swing) -> bool:
    """
    Fibonacci measures how deep price retraces BELOW the MA.
    Valid levels: 38.2%, 50%, 61.8%
    If price goes deeper than 61.8% (e.g., 78.6%), setup is INVALID.
    """
    from .types import Swing
    return swing.fib_level in {0.382, 0.5, 0.618}


def is_rejection_candle(candle: Candle, direction: str) -> bool:
    """Relaxed: Just needs some lower/upper wick"""
    body = abs(candle.close - candle.open)
    wick_up = candle.high - max(candle.open, candle.close)
    wick_down = min(candle.open, candle.close) - candle.low

    if direction == "BUY":
        return wick_down > body * 0.5 and candle.close > candle.open
    else:
        return wick_up > body * 0.5 and candle.close < candle.open


# Candlestick pattern detection - RELAXED
def is_bullish_pin_bar(candle: Candle) -> bool:
    """Relaxed: Lower wick > 1x body (not 2x)"""
    body = abs(candle.close - candle.open)
    lower_wick = min(candle.open, candle.close) - candle.low
    upper_wick = candle.high - max(candle.open, candle.close)
    return lower_wick >= body * 1.0 and upper_wick <= body and candle.close > candle.open


def is_bearish_pin_bar(candle: Candle) -> bool:
    """Relaxed: Upper wick > 1x body (not 2x)"""
    body = abs(candle.close - candle.open)
    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    return upper_wick >= body * 1.0 and lower_wick <= body and candle.close < candle.open


def is_bullish_engulfing(prev: Candle, curr: Candle) -> bool:
    """Relaxed: Just needs current to be bigger and close higher"""
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    return (prev.close < prev.open and  # Previous bearish
            curr.close > curr.open and  # Current bullish
            curr_body > prev_body * 0.5)  # Current body at least half


def is_bearish_engulfing(prev: Candle, curr: Candle) -> bool:
    """Relaxed: Just needs current to be bigger and close lower"""
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    return (prev.close > prev.open and  # Previous bullish
            curr.close < curr.open and  # Current bearish
            curr_body > prev_body * 0.5)  # Current body at least half


def is_morning_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """Relaxed: Just needs reversal direction"""
    return (c1.close < c1.open and  # Bearish
            c3.close > c3.open)  # Bullish


def is_evening_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """Relaxed: Just needs reversal direction"""
    return (c1.close > c1.open and  # Bullish
            c3.close < c3.open)  # Bearish


def is_tweezer_bottom(prev: Candle, curr: Candle, tolerance: float = 0.001) -> bool:
    """Relaxed: Lows within 0.1% of each other"""
    return abs(prev.low - curr.low) / max(prev.low, curr.low, 0.00001) < tolerance


def is_tweezer_top(prev: Candle, curr: Candle, tolerance: float = 0.001) -> bool:
    """Relaxed: Highs within 0.1% of each other"""
    return abs(prev.high - curr.high) / max(prev.high, curr.high, 0.00001) < tolerance


def is_one_white_soldier(prev: Candle, curr: Candle) -> bool:
    """Relaxed: Current just needs to be bullish and close higher than prev"""
    return (prev.close < prev.open and
            curr.close > curr.open and
            curr.close > prev.open)


def is_one_black_crow(prev: Candle, curr: Candle) -> bool:
    # Simplified: strong bearish candle after bullish
    return (prev.close > prev.open and
            curr.close < curr.open and
            curr.close < prev.low and
            curr.open > prev.close)


def has_candlestick_confirmation(candles: List[Candle], direction: str) -> bool:
    if len(candles) < 3:
        return False
    curr = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3] if len(candles) >= 3 else None

    if direction == "BUY":
        patterns = [
            is_bullish_pin_bar(curr),
            is_bullish_engulfing(prev, curr),
            is_one_white_soldier(prev, curr),
            is_tweezer_bottom(prev, curr)
        ]
        if prev2:
            patterns.append(is_morning_star(prev2, prev, curr))
        return any(patterns)
    else:  # SELL
        patterns = [
            is_bearish_pin_bar(curr),
            is_bearish_engulfing(prev, curr),
            is_one_black_crow(prev, curr),
            is_tweezer_top(prev, curr)
        ]
        if prev2:
            patterns.append(is_evening_star(prev2, prev, curr))
        return any(patterns)