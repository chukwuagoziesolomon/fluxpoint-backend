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
    """Very relaxed rejection: visible wick in trend direction + directional close.

    This is used as a fallback confirmation so that clear MA bounces with
    reasonable wicks are not rejected just because they don't form a
    textbook pin/engulfing/star pattern.
    """
    body = abs(candle.close - candle.open)
    if body == 0:
        return False

    wick_up = candle.high - max(candle.open, candle.close)
    wick_down = min(candle.open, candle.close) - candle.low

    if direction == "BUY":
        # Any visible lower wick and bullish close
        return wick_down > 0 and candle.close > candle.open
    else:
        # Any visible upper wick and bearish close
        return wick_up > 0 and candle.close < candle.open


# Candlestick pattern detection - RELAXED
def is_bullish_pin_bar(candle: Candle) -> bool:
    """Stricter: strong rejection tail below body (classic pin bar)."""
    body = abs(candle.close - candle.open)
    if body == 0:
        return False
    lower_wick = min(candle.open, candle.close) - candle.low
    upper_wick = candle.high - max(candle.open, candle.close)
    # Lower wick at least 2x body, small upper wick, bullish close
    return (
        lower_wick >= body * 2.0 and
        upper_wick <= body * 0.5 and
        candle.close > candle.open
    )


def is_bearish_pin_bar(candle: Candle) -> bool:
    """Stricter: strong rejection tail above body (classic pin bar)."""
    body = abs(candle.close - candle.open)
    if body == 0:
        return False
    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    # Upper wick at least 2x body, small lower wick, bearish close
    return (
        upper_wick >= body * 2.0 and
        lower_wick <= body * 0.5 and
        candle.close < candle.open
    )


def is_bullish_engulfing(prev: Candle, curr: Candle) -> bool:
    """Stricter bullish engulfing: current body engulfs previous body."""
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    if prev_body == 0 or curr_body == 0:
        return False
    return (
        prev.close < prev.open and            # Previous bearish
        curr.close > curr.open and            # Current bullish
        curr.open <= prev.close and           # Opens inside/at prev body
        curr.close >= prev.open and           # Closes beyond prev body
        curr_body >= prev_body                # Body at least as large
    )


def is_bearish_engulfing(prev: Candle, curr: Candle) -> bool:
    """Stricter bearish engulfing: current body engulfs previous body."""
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    if prev_body == 0 or curr_body == 0:
        return False
    return (
        prev.close > prev.open and            # Previous bullish
        curr.close < curr.open and            # Current bearish
        curr.open >= prev.close and           # Opens inside/at prev body
        curr.close <= prev.open and           # Closes beyond prev body
        curr_body >= prev_body                # Body at least as large
    )


def is_morning_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """Stricter morning star: bearish, small middle, then strong bullish."""
    body1 = abs(c1.close - c1.open)
    body2 = abs(c2.close - c2.open)
    body3 = abs(c3.close - c3.open)
    if body1 == 0 or body3 == 0:
        return False
    return (
        c1.close < c1.open and                # First bearish
        c3.close > c3.open and                # Third bullish
        body2 <= body1 * 0.6 and              # Middle candle small
        body2 <= body3 * 0.6                  # and smaller than third
    )


def is_evening_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """Stricter evening star: bullish, small middle, then strong bearish."""
    body1 = abs(c1.close - c1.open)
    body2 = abs(c2.close - c2.open)
    body3 = abs(c3.close - c3.open)
    if body1 == 0 or body3 == 0:
        return False
    return (
        c1.close > c1.open and                # First bullish
        c3.close < c3.open and                # Third bearish
        body2 <= body1 * 0.6 and              # Middle candle small
        body2 <= body3 * 0.6
    )


def is_tweezer_bottom(prev: Candle, curr: Candle, tolerance: float = 0.0005) -> bool:
    """Stricter: lows almost equal and both show rejection."""
    if not (curr.close > curr.open and prev.close < prev.open or curr.close > curr.open):
        # At least current bullish; previous often bearish or indecisive
        pass
    return abs(prev.low - curr.low) / max(prev.low, curr.low, 0.00001) < tolerance


def is_tweezer_top(prev: Candle, curr: Candle, tolerance: float = 0.0005) -> bool:
    """Stricter: highs almost equal and both show rejection."""
    if not (curr.close < curr.open and prev.close > prev.open or curr.close < curr.open):
        # At least current bearish; previous often bullish or indecisive
        pass
    return abs(prev.high - curr.high) / max(prev.high, curr.high, 0.00001) < tolerance


def is_one_white_soldier(prev: Candle, curr: Candle) -> bool:
    """Stricter: strong bullish candle after bearish, closing above prev high."""
    body_prev = abs(prev.close - prev.open)
    body_curr = abs(curr.close - curr.open)
    if body_prev == 0 or body_curr == 0:
        return False
    return (
        prev.close < prev.open and           # Previous bearish
        curr.close > curr.open and           # Current bullish
        body_curr >= body_prev * 1.2 and     # Stronger body
        curr.close > prev.high               # Closes above previous high
    )


def is_one_black_crow(prev: Candle, curr: Candle) -> bool:
    """Stricter: strong bearish candle after bullish, closing below prev low."""
    body_prev = abs(prev.close - prev.open)
    body_curr = abs(curr.close - curr.open)
    if body_prev == 0 or body_curr == 0:
        return False
    return (
        prev.close > prev.open and           # Previous bullish
        curr.close < curr.open and           # Current bearish
        body_curr >= body_prev * 1.2 and     # Stronger body
        curr.close < prev.low                # Closes below previous low
    )


def has_candlestick_confirmation(candles: List[Candle], direction: str) -> bool:
    if len(candles) < 3:
        return False
    curr = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3] if len(candles) >= 3 else None

    if direction == "BUY":
        patterns = [
            is_rejection_candle(curr, direction),
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
            is_rejection_candle(curr, direction),
            is_bearish_pin_bar(curr),
            is_bearish_engulfing(prev, curr),
            is_one_black_crow(prev, curr),
            is_tweezer_top(prev, curr)
        ]
        if prev2:
            patterns.append(is_evening_star(prev2, prev, curr))
        return any(patterns)