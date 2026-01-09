from typing import List
from .types import Candle, Indicators


def near_support_resistance(
    price: float,
    levels: List[float],
    tolerance: float = 0.002
) -> bool:
    for level in levels:
        if abs(price - level) / price < tolerance:
            return True
    return False


def at_ma_level(
    candle: Candle,
    indicators: Indicators,
    direction: str,
    tolerance: float = 0.05  # Relaxed to 5%
) -> bool:
    """
    TCE ONLY uses dynamic support (Moving Averages).
    NO horizontal S/R levels in TCE strategy.
    
    Price may go BELOW the MA (measured by Fibonacci).
    This checks if price is near/below a key MA.
    RELAXED: If price is within 5% of any MA, it counts.
    """
    price = candle.low if direction == "BUY" else candle.high
    
    # Check moving averages as dynamic support/resistance
    # Price can be AT or BELOW MA (Fib measures depth)
    mas = [indicators.ma6, indicators.ma18, indicators.ma50, indicators.ma200]
    
    for ma in mas:
        # Relaxed tolerance: price can be up to 5% away from MA
        if direction == "BUY":
            # Price can be below MA (within reasonable range)
            if candle.low <= ma and (ma - price) / ma < tolerance:
                return True
        else:  # SELL
            # Price can be above MA (within reasonable range)
            if candle.high >= ma and (price - ma) / ma < tolerance:
                return True
    
    return False


def has_ma_retest(
    recent_candles: List[Candle],
    indicators: Indicators,
    direction: str,
    timeframe: str = "M15",
    tolerance: float = 0.05  # Relaxed to 5%
) -> bool:
    """
    SIMPLIFIED: Check if price is currently at an MA that's been tested before.
    This allows price to bounce off MAs multiple times (normal market behavior).
    """
    if len(recent_candles) < 2:
        return False
    
    current = recent_candles[-1]
    mas = [indicators.ma6, indicators.ma18, indicators.ma50, indicators.ma200]
    
    # Check if current candle is near ANY MA
    for ma_value in mas:
        if direction == "BUY":
            # Current price below MA6, MA18, MA50
            if current.low < ma_value:
                # Check if any recent candle was also at/near this MA
                # This confirms the MA has been tested multiple times
                for prev_candle in recent_candles[-20:-1]:  # Check last 19 candles
                    if prev_candle.low <= ma_value <= prev_candle.high:
                        return True  # MA was tested before
        else:  # SELL
            # Current price above MA
            if current.high > ma_value:
                # Check if any recent candle tested this MA
                for prev_candle in recent_candles[-20:-1]:
                    if prev_candle.low <= ma_value <= prev_candle.high:
                        return True  # MA was tested before
    
    return False