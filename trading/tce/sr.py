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
    tolerance: float = 0.003
) -> bool:
    """
    TCE ONLY uses dynamic support (Moving Averages).
    NO horizontal S/R levels in TCE strategy.
    
    Price may go BELOW the MA (measured by Fibonacci).
    This checks if price is near/below a key MA.
    """
    price = candle.low if direction == "BUY" else candle.high
    
    # Check moving averages as dynamic support/resistance
    # Price can be AT or BELOW MA (Fib measures depth)
    mas = [indicators.ma6, indicators.ma18, indicators.ma50, indicators.ma200]
    
    for ma in mas:
        # Allow price to be near OR below MA for BUY
        # Allow price to be near OR above MA for SELL
        if direction == "BUY":
            # Price can be below MA (within reasonable range)
            if candle.low <= ma and abs(price - ma) / ma < 0.02:  # Within 2%
                return True
        else:  # SELL
            # Price can be above MA (within reasonable range)
            if candle.high >= ma and abs(price - ma) / ma < 0.02:
                return True
    
    return False


def has_ma_retest(
    recent_candles: List[Candle],
    indicators: Indicators,
    direction: str,
    timeframe: str = "M15",
    tolerance: float = 0.02
) -> bool:
    """
    Check if this is a RETEST of an MA, not the first touch.
    
    REQUIRED: 
    1. First touch: Price hit MA earlier (can be many candles back)
    2. Moved away: Price moved away from MA temporarily
    3. Retest: Price came BACK to same MA (current candle)
    
    You DON'T enter on first bounce - you wait for the RETEST.
    
    Lookback varies by timeframe:
    - M15/M30: 50 candles
    - H1/H4: 100 candles
    - D1/W1: 200 candles
    """
    # Dynamic lookback based on timeframe
    lookback_map = {
        'M1': 30, 'M5': 40, 'M15': 50, 'M30': 50,
        'H1': 100, 'H4': 100,
        'D1': 200, 'W1': 200, 'MN1': 200
    }
    lookback = lookback_map.get(timeframe, 50)
    if len(recent_candles) < 3:
        return False
    
    current = recent_candles[-1]
    mas = [
        ('ma6', indicators.ma6),
        ('ma18', indicators.ma18),
        ('ma50', indicators.ma50),
        ('ma200', indicators.ma200)
    ]
    
    # Check each MA for retest pattern
    for ma_name, ma_value in mas:
        # Is current candle at this MA?
        current_at_ma = False
        if direction == "BUY":
            if current.low <= ma_value and abs(current.low - ma_value) / ma_value < tolerance:
                current_at_ma = True
        else:  # SELL
            if current.high >= ma_value and abs(current.high - ma_value) / ma_value < tolerance:
                current_at_ma = True
        
        if not current_at_ma:
            continue
        
        # Now check if there was a FIRST touch and move away
        first_touch_found = False
        moved_away_found = False
        
        for i in range(len(recent_candles) - 2, max(0, len(recent_candles) - lookback - 1), -1):
            candle = recent_candles[i]
            
            # Check if this candle touched the MA (first touch)
            if direction == "BUY":
                if candle.low <= ma_value and abs(candle.low - ma_value) / ma_value < tolerance:
                    first_touch_found = True
            else:  # SELL
                if candle.high >= ma_value and abs(candle.high - ma_value) / ma_value < tolerance:
                    first_touch_found = True
            
            # Check if price moved away from MA
            if first_touch_found:
                if direction == "BUY":
                    # Price should have moved UP away from MA
                    if candle.low > ma_value * (1 + tolerance):
                        moved_away_found = True
                        break
                else:  # SELL
                    # Price should have moved DOWN away from MA
                    if candle.high < ma_value * (1 - tolerance):
                        moved_away_found = True
                        break
        
        # If we found first touch + move away + current retest = valid
        if first_touch_found and moved_away_found:
            return True
    
    return False