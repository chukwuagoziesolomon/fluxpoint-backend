from typing import List, Optional
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


def detect_active_ma(
    recent_candles: List[Candle],
    indicators: Indicators,
    direction: str,
    timeframe: str = "M15",
    consolidation_lookback: int = 7,
    min_consolidation_ratio: float = 0.5,
    penetration_factor: float = 0.1,
    close_factor: float = 0.2,
) -> Optional[str]:
    """Detect which MA (18/50/200) is actually acting as support/resistance.

    Logic:
    - Ignore MA6 for swing classification (used only for trend elsewhere).
    - Look at the last N candles for consolidation around each MA.
    - On the most recent candle, require penetration + rejection at that MA.
    - Return the first MA that satisfies both consolidation and bounce tests.
    """

    if not recent_candles:
        return None

    last = recent_candles[-1]

    # Use ATR if available; otherwise fall back to small percentage of price
    atr = getattr(indicators, "atr", None)
    ref_price = last.close if last.close else last.open
    if atr is None or atr <= 0:
        atr = ref_price * 0.001  # ~0.1% of price

    pen_tol = atr * penetration_factor
    close_tol = atr * close_factor

    ma_values = {
        "MA18": indicators.ma18,
        "MA50": indicators.ma50,
        "MA200": indicators.ma200,
    }

    # Consolidation window: last N candles (or all if fewer)
    if len(recent_candles) >= consolidation_lookback:
        window = recent_candles[-consolidation_lookback:]
    else:
        window = recent_candles

    required_hits = max(1, int(len(window) * min_consolidation_ratio))

    for ma_name, ma_val in ma_values.items():
        if ma_val is None:
            continue

        # 1) Consolidation around this MA
        consolidation_hits = 0
        for c in window:
            mid = (c.high + c.low) / 2.0
            if abs(mid - ma_val) <= close_tol:
                consolidation_hits += 1

        if consolidation_hits < required_hits:
            continue

        # 2) Penetration + rejection on the most recent candle
        if direction == "BUY":
            # Price dips below MA then closes back near/above it
            if last.low <= ma_val - pen_tol and last.close >= ma_val - close_tol:
                return ma_name
        else:  # SELL
            # Price spikes above MA then closes back near/below it
            if last.high >= ma_val + pen_tol and last.close <= ma_val + close_tol:
                return ma_name

    return None
