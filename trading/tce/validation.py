from typing import List, Dict, Tuple
from .types import Candle, Indicators, Swing, MarketStructure, HigherTFCandle
from .utils import has_candlestick_confirmation, valid_fib
from .sr import at_ma_level, has_ma_retest
from .structure import is_valid_uptrend, is_valid_downtrend, is_semi_circle_swing
from .risk_management import (
    calculate_stop_loss, 
    calculate_take_profit, 
    validate_risk_management,
    calculate_position_size,
    get_pip_value_per_lot
)
from .order_placement import (
    calculate_entry_order_price,
    create_pending_order,
    validate_order_placement,
    get_order_type
)
from .correlation import validate_correlation_directions
from .sr_detection import get_sr_analysis


def higher_timeframe_confirmed(
    higher_tf_candles: List[HigherTFCandle],
    direction: str
) -> bool:
    for htf in higher_tf_candles:
        if direction == "BUY" and not is_valid_uptrend(htf.indicators, MarketStructure([], [])):  # Note: structure not used here, but per feedback
            return False
        if direction == "SELL" and not is_valid_downtrend(htf.indicators, MarketStructure([], [])):
            return False
    return True


def correlation_confirmed(
    correlations: Dict[str, Tuple[float, str]],
    direction: str,
    threshold: float = 0.5
) -> bool:
    """
    Validate correlation pairs using correlation coefficients.
    
    Args:
        correlations: {
            'GBPUSD': (0.72, 'UP'),      # coefficient, trend_direction
            'GBPJPY': (-0.65, 'DOWN'),   # Negative = opposite direction
        }
        direction: Current setup direction ('BUY' or 'SELL')
        threshold: Minimum correlation strength (0.5 = moderate)
    
    Returns:
        True if all correlated pairs align correctly
    """
    # Convert BUY/SELL to UP/DOWN for correlation checking
    trend_direction = 'UP' if direction == 'BUY' else 'DOWN'
    
    return validate_correlation_directions(
        correlations=correlations,
        direction=trend_direction,
        threshold=threshold
    )


def ma_hit_confirmed(
    candle: Candle,
    indicators: Indicators,
    higher_tf_candles: List[HigherTFCandle],
    direction: str
) -> bool:
    """
    Additional MA hit rule:
    - If entry candle hits MA18, higher TF must hit MA6
    - If entry candle hits MA50, higher TF must hit MA18
    """
    hits_ma18 = False
    hits_ma50 = False

    if direction == "BUY":
        if candle.low < indicators.ma18:
            hits_ma18 = True
        if candle.low < indicators.ma50:
            hits_ma50 = True
    else:  # SELL
        if candle.high > indicators.ma18:
            hits_ma18 = True
        if candle.high > indicators.ma50:
            hits_ma50 = True

    # Check higher TF
    for htf in higher_tf_candles:
        if hits_ma18:
            # Higher TF must hit MA6
            if direction == "BUY":
                if htf.low >= htf.indicators.ma6:
                    return False
            else:
                if htf.high <= htf.indicators.ma6:
                    return False
        if hits_ma50:
            # Higher TF must hit MA18
            if direction == "BUY":
                if htf.low >= htf.indicators.ma18:
                    return False
            else:
                if htf.high <= htf.indicators.ma18:
                    return False
    return True


def validate_tce(
    candle: Candle,
    indicators: Indicators,
    swing: Swing,
    sr_levels: List[float],  # Not used in TCE - kept for compatibility
    higher_tf_candles: List[HigherTFCandle],
    correlations: Dict[str, float],
    structure: MarketStructure,
    recent_candles: List[Candle],
    timeframe: str = "M15",
    account_balance: float = 10000.0,
    risk_percentage: float = 1.0,
    symbol: str = "EURUSD"
) -> Dict:

    result = {
        "is_valid": False,
        "direction": None,
        "trend_ok": False,
        "fib_ok": False,
        "swing_ok": False,
        "ma_level_ok": False,
        "ma_retest_ok": False,
        "candlestick_ok": False,
        "multi_tf_ok": False,
        "ma_hit_ok": False,
        "correlation_ok": False,
        "risk_management_ok": False,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "sl_pips": None,
        "tp_pips": None,
        "position_size": None,
        "risk_amount": None,
        "pip_value_per_lot": None,
        "failure_reason": None,
        # NEW: Order Placement Details
        "order_type": None,  # "BUY_STOP" or "SELL_STOP"
        "entry_price": None,  # Entry price 2-3 pips above/below confirmation candle
        "pending_order": None,  # Full order specification dict
        "order_placement_valid": False,  # Whether order placement is valid
        "confirmation_candle_high": None,
        "confirmation_candle_low": None
    }

    # 1️⃣ Trend
    if is_valid_uptrend(indicators, structure):
        direction = "BUY"
    elif is_valid_downtrend(indicators, structure):
        direction = "SELL"
    else:
        result["failure_reason"] = "Trend not confirmed by structure"
        return result

    result["trend_ok"] = True
    result["direction"] = direction

    # 2️⃣ Fibonacci (measures depth BELOW MA)
    # Price often goes BELOW the MA - Fib measures how deep
    # Valid: 38.2%, 50%, 61.8% - Beyond 61.8% = INVALID setup
    if not valid_fib(swing):
        result["failure_reason"] = "Invalid Fibonacci - price retraced beyond 61.8%"
        return result
    result["fib_ok"] = True

    # 2.5️⃣ Semi-circle Swing Structure
    if not is_semi_circle_swing(structure.highs, structure.lows):
        result["failure_reason"] = "No proper swing structure - need curved retracement pattern"
        return result
    result["swing_ok"] = True

    # 3️⃣ MANDATORY: Must be at/near Moving Average (DYNAMIC SUPPORT ONLY)
    # TCE does NOT use horizontal S/R - ONLY Moving Averages + Fibonacci
    # Price can go BELOW MA (Fib measures depth) - that's normal
    # MA6, MA18, MA50, or MA200 act as dynamic support/resistance
    if not at_ma_level(candle, indicators, direction):
        result["failure_reason"] = "Not at Moving Average - TCE requires MA bounce (dynamic support only)"
        return result
    result["ma_level_ok"] = True

    # 3.5️⃣ MANDATORY: Must be RETEST of MA (not first touch)
    # Don't enter on FIRST bounce - wait for RETEST
    # Pattern: 1st touch → price moves away → comes back (retest) → ENTER
    if not has_ma_retest(recent_candles, indicators, direction, timeframe):
        result["failure_reason"] = "Not a retest - must wait for price to bounce first, move away, then retest MA"
        return result
    result["ma_retest_ok"] = True

    # 4️⃣ MANDATORY: Candlestick confirmation at the MA retest
    # Pattern must appear when price retests the MA (second touch)
    if not has_candlestick_confirmation(recent_candles, direction):
        result["failure_reason"] = "No candlestick confirmation pattern at MA retest"
        return result
    result["candlestick_ok"] = True

    # 5️⃣ Higher timeframe confirmation
    if not higher_timeframe_confirmed(higher_tf_candles, direction):
        result["failure_reason"] = "Higher timeframe disagreement"
        return result
    result["multi_tf_ok"] = True

    # 5.5️⃣ MA Hit Confirmation
    if not ma_hit_confirmed(candle, indicators, higher_tf_candles, direction):
        result["failure_reason"] = "MA hit rule not satisfied"
        return result
    result["ma_hit_ok"] = True

    # 6️⃣ Correlation
    if not correlation_confirmed(correlations, direction):
        result["failure_reason"] = "Correlation not aligned"
        return result
    result["correlation_ok"] = True

    # 6.5️⃣ Support/Resistance Filter - Entry must NOT be AT a S/R level
    # Price bouncing between S/R = clean move (overbought recovery)
    # Price bouncing AT S/R = might get stuck (rejection risk)
    sr_analysis = get_sr_analysis(
        candles=recent_candles,
        entry_price=candle.close,
        direction=direction,
        lookback=50,
        tolerance_pips=5
    )
    
    if not sr_analysis["is_valid"]:
        result["failure_reason"] = sr_analysis["failure_reason"]
        return result

    # 7️⃣ Risk Management - Calculate Stop Loss & Take Profit
    # Stop Loss: 1.5 * ATR, minimum 12 pips, below 61.8% Fib
    stop_loss = calculate_stop_loss(
        entry_price=candle.close,
        direction=direction,
        atr=indicators.atr,
        swing=swing
    )
    
    # Take Profit: Dynamic Risk:Reward (1:2 or 1:1.5 based on SL distance)
    take_profit, rr_ratio = calculate_take_profit(
        entry_price=candle.close,
        stop_loss=stop_loss,
        direction=direction
    )
    
    # Validate risk management setup
    risk_validation = validate_risk_management(
        entry_price=candle.close,
        stop_loss=stop_loss,
        take_profit=take_profit,
        direction=direction,
        swing=swing
    )
    
    if not risk_validation["is_valid"]:
        result["failure_reason"] = risk_validation["failure_reason"]
        return result
    
    result["risk_management_ok"] = True
    result["stop_loss"] = stop_loss
    result["take_profit"] = take_profit
    result["risk_reward_ratio"] = rr_ratio
    result["sl_pips"] = risk_validation["sl_pips"]
    result["tp_pips"] = risk_validation["tp_pips"]

    # 8️⃣ Position Sizing - Calculate number of lots
    # Formula: Lots = (Account Balance × Risk%) / (SL Distance × $Value-per-pip)
    pip_value_per_lot = get_pip_value_per_lot(symbol)
    
    position_size_info = calculate_position_size(
        account_balance=account_balance,
        risk_percentage=risk_percentage,
        stop_loss_pips=risk_validation["sl_pips"],
        pip_value_per_lot=pip_value_per_lot,
        symbol=symbol
    )
    
    result["position_size"] = position_size_info["lots"]
    result["risk_amount"] = position_size_info["risk_amount"]
    result["pip_value_per_lot"] = pip_value_per_lot

    # 9️⃣ ORDER PLACEMENT - Create pending order 2-3 pips above/below confirmation candle
    # Get the confirmation candle (should be the most recent candle)
    confirmation_candle = recent_candles[-1] if recent_candles else candle
    
    # Calculate entry price 2-3 pips above/below confirmation candle
    entry_price = calculate_entry_order_price(
        confirmation_candle_high=confirmation_candle.high,
        confirmation_candle_low=confirmation_candle.low,
        direction=direction,
        buffer_pips=2.5
    )
    
    # Create pending order specification
    pending_order = create_pending_order(
        symbol=symbol,
        direction=direction,
        confirmation_candle_high=confirmation_candle.high,
        confirmation_candle_low=confirmation_candle.low,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size_info["lots"],
        buffer_pips=2.5
    )
    
    # Validate order placement
    order_validation = validate_order_placement(
        pending_order,
        confirmation_candle.high,
        confirmation_candle.low
    )
    
    if not order_validation["is_valid"]:
        result["failure_reason"] = f"Order placement invalid: {', '.join(order_validation['issues'])}"
        return result
    
    # Store order placement details in result
    result["order_type"] = get_order_type(direction)  # "BUY_STOP" or "SELL_STOP"
    result["entry_price"] = entry_price
    result["pending_order"] = pending_order
    result["order_placement_valid"] = True
    result["confirmation_candle_high"] = confirmation_candle.high
    result["confirmation_candle_low"] = confirmation_candle.low

    # ✅ VALID TCE
    result["is_valid"] = True
    return result