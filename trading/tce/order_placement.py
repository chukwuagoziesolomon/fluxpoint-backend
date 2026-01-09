"""
Order Placement Module for TCE Strategy

Handles:
- Buy Stop / Sell Stop order creation
- Entry placement 2-3 pips above/below confirmation candle
- Order type selection based on trade direction
"""


def calculate_entry_order_price(
    confirmation_candle_high: float,
    confirmation_candle_low: float,
    direction: str,
    buffer_pips: float = 2.5
) -> float:
    """
    Calculate the entry order price 2-3 pips above/below confirmation candle.
    
    For BUY: Place 2-3 pips ABOVE the confirmation candle high
    For SELL: Place 2-3 pips BELOW the confirmation candle low
    
    This ensures we enter AFTER momentum confirms, not at the candle close.
    
    Args:
        confirmation_candle_high: High of the confirmation candle
        confirmation_candle_low: Low of the confirmation candle
        direction: "BUY" or "SELL"
        buffer_pips: Distance above/below candle (default 2.5 pips)
    
    Returns:
        float: Price at which to place the stop order
    """
    # Convert pips to price units (for forex, 1 pip = 0.0001 for most pairs)
    pip_value = 0.0001
    buffer_price = buffer_pips * pip_value
    
    if direction == "BUY":
        # BUY STOP: Place 2-3 pips ABOVE confirmation candle high
        entry_price = confirmation_candle_high + buffer_price
    elif direction == "SELL":
        # SELL STOP: Place 2-3 pips BELOW confirmation candle low
        entry_price = confirmation_candle_low - buffer_price
    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    return entry_price


def get_order_type(direction: str) -> str:
    """
    Get the appropriate stop order type based on trading direction.
    
    Args:
        direction: "BUY" or "SELL"
    
    Returns:
        str: Order type ("BUY_STOP" or "SELL_STOP")
    """
    if direction == "BUY":
        return "BUY_STOP"
    elif direction == "SELL":
        return "SELL_STOP"
    else:
        raise ValueError(f"Invalid direction: {direction}")


def create_pending_order(
    symbol: str,
    direction: str,
    confirmation_candle_high: float,
    confirmation_candle_low: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    position_size: float,
    buffer_pips: float = 2.5
) -> dict:
    """
    Create a complete pending order specification.
    
    Args:
        symbol: Trading pair (e.g., "EURUSD")
        direction: "BUY" or "SELL"
        confirmation_candle_high: High of confirmation candle
        confirmation_candle_low: Low of confirmation candle
        entry_price: Entry price (calculated or given)
        stop_loss: Stop loss price
        take_profit: Take profit price
        position_size: Position size in lots
        buffer_pips: Buffer above/below candle (default 2.5)
    
    Returns:
        dict: Complete order specification
    """
    # Recalculate entry if not provided
    if entry_price is None:
        entry_price = calculate_entry_order_price(
            confirmation_candle_high,
            confirmation_candle_low,
            direction,
            buffer_pips
        )
    
    order_type = get_order_type(direction)
    
    return {
        "symbol": symbol,
        "order_type": order_type,  # "BUY_STOP" or "SELL_STOP"
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "direction": direction,
        "confirmation_candle_high": confirmation_candle_high,
        "confirmation_candle_low": confirmation_candle_low,
        "buffer_pips": buffer_pips,
        "status": "PENDING",  # Order not yet filled
        "description": f"{order_type} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Size: {position_size:.2f} lots"
    }


def validate_order_placement(order: dict, confirmation_candle_high: float, confirmation_candle_low: float) -> dict:
    """
    Validate that order placement makes sense.
    
    For BUY STOP:
    - Entry price > confirmation candle high (placing above)
    - Stop loss < entry price (SL below entry)
    - Take profit > entry price (TP above entry)
    
    For SELL STOP:
    - Entry price < confirmation candle low (placing below)
    - Stop loss > entry price (SL above entry)
    - Take profit < entry price (TP below entry)
    
    Returns:
        dict: {"is_valid": bool, "issues": [list of problems]}
    """
    issues = []
    direction = order["direction"]
    entry = order["entry_price"]
    sl = order["stop_loss"]
    tp = order["take_profit"]
    
    if direction == "BUY":
        if entry <= confirmation_candle_high:
            issues.append(f"BUY: Entry {entry:.5f} not above candle high {confirmation_candle_high:.5f}")
        if sl >= entry:
            issues.append(f"BUY: SL {sl:.5f} must be below entry {entry:.5f}")
        if tp <= entry:
            issues.append(f"BUY: TP {tp:.5f} must be above entry {entry:.5f}")
        if sl >= tp:
            issues.append(f"BUY: SL {sl:.5f} must be below TP {tp:.5f}")
    
    elif direction == "SELL":
        if entry >= confirmation_candle_low:
            issues.append(f"SELL: Entry {entry:.5f} not below candle low {confirmation_candle_low:.5f}")
        if sl <= entry:
            issues.append(f"SELL: SL {sl:.5f} must be above entry {entry:.5f}")
        if tp >= entry:
            issues.append(f"SELL: TP {tp:.5f} must be below entry {entry:.5f}")
        if sl <= tp:
            issues.append(f"SELL: SL {sl:.5f} must be above TP {tp:.5f}")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }
