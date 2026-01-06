from typing import Tuple, Dict


def calculate_position_size(
    account_balance: float,
    risk_percentage: float,
    stop_loss_pips: float,
    pip_value_per_lot: float = 10.0,
    symbol: str = "EURUSD"
) -> Dict:
    """
    Calculate position size (number of lots) using risk management formula:
    
    No. of Lots = (Net Liquidation × %Risk Per Trade) / (Stop Loss Distance × $Value-per-pip)
    
    Args:
        account_balance: Net liquidation value (user's account balance)
        risk_percentage: % of account to risk per trade (e.g., 1.0 for 1%, 2.0 for 2%)
        stop_loss_pips: Stop loss distance in pips
        pip_value_per_lot: Dollar value per pip for 1 standard lot (default $10 for USD pairs)
        symbol: Trading pair symbol (used to determine pip value)
    
    Returns:
        Dict with position sizing details
    """
    # Amount willing to risk
    risk_amount = account_balance * (risk_percentage / 100.0)
    
    # Calculate pip value per pip for the SL distance
    total_pip_value = stop_loss_pips * pip_value_per_lot
    
    # Number of standard lots
    if total_pip_value > 0:
        num_lots = risk_amount / total_pip_value
    else:
        num_lots = 0
    
    # Round to 2 decimal places (0.01 lot = micro lot)
    num_lots = round(num_lots, 2)
    
    return {
        "lots": num_lots,
        "risk_amount": risk_amount,
        "risk_percentage": risk_percentage,
        "pip_value_per_lot": pip_value_per_lot,
        "stop_loss_pips": stop_loss_pips
    }


def get_pip_value_per_lot(symbol: str, account_currency: str = "USD") -> float:
    """
    Get the dollar value per pip for 1 standard lot based on the trading pair.
    
    Standard lot = 100,000 units
    1 pip = 0.0001 for most pairs (0.01 for JPY pairs)
    
    Args:
        symbol: Trading pair (e.g., 'EURUSD', 'GBPUSD', 'USDJPY')
        account_currency: Account currency (default 'USD')
    
    Returns:
        Dollar value per pip for 1 standard lot
    """
    # For pairs where USD is quote currency (XXXUSD)
    if symbol.endswith("USD"):
        return 10.0  # 1 standard lot = $10 per pip
    
    # For pairs where USD is base currency (USDXXX)
    elif symbol.startswith("USD"):
        if "JPY" in symbol:
            return 9.09  # Approximate for USDJPY (varies with rate)
        else:
            return 10.0  # Approximate, would need exchange rate
    
    # For cross pairs (no USD), need exchange rate
    else:
        # Default approximation
        return 10.0
    
    # Note: For exact calculations, you'd query current exchange rates


from .types import Candle, Indicators, Swing


def calculate_stop_loss(
    entry_price: float,
    direction: str,
    atr: float,
    swing: Swing,
    pip_value: float = 0.0001
) -> float:
    """
    Calculate stop loss based on TCE rules:
    1. Stop loss = 1.5 * ATR from entry
    2. Must be at least 12 pips minimum
    3. Must be below 61.8% Fibonacci level (BUY) or above (SELL)
    
    Args:
        entry_price: Entry price for the trade
        direction: 'BUY' or 'SELL'
        atr: Average True Range value
        swing: Swing object with fib_618_price
        pip_value: Value of 1 pip (default 0.0001 for forex)
    
    Returns:
        Stop loss price
    """
    # Calculate 1.5 * ATR stop loss
    atr_stop_distance = 1.5 * atr
    
    # Minimum 12 pips
    min_pips = 12
    min_stop_distance = min_pips * pip_value
    
    # Use the larger of ATR-based or minimum pip distance
    stop_distance = max(atr_stop_distance, min_stop_distance)
    
    if direction == "BUY":
        # Stop loss below entry
        sl_price = entry_price - stop_distance
        
        # Must be below 61.8% Fib level
        if swing.fib_618_price and sl_price > swing.fib_618_price:
            sl_price = swing.fib_618_price - (2 * pip_value)  # Place slightly below
    
    else:  # SELL
        # Stop loss above entry
        sl_price = entry_price + stop_distance
        
        # Must be above 61.8% Fib level
        if swing.fib_618_price and sl_price < swing.fib_618_price:
            sl_price = swing.fib_618_price + (2 * pip_value)  # Place slightly above
    
    return sl_price


def determine_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    pip_value: float = 0.0001
) -> float:
    """
    Determine optimal Risk:Reward ratio based on stop loss distance.
    
    Logic:
    - If SL distance <= 20 pips: Use 1:2 (conservative, good reward)
    - If SL distance > 20 pips and <= 40 pips: Use 1:1.5 (balanced)
    - If SL distance > 40 pips: Use 1:1.5 (avoid unrealistic targets)
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        pip_value: Value of 1 pip
    
    Returns:
        Risk:Reward ratio (e.g., 2.0 for 1:2, 1.5 for 1:1.5)
    """
    sl_distance = abs(entry_price - stop_loss)
    sl_pips = sl_distance / pip_value
    
    if sl_pips <= 20:
        return 2.0  # 1:2 ratio
    elif sl_pips <= 40:
        return 1.5  # 1:1.5 ratio
    else:
        return 1.5  # 1:1.5 ratio (avoid far targets)


def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    direction: str,
    risk_reward_ratio: float = None,
    pip_value: float = 0.0001
) -> Tuple[float, float]:
    """
    Calculate take profit based on dynamic risk:reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        direction: 'BUY' or 'SELL'
        risk_reward_ratio: Custom ratio, if None will be auto-calculated
        pip_value: Value of 1 pip
    
    Returns:
        Tuple of (take_profit_price, risk_reward_ratio_used)
    """
    # Calculate stop loss distance (risk)
    sl_distance = abs(entry_price - stop_loss)
    
    # Determine risk:reward ratio if not provided
    if risk_reward_ratio is None:
        risk_reward_ratio = determine_risk_reward_ratio(entry_price, stop_loss, pip_value)
    
    # Calculate take profit distance (reward)
    tp_distance = sl_distance * risk_reward_ratio
    
    if direction == "BUY":
        tp_price = entry_price + tp_distance
    else:  # SELL
        tp_price = entry_price - tp_distance
    
    return tp_price, risk_reward_ratio


def validate_risk_management(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    swing: Swing,
    pip_value: float = 0.0001
) -> Dict:
    """
    Validate that stop loss and take profit meet all TCE requirements.
    
    Returns:
        Dict with validation results
    """
    result = {
        "is_valid": True,
        "sl_valid": True,
        "tp_valid": True,
        "sl_pips": 0,
        "tp_pips": 0,
        "risk_reward": 0,
        "failure_reason": None
    }
    
    sl_distance = abs(entry_price - stop_loss)
    tp_distance = abs(entry_price - take_profit)
    
    result["sl_pips"] = sl_distance / pip_value
    result["tp_pips"] = tp_distance / pip_value
    result["risk_reward"] = tp_distance / sl_distance if sl_distance > 0 else 0
    
    # Check minimum 12 pips
    if result["sl_pips"] < 12:
        result["is_valid"] = False
        result["sl_valid"] = False
        result["failure_reason"] = "Stop loss less than 12 pips minimum"
        return result
    
    # Check SL position relative to 61.8% Fib
    if swing.fib_618_price:
        if direction == "BUY":
            if stop_loss >= swing.fib_618_price:
                result["is_valid"] = False
                result["sl_valid"] = False
                result["failure_reason"] = "Stop loss not below 61.8% Fib level"
                return result
        else:  # SELL
            if stop_loss <= swing.fib_618_price:
                result["is_valid"] = False
                result["sl_valid"] = False
                result["failure_reason"] = "Stop loss not above 61.8% Fib level"
                return result
    
    # Check minimum risk:reward of 1:1
    if result["risk_reward"] < 1.0:
        result["is_valid"] = False
        result["tp_valid"] = False
        result["failure_reason"] = "Risk:Reward ratio less than 1:1"
        return result
    
    return result
