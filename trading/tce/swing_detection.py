"""
SWING DETECTION & ANALYSIS

A SWING is the fundamental building block of TCE.
It establishes the directional reference and Fibonacci levels.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


# ============================================================================
# WHAT IS A SWING?
# ============================================================================

SWING_DEFINITION = """
🔄 WHAT IS A SWING?

A SWING is a directional price move with a clear START and END point.

UPSWING (for BUY setups):
├─ START: A swing low (valley/bottom)
├─ MIDDLE: Price rises
├─ END: A swing high (peak/top) - higher than previous highs
├─ Visual: ╱╲╱ (valley to peak)
└─ Confirms: Uptrend is in place

DOWNSWING (for SELL setups):
├─ START: A swing high (peak/top)
├─ MIDDLE: Price falls
├─ END: A swing low (valley/bottom) - lower than previous lows
├─ Visual: ╲╱╲ (peak to valley)
└─ Confirms: Downtrend is in place

MINIMUM SWING SIZE:
├─ Must be at least 30-50 pips (to avoid noise/small moves)
├─ Should span 10-20+ candles (time-based confirmation)
└─ Must have clear structure (not random bouncing)

EXAMPLE UPSWING:
                    PEAK (1.1100) ← Swing High
                         ▲
                        ╱│╲
                       ╱ │ ╲
                      ╱  │  ╲
                    ╱    │   ╲
                   ╱     │    ╲
                  ╱      │     ╲
              ▼──────────┼──────┴─▶ Retracement begins
           VALLEY    Swing Low
        (1.1000)

EXAMPLE DOWNSWING:
            PEAK (1.1100) ← Swing High
                 ▲
                ╱│╲
               ╱ │ ╲
              ╱  │  ╲
             ╱   │   ╲
            ╱    │    ╲
           ╱     │     ╲
          ┴──────┼──────╲──▶ Retracement begins
         Swing  │        ▼
         Low    VALLEY (1.1000)
        (1.0950)

KEY POINTS:
├─ A swing must have BOTH a start and end point
├─ The end point creates the reference for retracement
├─ Retracement is measured from swing high (upswing) or low (downswing)
├─ Once retracement happens, we WAIT FOR RETEST
└─ Retest = price returns to the MA that was bounced from during swing

"""

# ============================================================================
# SWING DETECTION ALGORITHM
# ============================================================================

SWING_DETECTION_ALGORITHM = """
🎯 HOW TO DETECT A SWING

STEP 1: FIND LOCAL HIGHS AND LOWS
├─ Local High: Candle where high > highs of N candles before and after
├─ Local Low: Candle where low < lows of N candles before and after
├─ Lookback period: 5 candles (before) + 5 candles (after) = 10 total
└─ Filters out noise and small fluctuations

STEP 2: IDENTIFY SIGNIFICANT MOVES
├─ For UPSWING:
│  ├─ Find a local low (valley)
│  ├─ Find the next local high that is HIGHER than the low
│  ├─ Difference must be > 30 pips (minimum swing size)
│  └─ This = VALID UPSWING
│
├─ For DOWNSWING:
│  ├─ Find a local high (peak)
│  ├─ Find the next local low that is LOWER than the high
│  ├─ Difference must be > 30 pips (minimum swing size)
│  └─ This = VALID DOWNSWING

STEP 3: CONFIRM SWING IS COMPLETE
├─ After finding local high/low, check:
│  ├─ Has price started pulling back?
│  ├─ Is retracement beginning? (at least 5% of swing size)
│  └─ Is the structure clear?
└─ Only then = SWING IS CONFIRMED

STEP 4: RECORD SWING POINTS
├─ Swing Low: Price, Index, DateTime
├─ Swing High: Price, Index, DateTime
├─ Swing Range: High - Low (in pips)
└─ Status: ACTIVE or COMPLETED

ALGORITHM CODE LOGIC:
─────────────────────

def find_swings(prices, lookback=5, min_swing_pips=30):
    '''Find all significant swings in price data'''
    
    swings = []
    local_highs = []
    local_lows = []
    
    # Find local extremes
    for i in range(lookback, len(prices) - lookback):
        
        # Local high: peak surrounded by lower prices
        if prices[i] == max(prices[i-lookback:i+lookback+1]):
            local_highs.append((i, prices[i]))
        
        # Local low: valley surrounded by higher prices
        if prices[i] == min(prices[i-lookback:i+lookback+1]):
            local_lows.append((i, prices[i]))
    
    # Match highs and lows into swings
    for i in range(len(local_lows) - 1):
        low_idx, low_price = local_lows[i]
        high_idx, high_price = local_highs[i] if i < len(local_highs) else None
        
        if high_idx and high_idx > low_idx:
            swing_range = high_price - low_price
            if swing_range > min_swing_pips * 0.0001:  # Convert pips to price
                swings.append({
                    'type': 'UP',
                    'low_idx': low_idx,
                    'low_price': low_price,
                    'high_idx': high_idx,
                    'high_price': high_price,
                    'range_pips': swing_range / 0.0001
                })
    
    return swings

"""

# ============================================================================
# SWING STATES & TRANSITIONS
# ============================================================================

SWING_STATES = """
📍 SWING STATES & ENTRY TIMING

A swing goes through multiple states. We enter at specific states.

STATE 1: SWING DEVELOPMENT
├─ Status: Upswing forming (price going up from low)
├─ What we see: Higher highs, higher lows
├─ MA alignment: MA6 > MA18 > MA50 > MA200 (uptrend)
├─ Action: WATCH - swing is developing
└─ Do we enter? NO

STATE 2: SWING COMPLETE
├─ Status: Upswing finished, peak reached
├─ What we see: Price made swing high, now starting to pull back
├─ MA alignment: MAs still lined up uptrend
├─ Action: MARK swing high - this is our Fibonacci reference
└─ Do we enter? NO

STATE 3: RETRACEMENT
├─ Status: Price pulling back from swing high
├─ What we see: Price coming down towards an MA level
├─ MA alignment: MAs still sloping up (trend intact)
├─ Depth: Moving towards 38.2%-61.8% range
├─ At this point:
│  ├─ Price bounces off MA (first touch)
│  ├─ This identifies the RETEST LEVEL
│  └─ Action: MARK this MA as retest level
└─ Do we enter? NO - waiting for second touch

STATE 4: RETEST (SECOND TOUCH)
├─ Status: Price came back to the SAME MA again
├─ What we see: Price touching same MA level second time
├─ MA alignment: MAs still correctly aligned
├─ Retracement depth: Confirmed in 38.2%-61.8% range
├─ Candlestick: Confirmation pattern forming
├─ Action: PREPARE ENTRY
└─ Do we enter? YES - if candlestick confirms

VISUAL TIMELINE:
────────────────

Candle 1:  ▲
Candle 2:  ▲▲  ← Swing developing (State 1)
Candle 3:  ▲▲▲
Candle 4:  ▲▲▲▲  ← SWING HIGH (State 2)
Candle 5:    ▼  ← Retracement starts (State 3)
Candle 6:    ▼▼  ← Price pulling back
Candle 7:    ▼▼─  ← Bounces at MA (first touch)
Candle 8:      ▲  ← Rally begins after first bounce
Candle 9:      ▲▲  ← Price rising again
Candle 10:     ▲▲  ← Approaching MA again
Candle 11:     ▲▼▼  ← RETEST at MA (State 4) ← ENTER HERE!
              ↑
          Confirmation

"""

# ============================================================================
# SWING DETECTION FUNCTION
# ============================================================================

def detect_swing(
    prices: np.ndarray,
    lookback_period: int = 5,
    min_swing_pips: float = 30.0
) -> List[Dict]:
    """
    Detect all significant swings in price data.
    
    A swing is:
    - For UPSWING: A valley (low) followed by a peak (high)
    - For DOWNSWING: A peak (high) followed by a valley (low)
    
    Args:
        prices: Array of prices (typically closing prices)
        lookback_period: Bars before/after to compare (default 5)
        min_swing_pips: Minimum swing size in pips (default 30)
    
    Returns:
        List of swings: [
            {
                'type': 'UP' or 'DOWN',
                'low_idx': index of swing low,
                'high_idx': index of swing high,
                'low_price': swing low price,
                'high_price': swing high price,
                'range_pips': range in pips,
                'status': 'DEVELOPING' or 'COMPLETE'
            }
        ]
    """
    
    swings = []
    
    if len(prices) < (lookback_period * 2 + 10):
        return swings
    
    # Find local extremes
    local_highs = []  # (index, price)
    local_lows = []   # (index, price)
    
    for i in range(lookback_period, len(prices) - lookback_period):
        
        # Check if it's a local high
        is_local_high = True
        for j in range(i - lookback_period, i):
            if prices[j] > prices[i]:
                is_local_high = False
                break
        for j in range(i + 1, i + lookback_period + 1):
            if prices[j] >= prices[i]:
                is_local_high = False
                break
        
        if is_local_high:
            local_highs.append((i, prices[i]))
        
        # Check if it's a local low
        is_local_low = True
        for j in range(i - lookback_period, i):
            if prices[j] < prices[i]:
                is_local_low = False
                break
        for j in range(i + 1, i + lookback_period + 1):
            if prices[j] <= prices[i]:
                is_local_low = False
                break
        
        if is_local_low:
            local_lows.append((i, prices[i]))
    
    # Match lows and highs into swings
    for low_idx, low_price in local_lows:
        
        # Find next high after this low
        next_highs = [h for h in local_highs if h[0] > low_idx]
        
        if next_highs:
            high_idx, high_price = next_highs[0]
            
            # Calculate swing range
            swing_range_pips = (high_price - low_price) / 0.0001
            
            if swing_range_pips >= min_swing_pips:
                swings.append({
                    'type': 'UP',
                    'low_idx': low_idx,
                    'high_idx': high_idx,
                    'low_price': low_price,
                    'high_price': high_price,
                    'range_pips': swing_range_pips,
                    'status': 'COMPLETE'
                })
    
    return swings


def identify_current_swing_state(
    recent_prices: np.ndarray,
    recent_highs: np.ndarray,
    recent_lows: np.ndarray,
    ma6: float,
    ma18: float,
    ma50: float,
    ma200: float,
    slope50: float,
    slope200: float,
    direction: str
) -> Dict:
    """
    Identify what state the current swing is in.
    
    Returns:
        {
            'state': 'SWING_DEV' | 'SWING_COMPLETE' | 'RETRACEMENT' | 'RETEST',
            'swing_low': float,
            'swing_high': float,
            'retracement_depth': float (0-1),
            'bounce_ma': str (MA6, MA18, MA50),
            'reason': str
        }
    """
    
    current_price = recent_prices[-1]
    swing_high = np.max(recent_highs)
    swing_low = np.min(recent_lows)
    
    result = {
        'state': None,
        'swing_low': swing_low,
        'swing_high': swing_high,
        'retracement_depth': None,
        'bounce_ma': None,
        'reason': ''
    }
    
    if direction == "BUY":
        
        # Check if price is near swing high (still developing)
        distance_to_high = swing_high - current_price
        
        if distance_to_high < (swing_high - swing_low) * 0.1:  # Within 10% of high
            result['state'] = 'SWING_DEV'
            result['reason'] = 'Price still near swing high, swing developing'
            return result
        
        # Check retracement depth
        total_swing = swing_high - swing_low
        if total_swing > 0:
            retracement_depth = (swing_high - current_price) / total_swing
            result['retracement_depth'] = retracement_depth
            
            # Check if in retracement zone
            if retracement_depth < 0.382:
                result['state'] = 'SWING_COMPLETE'
                result['reason'] = 'Retracement not yet started'
                return result
            
            elif 0.382 <= retracement_depth <= 0.618:
                result['state'] = 'RETRACEMENT'
                
                # Identify which MA price bounced from
                if abs(current_price - ma50) < abs(current_price - ma18) and abs(current_price - ma50) < abs(current_price - ma6):
                    result['bounce_ma'] = 'MA50'
                elif abs(current_price - ma18) < abs(current_price - ma6):
                    result['bounce_ma'] = 'MA18'
                else:
                    result['bounce_ma'] = 'MA6'
                
                result['reason'] = f'In retracement zone, bouncing from {result["bounce_ma"]}'
                return result
            
            else:  # > 0.618
                result['state'] = 'RETEST'
                result['reason'] = 'Price deep in retracement, likely at retest'
                return result
    
    elif direction == "SELL":
        
        # Check if price is near swing low (still developing)
        distance_to_low = current_price - swing_low
        
        if distance_to_low < (swing_high - swing_low) * 0.1:  # Within 10% of low
            result['state'] = 'SWING_DEV'
            result['reason'] = 'Price still near swing low, swing developing'
            return result
        
        # Check retracement depth
        total_swing = swing_high - swing_low
        if total_swing > 0:
            retracement_depth = (current_price - swing_low) / total_swing
            result['retracement_depth'] = retracement_depth
            
            # Check if in retracement zone
            if retracement_depth < 0.382:
                result['state'] = 'SWING_COMPLETE'
                result['reason'] = 'Retracement not yet started'
                return result
            
            elif 0.382 <= retracement_depth <= 0.618:
                result['state'] = 'RETRACEMENT'
                
                # Identify which MA price bounced from
                if abs(current_price - ma50) < abs(current_price - ma18) and abs(current_price - ma50) < abs(current_price - ma6):
                    result['bounce_ma'] = 'MA50'
                elif abs(current_price - ma18) < abs(current_price - ma6):
                    result['bounce_ma'] = 'MA18'
                else:
                    result['bounce_ma'] = 'MA6'
                
                result['reason'] = f'In retracement zone, bouncing from {result["bounce_ma"]}'
                return result
            
            else:  # > 0.618
                result['state'] = 'RETEST'
                result['reason'] = 'Price deep in retracement, likely at retest'
                return result
    
    return result


# Print definitions
if __name__ == "__main__":
    print(SWING_DEFINITION)
    print("\n" + "="*80 + "\n")
    print(SWING_DETECTION_ALGORITHM)
    print("\n" + "="*80 + "\n")
    print(SWING_STATES)
