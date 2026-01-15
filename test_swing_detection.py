#!/usr/bin/env python
"""
Test proper swing detection with clear uptrend/downtrend examples.
This demonstrates how to identify swing highs/lows correctly.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*80)
print("SWING DETECTION TEST - IDENTIFYING PEAKS AND VALLEYS")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS (same as in CELL4)
# ============================================================================

def find_swing_points(prices, lookback=5):
    """
    Find local highs and lows (swing points) with proper lookback validation.
    A swing high must be higher than N candles before AND after it.
    A swing low must be lower than N candles before AND after it.
    """
    local_highs = []  # (index, price)
    local_lows = []   # (index, price)
    
    for i in range(lookback, len(prices) - lookback):
        # Check if it's a local high (peak)
        is_high = True
        for j in range(i - lookback, i):
            if prices[j] > prices[i]:
                is_high = False
                break
        if is_high:
            for j in range(i + 1, i + lookback + 1):
                if prices[j] >= prices[i]:
                    is_high = False
                    break
        if is_high:
            local_highs.append((i, prices[i]))
        
        # Check if it's a local low (valley)
        is_low = True
        for j in range(i - lookback, i):
            if prices[j] < prices[i]:
                is_low = False
                break
        if is_low:
            for j in range(i + 1, i + lookback + 1):
                if prices[j] <= prices[i]:
                    is_low = False
                    break
        if is_low:
            local_lows.append((i, prices[i]))
    
    return local_highs, local_lows


# ============================================================================
# TEST 1: CLEAR UPTREND with HIGHER LOWS
# ============================================================================

print("\n" + "="*80)
print("TEST 1: UPTREND - Price making HIGHER LOWS (strengthening uptrend)")
print("="*80)

# Create synthetic uptrend data with clear swing points
# Pattern: Valley -> Peak -> Higher Valley -> Higher Peak
uptrend_prices = np.array([
    1.0800, 1.0810, 1.0820, 1.0830, 1.0840,  # Rising
    1.0850, 1.0860, 1.0870, 1.0880, 1.0890,  # Keep rising
    1.0900, 1.0910, 1.0920, 1.0930, 1.0940,  # Peak forming
    1.0950, 1.0960, 1.0970, 1.0980, 1.0990,  # FIRST PEAK (swing high)
    1.0980, 1.0970, 1.0960, 1.0950, 1.0940,  # Pullback starts
    1.0930, 1.0920, 1.0910, 1.0900, 1.0890,  # FIRST VALLEY (swing low) 
    1.0895, 1.0900, 1.0910, 1.0920, 1.0930,  # Bouncing from MA
    1.0940, 1.0950, 1.0960, 1.0970, 1.0980,  # Rising again
    1.0990, 1.1000, 1.1010, 1.1020, 1.1030,  # SECOND PEAK (higher)
    1.1020, 1.1010, 1.1000, 0.9990, 1.0980,  # Second pullback
    1.0970, 1.0960, 1.0950, 1.0940, 1.0930,  # Going down
    1.0925, 1.0920, 1.0915, 1.0920, 1.0925,  # SECOND VALLEY (swing low)
])

print(f"\nPrice data: {len(uptrend_prices)} candles")
print(f"Start: {uptrend_prices[0]:.4f}")
print(f"End: {uptrend_prices[-1]:.4f}")
print(f"Direction: {'UP' if uptrend_prices[-1] > uptrend_prices[0] else 'DOWN'}")

# Find swing points
highs, lows = find_swing_points(uptrend_prices, lookback=5)

print(f"\n✅ SWING HIGHS (Peaks): {len(highs)} found")
for idx, (i, price) in enumerate(highs):
    print(f"   {idx+1}. Index {i:3d}: {price:.4f}")

print(f"\n✅ SWING LOWS (Valleys): {len(lows)} found")
for idx, (i, price) in enumerate(lows):
    print(f"   {idx+1}. Index {i:3d}: {price:.4f}")

# Check for higher lows (uptrend confirmation)
if len(lows) >= 2:
    print(f"\n📊 HIGHER LOWS CHECK (Uptrend Confirmation):")
    for i in range(1, len(lows)):
        prev_low = lows[i-1][1]
        curr_low = lows[i][1]
        is_higher = curr_low > prev_low
        print(f"   Low {i}: {prev_low:.4f} -> {curr_low:.4f} = {'✅ HIGHER' if is_higher else '❌ LOWER'}")


# ============================================================================
# TEST 2: CLEAR DOWNTREND with LOWER HIGHS
# ============================================================================

print("\n" + "="*80)
print("TEST 2: DOWNTREND - Price making LOWER HIGHS (strengthening downtrend)")
print("="*80)

# Create synthetic downtrend data with clear swing points
# Pattern: Peak -> Valley -> Lower Peak -> Lower Valley
downtrend_prices = np.array([
    1.1000, 1.0990, 1.0980, 1.0970, 1.0960,  # Falling
    1.0950, 1.0940, 1.0930, 1.0920, 1.0910,  # Keep falling
    1.0900, 1.0890, 1.0880, 1.0870, 1.0860,  # Valley forming
    1.0850, 1.0840, 1.0830, 1.0820, 1.0810,  # FIRST VALLEY (swing low)
    1.0820, 1.0830, 1.0840, 1.0850, 1.0860,  # Bounce/rally
    1.0870, 1.0880, 1.0890, 1.0900, 1.0910,  # FIRST PEAK (swing high)
    1.0905, 1.0900, 1.0890, 1.0880, 1.0870,  # Falling again
    1.0860, 1.0850, 1.0840, 1.0830, 1.0820,  # Going down
    1.0810, 1.0800, 1.0790, 1.0780, 1.0770,  # SECOND VALLEY (lower)
    1.0780, 1.0790, 1.0800, 1.0810, 1.0820,  # Second rally
    1.0830, 1.0840, 1.0850, 1.0860, 1.0870,  # Rising
    1.0875, 1.0880, 1.0885, 1.0880, 1.0875,  # SECOND PEAK (lower than first)
])

print(f"\nPrice data: {len(downtrend_prices)} candles")
print(f"Start: {downtrend_prices[0]:.4f}")
print(f"End: {downtrend_prices[-1]:.4f}")
print(f"Direction: {'UP' if downtrend_prices[-1] > downtrend_prices[0] else 'DOWN'}")

# Find swing points
highs, lows = find_swing_points(downtrend_prices, lookback=5)

print(f"\n✅ SWING HIGHS (Peaks): {len(highs)} found")
for idx, (i, price) in enumerate(highs):
    print(f"   {idx+1}. Index {i:3d}: {price:.4f}")

print(f"\n✅ SWING LOWS (Valleys): {len(lows)} found")
for idx, (i, price) in enumerate(lows):
    print(f"   {idx+1}. Index {i:3d}: {price:.4f}")

# Check for lower highs (downtrend confirmation)
if len(highs) >= 2:
    print(f"\n📊 LOWER HIGHS CHECK (Downtrend Confirmation):")
    for i in range(1, len(highs)):
        prev_high = highs[i-1][1]
        curr_high = highs[i][1]
        is_lower = curr_high < prev_high
        print(f"   High {i}: {prev_high:.4f} -> {curr_high:.4f} = {'✅ LOWER' if is_lower else '❌ HIGHER'}")


# ============================================================================
# TEST 3: NO CLEAR SWINGS (choppy market)
# ============================================================================

print("\n" + "="*80)
print("TEST 3: CHOPPY MARKET - No clear swing structure")
print("="*80)

# Create noisy/choppy data
np.random.seed(42)
choppy_prices = 1.0900 + np.random.randn(50) * 0.0010

print(f"\nPrice data: {len(choppy_prices)} candles")
print(f"Start: {choppy_prices[0]:.4f}")
print(f"End: {choppy_prices[-1]:.4f}")

# Find swing points
highs, lows = find_swing_points(choppy_prices, lookback=5)

print(f"\n✅ SWING HIGHS (Peaks): {len(highs)} found")
if len(highs) <= 3:
    for idx, (i, price) in enumerate(highs):
        print(f"   {idx+1}. Index {i:3d}: {price:.4f}")
else:
    print(f"   (Too many small swings - market is choppy)")

print(f"\n✅ SWING LOWS (Valleys): {len(lows)} found")
if len(lows) <= 3:
    for idx, (i, price) in enumerate(lows):
        print(f"   {idx+1}. Index {i:3d}: {price:.4f}")
else:
    print(f"   (Too many small swings - market is choppy)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
✅ PROPER SWING DETECTION REQUIRES:

1. LOOKBACK VALIDATION
   - A swing high must be higher than N candles before AND after
   - A swing low must be lower than N candles before AND after
   - Default lookback: 5 candles

2. TREND CONFIRMATION
   - UPTREND: Current swing low > Previous swing low (HIGHER LOWS)
   - DOWNTREND: Current swing high < Previous swing high (LOWER HIGHS)

3. MOVING AVERAGE RETEST
   - First swing creates the reference (swing high or swing low)
   - Price retraces to an MA level (38.2%, 50%, or 61.8% Fibonacci)
   - Second touch/test of the MA = entry opportunity

4. CANDLESTICK CONFIRMATION
   - Rejection candle, pin bar, or engulfing at the MA retest
   - Confirms price is bouncing from the level

❌ WHAT NOT TO DO:
   - Don't use simple min/max of last N candles
   - Don't ignore swing structure (higher lows / lower highs)
   - Don't skip lookback validation
   - Don't enter on first touch of MA (wait for retest)
""")
