# ============================================================================
# DEBUG SCRIPT - Test if TCE Validation Rules are Working
# ============================================================================
# Run this locally to verify all validation rules work before Colab training
# This helps identify which rule is causing issues

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce
from trading.tce.utils import is_uptrend, is_downtrend, valid_fib
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend

print("\n" + "="*80)
print("TCE VALIDATION RULES DEBUG TEST")
print("="*80)

# ============================================================================
# TEST 1: Moving Average Trend Detection
# ============================================================================
print("\n1️⃣ TEST: Moving Average Trend Detection")
print("-" * 80)

test_indicators_uptrend = Indicators(
    ma6=1.1050, ma18=1.1040, ma50=1.1020, ma200=1.0980,
    slope6=0.001, slope18=0.0005, slope50=0.0002, slope200=0.00005,
    atr=0.0010
)

test_indicators_downtrend = Indicators(
    ma6=1.0980, ma18=1.0990, ma50=1.1010, ma200=1.1050,
    slope6=-0.001, slope18=-0.0005, slope50=-0.0002, slope200=-0.00005,
    atr=0.0010
)

is_up = is_uptrend(test_indicators_uptrend)
is_down = is_downtrend(test_indicators_downtrend)

print(f"  Uptrend Check: {is_up}")
print(f"    • MA6 (1.1050) > MA18 (1.1040) > MA50 (1.1020) > MA200 (1.0980): ✅")
print(f"    • slope6 (0.001) > 0: ✅")
print(f"    • slope18 (0.0005) > 0: ✅")
print(f"    • slope50 (0.0002) > 0: ✅")
print(f"  Result: {'✅ PASS' if is_up else '❌ FAIL'}")

print(f"\n  Downtrend Check: {is_down}")
print(f"    • MA200 (1.1050) > MA50 (1.1010) > MA18 (1.0990) > MA6 (1.0980): ✅")
print(f"    • slope6 (-0.001) < 0: ✅")
print(f"    • slope18 (-0.0005) < 0: ✅")
print(f"    • slope50 (-0.0002) < 0: ✅")
print(f"  Result: {'✅ PASS' if is_down else '❌ FAIL'}")

# ============================================================================
# TEST 2: Fibonacci Validation
# ============================================================================
print("\n2️⃣ TEST: Fibonacci Validation")
print("-" * 80)

test_swings = [
    Swing(type='low', price=1.1000, fib_level=0.382),
    Swing(type='low', price=1.1000, fib_level=0.500),
    Swing(type='low', price=1.1000, fib_level=0.618),
    Swing(type='low', price=1.1000, fib_level=0.786),  # INVALID - too deep
]

for swing in test_swings:
    result = valid_fib(swing)
    print(f"  Fibonacci Level {swing.fib_level:.3f}: {'✅ VALID' if result else '❌ INVALID'}")

# ============================================================================
# TEST 3: Structure Analysis (Higher Highs/Lows)
# ============================================================================
print("\n3️⃣ TEST: Structure Analysis (Trend Confirmation)")
print("-" * 80)

# Uptrend structure: higher highs and higher lows
uptrend_highs = [1.1000, 1.1005, 1.1010, 1.1015, 1.1020]
uptrend_lows = [1.0990, 1.0995, 1.1000, 1.1005, 1.1010]

uptrend_structure = MarketStructure(highs=uptrend_highs, lows=uptrend_lows)

is_valid_up = is_valid_uptrend(test_indicators_uptrend, uptrend_structure)
print(f"  Uptrend Structure: {'✅ PASS' if is_valid_up else '❌ FAIL'}")
print(f"    • Highs: {uptrend_highs} (higher highs)")
print(f"    • Lows: {uptrend_lows} (higher lows)")

# Downtrend structure: lower highs and lower lows
downtrend_highs = [1.1050, 1.1045, 1.1040, 1.1035, 1.1030]
downtrend_lows = [1.1010, 1.1005, 1.1000, 1.0995, 1.0990]

downtrend_structure = MarketStructure(highs=downtrend_highs, lows=downtrend_lows)

is_valid_down = is_valid_downtrend(test_indicators_downtrend, downtrend_structure)
print(f"\n  Downtrend Structure: {'✅ PASS' if is_valid_down else '❌ FAIL'}")
print(f"    • Highs: {downtrend_highs} (lower highs)")
print(f"    • Lows: {downtrend_lows} (lower lows)")

# ============================================================================
# TEST 4: Complete TCE Validation
# ============================================================================
print("\n4️⃣ TEST: Complete TCE Validation (All 7 Rules)")
print("-" * 80)

# Create a realistic test case
test_candle = Candle(
    open=1.1010,
    high=1.1020,
    low=1.1005,
    close=1.1015,
    timestamp="2026-01-08 10:00:00"
)

test_swing = Swing(
    type='low',
    price=1.1005,
    fib_level=0.382
)

# Create recent candles with pattern (bullish pin bar)
recent_candles = [
    Candle(open=1.1025, high=1.1030, low=1.1010, close=1.1025, timestamp="2026-01-08 09:00:00"),  # Pullback
    Candle(open=1.1025, high=1.1027, low=1.1005, close=1.1015, timestamp="2026-01-08 10:00:00"),  # Bullish pin bar at MA
]

# Extend with more historical candles
for i in range(48):
    recent_candles.insert(0, Candle(
        open=1.1020 + np.random.randn() * 0.0001,
        high=1.1025 + np.random.randn() * 0.0001,
        low=1.1015 + np.random.randn() * 0.0001,
        close=1.1020 + np.random.randn() * 0.0001,
        timestamp="2026-01-08 08:00:00"
    ))

test_result = validate_tce(
    candle=test_candle,
    indicators=test_indicators_uptrend,
    swing=test_swing,
    sr_levels=[],
    higher_tf_candles=[],
    correlations={},
    structure=uptrend_structure,
    recent_candles=recent_candles,
    timeframe="H1",
    account_balance=10000.0,
    risk_percentage=1.0,
    symbol="EURUSD"
)

print(f"  Overall Result: {'✅ VALID SETUP' if test_result['is_valid'] else '❌ INVALID SETUP'}")
print(f"\n  Individual Rules:")
print(f"    1️⃣  Trend: {'✅' if test_result['trend_ok'] else '❌'} - {test_result.get('direction', 'N/A')}")
print(f"    2️⃣  Fibonacci: {'✅' if test_result['fib_ok'] else '❌'}")
print(f"    2.5️⃣ Swing: {'✅' if test_result['swing_ok'] else '❌'}")
print(f"    3️⃣  MA Level: {'✅' if test_result['ma_level_ok'] else '❌'}")
print(f"    3.5️⃣ MA Retest: {'✅' if test_result['ma_retest_ok'] else '❌'}")
print(f"    4️⃣  Candlestick: {'✅' if test_result['candlestick_ok'] else '❌'}")
print(f"    5️⃣  Multi-TF: {'✅' if test_result['multi_tf_ok'] else '❌'}")
print(f"    6️⃣  Correlation: {'✅' if test_result['correlation_ok'] else '❌'}")
print(f"    7️⃣  Risk Mgmt: {'✅' if test_result['risk_management_ok'] else '❌'}")

if test_result['is_valid']:
    print(f"\n  Risk Management Details:")
    print(f"    • Stop Loss: {test_result['stop_loss']:.5f} ({test_result['sl_pips']:.1f} pips)")
    print(f"    • Take Profit: {test_result['take_profit']:.5f} ({test_result['tp_pips']:.1f} pips)")
    print(f"    • Risk/Reward Ratio: {test_result['risk_reward_ratio']:.2f}:1")
    print(f"    • Position Size: {test_result['position_size']:.2f} lots")
    print(f"    • Risk Amount: ${test_result['risk_amount']:.2f}")
else:
    if test_result['failure_reason']:
        print(f"\n  ❌ Failure Reason: {test_result['failure_reason']}")

print("\n" + "="*80)
print("✅ DEBUG TEST COMPLETE")
print("="*80 + "\n")
