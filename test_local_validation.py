"""
LOCAL TEST: Validate TCE rules with synthetic data
This tests the validation system locally WITHOUT needing actual MT5 data
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce

print("\n" + "="*100)
print("LOCAL TEST: TCE VALIDATION RULES")
print("="*100)

def calculate_slope(values, period=20):
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

# ============================================================================
# TEST 1: Create a synthetic UPTREND setup
# ============================================================================

print("\n🔍 TEST 1: SYNTHETIC UPTREND BUY SETUP")
print("-" * 100)

# Create 300 candles with uptrend pattern
np.random.seed(42)
start_price = 100.0
candles_data = []
base_trend = 0.1  # uptrend

for i in range(300):
    # Price with uptrend + small noise
    close = start_price + (i * base_trend) + np.random.randn() * 0.5
    high = close + np.random.rand() * 0.5
    low = close - np.random.rand() * 0.5
    open_price = close - 0.2
    
    candles_data.append({
        'Date': datetime.now() - timedelta(hours=300-i),
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close
    })

df = pd.DataFrame(candles_data)
close_arr = df['Close'].values
high_arr = df['High'].values
low_arr = df['Low'].values
open_arr = df['Open'].values

# Calculate indicators for candle 250 (first valid candle)
row_idx = 250
close_window = close_arr[row_idx-240:row_idx+1]
high_window = high_arr[row_idx-240:row_idx+1]
low_window = low_arr[row_idx-240:row_idx+1]

ma6 = np.mean(close_window[-6:])
ma18 = np.mean(close_window[-18:])
ma50 = np.mean(close_window[-50:])
ma200 = np.mean(close_window[-200:])

slope6 = calculate_slope(close_window, 6)
slope18 = calculate_slope(close_window, 18)
slope50 = calculate_slope(close_window, 50)
slope200 = calculate_slope(close_window, 200)

# ATR
tr_list = []
for i in range(1, min(15, len(high_window))):
    tr = max(
        high_window[i] - low_window[i],
        abs(high_window[i] - close_window[i-1]),
        abs(low_window[i] - close_window[i-1])
    )
    tr_list.append(tr)
atr = np.mean(tr_list) if tr_list else 0.1

print(f"\n📊 CANDLE DATA (Index {row_idx}):")
print(f"  Close: {close_arr[row_idx]:.4f}")
print(f"  High:  {high_arr[row_idx]:.4f}")
print(f"  Low:   {low_arr[row_idx]:.4f}")

print(f"\n📈 MOVING AVERAGES:")
print(f"  MA6:   {ma6:.4f}")
print(f"  MA18:  {ma18:.4f}")
print(f"  MA50:  {ma50:.4f}")
print(f"  MA200: {ma200:.4f}")
print(f"  ATR:   {atr:.4f}")

# Check MA alignment
print(f"\n📐 MA ALIGNMENT CHECK:")
print(f"  MA6 ({ma6:.4f}) > MA18 ({ma18:.4f})? {ma6 > ma18} {'✅' if ma6 > ma18 else '❌'}")
print(f"  MA18 ({ma18:.4f}) > MA50 ({ma50:.4f})? {ma18 > ma50} {'✅' if ma18 > ma50 else '❌'}")
print(f"  MA50 ({ma50:.4f}) > MA200 ({ma200:.4f})? {ma50 > ma200} {'✅' if ma50 > ma200 else '❌'}")

print(f"\n📊 SLOPES:")
print(f"  Slope6:  {slope6:+.6f} {'↗️ UP' if slope6 > 0 else '↘️ DOWN'}")
print(f"  Slope18: {slope18:+.6f} {'↗️ UP' if slope18 > 0 else '↘️ DOWN'}")
print(f"  Slope50: {slope50:+.6f} {'↗️ UP' if slope50 > 0 else '↘️ DOWN'}")
slopes_up = sum([slope6 > 0, slope18 > 0, slope50 > 0])
print(f"  → {slopes_up}/3 slopes positive {'✅' if slopes_up >= 2 else '❌'}")

# Create validation objects
indicators = Indicators(
    ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
    slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200,
    atr=atr
)

# Current candle
candle = Candle(
    open=float(open_arr[row_idx]),
    high=float(high_arr[row_idx]),
    low=float(low_arr[row_idx]),
    close=float(close_arr[row_idx]),
    timestamp=df['Date'].iloc[row_idx].isoformat()
)

# Fibonacci (price bounced off MA18)
ma_ref = ma18
fib_level = 0.618
if candle.low < ma_ref:
    depth = (ma_ref - candle.low) / atr if atr > 0 else 0
    if depth < 0.5:
        fib_level = 0.382
    elif depth < 1.0:
        fib_level = 0.5
print(f"\n🎯 FIBONACCI LEVEL: {int(fib_level*100)}%")

swing = Swing(
    type='low',
    price=float(candle.low),
    fib_level=fib_level
)

# Recent candles for retest check
recent_close = close_arr[max(0, row_idx-50):row_idx+1]
recent_high = high_arr[max(0, row_idx-50):row_idx+1]
recent_low = low_arr[max(0, row_idx-50):row_idx+1]
recent_open = open_arr[max(0, row_idx-50):row_idx+1]

recent_candles = [
    Candle(
        open=float(recent_open[i]),
        high=float(recent_high[i]),
        low=float(recent_low[i]),
        close=float(recent_close[i]),
        timestamp=(datetime.now() - timedelta(hours=50-i)).isoformat()
    )
    for i in range(len(recent_close))
]

structure = MarketStructure(
    highs=list(recent_high[-50:]),
    lows=list(recent_low[-50:])
)

# ✅ RUN VALIDATION
print(f"\n" + "="*100)
print("RUNNING VALIDATION...")
print("="*100)

result = validate_tce(
    candle=candle,
    indicators=indicators,
    swing=swing,
    sr_levels=[],
    higher_tf_candles=[],
    correlations={},
    structure=structure,
    recent_candles=recent_candles,
    timeframe="H1",
    account_balance=10000.0,
    risk_percentage=1.0,
    symbol="SYNTHETIC"
)

print(f"\n📋 VALIDATION RESULTS:")
print(f"  ✅ Trend:             {result['trend_ok']}")
print(f"  ✅ Fibonacci:         {result['fib_ok']}")
print(f"  ✅ Swing:            {result['swing_ok']}")
print(f"  ✅ MA Level:         {result['ma_level_ok']}")
print(f"  ✅ MA Retest:        {result['ma_retest_ok']}")
print(f"  ✅ Candlestick:      {result['candlestick_ok']}")
print(f"  ✅ Multi-TF:         {result['multi_tf_ok']}")
print(f"  ✅ Correlation:      {result['correlation_ok']}")
print(f"  ✅ Risk Management:  {result['risk_management_ok']}")

print(f"\n{'='*100}")
if result["is_valid"]:
    print("🎉 SETUP VALID!")
    print(f"   Direction:    {result['direction']}")
    print(f"   Entry:        {result['entry_price']:.4f}")
    print(f"   Stop Loss:    {result['stop_loss']:.4f}")
    print(f"   Take Profit:  {result['take_profit']:.4f}")
    print(f"   Risk/Reward:  {result['risk_reward_ratio']:.2f}:1")
    print(f"   Position Size: {result['position_size']:.2f} lots")
    print(f"   Risk Amount:  ${result['risk_amount']:.2f}")
else:
    print("❌ SETUP INVALID")
    print(f"   Reason: {result['failure_reason']}")

print("="*100)

# ============================================================================
# TEST 2: Run validation on multiple synthetic setups
# ============================================================================

print("\n\n🔍 TEST 2: SCAN MULTIPLE CANDLES FOR VALID SETUPS")
print("-" * 100)

valid_count = 0
trend_pass = 0
fib_pass = 0
ma_level_pass = 0
ma_retest_pass = 0
candlestick_pass = 0

# Scan every 10 candles
total_checked = len(range(250, len(df) - 50, 10)) if len(df) > 300 else 1

for scan_idx in range(250, min(len(df) - 50, 270), 10) if len(df) > 300 else range(250, 260, 10):
    close_window = close_arr[scan_idx-240:scan_idx+1]
    high_window = high_arr[scan_idx-240:scan_idx+1]
    low_window = low_arr[scan_idx-240:scan_idx+1]
    
    ma6 = np.mean(close_window[-6:])
    ma18 = np.mean(close_window[-18:])
    ma50 = np.mean(close_window[-50:])
    ma200 = np.mean(close_window[-200:])
    
    slope6 = calculate_slope(close_window, 6)
    slope18 = calculate_slope(close_window, 18)
    slope50 = calculate_slope(close_window, 50)
    slope200 = calculate_slope(close_window, 200)
    
    tr_list = []
    for i in range(1, min(15, len(high_window))):
        tr = max(
            high_window[i] - low_window[i],
            abs(high_window[i] - close_window[i-1]),
            abs(low_window[i] - close_window[i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0.1
    
    indicators = Indicators(
        ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
        slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200,
        atr=atr
    )
    
    candle = Candle(
        open=float(open_arr[scan_idx]),
        high=float(high_arr[scan_idx]),
        low=float(low_arr[scan_idx]),
        close=float(close_arr[scan_idx]),
        timestamp=df['Date'].iloc[scan_idx].isoformat()
    )
    
    ma_ref = ma18
    fib_level = 0.618
    if candle.low < ma_ref and atr > 0:
        depth = (ma_ref - candle.low) / atr
        if depth < 0.5:
            fib_level = 0.382
        elif depth < 1.0:
            fib_level = 0.5
    
    swing = Swing(
        type='low' if close_arr[scan_idx] > np.mean(close_arr[max(0, scan_idx-50):scan_idx]) else 'high',
        price=float(candle.low if close_arr[scan_idx] > np.mean(close_arr[max(0, scan_idx-50):scan_idx]) else candle.high),
        fib_level=fib_level
    )
    
    recent_close = close_arr[max(0, scan_idx-50):scan_idx+1]
    recent_high = high_arr[max(0, scan_idx-50):scan_idx+1]
    recent_low = low_arr[max(0, scan_idx-50):scan_idx+1]
    recent_open = open_arr[max(0, scan_idx-50):scan_idx+1]
    
    recent_candles = [
        Candle(
            open=float(recent_open[i]),
            high=float(recent_high[i]),
            low=float(recent_low[i]),
            close=float(recent_close[i]),
            timestamp=(datetime.now() - timedelta(hours=50-i)).isoformat()
        )
        for i in range(len(recent_close))
    ]
    
    structure = MarketStructure(
        highs=list(recent_high[-50:]),
        lows=list(recent_low[-50:])
    )
    
    result = validate_tce(
        candle=candle,
        indicators=indicators,
        swing=swing,
        sr_levels=[],
        higher_tf_candles=[],
        correlations={},
        structure=structure,
        recent_candles=recent_candles,
        timeframe="H1",
        account_balance=10000.0,
        risk_percentage=1.0,
        symbol="SYNTHETIC"
    )
    
    if result['trend_ok']:
        trend_pass += 1
    if result['fib_ok']:
        fib_pass += 1
    if result['ma_level_ok']:
        ma_level_pass += 1
    if result['ma_retest_ok']:
        ma_retest_pass += 1
    if result['candlestick_ok']:
        candlestick_pass += 1
    
    if result["is_valid"]:
        valid_count += 1
        print(f"  ✅ Setup #{valid_count} at index {scan_idx}: {result['direction']} @ {close_arr[scan_idx]:.4f}")

if total_checked > 0:
    print(f"\n{'='*100}")
    print(f"📊 RESULTS FROM {total_checked} CANDLE CHECKS:")
    print(f"{'='*100}")
    print(f"  ✅ Valid setups:     {valid_count}/{total_checked} ({100*valid_count/total_checked:.1f}%)")
    print(f"  ✅ Trend passed:     {trend_pass}/{total_checked} ({100*trend_pass/total_checked:.1f}%)")
    print(f"  ✅ Fibonacci passed: {fib_pass}/{total_checked} ({100*fib_pass/total_checked:.1f}%)")
    print(f"  ✅ MA Level passed:  {ma_level_pass}/{total_checked} ({100*ma_level_pass/total_checked:.1f}%)")
    print(f"  ✅ MA Retest passed: {ma_retest_pass}/{total_checked} ({100*ma_retest_pass/total_checked:.1f}%)")
    print(f"  ✅ Candlestick:      {candlestick_pass}/{total_checked} ({100*candlestick_pass/total_checked:.1f}%)")
    
    if valid_count > 0:
        print(f"\n✅ LOCAL TEST PASSED - System is working correctly!")
    else:
        print(f"\n⚠️  No valid setups found - Adjust synthetic data parameters")
else:
    print(f"\n⚠️  Dataset too small to scan")

print(f"\n{'='*100}\n")
