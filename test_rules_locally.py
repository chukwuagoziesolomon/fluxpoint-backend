#!/usr/bin/env python
"""
Local test of validation rules to verify they're working BEFORE Colab
Tests the relaxed validation rules on real data
"""

import os
import django
import sys
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce
from trading.tce.utils import is_uptrend, is_downtrend
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend

def calculate_slope(values, period=20):
    """Calculate slope of a moving average"""
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

# Find and load data
data_dir = Path('./training_data')
if not data_dir.exists():
    print(f"ERROR: {data_dir} not found")
    sys.exit(1)

csv_files = list(data_dir.glob('*.csv'))
if not csv_files:
    print(f"ERROR: No CSV files in {data_dir}")
    sys.exit(1)

print("="*80)
print("LOCAL TEST: VALIDATION RULES")
print("="*80 + "\n")

# Test on first pair
csv_file = csv_files[0]
print(f"Testing with: {csv_file.name}\n")

df = pd.read_csv(csv_file)
df.columns = [col.lower().strip() for col in df.columns]

# Handle column name mapping
if 'price' in df.columns and 'close' not in df.columns:
    df['close'] = df['price']
for col in df.columns:
    if 'open' in col.lower() and 'open' not in df.columns:
        df['open'] = df[col]
    elif 'high' in col.lower() and 'high' not in df.columns:
        df['high'] = df[col]
    elif 'low' in col.lower() and 'low' not in df.columns:
        df['low'] = df[col]

# Convert to float
for col in ['open', 'high', 'low', 'close']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
print(f"Loaded {len(df)} candles\n")

if len(df) < 250:
    print(f"ERROR: Need at least 250 candles, got {len(df)}")
    sys.exit(1)

# Test statistics
stats = {
    'total_checked': 0,
    'trend_ok': 0,
    'ma_level_ok': 0,
    'ma_retest_ok': 0,
    'candlestick_ok': 0,
    'all_pass': 0,
    'failures': {}
}

print("Scanning setups...")
print("-" * 80 + "\n")

# Check every 10 candles (faster test)
test_indices = list(range(250, len(df) - 50, 10))

for row_idx in test_indices:
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values
    
    # Get window for indicators
    close_window = close[row_idx-240:row_idx+1]
    high_window = high[row_idx-240:row_idx+1]
    low_window = low[row_idx-240:row_idx+1]
    
    # Calculate MAs
    ma6 = np.mean(close_window[-6:]) if len(close_window) >= 6 else 0
    ma18 = np.mean(close_window[-18:]) if len(close_window) >= 18 else 0
    ma50 = np.mean(close_window[-50:]) if len(close_window) >= 50 else 0
    ma200 = np.mean(close_window[-200:]) if len(close_window) >= 200 else 0
    
    # Calculate slopes
    slope6 = calculate_slope(close_window, 6)
    slope18 = calculate_slope(close_window, 18)
    slope50 = calculate_slope(close_window, 50)
    
    # ATR
    tr_list = []
    for i in range(1, min(15, len(high_window))):
        tr = max(
            high_window[i] - low_window[i],
            abs(high_window[i] - close_window[i-1]),
            abs(low_window[i] - close_window[i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0.001
    
    # Create objects
    indicators = Indicators(
        ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
        slope6=slope6, slope18=slope18, slope50=slope50, slope200=0,
        atr=atr
    )
    
    candle = Candle(
        open=float(open_price[row_idx]),
        high=float(high[row_idx]),
        low=float(low[row_idx]),
        close=float(close[row_idx]),
        timestamp=str(row_idx)
    )
    
    # Fibonacci estimate
    ma_ref = ma18
    fib_level = 0.618
    if candle.low < ma_ref and atr > 0:
        depth = (ma_ref - candle.low) / atr
        if depth < 0.5:
            fib_level = 0.382
        elif depth < 1.0:
            fib_level = 0.5
    
    swing = Swing(
        type='high' if close[row_idx] > np.mean(close[row_idx-50:row_idx]) else 'low',
        price=float(close[row_idx]),
        fib_level=fib_level
    )
    
    # Recent candles
    recent_close = close[max(0, row_idx-50):row_idx+1]
    recent_high = high[max(0, row_idx-50):row_idx+1]
    recent_low = low[max(0, row_idx-50):row_idx+1]
    recent_open = open_price[max(0, row_idx-50):row_idx+1]
    
    recent_candles = [
        Candle(
            open=float(recent_open[i]),
            high=float(recent_high[i]),
            low=float(recent_low[i]),
            close=float(recent_close[i]),
            timestamp=str(i)
        )
        for i in range(len(recent_close))
    ]
    
    structure = MarketStructure(
        highs=list(recent_high[-50:]),
        lows=list(recent_low[-50:])
    )
    
    # Validate
    try:
        # Debug first few
        if stats['total_checked'] < 3:
            print(f"\nDEBUG Row {row_idx}:")
            print(f"  MAs: {ma6:.2f} > {ma18:.2f} > {ma50:.2f} > {ma200:.2f}")
            print(f"  is_uptrend: {is_uptrend(indicators)}")
            print(f"  is_downtrend: {is_downtrend(indicators)}")
            print(f"  is_valid_uptrend: {is_valid_uptrend(indicators, structure)}")
        
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
            symbol=csv_file.stem.upper()
        )
        
        stats['total_checked'] += 1
        
        # Track individual rules
        if result.get('trend_ok'):
            stats['trend_ok'] += 1
        if result.get('ma_level_ok'):
            stats['ma_level_ok'] += 1
        if result.get('ma_retest_ok'):
            stats['ma_retest_ok'] += 1
        if result.get('candlestick_ok'):
            stats['candlestick_ok'] += 1
        
        if result.get('is_valid'):
            stats['all_pass'] += 1
            print(f"FOUND VALID SETUP at candle {row_idx}")
            print(f"  Price: {candle.close:.5f}")
            print(f"  MAs: {ma6:.5f} > {ma18:.5f} > {ma50:.5f} > {ma200:.5f}")
            print(f"  Slopes: {slope6:+.6f}, {slope18:+.6f}, {slope50:+.6f}")
            print(f"  Direction: {result.get('direction')}")
            print(f"  Rules: Trend={result.get('trend_ok')} Fib={result.get('fib_ok')} MA={result.get('ma_level_ok')} Retest={result.get('ma_retest_ok')} Candle={result.get('candlestick_ok')}")
            print()
        else:
            reason = result.get('failure_reason', 'Unknown')
            stats['failures'][reason] = stats['failures'].get(reason, 0) + 1
    
    except Exception as e:
        pass

print("\n" + "="*80)
print("RESULTS")
print("="*80 + "\n")

print(f"Total checked: {stats['total_checked']}")
print(f"All rules pass (Valid setups): {stats['all_pass']}")
if stats['total_checked'] > 0:
    print(f"  Percentage: {100*stats['all_pass']/stats['total_checked']:.1f}%\n")
    
    print(f"Rule Pass Rates:")
    print(f"  1. Trend OK: {stats['trend_ok']}/{stats['total_checked']} ({100*stats['trend_ok']/stats['total_checked']:.1f}%)")
    print(f"  2. MA Level OK: {stats['ma_level_ok']}/{stats['total_checked']} ({100*stats['ma_level_ok']/stats['total_checked']:.1f}%)")
    print(f"  3. MA Retest OK: {stats['ma_retest_ok']}/{stats['total_checked']} ({100*stats['ma_retest_ok']/stats['total_checked']:.1f}%)")
    print(f"  4. Candlestick OK: {stats['candlestick_ok']}/{stats['total_checked']} ({100*stats['candlestick_ok']/stats['total_checked']:.1f}%)\n")
    
    print(f"Top Failure Reasons:")
    sorted_failures = sorted(stats['failures'].items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_failures[:5]:
        pct = 100 * count / stats['total_checked']
        print(f"  - {reason}: {count}x ({pct:.1f}%)")

print("\n" + "="*80)
if stats['all_pass'] > 0:
    print("SUCCESS: Found valid setups with relaxed rules!")
else:
    print("WARNING: Still finding 0 valid setups - needs more investigation")
print("="*80)
