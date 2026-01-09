#!/usr/bin/env python
"""
Direct test of validation.py to debug why 0 setups are passing.
Run with: python manage.py shell < debug_validation.py
OR: python debug_validation.py
"""

import os
import django
import sys
import numpy as np
import pandas as pd

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce
from trading.tce.utils import is_uptrend, is_downtrend
from trading.tce.structure import (
    is_valid_uptrend, is_valid_downtrend,
    has_higher_highs, has_higher_lows,
    has_lower_highs, has_lower_lows
)

def calculate_slope(values, period=20):
    """Calculate slope of a moving average"""
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

# Load a sample pair - look in trading/data directory
data_dir = "trading/data"  # Check this first
if not os.path.exists(data_dir):
    print(f"WARNING: {data_dir} not found, searching workspace...")
    # Search for OHLC CSV files with common columns
    possible_dirs = [
        "/content/drive/MyDrive/forex_data",
        "data",
        "forex_data",
        "trading/forex_data"
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break

# Try to find any CSV file with OHLC data
csv_files = []
for root, dirs, files in os.walk(data_dir if os.path.exists(data_dir) else '.'):
    for file in files:
        if file.endswith('.csv'):
            # Check if it's an OHLC file
            path = os.path.join(root, file)
            try:
                test_df = pd.read_csv(path, nrows=1)
                # Look for OHLC columns (case insensitive)
                cols = [col.lower().strip() for col in test_df.columns]
                if any(c in cols for c in ['close', 'high', 'low', 'open']):
                    csv_files.append(path)
            except:
                pass

# If still no files, create synthetic data for testing
if not csv_files:
    print("No OHLC CSV files found. Creating synthetic data for testing...")
    np.random.seed(42)
    n = 1000
    prices = 1.0 + np.cumsum(np.random.randn(n) * 0.0001)
    df = pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.00005,
        'high': prices + np.abs(np.random.randn(n) * 0.0001),
        'low': prices - np.abs(np.random.randn(n) * 0.0001),
        'close': prices,
        'date': pd.date_range('2020-01-01', periods=n, freq='H')
    })
    print(f"Created synthetic {len(df)} candles")

print(f"Found {len(csv_files)} OHLC CSV files")

# Load data
if csv_files:
    csv_file = csv_files[0]
    print(f"\nLoading: {csv_file}")
    df = pd.read_csv(csv_file)
else:
    print("Using synthetic data")

print(f"   Rows: {len(df)}, Columns: {df.columns.tolist()}")

# Normalize column names
df.columns = [col.lower().strip() for col in df.columns]
df = df.dropna()

# Handle different column name variations
if 'price' in df.columns and 'close' not in df.columns:
    df['close'] = df['price']
if 'close' not in df.columns:
    # Try to find close column
    for col in df.columns:
        if 'close' in col.lower():
            df['close'] = df[col]
            break

# Same for OHLC
for ohlc in ['open', 'high', 'low']:
    found = False
    for col in df.columns:
        if ohlc in col.lower():
            df[ohlc] = df[col]
            found = True
            break
    if not found and ohlc not in df.columns:
        df[ohlc] = df['close']  # Fallback

if 'date' not in df.columns:
    df['date'] = range(len(df))
if len(df) < 250:
    print(f"ERROR: Not enough data ({len(df)} rows, need 250+)")
    sys.exit(1)

print(f"Loaded {len(df)} candles\n")

# Test on a few rows
test_indices = [250, 300, 350, 400, 450]

print("="*80)
print("TESTING TREND VALIDATION ON SAMPLE CANDLES")
print("="*80 + "\n")

passing_count = 0
failing_count = 0

for row_idx in test_indices:
    if row_idx >= len(df):
        continue
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values
    
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
    slope200 = calculate_slope(close_window, 200)
    
    # Create Indicators object
    indicators = Indicators(
        ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
        slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200,
        atr=0.001
    )
    
    # Get recent highs/lows for structure
    recent_high = high_window[-50:]
    recent_low = low_window[-50:]
    
    structure = MarketStructure(
        highs=list(recent_high),
        lows=list(recent_low)
    )
    
    # Test each validation step
    print(f"ROW {row_idx}: Price={close[row_idx]:.5f}")
    print(f"   MAs: 6={ma6:.5f}, 18={ma18:.5f}, 50={ma50:.5f}, 200={ma200:.5f}")
    print(f"   Slopes: 6={slope6:+.6f}, 18={slope18:+.6f}, 50={slope50:+.6f}")
    
    # Check MA alignment
    ma_aligned_up = ma6 > ma18 > ma50 > ma200
    ma_aligned_down = ma200 > ma50 > ma18 > ma6
    print(f"   MA aligned (UP): {ma_aligned_up}")
    print(f"   MA aligned (DOWN): {ma_aligned_down}")
    
    # Check slopes
    positive_slopes = sum([slope6 > 0, slope18 > 0, slope50 > 0])
    negative_slopes = sum([slope6 < 0, slope18 < 0, slope50 < 0])
    print(f"   Positive slopes: {positive_slopes}/3")
    print(f"   Negative slopes: {negative_slopes}/3")
    
    # Check trend functions
    uptrend = is_uptrend(indicators)
    downtrend = is_downtrend(indicators)
    print(f"   is_uptrend(): {uptrend}")
    print(f"   is_downtrend(): {downtrend}")
    
    # Check structure functions
    hh = has_higher_highs(recent_high)
    hl = has_higher_lows(recent_low)
    lh = has_lower_highs(recent_high)
    ll = has_lower_lows(recent_low)
    print(f"   Structure - HH: {hh}, HL: {hl}, LH: {lh}, LL: {ll}")
    
    # Check if valid trend
    valid_up = is_valid_uptrend(indicators, structure)
    valid_down = is_valid_downtrend(indicators, structure)
    print(f"   Valid UPTREND: {valid_up}")
    print(f"   Valid DOWNTREND: {valid_down}")
    
    if not (valid_up or valid_down):
        failing_count += 1
        print(f"   [FAIL] NO VALID TREND!")
        if uptrend and not hh:
            print(f"      - MA uptrend OK but has_higher_highs FAILED")
        if uptrend and not hl:
            print(f"      - MA uptrend OK but has_higher_lows FAILED")
        if downtrend and not lh:
            print(f"      - MA downtrend OK but has_lower_highs FAILED")
        if downtrend and not ll:
            print(f"      - MA downtrend OK but has_lower_lows FAILED")
    else:
        passing_count += 1
        print(f"   [PASS] Valid trend found!")
    
    print()

print("\n" + "="*80)
print("SUMMARY: Identify which validation rule is the bottleneck")
print("="*80)
print(f"Passed: {passing_count}, Failed: {failing_count}")

