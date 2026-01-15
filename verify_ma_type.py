"""
Quick script to verify which MA type (SMA vs EMA) matches your broker
Run this to compare MA50 values for EURCHF H1 at specific timestamps
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Your example data
symbol = "EURCHF"
timeframe = "H1"
first_test_time = "2024-02-20 21:00:00"
second_test_time = "2024-02-22 23:00:00"

# Your broker's values
broker_ma50_first_test = 0.950615
broker_ma50_second_test = 0.95098

print("="*80)
print("🔍 MA TYPE VERIFICATION - SMA vs EMA")
print("="*80)
print(f"\nSymbol: {symbol} {timeframe}")
print(f"First Test:  {first_test_time} - Broker MA50: {broker_ma50_first_test}")
print(f"Second Test: {second_test_time} - Broker MA50: {broker_ma50_second_test}\n")

# Load your MT5 data
mt5_data_path = Path("training_data_mt5")
csv_file = mt5_data_path / f"{symbol}_{timeframe}.csv"

if not csv_file.exists():
    print(f"❌ Data file not found: {csv_file}")
    print("   Make sure you have downloaded MT5 data first!")
    exit(1)

# Load and prepare data
df = pd.read_csv(csv_file)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})

# Calculate BOTH SMA and EMA
print("📊 Calculating both SMA and EMA...\n")

# SMA (Simple Moving Average)
df['SMA50'] = df['Close'].rolling(window=50, min_periods=50).mean()

# EMA (Exponential Moving Average)
df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

# Find the specific timestamps
try:
    first_test_row = df.loc[first_test_time]
    second_test_row = df.loc[second_test_time]
    
    sma50_first = first_test_row['SMA50']
    sma50_second = second_test_row['SMA50']
    ema50_first = first_test_row['EMA50']
    ema50_second = second_test_row['EMA50']
    
    print("="*80)
    print("RESULTS:")
    print("="*80)
    
    print(f"\n📅 FIRST TEST: {first_test_time}")
    print(f"   Your Broker:  {broker_ma50_first_test:.6f}")
    print(f"   Our SMA50:    {sma50_first:.6f} (diff: {abs(sma50_first - broker_ma50_first_test):.6f})")
    print(f"   Our EMA50:    {ema50_first:.6f} (diff: {abs(ema50_first - broker_ma50_first_test):.6f})")
    
    print(f"\n📅 SECOND TEST: {second_test_time}")
    print(f"   Your Broker:  {broker_ma50_second_test:.6f}")
    print(f"   Our SMA50:    {sma50_second:.6f} (diff: {abs(sma50_second - broker_ma50_second_test):.6f})")
    print(f"   Our EMA50:    {ema50_second:.6f} (diff: {abs(ema50_second - broker_ma50_second_test):.6f})")
    
    # Determine which matches better
    sma_error = abs(sma50_first - broker_ma50_first_test) + abs(sma50_second - broker_ma50_second_test)
    ema_error = abs(ema50_first - broker_ma50_first_test) + abs(ema50_second - broker_ma50_second_test)
    
    print(f"\n{'='*80}")
    print("✅ RECOMMENDATION:")
    print("="*80)
    
    if sma_error < ema_error:
        print(f"   Use MA_TYPE_TO_USE = 'SMA'")
        print(f"   Total error: {sma_error:.6f} (SMA is closer)")
    else:
        print(f"   Use MA_TYPE_TO_USE = 'EMA'")
        print(f"   Total error: {ema_error:.6f} (EMA is closer)")
    
    if min(sma_error, ema_error) > 0.001:
        print(f"\n   ⚠️  WARNING: Both have significant error (>{0.001})")
        print(f"   Possible causes:")
        print(f"   1. Data timezone mismatch")
        print(f"   2. Broker uses different MA period or type")
        print(f"   3. Data source difference")
        
    print(f"\n{'='*80}\n")
    
    # Show last 50 closes for manual verification
    print("📋 Last 50 CLOSE prices before first test (for manual calculation):")
    closes_before_first = df.loc[:first_test_time, 'Close'].tail(50)
    print(closes_before_first.to_string())
    
    manual_sma50 = closes_before_first.mean()
    print(f"\n✋ Manual SMA50 calculation: {manual_sma50:.6f}")
    print(f"   Our SMA50:  {sma50_first:.6f}")
    print(f"   Match: {'✅ YES' if abs(manual_sma50 - sma50_first) < 0.00001 else '❌ NO'}")
    
except KeyError as e:
    print(f"❌ Timestamp not found in data: {e}")
    print(f"   Available date range: {df.index.min()} to {df.index.max()}")
