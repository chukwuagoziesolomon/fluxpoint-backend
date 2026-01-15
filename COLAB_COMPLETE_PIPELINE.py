# ════════════════════════════════════════════════════════════════════════════════
# TCE COMPLETE PIPELINE - MULTI-CELL VERSION FOR GOOGLE COLAB
# ════════════════════════════════════════════════════════════════════════════════
#
# 🎯 WORKFLOW (Run cells in order):
# 
# CELL 1: Setup - Clone repo and install dependencies
# CELL 2: Extract Data - Upload CSVs and extract training examples
# CELL 3: View Examples - Inspect extracted data before training
# CELL 4: Train Model - Train DL model on extracted data
# CELL 5: Download Results - Get model and training data
#
# ⏱️ TOTAL TIME: 5-10 minutes
#
# 📋 BEFORE STARTING:
# 1. Download MT5 data locally: python download_mt5_multi_timeframe.py
# 2. Have CSV files ready (multiple pairs and timeframes recommended)
#
# 🚀 RUN EACH CELL IN ORDER (Ctrl+Enter or click Play)
#
# ════════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════
# CELL 1: SETUP - CLONE REPO & INSTALL DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════════

import os
import sys
import subprocess
from pathlib import Path

print("="*80)
print("🚀 STEP 1: CLONE GITHUB REPOSITORY")
print("="*80)

# Repository details (update if using your own fork)
REPO_URL = "https://github.com/YOUR_USERNAME/fluxpointai-backend.git"
REPO_NAME = "fluxpointai-backend"
REPO_PATH = Path(f"/content/{REPO_NAME}")

print(f"\n🔗 Repository: {REPO_URL}")

if REPO_PATH.exists():
    print("📁 Repository exists - pulling latest changes...")
    os.chdir(REPO_PATH)
    try:
        subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        print("✅ Updated to latest version")
    except:
        print("⚠️  Using existing code")
else:
    print("📥 Cloning repository...")
    try:
        subprocess.run(["git", "clone", REPO_URL, str(REPO_PATH)], 
                      capture_output=True, text=True, check=True)
        print("✅ Repository cloned")
    except:
        print("⚠️  Clone failed - will use standalone mode")
        REPO_PATH.mkdir(parents=True, exist_ok=True)

# Add to Python path
fluxpoint_path = str(REPO_PATH / "fluxpoint")
if fluxpoint_path not in sys.path:
    sys.path.insert(0, fluxpoint_path)

print("\n" + "="*80)
print("📦 STEP 2: INSTALL DEPENDENCIES")
print("="*80)

packages = ['pandas', 'numpy', 'torch', 'scikit-learn']
print(f"\nInstalling: {', '.join(packages)}")

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ All dependencies installed")

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from google.colab import files

print("\n" + "="*80)
print("✅ CELL 1 COMPLETE: Setup finished")
print("="*80)
print("\n💡 Next: Run CELL 2 to load CSVs from Google Drive and extract training data")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 2: EXTRACT TRAINING DATA FROM GOOGLE DRIVE
# ════════════════════════════════════════════════════════════════════════════════

from google.colab import drive

print("\n" + "="*80)
print("📊 CELL 2: EXTRACT TRAINING DATA FROM GOOGLE DRIVE")
print("="*80)

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Google Drive mounted")

# Specify data directory
data_dir = Path('/content/drive/MyDrive/forex_data/training_data_mt5')
print(f"\n📂 Looking for MT5 data in: {data_dir}")

# Check if directory exists
if not data_dir.exists():
    print(f"\n❌ Directory not found!")
    print(f"\n📝 TO FIX:")
    print(f"   1. Create folder in Google Drive: MyDrive/forex_data/training_data_mt5/")
    print(f"   2. Upload your CSV files there (USDJPY_H1.csv, EURUSD_H4.csv, etc.)")
    print(f"   3. Re-run this cell")
    all_training_data = []
else:
    # Find all CSV files recursively
    csv_files = []
    
    # Check for timeframe folders (H1, H4, D1, etc.)
    for item in data_dir.iterdir():
        if item.is_dir():
            csv_files.extend(list(item.glob('*.csv')))
    
    # Also check root directory
    csv_files.extend(list(data_dir.glob('*.csv')))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in {data_dir}")
        print(f"\n📝 Expected structure:")
        print(f"   MyDrive/forex_data/training_data_mt5/")
        print(f"   ├── H1/")
        print(f"   │   ├── USDJPY.csv")
        print(f"   │   └── EURUSD.csv")
        print(f"   ├── H4/")
        print(f"   │   └── USDJPY.csv")
        print(f"   └── D1/")
        print(f"       └── USDJPY.csv")
        print(f"\n   OR flat structure:")
        print(f"   ├── USDJPY_H1.csv")
        print(f"   ├── USDJPY_H4.csv")
        print(f"   └── EURUSD_H1.csv")
        all_training_data = []
    else:
        print(f"\n✅ Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"   • {csv_file.relative_to(data_dir)}")
        
        print("\n" + "="*80)
        
        # ============================================================================
        # HELPER FUNCTIONS
        # ============================================================================
        
        def calculate_indicators(df):
            """Calculate MAs, ATR, RSI, momentum indicators, and Force Index"""
            df['ma6'] = df['close'].rolling(window=6).mean()
            df['ma18'] = df['close'].rolling(window=18).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            df['ma200'] = df['close'].rolling(window=200).mean()
            
            # Slopes
            for period in [6, 18, 50, 200]:
                df[f'slope{period}'] = df[f'ma{period}'].diff(5)
            
            # ATR
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
            
            # RSI (14-period)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Force Index (catches fake breakouts & divergences)
            # Formula: (Close - Previous Close) × Volume
            # Since forex has no volume, use candle range as proxy
            price_change = df['close'].diff()
            candle_range = df['high'] - df['low']
            df['force_index_raw'] = price_change * candle_range
            df['force_index'] = df['force_index_raw'].rolling(window=13).mean()  # 13-period EMA
            
            # Correlation (trend strength) - 20-period correlation between price and MA50
            df['correlation'] = df['close'].rolling(window=20).corr(df['ma50'])
            
            return df
        
        
        def basic_trend_direction(row):
            """Simple trend check for BUY/SELL labeling"""
            if row['ma6'] > row['ma18'] > row['ma50']:
                return 'BUY'
            elif row['ma6'] < row['ma18'] < row['ma50']:
                return 'SELL'
            return None
        
        
        # ============================================================================
        # CANDLESTICK PATTERN DETECTION (from validation.py)
        # ============================================================================
        
        class Candle:
            def __init__(self, time, open, high, low, close):
                self.time = time
                self.open = open
                self.high = high
                self.low = low
                self.close = close
        
        def is_rejection_candle(candle, direction):
            """Basic rejection with visible wick"""
            body = abs(candle.close - candle.open)
            if body == 0:
                return False
            wick_up = candle.high - max(candle.open, candle.close)
            wick_down = min(candle.open, candle.close) - candle.low
            if direction == "BUY":
                return wick_down > 0 and candle.close > candle.open
            else:
                return wick_up > 0 and candle.close < candle.open
        
        def is_bullish_pin_bar(candle):
            """Lower wick 2x body, small upper wick"""
            body = abs(candle.close - candle.open)
            if body == 0:
                return False
            lower_wick = min(candle.open, candle.close) - candle.low
            upper_wick = candle.high - max(candle.open, candle.close)
            return (lower_wick >= body * 2.0 and upper_wick <= body * 0.5 and candle.close > candle.open)
        
        def is_bearish_pin_bar(candle):
            """Upper wick 2x body, small lower wick"""
            body = abs(candle.close - candle.open)
            if body == 0:
                return False
            upper_wick = candle.high - max(candle.open, candle.close)
            lower_wick = min(candle.open, candle.close) - candle.low
            return (upper_wick >= body * 2.0 and lower_wick <= body * 0.5 and candle.close < candle.open)
        
        def is_bullish_engulfing(prev, curr):
            """Current bullish body engulfs previous bearish body"""
            prev_body = abs(prev.close - prev.open)
            curr_body = abs(curr.close - curr.open)
            if prev_body == 0 or curr_body == 0:
                return False
            return (prev.close < prev.open and curr.close > curr.open and 
                    curr.open <= prev.close and curr.close >= prev.open and curr_body >= prev_body)
        
        def is_bearish_engulfing(prev, curr):
            """Current bearish body engulfs previous bullish body"""
            prev_body = abs(prev.close - prev.open)
            curr_body = abs(curr.close - curr.open)
            if prev_body == 0 or curr_body == 0:
                return False
            return (prev.close > prev.open and curr.close < curr.open and 
                    curr.open >= prev.close and curr.close <= prev.open and curr_body >= prev_body)
        
        def is_one_white_soldier(prev, curr):
            """Strong bullish after bearish, closes above prev high"""
            body_prev = abs(prev.close - prev.open)
            body_curr = abs(curr.close - curr.open)
            if body_prev == 0 or body_curr == 0:
                return False
            return (prev.close < prev.open and curr.close > curr.open and 
                    body_curr >= body_prev * 1.2 and curr.close > prev.high)
        
        def is_one_black_crow(prev, curr):
            """Strong bearish after bullish, closes below prev low"""
            body_prev = abs(prev.close - prev.open)
            body_curr = abs(curr.close - curr.open)
            if body_prev == 0 or body_curr == 0:
                return False
            return (prev.close > prev.open and curr.close < curr.open and 
                    body_curr >= body_prev * 1.2 and curr.close < prev.low)
        
        def is_tweezer_bottom(prev, curr):
            """Lows almost equal (bottom rejection)"""
            return abs(prev.low - curr.low) / max(prev.low, curr.low, 0.00001) < 0.0005
        
        def is_tweezer_top(prev, curr):
            """Highs almost equal (top rejection)"""
            return abs(prev.high - curr.high) / max(prev.high, curr.high, 0.00001) < 0.0005
        
        def is_morning_star(c1, c2, c3):
            """Bearish, small middle, strong bullish"""
            body1 = abs(c1.close - c1.open)
            body2 = abs(c2.close - c2.open)
            body3 = abs(c3.close - c3.open)
            if body1 == 0 or body3 == 0:
                return False
            return (c1.close < c1.open and c3.close > c3.open and 
                    body2 <= body1 * 0.6 and body2 <= body3 * 0.6)
        
        def is_evening_star(c1, c2, c3):
            """Bullish, small middle, strong bearish"""
            body1 = abs(c1.close - c1.open)
            body2 = abs(c2.close - c2.open)
            body3 = abs(c3.close - c3.open)
            if body1 == 0 or body3 == 0:
                return False
            return (c1.close > c1.open and c3.close < c3.open and 
                    body2 <= body1 * 0.6 and body2 <= body3 * 0.6)
        
        def detect_candlestick_patterns(df, i, direction, ma_level, row):
            """
            Detect candlestick patterns at position i (second bounce/entry point)
            ONLY valid if pattern occurs at rally/retracement S/R level (MA touch)
            
            Args:
                df: DataFrame with price data
                i: Current index (second bounce - COMPLETED candle)
                direction: 'BUY' or 'SELL'
                ma_level: Which MA was touched ('MA18', 'MA50', 'MA200')
                row: Current row data (for MA prices)
            """
            if i < 2:
                return {
                    'has_rejection': 0, 'has_pin_bar': 0, 'has_engulfing': 0,
                    'has_soldier_crow': 0, 'has_tweezer': 0, 'has_star': 0,
                    'has_any_pattern': 0, 'pattern_at_sr': 0
                }
            
            # Check if pattern is at S/R level (MA touch)
            # Pattern must be within 5% of the MA level that was touched
            ma_price = row[ma_level.lower()]  # Get the MA price (ma18, ma50, or ma200)
            price_at_sr = row['low'] if direction == 'BUY' else row['high']
            
            # Calculate distance from MA
            distance_from_ma = abs(price_at_sr - ma_price) / ma_price if ma_price > 0 else 1.0
            pattern_at_sr_level = 1 if distance_from_ma <= 0.05 else 0
            
            # If pattern is NOT at S/R level, return all zeros
            if not pattern_at_sr_level:
                return {
                    'has_rejection': 0, 'has_pin_bar': 0, 'has_engulfing': 0,
                    'has_soldier_crow': 0, 'has_tweezer': 0, 'has_star': 0,
                    'has_any_pattern': 0, 'pattern_at_sr': 0
                }
            
            # Create candle objects (pattern detected at S/R level)
            curr = Candle(df.iloc[i]['time'], df.iloc[i]['open'], 
                         df.iloc[i]['high'], df.iloc[i]['low'], df.iloc[i]['close'])
            prev = Candle(df.iloc[i-1]['time'], df.iloc[i-1]['open'],
                         df.iloc[i-1]['high'], df.iloc[i-1]['low'], df.iloc[i-1]['close'])
            prev2 = Candle(df.iloc[i-2]['time'], df.iloc[i-2]['open'],
                          df.iloc[i-2]['high'], df.iloc[i-2]['low'], df.iloc[i-2]['close'])
            
            # Detect patterns based on direction
            if direction == "BUY":
                patterns = {
                    'has_rejection': int(is_rejection_candle(curr, direction)),
                    'has_pin_bar': int(is_bullish_pin_bar(curr)),
                    'has_engulfing': int(is_bullish_engulfing(prev, curr)),
                    'has_soldier_crow': int(is_one_white_soldier(prev, curr)),
                    'has_tweezer': int(is_tweezer_bottom(prev, curr)),
                    'has_star': int(is_morning_star(prev2, prev, curr)),
                    'pattern_at_sr': pattern_at_sr_level
                }
            else:  # SELL
                patterns = {
                    'has_rejection': int(is_rejection_candle(curr, direction)),
                    'has_pin_bar': int(is_bearish_pin_bar(curr)),
                    'has_engulfing': int(is_bearish_engulfing(prev, curr)),
                    'has_soldier_crow': int(is_one_black_crow(prev, curr)),
                    'has_tweezer': int(is_tweezer_top(prev, curr)),
                    'has_star': int(is_evening_star(prev2, prev, curr)),
                    'pattern_at_sr': pattern_at_sr_level
                }
            
            # Overall confirmation flag
            patterns['has_any_pattern'] = int(any([
                patterns['has_rejection'], patterns['has_pin_bar'], 
                patterns['has_engulfing'], patterns['has_soldier_crow'],
                patterns['has_tweezer'], patterns['has_star']
            ]))
            
            return patterns
        
        
        def calculate_ma_distance(row, direction):
            """Calculate distance from price to each MA (from validation.py logic)"""
            price = row['low'] if direction == 'BUY' else row['high']
            
            # Distance as percentage
            dist_ma18 = abs(price - row['ma18']) / row['ma18'] if row['ma18'] > 0 else 0
            dist_ma50 = abs(price - row['ma50']) / row['ma50'] if row['ma50'] > 0 else 0
            dist_ma200 = abs(price - row['ma200']) / row['ma200'] if row['ma200'] > 0 else 0
            
            # Is price at MA? (within 5% tolerance like validation.py)
            at_ma18 = 1 if dist_ma18 < 0.05 else 0
            at_ma50 = 1 if dist_ma50 < 0.05 else 0
            at_ma200 = 1 if dist_ma200 < 0.05 else 0
            
            return {
                'dist_ma18': dist_ma18,
                'dist_ma50': dist_ma50,
                'dist_ma200': dist_ma200,
                'at_ma18': at_ma18,
                'at_ma50': at_ma50,
                'at_ma200': at_ma200
            }
        
        
        def calculate_fibonacci_retracement(df, first_bounce_idx, current_idx, direction):
            """
            Calculate Fibonacci retracement level - INVALID if > 61.8%
            
            For BUY: Measures how much price retraced from swing high back toward first bounce low
            For SELL: Measures how much price retraced from swing low back toward first bounce high
            """
            if first_bounce_idx >= current_idx or first_bounce_idx < 0:
                return 0.0, True  # Return 0% and valid=True as fallback
            
            # Get first bounce price and swing slice
            first_bounce_price = df.iloc[first_bounce_idx]['low'] if direction == 'BUY' else df.iloc[first_bounce_idx]['high']
            swing_slice = df.iloc[first_bounce_idx:current_idx+1]
            
            if direction == 'BUY':
                # For BUY: Price bounced from low, rallied to high, now retracing back down
                swing_low = first_bounce_price  # Starting point (first bounce)
                swing_high = swing_slice['high'].max()  # Peak of the rally
                current_low = df.iloc[current_idx]['low']  # Current retracement level
                
                # Calculate retracement: how far back down from the high
                swing_range = swing_high - swing_low
                if swing_range > 0:
                    # Retracement = distance from high to current / original swing range
                    retracement = (swing_high - current_low) / swing_range
                else:
                    retracement = 0.0
            else:  # SELL
                # For SELL: Price bounced from high, declined to low, now retracing back up
                swing_high = first_bounce_price  # Starting point (first bounce)
                swing_low = swing_slice['low'].min()  # Bottom of the decline
                current_high = df.iloc[current_idx]['high']  # Current retracement level
                
                # Calculate retracement: how far back up from the low
                swing_range = swing_high - swing_low
                if swing_range > 0:
                    # Retracement = distance from low to current / original swing range
                    retracement = (current_high - swing_low) / swing_range
                else:
                    retracement = 0.0
            
            # INVALID if retracement > 0.618 (61.8% Fibonacci level)
            # Higher retracement = deeper pullback = weaker trend = skip
            is_valid = retracement <= 0.618
            
            return retracement, is_valid
        
        
        def detect_ma_bounce(row, direction, pip_size=0.01):
            """Detect if candle bounced from any MA (18, 50, 200) - returns ALL touched MAs"""
            tol = 2 * pip_size  # 2 pips
            
            bounces = []
            
            if direction == 'BUY':
                # Low touches MA, close above MA
                if row['low'] <= row['ma18'] + tol and row['close'] >= row['ma18']:
                    bounces.append('MA18')
                if row['low'] <= row['ma50'] + tol and row['close'] >= row['ma50']:
                    bounces.append('MA50')
                if row['low'] <= row['ma200'] + tol and row['close'] >= row['ma200']:
                    bounces.append('MA200')
            else:  # SELL
                # High touches MA, close below MA
                if row['high'] >= row['ma18'] - tol and row['close'] <= row['ma18']:
                    bounces.append('MA18')
                if row['high'] >= row['ma50'] - tol and row['close'] <= row['ma50']:
                    bounces.append('MA50')
                if row['high'] >= row['ma200'] - tol and row['close'] <= row['ma200']:
                    bounces.append('MA200')
            
            # Return ALL touched MAs (not just one)
            return bounces
        
        
        def extract_training_examples(df, symbol='USDJPY', timeframe='H1', pip_size=0.01):
            """Extract all MA bounce retests as training examples - ALL MAs tracked separately"""
            
            df = calculate_indicators(df)
            
            training_data = []
            ma_bounces = {'MA18': [], 'MA50': [], 'MA200': []}
            
            print(f"\n🔍 Extracting from {symbol} {timeframe}...")
            print(f"   📊 Total candles: {len(df)}")
            
            # Debug counters
            debug_counts = {
                'total_processed': 0,
                'has_trend': 0,
                'has_ma_bounce': 0,
                'has_previous_bounce': 0,
                'time_too_short': 0,
                'fib_invalid': 0,
                'extracted': 0
            }
            
            for i in range(250, len(df)):
                if pd.isna(df.iloc[i]['ma200']) or pd.isna(df.iloc[i]['atr']):
                    continue
                
                debug_counts['total_processed'] += 1
                row = df.iloc[i]
                
                # Get trend direction
                trend = basic_trend_direction(row)
                if not trend:
                    continue
                
                debug_counts['has_trend'] += 1
                
                # Detect MA bounces (can be multiple MAs at once)
                mas_touched = detect_ma_bounce(row, trend, pip_size)
                if not mas_touched:
                    continue
                
                debug_counts['has_ma_bounce'] += 1
                
                bounce_price = row['low'] if trend == 'BUY' else row['high']
                
                # Process EACH MA that was touched separately
                for ma_touched in mas_touched:
                    # Record bounce for this specific MA
                    ma_bounces[ma_touched].append({
                        'index': i,
                        'time': row['time'],
                        'price': bounce_price,
                        'direction': trend
                    })
                    
                    # Look for retest (previous bounce of SAME MA/direction)
                    # IMPORTANT: First bounce = test, Second bounce (current) = entry point
                    recent_bounces = [
                        b for b in ma_bounces[ma_touched]
                        if b['index'] < i 
                        and b['index'] > i - 100
                        and b['direction'] == trend
                    ]
                    
                    if len(recent_bounces) == 0:
                        continue
                    
                    debug_counts['has_previous_bounce'] += 1
                    
                    first_bounce = recent_bounces[-1]  # This is the TEST bounce (we don't trade here)
                    # Current bounce (i) is the RETEST/ENTRY (we trade here)
                    
                    # Calculate time between bounces
                    time_diff_hours = (row['time'] - first_bounce['time']).total_seconds() / 3600
                    
                    # Minimum 4 hours between bounces
                    if time_diff_hours < 4:
                        debug_counts['time_too_short'] += 1
                        continue
                    
                    # Calculate Fibonacci retracement - SKIP if > 61.8%
                    fib_retracement, fib_valid = calculate_fibonacci_retracement(
                        df, first_bounce['index'], i, trend
                    )
                    
                    if not fib_valid:
                        # Skip this setup - retracement too deep (> 61.8%)
                        debug_counts['fib_invalid'] += 1
                        continue
                    
                    # Calculate outcome (did it move in trend direction?)
                    # We measure from ENTRY (current bounce), not from first bounce
                    future_window = min(20, len(df) - i - 1)
                    if future_window < 5:
                        continue
                    
                    future_slice = df.iloc[i+1:i+1+future_window]
                    
                    if trend == 'BUY':
                        max_profit = (future_slice['high'].max() - row['close']) / pip_size
                        max_loss = (row['close'] - future_slice['low'].min()) / pip_size
                    else:
                        max_profit = (row['close'] - future_slice['low'].min()) / pip_size
                        max_loss = (future_slice['high'].max() - row['close']) / pip_size
                    
                    # Detect candlestick patterns at S/R level (MA touch)
                    candle_patterns = detect_candlestick_patterns(df, i, trend, ma_touched, row)
                    ma_distance_features = calculate_ma_distance(row, trend)
                    
                    # Higher Timeframe Trend Confirmation
                    htf_trend_aligned = 0
                    if timeframe == 'H1' and symbol in all_data and 'H4' in all_data[symbol]:
                        h4_df = all_data[symbol]['H4']
                        h4_match = h4_df[h4_df['time'] <= row['time']]
                        if len(h4_match) > 0:
                            h4_row = h4_match.iloc[-1]
                            if not pd.isna(h4_row['ma50']):
                                h4_trend = 'BUY' if h4_row['close'] > h4_row['ma50'] else 'SELL'
                                htf_trend_aligned = 1 if h4_trend == trend else 0
                    elif timeframe == 'H4' and symbol in all_data and 'D1' in all_data[symbol]:
                        d1_df = all_data[symbol]['D1']
                        d1_match = d1_df[d1_df['time'] <= row['time']]
                        if len(d1_match) > 0:
                            d1_row = d1_match.iloc[-1]
                            if not pd.isna(d1_row['ma50']):
                                d1_trend = 'BUY' if d1_row['close'] > d1_row['ma50'] else 'SELL'
                                htf_trend_aligned = 1 if d1_trend == trend else 0
                    
                    # Build training example with all features
                    example = {
                        'time': str(row['time']),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': trend,
                        'ma_level': ma_touched,
                        
                        # Entry point
                        'entry_price': float(row['close']),
                        'entry_high': float(row['high']),
                        'entry_low': float(row['low']),
                        'entry_open': float(row['open']),
                        
                        # MA values at entry
                        'ma6': float(row['ma6']),
                        'ma18': float(row['ma18']),
                        'ma50': float(row['ma50']),
                        'ma200': float(row['ma200']),
                        
                        # Slopes
                        'slope6': float(row['slope6']),
                        'slope18': float(row['slope18']),
                        'slope50': float(row['slope50']),
                        'slope200': float(row['slope200']),
                        
                        # Volatility
                        'atr': float(row['atr']),
                        
                        # RSI & Force Index
                        'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else 50.0,
                        'force_index': float(row['force_index']) if not pd.isna(row['force_index']) else 0.0,
                        
                        # Correlation (trend strength) & HTF confirmation
                        'correlation': float(row['correlation']) if not pd.isna(row['correlation']) else 0.0,
                        'htf_trend_aligned': htf_trend_aligned,
                        
                        # Fibonacci retracement (already validated <= 61.8%)
                        'fib_retracement': float(fib_retracement),
                        'fib_valid': int(fib_valid),  # Always 1 since we filtered out invalid ones
                        
                        # Candlestick patterns (from validation.py) - ONLY at S/R level
                        'pattern_at_sr': candle_patterns['pattern_at_sr'],
                        'has_rejection': candle_patterns['has_rejection'],
                        'has_pin_bar': candle_patterns['has_pin_bar'],
                        'has_engulfing': candle_patterns['has_engulfing'],
                        'has_soldier_crow': candle_patterns['has_soldier_crow'],
                        'has_tweezer': candle_patterns['has_tweezer'],
                        'has_star': candle_patterns['has_star'],
                        'has_any_pattern': candle_patterns['has_any_pattern'],
                        
                        # Distance from MAs
                        'dist_ma18': float(ma_distance_features['dist_ma18']),
                        'dist_ma50': float(ma_distance_features['dist_ma50']),
                        'dist_ma200': float(ma_distance_features['dist_ma200']),
                        'at_ma18': int(ma_distance_features['at_ma18']),
                        'at_ma50': int(ma_distance_features['at_ma50']),
                        'at_ma200': int(ma_distance_features['at_ma200']),
                        
                        # Swing info
                        'first_bounce_time': str(first_bounce['time']),
                        'first_bounce_price': float(first_bounce['price']),
                        'swing_duration_hours': float(time_diff_hours),
                        
                        # Outcomes (for supervised learning labels)
                        'max_profit_pips': float(max_profit),
                        'max_loss_pips': float(max_loss),
                        'profit_loss_ratio': float(max_profit / max_loss) if max_loss > 0 else 0,
                        
                        # Success labels (simple binary)
                        'profitable': bool(max_profit > max_loss * 1.5),  # At least 1.5:1
                        'very_profitable': bool(max_profit > max_loss * 2.0),  # At least 2:1
                    }
                    
                    training_data.append(example)
                    debug_counts['extracted'] += 1
                
                if len(training_data) % 50 == 0 and len(training_data) > 0:
                    print(f"  Extracted {len(training_data)} examples...")
            
            # Show distribution for this pair/timeframe
            ma_counts = {}
            for ex in training_data:
                ma = ex['ma_level']
                ma_counts[ma] = ma_counts.get(ma, 0) + 1
            
            print(f"✅ Extracted {len(training_data)} examples from {symbol} {timeframe}")
            
            # Show debug information if no examples found
            if len(training_data) == 0:
                print(f"   🔍 DEBUG: Why no examples?")
                print(f"      Candles processed: {debug_counts['total_processed']}")
                print(f"      Had valid trend: {debug_counts['has_trend']}")
                print(f"      Had MA bounce: {debug_counts['has_ma_bounce']}")
                print(f"      Had previous bounce: {debug_counts['has_previous_bounce']}")
                print(f"      Time too short (<4h): {debug_counts['time_too_short']}")
                print(f"      Fib invalid (>61.8%): {debug_counts['fib_invalid']}")
            
            if ma_counts:
                print(f"   Distribution: ", end="")
                print(" | ".join([f"{ma}: {count}" for ma, count in sorted(ma_counts.items())]))
            
            return training_data
        
        
        def load_csv_file(filename):
            """Load CSV file with proper time parsing"""
            try:
                df = pd.read_csv(filename)
                
                # Check if empty
                if df.empty or len(df) == 0:
                    print(f"\n⚠️  Warning: {filename} is empty, skipping...")
                    return None
                
                # Normalize column names to lowercase
                df.columns = df.columns.str.lower()
                
                # Handle different time column names
                time_columns = ['time', 'datetime', 'timestamp', 'date']
                time_col = None
                for col in time_columns:
                    if col in df.columns:
                        time_col = col
                        break
                
                # If first column looks like datetime, use it
                if time_col is None and len(df.columns) > 0:
                    first_col = df.columns[0]
                    # Try to parse first column as datetime
                    try:
                        pd.to_datetime(df[first_col])
                        time_col = first_col
                    except:
                        pass
                
                if time_col is None:
                    print(f"\n⚠️  Warning: Could not find time column in {filename}. Columns: {list(df.columns)}")
                    return None
                
                # Rename to 'time' and parse
                if time_col != 'time':
                    df = df.rename(columns={time_col: 'time'})
                
                df['time'] = pd.to_datetime(df['time'])
                
                return df
                
            except pd.errors.EmptyDataError:
                print(f"\n⚠️  Warning: {filename} is empty or corrupted, skipping...")
                return None
            except Exception as e:
                print(f"\n⚠️  Warning: Error loading {filename}: {e}, skipping...")
                return None
        
        
        # ============================================================================
        # MAIN EXECUTION
        # ============================================================================
        
        print("\n" + "="*80)
        
        # Step 1: Load ALL CSV files into memory organized by symbol/timeframe
        print("📥 STEP 1: Loading all CSV files into memory...")
        all_data = {}  # {symbol: {timeframe: df}}
        
        for csv_file in csv_files:
            filename = csv_file.name
            
            # Parse symbol and timeframe
            parent_folder = csv_file.parent.name
            if parent_folder in ['H1', 'H4', 'D1', 'W1', 'M15', 'M30', 'M1', 'M5']:
                timeframe = parent_folder
                symbol = filename.replace('.csv', '').upper()
            else:
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 2:
                    symbol = parts[0].upper()
                    timeframe = parts[1].upper()
                else:
                    symbol = filename.replace('.csv', '').upper()
                    timeframe = 'UNKNOWN'
            
            # Clean symbol name (remove _h4, _H4 suffixes if present)
            symbol = symbol.replace('_H1', '').replace('_H4', '').replace('_D1', '').replace('_W1', '')
            symbol = symbol.replace('_M1', '').replace('_M5', '').replace('_M15', '').replace('_M30', '')
            
            print(f"  Loading {symbol} {timeframe}...", end=" ")
            df = load_csv_file(str(csv_file))
            
            # Skip if file is empty or failed to load
            if df is None:
                continue
            
            print(f"✅ {len(df)} candles")
            
            # Calculate indicators for this dataframe (needed for HTF confirmation)
            df = calculate_indicators(df)
            
            # Store in nested dictionary
            if symbol not in all_data:
                all_data[symbol] = {}
            all_data[symbol][timeframe] = df
        
        print(f"\n✅ Loaded {len(csv_files)} files for {len(all_data)} symbols")
        
        # Step 2: Extract training examples from each file
        print("\n" + "="*80)
        print("📊 STEP 2: Extracting training examples...")
        all_training_data = []
        
        # Process each symbol/timeframe combination
        for symbol in all_data:
            for timeframe in all_data[symbol]:
                df = all_data[symbol][timeframe]
                
                # Determine pip size
                pip_size = 0.01 if 'JPY' in symbol else 0.0001
                
                # Extract training examples
                examples = extract_training_examples(df, symbol, timeframe, pip_size)
                all_training_data.extend(examples)

# Save all training data
print("\n" + "=" * 80)
print(f"💾 Total training examples: {len(all_training_data)}")
print("=" * 80)

output_filename = 'training_data.json'
with open(output_filename, 'w') as f:
    json.dump(all_training_data, f, indent=2)

print(f"✅ Saved to {output_filename}")

# Show statistics
if all_training_data:
    profitable = sum(1 for ex in all_training_data if ex['profitable'])
    unprofitable = len(all_training_data) - profitable
    
    print(f"\n📊 Statistics:")
    print(f"   Total: {len(all_training_data)}")
    print(f"   ✅ Profitable (1.5:1+): {profitable} ({profitable/len(all_training_data)*100:.1f}%)")
    print(f"   ❌ Unprofitable: {unprofitable} ({unprofitable/len(all_training_data)*100:.1f}%)")
    
    # Distribution by MA
    ma_counts = {}
    for ex in all_training_data:
        ma = ex['ma_level']
        ma_counts[ma] = ma_counts.get(ma, 0) + 1
    
    print(f"\n📊 Distribution by MA Level:")
    for ma, count in sorted(ma_counts.items()):
        print(f"   {ma}: {count} ({count/len(all_training_data)*100:.1f}%)")
    
    # Distribution by timeframe
    tf_counts = {}
    for ex in all_training_data:
        tf = ex['timeframe']
        tf_counts[tf] = tf_counts.get(tf, 0) + 1
    
    print(f"\n📊 Distribution by Timeframe:")
    for tf, count in sorted(tf_counts.items()):
        print(f"   {tf}: {count} ({count/len(all_training_data)*100:.1f}%)")
    
    # Sample
    print(f"\n📋 Sample example (showing 2-bounce structure):")
    sample = all_training_data[0]
    print(f"   {sample['symbol']} {sample['timeframe']} | {sample['direction']} @ {sample['ma_level']}")
    print(f"   ❶ 1st Bounce (TEST - don't trade):  {sample['first_bounce_time']} @ {sample['first_bounce_price']}")
    print(f"   ❷ 2nd Bounce (ENTRY - we trade):    {sample['time']} @ {sample['entry_price']}")
    print(f"   ⏱️  Time between bounces: {sample['swing_duration_hours']:.1f} hours")
    print(f"   📊 Outcome: Profit={sample['max_profit_pips']:.1f}p, Loss={sample['max_loss_pips']:.1f}p, Ratio={sample['profit_loss_ratio']:.2f}")
    print(f"   ✅ Profitable: {sample['profitable']}")

print("\n" + "=" * 80)
print("✅ CELL 2 COMPLETE: Extracted training data")
print("=" * 80)
print(f"💾 Total examples: {len(all_training_data)}")
print(f"\n🔑 KEY REMINDER: Entry happens at the 2ND BOUNCE (retest)")
print(f"   • 1st bounce = test/setup (DON'T trade)")
print(f"   • 2nd bounce = entry/retest (TRADE HERE)")
print(f"\n💡 Next: Run CELL 3 to view detailed examples before training")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 3: VIEW & INSPECT TRAINING EXAMPLES
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("👁️  CELL 3: VIEW TRAINING EXAMPLES")
print("="*80)

if not all_training_data:
    print("\n❌ No training data found. Run CELL 2 first.")
else:
    print(f"\n📊 Total examples: {len(all_training_data)}")
    
    # Overall statistics
    profitable = sum(1 for ex in all_training_data if ex['profitable'])
    unprofitable = len(all_training_data) - profitable
    very_profitable = sum(1 for ex in all_training_data if ex['very_profitable'])
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"✅ Profitable (1.5:1+):      {profitable:>6} ({profitable/len(all_training_data)*100:>5.1f}%)")
    print(f"⭐ Very Profitable (2:1+):   {very_profitable:>6} ({very_profitable/len(all_training_data)*100:>5.1f}%)")
    print(f"❌ Unprofitable:             {unprofitable:>6} ({unprofitable/len(all_training_data)*100:>5.1f}%)")
    
    # MA distribution
    ma_counts = {}
    for ex in all_training_data:
        ma = ex['ma_level']
        ma_counts[ma] = ma_counts.get(ma, 0) + 1
    
    print(f"\n{'='*80}")
    print("DISTRIBUTION BY MA LEVEL")
    print(f"{'='*80}")
    for ma in ['MA18', 'MA50', 'MA200']:
        count = ma_counts.get(ma, 0)
        if count > 0:
            print(f"{ma}:  {count:>6} ({count/len(all_training_data)*100:>5.1f}%)")
    
    # Timeframe distribution
    tf_counts = {}
    for ex in all_training_data:
        tf = ex['timeframe']
        tf_counts[tf] = tf_counts.get(tf, 0) + 1
    
    print(f"\n{'='*80}")
    print("DISTRIBUTION BY TIMEFRAME")
    print(f"{'='*80}")
    for tf, count in sorted(tf_counts.items()):
        print(f"{tf:>6}:  {count:>6} ({count/len(all_training_data)*100:>5.1f}%)")
    
    # Symbol distribution
    symbol_counts = {}
    for ex in all_training_data:
        symbol = ex['symbol']
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    print(f"\n{'='*80}")
    print("DISTRIBUTION BY SYMBOL")
    print(f"{'='*80}")
    for symbol, count in sorted(symbol_counts.items()):
        print(f"{symbol:>10}:  {count:>6} ({count/len(all_training_data)*100:>5.1f}%)")
    
    # Show winning examples
    winning = [ex for ex in all_training_data if ex['profitable']]
    losing = [ex for ex in all_training_data if not ex['profitable']]
    
    print(f"\n{'='*80}")
    print("SAMPLE WINNING TRADES (First 5)")
    print(f"{'='*80}")
    for i, ex in enumerate(winning[:5], 1):
        print(f"\n{i}. {ex['symbol']} {ex['timeframe']} | {ex['direction']} @ {ex['ma_level']}")
        print(f"   ❶ 1st Bounce (TEST - DON'T TRADE): {ex['first_bounce_time']} @ {ex['first_bounce_price']:.5f}")
        print(f"   ❷ 2nd Bounce (ENTRY - TRADE HERE): {ex['time']} @ {ex['entry_price']:.5f}")
        print(f"      ⏱️  {ex['swing_duration_hours']:.1f} hours between bounces")
        print(f"      📊 Outcome: ✅ Profit={ex['max_profit_pips']:.1f}p | Loss={ex['max_loss_pips']:.1f}p | Ratio={ex['profit_loss_ratio']:.2f}")
    
    print(f"\n{'='*80}")
    print("SAMPLE LOSING TRADES (First 5)")
    print(f"{'='*80}")
    for i, ex in enumerate(losing[:5], 1):
        print(f"\n{i}. {ex['symbol']} {ex['timeframe']} | {ex['direction']} @ {ex['ma_level']}")
        print(f"   ❶ 1st Bounce (TEST - DON'T TRADE): {ex['first_bounce_time']} @ {ex['first_bounce_price']:.5f}")
        print(f"   ❷ 2nd Bounce (ENTRY - TRADE HERE): {ex['time']} @ {ex['entry_price']:.5f}")
        print(f"      ⏱️  {ex['swing_duration_hours']:.1f} hours between bounces")
        print(f"      📊 Outcome: ❌ Profit={ex['max_profit_pips']:.1f}p | Loss={ex['max_loss_pips']:.1f}p | Ratio={ex['profit_loss_ratio']:.2f}")
    
    print(f"\n{'='*80}")
    print("DATA QUALITY CHECK")
    print(f"{'='*80}")
    print(f"✓ All MAs represented: {len(ma_counts) == 3}")
    print(f"✓ Multiple timeframes: {len(tf_counts) > 1}")
    print(f"✓ Balanced win/loss: {0.3 < profitable/len(all_training_data) < 0.7}")
    print(f"✓ Sufficient examples: {len(all_training_data) >= 100}")

print("\n" + "=" * 80)
print("✅ CELL 3 COMPLETE: Reviewed examples")
print("=" * 80)
print(f"\n💡 Next: Run CELL 4 to train DL model (if data looks good)")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 4: TRAIN DEEP LEARNING MODEL
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧠 CELL 4: TRAIN DEEP LEARNING MODEL")
print("="*80)

if not all_training_data:
    print("❌ No training data - skipping model training")
else:
    # Prepare features and labels
    print("\n📊 Preparing training data...")
    
    features = []
    labels = []
    
    for ex in all_training_data:
        # Features: MAs, slopes, ATR, RSI, Force Index, correlation, HTF, Fib, patterns, MA distances
        feature_vector = [
            # Moving averages
            ex['ma6'], ex['ma18'], ex['ma50'], ex['ma200'],
            # MA slopes
            ex['slope6'], ex['slope18'], ex['slope50'], ex['slope200'],
            # Volatility
            ex['atr'],
            # RSI & Force Index
            ex['rsi'], ex['force_index'],
            # Correlation (trend strength) & HTF confirmation
            ex['correlation'], ex['htf_trend_aligned'],
            # Fibonacci retracement (validated <= 61.8%)
            ex['fib_retracement'], ex['fib_valid'],
            # Candlestick patterns (ONLY at S/R level - from validation.py)
            ex['pattern_at_sr'], ex['has_rejection'], ex['has_pin_bar'], ex['has_engulfing'],
            ex['has_soldier_crow'], ex['has_tweezer'], ex['has_star'],
            ex['has_any_pattern'],
            # Distance from MAs
            ex['dist_ma18'], ex['dist_ma50'], ex['dist_ma200'],
            ex['at_ma18'], ex['at_ma50'], ex['at_ma200'],
            # Timing
            ex['swing_duration_hours'],
            # Direction encoding
            1 if ex['direction'] == 'BUY' else 0,
            # MA level encoding
            1 if ex['ma_level'] == 'MA18' else (2 if ex['ma_level'] == 'MA50' else 3)
        ]
        features.append(feature_vector)
        
        # Label: profitable or not
        labels.append(1 if ex['profitable'] else 0)
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"   Features shape: {X.shape}")
    print(f"   Labels: {sum(labels)} profitable, {len(labels) - sum(labels)} unprofitable")
    
    # Calculate class weights for balanced training
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive
    total = len(labels)
    
    # Aggressive weight for minority class (multiply by 3 to catch more winners)
    pos_weight = (total / (2 * num_positive)) * 3.0 if num_positive > 0 else 1.0
    neg_weight = total / (2 * num_negative) if num_negative > 0 else 1.0
    
    print(f"   Class weights: positive={pos_weight:.2f}, negative={neg_weight:.2f}")
    print(f"   Class distribution: {num_positive} profitable ({num_positive/total*100:.1f}%), {num_negative} unprofitable ({num_negative/total*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    print(f"\n   Train set: {len(X_train)} examples")
    print(f"   Test set: {len(X_test)} examples")
    
    # Define neural network (BIGGER to fix HIGH BIAS)
    class TCEClassifier(nn.Module):
        def __init__(self, input_size):
            super(TCEClassifier, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),  # Increased from 64
                nn.ReLU(),
                nn.Dropout(0.2),  # Reduced from 0.3 (less regularization for underfitting)
                nn.Linear(128, 64),  # Added extra layer
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
                # Note: No Sigmoid here - using BCEWithLogitsLoss which includes it
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    model = TCEClassifier(input_size=X_train.shape[1]).to(device)
    
    # Use weighted BCE loss to handle class imbalance
    pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n🏋️  Training model...")
    epochs = 100  # Increased from 50 to fix underfitting
    batch_size = 32
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_logits = model(X_train)
                train_pred = torch.sigmoid(train_logits)
                train_acc = ((train_pred > 0.5).float() == y_train).float().mean()
                
                test_logits = model(X_test)
                test_pred = torch.sigmoid(test_logits)
                test_acc = ((test_pred > 0.5).float() == y_test).float().mean()
            
            print(f"   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.4f} | "
                  f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = torch.sigmoid(test_logits)
        
        # Find optimal threshold by maximizing F1 score
        print(f"\n🎯 Finding optimal decision threshold...")
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred_labels = (test_pred > threshold).float()
            tp = ((pred_labels == 1) & (y_test == 1)).sum().item()
            fp = ((pred_labels == 1) & (y_test == 0)).sum().item()
            fn = ((pred_labels == 0) & (y_test == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': precision, 'recall': recall, 'f1': f1
                }
        
        print(f"   Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")
        
        # Use optimal threshold for final evaluation
        pred_labels = (test_pred > best_threshold).float()
        test_acc = ((pred_labels == y_test).float()).mean()
        
        true_positives = best_metrics['tp']
        false_positives = best_metrics['fp']
        false_negatives = best_metrics['fn']
        true_negatives = ((pred_labels == 0) & (y_test == 0)).sum().item()
        
        precision = best_metrics['precision']
        recall = best_metrics['recall']
        f1 = best_metrics['f1']
    
    print(f"\n📊 Final Model Performance (threshold={best_threshold:.2f}):")
    print(f"   Accuracy: {test_acc:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Positives:  {int(true_positives):>6} (correctly predicted profitable)")
    print(f"   False Positives: {int(false_positives):>6} (incorrectly predicted profitable)")
    print(f"   True Negatives:  {int(true_negatives):>6} (correctly predicted unprofitable)")
    print(f"   False Negatives: {int(false_negatives):>6} (missed profitable trades)")
    
    total_profitable = int(true_positives + false_negatives)
    total_unprofitable = int(true_negatives + false_positives)
    print(f"\n📊 Trade Capture:")
    print(f"   Caught {int(true_positives)}/{total_profitable} profitable trades ({recall*100:.1f}%)")
    print(f"   Avoided {int(true_negatives)}/{total_unprofitable} unprofitable trades ({true_negatives/(true_negatives + false_positives)*100:.1f}%)")
    
    print(f"\n💡 Interpretation:")
    if recall < 0.5:
        print(f"   ⚠️  Low recall ({recall:.1%}) - model is missing many profitable trades")
        print(f"   → Consider: More training data, different features, or lower threshold")
    elif recall >= 0.5 and recall < 0.7:
        print(f"   ✅ Moderate recall ({recall:.1%}) - catching half of profitable trades")
    else:
        print(f"   ✅ Good recall ({recall:.1%}) - catching most profitable trades")
    
    if precision < 0.5:
        print(f"   ⚠️  Low precision ({precision:.1%}) - many false positives")
        print(f"   → Model predictions have high false alarm rate")
    elif precision >= 0.5 and precision < 0.7:
        print(f"   ✅ Moderate precision ({precision:.1%}) - reasonable accuracy")
    else:
        print(f"   ✅ Good precision ({precision:.1%}) - few false positives")
    
    if f1 > 0.6:
        print(f"   ✅✅ Strong F1 score ({f1:.1%}) - balanced performance")
    elif f1 > 0.4:
        print(f"   ✅ Acceptable F1 score ({f1:.1%}) - usable model")
    else:
        print(f"   ⚠️  Low F1 score ({f1:.1%}) - needs improvement")
    
    # Save model with optimal threshold
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'optimal_threshold': best_threshold,
        'feature_names': ['ma6', 'ma18', 'ma50', 'ma200', 
                         'slope6', 'slope18', 'slope50', 'slope200',
                         'atr', 'rsi', 'force_index',
                         'correlation', 'htf_trend_aligned',
                         'fib_retracement', 'fib_valid',
                         'pattern_at_sr', 'has_rejection', 'has_pin_bar', 'has_engulfing',
                         'has_soldier_crow', 'has_tweezer', 'has_star', 'has_any_pattern',
                         'dist_ma18', 'dist_ma50', 'dist_ma200',
                         'at_ma18', 'at_ma50', 'at_ma200',
                         'swing_duration', 'direction', 'ma_level']
    }, 'tce_model.pth')
    
    print(f"✅ Model saved as tce_model.pth (optimal threshold: {best_threshold:.2f})")
    
    # ============================================================================
    # BIAS-VARIANCE DIAGNOSTICS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("🔬 BIAS-VARIANCE DIAGNOSTICS")
    print("=" * 80)
    
    # Evaluate on both train and test sets at optimal threshold
    model.eval()
    with torch.no_grad():
        # Train set performance
        train_logits = model(X_train)
        train_pred = torch.sigmoid(train_logits)
        train_pred_labels = (train_pred > best_threshold).float()
        
        train_tp = ((train_pred_labels == 1) & (y_train == 1)).sum().item()
        train_fp = ((train_pred_labels == 1) & (y_train == 0)).sum().item()
        train_fn = ((train_pred_labels == 0) & (y_train == 1)).sum().item()
        
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        train_acc = ((train_pred_labels == y_train).float()).mean().item()
    
    # Compare train vs test
    print(f"\n📊 Train vs Test Performance (at threshold={best_threshold:.2f}):")
    print(f"\n{'Metric':<15} {'Train':<12} {'Test':<12} {'Difference':<12}")
    print(f"{'-'*51}")
    print(f"{'Accuracy':<15} {train_acc:<12.3f} {test_acc.item():<12.3f} {abs(train_acc - test_acc.item()):<12.3f}")
    print(f"{'Precision':<15} {train_precision:<12.3f} {precision:<12.3f} {abs(train_precision - precision):<12.3f}")
    print(f"{'Recall':<15} {train_recall:<12.3f} {recall:<12.3f} {abs(train_recall - recall):<12.3f}")
    print(f"{'F1 Score':<15} {train_f1:<12.3f} {f1:<12.3f} {abs(train_f1 - f1):<12.3f}")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    
    gap = abs(train_f1 - f1)
    train_perf = train_f1
    test_perf = f1
    
    if gap < 0.05:
        print(f"   ✅ Low train-test gap ({gap:.3f}) - Good generalization")
        if test_perf < 0.6:
            print(f"   ⚠️  But test F1 is low ({test_perf:.3f}) - HIGH BIAS (underfitting)")
            print(f"   → Model too simple to capture patterns")
        else:
            print(f"   ✅ Test F1 is good ({test_perf:.3f}) - Well-balanced model")
    elif gap >= 0.05 and gap < 0.15:
        print(f"   ⚠️  Moderate train-test gap ({gap:.3f}) - Some overfitting")
        print(f"   → Model memorizing training data slightly")
    else:
        print(f"   ❌ Large train-test gap ({gap:.3f}) - HIGH VARIANCE (overfitting)")
        print(f"   → Model memorizing training data instead of learning patterns")
    
    # Detailed recommendations
    print(f"\n💡 Recommendations to Improve:")
    
    if gap >= 0.15:  # High variance
        print(f"\n   🎯 HIGH VARIANCE DETECTED - To reduce overfitting:")
        print(f"      1. Add more training data (currently {len(X_train)} examples)")
        print(f"      2. Increase dropout (currently 0.3 → try 0.5)")
        print(f"      3. Reduce model complexity (fewer layers/neurons)")
        print(f"      4. Add L2 regularization (weight_decay in optimizer)")
        print(f"      5. Use early stopping based on validation loss")
        
    if test_perf < 0.6 and gap < 0.1:  # High bias
        print(f"\n   🎯 HIGH BIAS DETECTED - To reduce underfitting:")
        print(f"      1. Increase model complexity (add layers: 64→128→64→32→16)")
        print(f"      2. Add more features (e.g., RSI, volume, price patterns)")
        print(f"      3. Train longer (currently {epochs} epochs → try 100-200)")
        print(f"      4. Lower dropout (currently 0.3 → try 0.2)")
        print(f"      5. Use different activation functions (try LeakyReLU, ELU)")
    
    if precision < 0.5:  # Low precision
        print(f"\n   🎯 LOW PRECISION - To reduce false positives:")
        print(f"      1. Add quality filters (e.g., require strong trend, low volatility)")
        print(f"      2. Engineer better features (distance from MA, trend strength)")
        print(f"      3. Collect more negative examples (losing trades)")
        print(f"      4. Try ensemble methods (train multiple models, vote)")
        print(f"      5. Increase threshold (currently {best_threshold:.2f} → try 0.65-0.70)")
    
    if recall < 0.7:  # Low recall
        print(f"\n   🎯 LOW RECALL - To catch more winners:")
        print(f"      1. Increase positive class weight (currently {pos_weight:.2f} → try 4-5x)")
        print(f"      2. Lower threshold (currently {best_threshold:.2f} → try 0.4-0.5)")
        print(f"      3. Add more positive examples (winning trades)")
        print(f"      4. Use SMOTE or data augmentation for minority class")
    
    # General improvements
    print(f"\n   📈 GENERAL IMPROVEMENTS:")
    print(f"      1. **More Data**: Collect 6-12 months across multiple pairs/timeframes")
    print(f"      2. **Better Features**:")
    print(f"         • Price action: candle patterns, wicks, body size")
    print(f"         • Volatility: ATR ratio, Bollinger Bands")
    print(f"         • Momentum: RSI, MACD, Stochastic")
    print(f"         • Volume: if available from broker")
    print(f"         • Time of day: London/NY session indicators")
    print(f"      3. **Different Models**:")
    print(f"         • Try LightGBM, XGBoost (often better for tabular data)")
    print(f"         • Try LSTM/GRU for sequential patterns")
    print(f"         • Try ensemble: combine multiple models")
    print(f"      4. **Cross-Validation**: Use k-fold CV to verify generalization")
    print(f"      5. **Hyperparameter Tuning**: Grid search learning rate, batch size, architecture")
    
    # Feature importance (approximate with gradient magnitudes)
    print(f"\n📊 Feature Importance (gradient magnitude):")
    model.eval()
    X_sample = X_train[:1000].clone().requires_grad_(True)
    output = model(X_sample)
    output.sum().backward()
    
    feature_importance = X_sample.grad.abs().mean(dim=0).cpu().numpy()
    feature_names = ['ma6', 'ma18', 'ma50', 'ma200', 'slope6', 'slope18', 
                     'slope50', 'slope200', 'atr', 'rsi', 'force_index',
                     'correlation', 'htf_trend_aligned',
                     'fib_retracement', 'fib_valid',
                     'pattern_at_sr', 'has_rejection', 'has_pin_bar', 'has_engulfing',
                     'has_soldier_crow', 'has_tweezer', 'has_star', 'has_any_pattern',
                     'dist_ma18', 'dist_ma50', 'dist_ma200',
                     'at_ma18', 'at_ma50', 'at_ma200',
                     'swing_hours', 'direction', 'ma_level']
    
    sorted_idx = np.argsort(feature_importance)[::-1]
    print(f"\n   Top 5 most important features:")
    for i, idx in enumerate(sorted_idx[:5], 1):
        print(f"      {i}. {feature_names[idx]:<15} (importance: {feature_importance[idx]:.4f})")

print("\n" + "=" * 80)
print("✅ CELL 4 COMPLETE: Model trained & analyzed")
print("=" * 80)
print(f"\n💡 Next: Run CELL 5 to download results")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 5: DOWNLOAD RESULTS
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("⬇️  CELL 5: DOWNLOAD RESULTS")
print("="*80)

if all_training_data:
    print("\n📥 Downloading files...")
    files.download('training_data.json')
    print("✅ Downloaded: training_data.json")
    
    if os.path.exists('tce_model.pth'):
        files.download('tce_model.pth')
        print("✅ Downloaded: tce_model.pth")

print("\n" + "=" * 80)
print("🎉 COMPLETE! PIPELINE FINISHED SUCCESSFULLY")
print("=" * 80)
print("\nKey Points:")
print("  • Entry is ALWAYS at 2nd bounce (retest)")
print("  • 1st bounce = test/setup, 2nd bounce = entry signal")
print("  • Both winners and losers included for balanced training")
print("  • Models will learn which patterns work from features")
