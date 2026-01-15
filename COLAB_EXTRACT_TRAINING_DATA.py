"""
COLAB CELL: Extract Training Data for DL/RL Models
====================================================

Upload this to Google Colab and run to generate training examples
from your MT5 CSV files.

Steps:
1. Upload your CSV files (USDJPY_H1.csv, etc.)
2. Run this cell
3. Download the generated training_data.json

The script extracts MA bounce patterns with minimal filtering,
letting your DL/RL models learn optimal patterns from labeled examples.
"""

import pandas as pd
import numpy as np
import json
from google.colab import files

def calculate_indicators(df):
    """Calculate MAs and ATR for feature extraction"""
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
    
    return df


def basic_trend_direction(row):
    """Simple trend check for BUY/SELL labeling"""
    if row['ma6'] > row['ma18'] > row['ma50']:
        return 'BUY'
    elif row['ma6'] < row['ma18'] < row['ma50']:
        return 'SELL'
    return None


def detect_ma_bounce(row, direction, pip_size=0.01):
    """Detect if candle bounced from any MA (18, 50, 200)"""
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
    
    # Return closest MA (prefer MA50 > MA18 > MA200)
    if not bounces:
        return None
    
    if 'MA50' in bounces:
        return 'MA50'
    if 'MA18' in bounces:
        return 'MA18'
    return 'MA200'


def extract_training_examples(df, symbol='USDJPY', timeframe='H1', pip_size=0.01):
    """Extract all MA bounce retests as training examples"""
    
    df = calculate_indicators(df)
    
    training_data = []
    ma_bounces = {'MA18': [], 'MA50': [], 'MA200': []}
    
    print(f"\n🔍 Extracting from {symbol} {timeframe}...")
    print("="*80)
    
    for i in range(250, len(df)):
        if pd.isna(df.iloc[i]['ma200']) or pd.isna(df.iloc[i]['atr']):
            continue
        
        row = df.iloc[i]
        
        # Get trend direction
        trend = basic_trend_direction(row)
        if not trend:
            continue
        
        # Detect MA bounce
        ma_touched = detect_ma_bounce(row, trend, pip_size)
        if not ma_touched:
            continue
        
        # Record bounce
        bounce_price = row['low'] if trend == 'BUY' else row['high']
        ma_bounces[ma_touched].append({
            'index': i,
            'time': row['time'],
            'price': bounce_price,
            'direction': trend
        })
        
        # Look for retest (previous bounce of same MA/direction)
        # IMPORTANT: First bounce = test, Second bounce (current) = entry point
        recent_bounces = [
            b for b in ma_bounces[ma_touched]
            if b['index'] < i 
            and b['index'] > i - 100
            and b['direction'] == trend
        ]
        
        if len(recent_bounces) == 0:
            continue
        
        first_bounce = recent_bounces[-1]  # This is the TEST bounce (we don't trade here)
        # Current bounce (i) is the RETEST/ENTRY (we trade here)
        
        # Calculate time between bounces
        time_diff_hours = (row['time'] - first_bounce['time']).total_seconds() / 3600
        
        # Minimum 4 hours between bounces
        if time_diff_hours < 4:
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
        
        if len(training_data) % 50 == 0:
            print(f"  Extracted {len(training_data)} examples...")
    
    print(f"✅ Extracted {len(training_data)} examples from {symbol} {timeframe}")
    return training_data


def load_csv_file(filename):
    """Load CSV file with proper time parsing"""
    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'])
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("📁 Upload your CSV files (USDJPY_H1.csv, EURUSD_H4.csv, etc.)")
print("=" * 80)

uploaded = files.upload()

all_training_data = []

# Process each uploaded file
for filename in uploaded.keys():
    print(f"\n📊 Processing {filename}...")
    
    # Parse symbol and timeframe from filename
    # Expected format: SYMBOL_TIMEFRAME.csv (e.g., USDJPY_H1.csv)
    parts = filename.replace('.csv', '').split('_')
    if len(parts) >= 2:
        symbol = parts[0]
        timeframe = parts[1]
    else:
        symbol = filename.replace('.csv', '')
        timeframe = 'UNKNOWN'
    
    # Determine pip size (JPY pairs use 0.01, others use 0.0001)
    pip_size = 0.01 if 'JPY' in symbol else 0.0001
    
    # Load data
    df = load_csv_file(filename)
    print(f"  Loaded {len(df)} candles")
    
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
    print(f"\n📊 Statistics:")
    print(f"   Total: {len(all_training_data)}")
    print(f"   Profitable (1.5:1+): {profitable} ({profitable/len(all_training_data)*100:.1f}%)")
    print(f"   Unprofitable: {len(all_training_data) - profitable} ({(len(all_training_data) - profitable)/len(all_training_data)*100:.1f}%)")
    
    # Sample
    print(f"\n📋 Sample example:")
    print(json.dumps(all_training_data[0], indent=2))
    
    # Download
    print(f"\n⬇️ Downloading {output_filename}...")
    files.download(output_filename)

print("\n✅ Complete! Use this data to train your DL/RL models.")
