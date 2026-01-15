"""
Debug TCE Validation - Show Step-by-Step Rejections
====================================================

This will show us exactly which validation step is rejecting setups
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from trading.tce.types import Candle, Indicators, Swing, MarketStructure, HigherTFCandle
from trading.tce.validation import validate_tce
from trading.tce.utils import valid_fib, has_candlestick_confirmation
from trading.tce.sr import at_ma_level, has_ma_retest
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend


def initialize_mt5():
    if not mt5.initialize():
        return False
    print("✅ MT5 connected")
    return True


def download_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_indicators(df):
    df['ma6'] = df['close'].rolling(window=6).mean()
    df['ma18'] = df['close'].rolling(window=18).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    for period in [6, 18, 50, 200]:
        df[f'slope{period}'] = df[f'ma{period}'].diff()
    
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    return df


def find_swing_lows(df, lookback=5):
    swings = []
    for i in range(lookback, len(df) - lookback):
        low = df.iloc[i]['low']
        is_swing = all(df.iloc[i-j]['low'] > low and df.iloc[i+j]['low'] > low 
                       for j in range(1, lookback + 1))
        if is_swing:
            swings.append({'index': i, 'price': low})
    return swings


def find_swing_highs(df, lookback=5):
    swings = []
    for i in range(lookback, len(df) - lookback):
        high = df.iloc[i]['high']
        is_swing = all(df.iloc[i-j]['high'] < high and df.iloc[i+j]['high'] < high 
                       for j in range(1, lookback + 1))
        if is_swing:
            swings.append({'index': i, 'price': high})
    return swings


def debug_validation(df, htf_df, symbol='USDJPY'):
    """
    Test each validation step separately to see where setups are failing
    """
    
    df = calculate_indicators(df)
    htf_df = calculate_indicators(htf_df)
    
    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    
    print(f"\n📊 Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
    
    # Counters for each validation step
    step_counts = {
        'total_checked': 0,
        'has_swing': 0,
        'trend_ok': 0,
        'fib_ok': 0,
        'at_ma': 0,
        'has_retest': 0,
        'has_candle_confirmation': 0,
        'structure_ok': 0,
        'htf_ok': 0,
        'fully_valid': 0
    }
    
    # Store some examples
    examples = []
    
    for i in range(250, len(df) - 10):
        if pd.isna(df.iloc[i]['ma200']) or pd.isna(df.iloc[i]['atr']):
            continue
        
        step_counts['total_checked'] += 1
        row = df.iloc[i]
        
        # Find swing
        swing = None
        direction = None
        
        # Check for BUY (uptrend)
        if row['ma18'] > row['ma50'] > row['ma200']:
            direction = 'BUY'
            for sl in reversed(swing_lows):
                if sl['index'] < i and sl['index'] > i - 50:
                    swing_low = sl['price']
                    swing_high = df.iloc[sl['index']:i]['high'].max()
                    fib_range = swing_high - swing_low
                    
                    if fib_range > 0:
                        retracement = (swing_high - row['low']) / fib_range
                        
                        # Check if price is at valid Fib zone
                        if 0.33 <= retracement <= 0.43:  # 38.2%
                            fib_level = 0.382
                        elif 0.45 <= retracement <= 0.55:  # 50%
                            fib_level = 0.5
                        elif 0.57 <= retracement <= 0.67:  # 61.8%
                            fib_level = 0.618
                        else:
                            continue  # Not at valid Fib
                        
                        fib_618_price = swing_high - (fib_range * 0.618)
                        
                        swing = Swing(
                            type='low',
                            price=swing_low,
                            fib_level=fib_level,
                            fib_618_price=fib_618_price
                        )
                        break
        
        # Check for SELL (downtrend)
        elif row['ma18'] < row['ma50'] < row['ma200']:
            direction = 'SELL'
            for sh in reversed(swing_highs):
                if sh['index'] < i and sh['index'] > i - 50:
                    swing_high = sh['price']
                    swing_low = df.iloc[sh['index']:i]['low'].min()
                    fib_range = swing_high - swing_low
                    
                    if fib_range > 0:
                        retracement = (row['high'] - swing_low) / fib_range
                        
                        if 0.33 <= retracement <= 0.43:
                            fib_level = 0.382
                        elif 0.45 <= retracement <= 0.55:
                            fib_level = 0.5
                        elif 0.57 <= retracement <= 0.67:
                            fib_level = 0.618
                        else:
                            continue
                        
                        fib_618_price = swing_low + (fib_range * 0.618)
                        
                        swing = Swing(
                            type='high',
                            price=swing_high,
                            fib_level=fib_level,
                            fib_618_price=fib_618_price
                        )
                        break
        
        if swing is None:
            continue
        
        step_counts['has_swing'] += 1
        
        # Create objects
        candle = Candle(
            timestamp=row['time'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close']
        )
        
        indicators = Indicators(
            ma6=row['ma6'], ma18=row['ma18'], ma50=row['ma50'], ma200=row['ma200'],
            slope6=row['slope6'], slope18=row['slope18'], slope50=row['slope50'], slope200=row['slope200'],
            atr=row['atr']
        )
        
        structure = MarketStructure(
            highs=df.iloc[max(0, i-50):i]['high'].tolist(),
            lows=df.iloc[max(0, i-50):i]['low'].tolist()
        )
        
        recent_candles = []
        for j in range(max(0, i-20), i+1):
            r = df.iloc[j]
            recent_candles.append(Candle(r['time'], r['open'], r['high'], r['low'], r['close']))
        
        # Step 1: Trend (basic MA check already done)
        step_counts['trend_ok'] += 1
        
        # Step 2: Fib
        if valid_fib(swing):
            step_counts['fib_ok'] += 1
        else:
            continue
        
        # Step 3: At MA level
        if at_ma_level(candle, indicators, direction):
            step_counts['at_ma'] += 1
        else:
            continue
        
        # Step 4: MA retest
        if has_ma_retest(recent_candles, indicators, direction, "H1"):
            step_counts['has_retest'] += 1
        else:
            continue
        
        # Step 5: Candlestick confirmation
        if has_candlestick_confirmation(recent_candles, direction):
            step_counts['has_candle_confirmation'] += 1
        else:
            continue
        
        # Step 6: Structure
        if direction == 'BUY':
            struct_ok = is_valid_uptrend(indicators, structure)
        else:
            struct_ok = is_valid_downtrend(indicators, structure)
        
        if struct_ok:
            step_counts['structure_ok'] += 1
        else:
            if len(examples) < 5:
                examples.append({
                    'time': row['time'],
                    'direction': direction,
                    'failed_at': 'structure',
                    'ma18': row['ma18'],
                    'ma50': row['ma50'],
                    'ma200': row['ma200']
                })
            continue
        
        # If we got here, the setup passed all basic checks!
        step_counts['fully_valid'] += 1
        
        if len(examples) < 5:
            examples.append({
                'time': row['time'],
                'direction': direction,
                'failed_at': 'PASSED',
                'entry': row['close']
            })
    
    # Print results
    print("\n" + "="*80)
    print("📊 VALIDATION FUNNEL")
    print("="*80)
    print(f"Total candles checked:        {step_counts['total_checked']:6d}")
    print(f"  ↓ Has swing + direction:    {step_counts['has_swing']:6d} ({step_counts['has_swing']/max(1,step_counts['total_checked'])*100:.1f}%)")
    print(f"  ↓ Trend OK (MA alignment):  {step_counts['trend_ok']:6d} ({step_counts['trend_ok']/max(1,step_counts['has_swing'])*100:.1f}%)")
    print(f"  ↓ Valid Fibonacci:          {step_counts['fib_ok']:6d} ({step_counts['fib_ok']/max(1,step_counts['trend_ok'])*100:.1f}%)")
    print(f"  ↓ At MA level:              {step_counts['at_ma']:6d} ({step_counts['at_ma']/max(1,step_counts['fib_ok'])*100:.1f}%)")
    print(f"  ↓ MA retest:                {step_counts['has_retest']:6d} ({step_counts['has_retest']/max(1,step_counts['at_ma'])*100:.1f}%)")
    print(f"  ↓ Candlestick confirmation: {step_counts['has_candle_confirmation']:6d} ({step_counts['has_candle_confirmation']/max(1,step_counts['has_retest'])*100:.1f}%)")
    print(f"  ↓ Structure validation:     {step_counts['structure_ok']:6d} ({step_counts['structure_ok']/max(1,step_counts['has_candle_confirmation'])*100:.1f}%)")
    print(f"  ✅ FULLY VALID:             {step_counts['fully_valid']:6d}")
    print("="*80)
    
    if examples:
        print("\n📋 EXAMPLE SETUPS:")
        for ex in examples:
            print(f"  {ex['time']} | {ex['direction']:4s} | Failed at: {ex['failed_at']}")


def main():
    print("="*80)
    print("🔍 TCE VALIDATION DEBUGGER")
    print("="*80)
    
    if not initialize_mt5():
        return
    
    try:
        SYMBOL = 'USDJPY'
        
        print(f"\n📥 Downloading {SYMBOL} H1...")
        df = download_data(SYMBOL, mt5.TIMEFRAME_H1, 10000)
        
        print(f"📥 Downloading {SYMBOL} H4...")
        htf_df = download_data(SYMBOL, mt5.TIMEFRAME_H4, 3000)
        
        if df is None or htf_df is None:
            return
        
        debug_validation(df, htf_df, SYMBOL)
        
    finally:
        mt5.shutdown()
        print("\n✅ Complete")


if __name__ == "__main__":
    main()
