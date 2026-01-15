"""
TCE Setup Detection - CORRECT LOGIC
====================================

TCE Requirements:
1. Clear trend: MA6 > MA18 > MA50 > MA200, all sloping up (or down for sell)
2. First swing: Price retraces to a specific MA and bounces
3. Makes HH (buy) or LL (sell) - trend continues
4. Second swing: Price retraces AGAIN to SAME MA (retest)
5. Enter on this retest with confirmation

This is what "MA retest" actually means!
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Ensure project modules are importable (trading.tce.*)
sys.path.insert(0, str(Path(__file__).parent))

from trading.tce.types import Candle, Indicators, Swing, MarketStructure, HigherTFCandle
from trading.tce.validation import validate_tce


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


def download_higher_timeframe(symbol, timeframe, bars):
    """Download higher timeframe data for multi-TF confirmation."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_indicators(df):
    """Calculate MAs and slopes"""
    df['ma6'] = df['close'].rolling(window=6).mean()
    df['ma18'] = df['close'].rolling(window=18).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    # Calculate slopes (5-period rate of change)
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


def is_clear_uptrend(row):
    """
    UPTREND: MA6 > MA18 > MA50 > MA200 AND all sloping UP
    """
    ma_order = (row['ma6'] > row['ma18'] > row['ma50'] > row['ma200'])
    all_sloping_up = (row['slope6'] > 0 and row['slope18'] > 0 and 
                      row['slope50'] > 0 and row['slope200'] > 0)
    return ma_order and all_sloping_up


def is_clear_downtrend(row):
    """
    DOWNTREND: MA6 < MA18 < MA50 < MA200 AND all sloping DOWN
    """
    ma_order = (row['ma6'] < row['ma18'] < row['ma50'] < row['ma200'])
    all_sloping_down = (row['slope6'] < 0 and row['slope18'] < 0 and 
                        row['slope50'] < 0 and row['slope200'] < 0)
    return ma_order and all_sloping_down


def which_ma_touched(row, direction, tolerance_multiplier=0.5):
    """Determine which MA (18/50/200) price is actually bouncing from.

    Key change:
    - We now choose the MA whose price is *closest* to the candle (true bounce),
      not the one with the biggest "penetration" distance.

    For BUY: price trades around/below MA and closes near/above it (support).
    For SELL: price trades around/above MA and closes near/below it (resistance).

    Returns: 'MA18', 'MA50', 'MA200', or None.
    """

    if pd.isna(row['atr']) or row['atr'] <= 0:
        return None

    # Use a very tight fixed tolerance (2 pips) for actual MA contact
    # instead of ATR-based which can be too loose
    pip_size = 0.01 if 'JPY' in str(row.get('symbol', '')) else 0.0001
    tol = 2 * pip_size  # 2 pips tolerance
    mas_tested = []

    if direction == 'BUY':
        # Support bounce: use LOW wick vs MA (where price actually tags support)

        # MA18
        if row['low'] <= row['ma18'] + tol and row['close'] >= row['ma18']:
            distance = abs(row['low'] - row['ma18'])
            mas_tested.append(('MA18', distance))

        # MA50
        if row['low'] <= row['ma50'] + tol and row['close'] >= row['ma50']:
            distance = abs(row['low'] - row['ma50'])
            mas_tested.append(('MA50', distance))

        # MA200
        if row['low'] <= row['ma200'] + tol and row['close'] >= row['ma200']:
            distance = abs(row['low'] - row['ma200'])
            mas_tested.append(('MA200', distance))

    else:  # SELL
        # Resistance bounce: use HIGH wick vs MA (where price actually tags resistance)

        # MA18
        if row['high'] >= row['ma18'] - tol and row['close'] <= row['ma18']:
            distance = abs(row['high'] - row['ma18'])
            mas_tested.append(('MA18', distance))

        # MA50
        if row['high'] >= row['ma50'] - tol and row['close'] <= row['ma50']:
            distance = abs(row['high'] - row['ma50'])
            mas_tested.append(('MA50', distance))

        # MA200
        if row['high'] >= row['ma200'] - tol and row['close'] <= row['ma200']:
            distance = abs(row['high'] - row['ma200'])
            mas_tested.append(('MA200', distance))

    if not mas_tested:
        return None

    # Choose the MA the wick is *closest* to (smallest distance)
    mas_tested.sort(key=lambda x: x[1])
    best_ma, best_dist = mas_tested[0]

    # If MA18 is only slightly closer than MA50, prefer MA50 as primary level
    if best_ma == 'MA18':
        for name, dist in mas_tested[1:]:
            if name == 'MA50' and dist <= best_dist + tol * 0.25:
                return 'MA50'

    return best_ma


def get_swing_time_requirements(ma_name, timeframe):
    """
    Get minimum and maximum time (in hours) between MA bounces for valid swing
    
    M15:
      MA18: 1-4 hours
      MA50: 4-12 hours
    
        H1:
            MA18: 4-16 hours
            MA50: 24-72 hours (up to 3 days)
    
        H4:
            MA18: 16-64 hours
            MA50: 96-288 hours (up to 12 days)
    """
    requirements = {
        'M15': {
            'MA18': (1, 4),
            'MA50': (4, 12),
            'MA200': (12, 48)
        },
        'H1': {
            'MA18': (4, 16),
            'MA50': (24, 72),
            'MA200': (60, 120)
        },
        'H4': {
            'MA18': (16, 64),
            'MA50': (96, 288),
            'MA200': (168, 336)  # 1-2 weeks
        }
    }
    
    return requirements.get(timeframe, {}).get(ma_name, (4, 24))


def get_primary_ma_lookback(timeframe: str) -> int:
    """Candles to look back for primary MA50 bounce before allowing MA18.

    H1: 100 candles
    H4: 400 candles (4× H1)
    Default: 50 candles for other timeframes.
    """
    if timeframe == 'H1':
        return 100
    if timeframe == 'H4':
        return 400
    return 50


def detect_tce_setups(df, symbol='USDJPY', timeframe='H1', htf_df=None):
    """
    Detect TCE setups with CORRECT logic:
    1. Find clear trend
    2. Find first MA bounce (swing 1)
    3. Confirm price made HH/LL
    4. Find second bounce at SAME MA (swing 2 = entry)
    5. CHECK TIME DURATION between bounces (validates real swing)
    """
    
    df = calculate_indicators(df)

    # Prepare higher timeframe indicators if provided
    if htf_df is not None:
        htf_df = calculate_indicators(htf_df)
    
    setups = []

    # Pip size approximation for HH/LL distance checks
    pip_size = 0.01 if symbol.endswith('JPY') else 0.0001
    
    # Track MA bounces for each MA
    ma_bounces = {
        'MA18': [],
        'MA50': [],
        'MA200': []
    }
    
    print("\n🔍 Scanning for TCE setups...")
    print("="*80)

    def get_swing_segment_for_ma(ma_name: str, direction: str, current_index: int, max_lookback: int = 100):
        """Return explicit swing segment for a given MA and direction.

        A swing is defined as the move from:
        - start_index: previous touch of the same MA in the same direction
        - end_index: current touch of that MA (retest / potential entry)

        We only consider bounces within a recent lookback window so that
        very old touches don't create unrealistic swing durations.
        """

        candidates = [
            b for b in ma_bounces[ma_name]
            if b['direction'] == direction
            and b['index'] < current_index
            and b['index'] > current_index - max_lookback
        ]
        if not candidates:
            return None

        first_bounce = candidates[-1]
        return {
            'start_index': first_bounce['index'],
            'end_index': current_index,
            'first_bounce': first_bounce,
        }
    
    # Debug counters
    ma_touch_counts = {'MA18': 0, 'MA50': 0, 'MA200': 0}
    ma_valid_setups = {'MA18': 0, 'MA50': 0, 'MA200': 0}
    validation_passed = 0
    ma18_skipped_for_50_buy = 0
    ma18_skipped_for_50_sell = 0

    # MA50 debug funnel (where do MA50 swings get rejected?)
    ma50_retests = {'BUY': 0, 'SELL': 0}
    ma50_after_hhll = {'BUY': 0, 'SELL': 0}
    ma50_after_fib = {'BUY': 0, 'SELL': 0}
    ma50_valid = {'BUY': 0, 'SELL': 0}
    ma50_validation_fail_reasons = {}
    
    for i in range(250, len(df) - 5):
        
        if pd.isna(df.iloc[i]['ma200']) or pd.isna(df.iloc[i]['atr']):
            continue
        
        row = df.iloc[i]

        # Detect MA bounces - simplified to just detect first and second touch
        ma_touch_buy = which_ma_touched(row, 'BUY')
        ma_touch_sell = which_ma_touched(row, 'SELL')

        # ====================================================================
        # BUY BOUNCES (detecting all MA touches first)
        # ====================================================================
        if ma_touch_buy:
            ma_touched = ma_touch_buy
            
            # MA50 is PRIMARY, MA18 is SECONDARY
            # Skip MA18 if there's a recent MA50 bounce
            if ma_touched == 'MA18':
                lookback = get_primary_ma_lookback(timeframe)
                recent_primary_50 = [
                    b for b in ma_bounces['MA50']
                    if b['direction'] == 'BUY' and b['index'] >= i - lookback
                ]
                if recent_primary_50:
                    ma18_skipped_for_50_buy += 1
                    continue
            
            ma_touch_counts[ma_touched] += 1
            
            # Record this bounce
            ma_bounces[ma_touched].append({
                'index': i,
                'time': row['time'],
                'price': row['low'],
                'direction': 'BUY'
            })
            
            # Check for previous bounce (retest detection)
            swing_segment = get_swing_segment_for_ma(ma_touched, 'BUY', i, max_lookback=100)
            
            if swing_segment is not None:
                # Found a retest - check minimum time
                first_bounce = swing_segment['first_bounce']
                
                # Calculate time between bounces
                time_diff = (row['time'] - first_bounce['time']).total_seconds() / 3600
                
                # MINIMUM TIME REQUIREMENT: at least 4 hours between bounces
                if time_diff < 4:
                    continue
                
                # TREND VALIDATION: entry must be in valid uptrend
                if not is_clear_uptrend(row):
                    continue
                
                # Check for duplicate
                if any(s['time'] == row['time'] and s['ma'] == ma_touched and s['direction'] == 'BUY' for s in setups):
                    continue
                
                validation_passed += 1
                ma_valid_setups[ma_touched] += 1
                
                entry = row['close']
                sl = entry - (1.5 * row['atr'])
                sl_distance = entry - sl
                tp = entry + (sl_distance * 2)
                
                setups.append({
                    'time': row['time'],
                    'direction': 'BUY',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'ma': ma_touched,
                    'first_bounce_time': first_bounce['time'],
                    'first_bounce_price': first_bounce['price'],
                    'swing_duration_hours': time_diff,
                    'sl_pips': sl_distance / pip_size,
                    'tp_pips': (tp - entry) / pip_size,
                    'rr': 2.0
                })
                
                print(f"\n✅ BUY BOUNCE RETEST:")
                print(f"   Time: {row['time']}")
                print(f"   MA: {ma_touched}")
                print(f"   Swing duration: {time_diff:.1f} hours")
                print(f"   First bounce: {first_bounce['time']} @ {first_bounce['price']:.5f} (low={df.iloc[first_bounce['index']]['low']:.5f}, close={df.iloc[first_bounce['index']]['close']:.5f})")
                print(f"   MA value at first: {df.iloc[first_bounce['index']][ma_touched.lower()]:.5f}")
                print(f"   Second bounce: {row['time']} @ {row['low']:.5f} (close={row['close']:.5f})")
                print(f"   MA value at second: {row[ma_touched.lower()]:.5f}")
        
        # ====================================================================
        # SELL BOUNCES (detecting all MA touches first)
        # ====================================================================
        if ma_touch_sell:
            ma_touched = ma_touch_sell
            
            # MA50 is PRIMARY, MA18 is SECONDARY
            # Skip MA18 if there's a recent MA50 bounce
            if ma_touched == 'MA18':
                lookback = get_primary_ma_lookback(timeframe)
                recent_primary_50 = [
                    b for b in ma_bounces['MA50']
                    if b['direction'] == 'SELL' and b['index'] >= i - lookback
                ]
                if recent_primary_50:
                    ma18_skipped_for_50_sell += 1
                    continue
            
            ma_touch_counts[ma_touched] += 1
            
            # Record this bounce
            ma_bounces[ma_touched].append({
                'index': i,
                'time': row['time'],
                'price': row['high'],
                'direction': 'SELL'
            })
            
            # Check for previous bounce (retest detection)
            swing_segment = get_swing_segment_for_ma(ma_touched, 'SELL', i, max_lookback=100)
            
            if swing_segment is not None:
                # Found a retest - check minimum time
                first_bounce = swing_segment['first_bounce']
                
                # Calculate time between bounces
                time_diff = (row['time'] - first_bounce['time']).total_seconds() / 3600
                
                # MINIMUM TIME REQUIREMENT: at least 4 hours between bounces
                if time_diff < 4:
                    continue
                
                # TREND VALIDATION: entry must be in valid downtrend
                if not is_clear_downtrend(row):
                    continue
                
                # Check for duplicate
                if any(s['time'] == row['time'] and s['ma'] == ma_touched and s['direction'] == 'SELL' for s in setups):
                    continue
                
                validation_passed += 1
                ma_valid_setups[ma_touched] += 1
                
                entry = row['close']
                sl = entry + (1.5 * row['atr'])
                sl_distance = sl - entry
                tp = entry - (sl_distance * 2)
                
                setups.append({
                    'time': row['time'],
                    'direction': 'SELL',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'ma': ma_touched,
                    'first_bounce_time': first_bounce['time'],
                    'first_bounce_price': first_bounce['price'],
                    'sl_pips': sl_distance / pip_size,
                    'tp_pips': (entry - tp) / pip_size,
                    'rr': 2.0
                })
                
                print(f"\n✅ SELL BOUNCE RETEST:")
                print(f"   Time: {row['time']}")
                print(f"   MA: {ma_touched}")
                print(f"   First bounce: {first_bounce['time']} @ {first_bounce['price']:.5f} (high={df.iloc[first_bounce['index']]['high']:.5f}, close={df.iloc[first_bounce['index']]['close']:.5f})")
                print(f"   MA value at first: {df.iloc[first_bounce['index']][ma_touched.lower()]:.5f}")
                print(f"   Second bounce: {row['time']} @ {row['high']:.5f} (close={row['close']:.5f})")
                print(f"   MA value at second: {row[ma_touched.lower()]:.5f}")
    
    # Print MA statistics
                
                entry = row['close']
                sl = entry + (1.5 * row['atr'])
                sl_distance = sl - entry
                tp = entry - (sl_distance * 2)
                
                setups.append({
                    'time': row['time'],
                    'direction': 'SELL',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'ma': ma_touched,
                    'first_bounce_time': first_bounce['time'],
                    'first_bounce_price': first_bounce['price'],
                    'sl_pips': sl_distance / pip_size,
                    'tp_pips': (entry - tp) / pip_size,
                    'rr': 2.0
                })
                
                print(f"\n✅ SELL BOUNCE RETEST:")
                print(f"   Time: {row['time']}")
                print(f"   MA: {ma_touched}")
                print(f"   First bounce: {first_bounce['time']} @ {first_bounce['price']:.5f} (high={df.iloc[first_bounce['index']]['high']:.5f}, close={df.iloc[first_bounce['index']]['close']:.5f})")
                print(f"   MA value at first: {df.iloc[first_bounce['index']][ma_touched.lower()]:.5f}")
                print(f"   Second bounce: {row['time']} @ {row['high']:.5f} (close={row['close']:.5f})")
                print(f"   MA value at second: {row[ma_touched.lower()]:.5f}")
    
    # Print MA statistics
    print("\n" + "="*80)
    print("📊 MA STATISTICS")
    print("="*80)
    print(f"MA touches detected:")
    for ma, count in ma_touch_counts.items():
        print(f"  {ma}: {count} touches")
    print(f"\nValid setups by MA (after full validation):")
    for ma, count in ma_valid_setups.items():
        print(f"  {ma}: {count} valid setups")
    print(f"\nTotal setups passing validation.py: {validation_passed}")
    print(f"\nMA18 skipped due to recent MA50 (primary):")
    print(f"  BUY side:  {ma18_skipped_for_50_buy} skips (lookback={get_primary_ma_lookback(timeframe)} candles)")
    print(f"  SELL side: {ma18_skipped_for_50_sell} skips (lookback={get_primary_ma_lookback(timeframe)} candles)")
    print("\nMA50 DEBUG:")
    print(f"  BUY:  retests={ma50_retests['BUY']}, after_HH/LL={ma50_after_hhll['BUY']}, after_Fib={ma50_after_fib['BUY']}, valid={ma50_valid['BUY']}")
    print(f"  SELL: retests={ma50_retests['SELL']}, after_HH/LL={ma50_after_hhll['SELL']}, after_Fib={ma50_after_fib['SELL']}, valid={ma50_valid['SELL']}")
    if ma50_validation_fail_reasons:
        print("  Top MA50 validation failure reasons:")
        for reason, count in sorted(ma50_validation_fail_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {count} × {reason}")
    print("="*80)
    
    return setups


def display_results(setups, symbol):
    """Display all found setups"""
    
    print("\n" + "="*80)
    print(f"🎯 TCE SETUPS FOR {symbol}")
    print("="*80)
    print(f"Total setups found: {len(setups)}\n")
    
    if len(setups) == 0:
        print("⚠️  No valid setups found")
        print("\nRemember TCE requires:")
        print("  1. Clear trend (MAs aligned + all sloping same direction)")
        print("  2. First MA bounce")
        print("  3. Price makes HH (buy) or LL (sell)")
        print("  4. Second MA bounce at SAME MA = entry")
        print("  5. Valid swing duration (time between bounces)")
        return
    
    for i, setup in enumerate(setups[:10], 1):
        # Debug: print what keys are in setup
        if 'entry' not in setup:
            print(f"\n⚠️  Setup {i} missing 'entry' key. Keys: {list(setup.keys())}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Setup #{i}")
        print(f"{'='*80}")
        print(f"⏰ Entry Time:     {setup['time']}")
        print(f"📍 Direction:      {setup['direction']}")
        print(f"💰 Entry Price:    {setup['entry']:.5f}")
        print(f"🛑 Stop Loss:      {setup['sl']:.5f} ({setup['sl_pips']:.1f} pips)")
        print(f"🎯 Take Profit:    {setup['tp']:.5f} ({setup['tp_pips']:.1f} pips)")
        print(f"📊 Risk:Reward:    1:{setup['rr']:.1f}")
        print(f"📐 MA Retested:    {setup['ma']}")
        print(f"⏱️  Swing Duration: {setup.get('swing_duration_hours', 0):.1f} hours")
        print(f"")
        print(f"📈 First Bounce:   {setup['first_bounce_time']} @ {setup['first_bounce_price']:.5f}")
        if 'highest_between' in setup:
            print(f"🔺 Highest High:   {setup['highest_between']:.5f} (confirmed HH)")
        if 'lowest_between' in setup:
            print(f"🔻 Lowest Low:     {setup['lowest_between']:.5f} (confirmed LL)")
        print(f"🔄 Second Bounce:  {setup['time']} @ {setup['entry']:.5f} (ENTRY)")
    
    if len(setups) > 10:
        print(f"\n... and {len(setups) - 10} more setups")
    
    print(f"\n{'='*80}")


def main():
    print("="*80)
    print("🚀 TCE SETUP DETECTOR - CORRECT LOGIC")
    print("="*80)
    print("\nTCE Logic:")
    print("1. Clear trend (MA6 > MA18 > MA50 > MA200, all sloping up)")
    print("2. First swing: Price bounces off specific MA")
    print("3. Makes HH (buy) or LL (sell)")
    print("4. Second swing: Price retests SAME MA (ENTRY)")
    print("="*80)
    
    if not initialize_mt5():
        return
    
    try:
        SYMBOL = 'USDJPY'

        print(f"\n📥 Downloading {SYMBOL} H1 data...")
        df = download_data(SYMBOL, mt5.TIMEFRAME_H1, 10000)

        if df is None:
            print("❌ Failed to download data")
            return

        print(f"✅ Got {len(df)} candles")

        # Higher timeframe (H4) for validation.py multi-TF checks
        print(f"\n📥 Downloading {SYMBOL} H4 data (higher TF)...")
        htf_df = download_higher_timeframe(SYMBOL, mt5.TIMEFRAME_H4, 2000)
        if htf_df is None:
            print("⚠️ Failed to download higher timeframe data - continuing without HTF")
            htf_df = None

        # Detect setups (then filter via validation.py)
        setups = detect_tce_setups(df, SYMBOL, timeframe='H1', htf_df=htf_df)
        
        # Display results
        display_results(setups, SYMBOL)
        
    finally:
        mt5.shutdown()
        print("\n✅ Complete")


if __name__ == "__main__":
    main()
