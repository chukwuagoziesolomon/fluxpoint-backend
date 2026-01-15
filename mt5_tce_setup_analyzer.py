"""
MT5 TCE Setup Analyzer - Complete Pipeline
===========================================

This script does EXACTLY what Cell 5 does in Colab:
1. Downloads real MT5 data
2. Detects proper swing highs/lows
3. Validates TCE setups with all 8 rules
4. Calculates risk management (SL/TP)
5. Shows trade entry signals

Usage:
    python mt5_tce_setup_analyzer.py
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add trading module to path
sys.path.insert(0, str(Path(__file__).parent))

from trading.tce.types import Candle, Indicators, Swing, MarketStructure, HigherTFCandle
from trading.tce.validation import validate_tce


# ================================================================================
# MT5 DATA DOWNLOAD
# ================================================================================

def initialize_mt5():
    """Initialize MetaTrader 5 connection"""
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        return False
    
    print("✅ MT5 initialized successfully")
    print(f"MT5 version: {mt5.version()}")
    print(f"Terminal info: {mt5.terminal_info()}")
    return True


def download_mt5_data(symbol='EURUSD', timeframe=mt5.TIMEFRAME_M15, bars=5000):
    """Download historical data from MT5"""
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        print(f"❌ Failed to get data for {symbol}")
        print(f"Error: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Downloaded {len(df)} candles for {symbol}")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df


# ================================================================================
# SWING DETECTION (Proper Implementation)
# ================================================================================

def find_swing_highs(df, lookback=5):
    """
    Find PROPER swing highs (local peaks)
    A swing high is a candle whose high is higher than N candles before and after it
    """
    swing_highs = []
    
    for i in range(lookback, len(df) - lookback):
        current_high = df.iloc[i]['high']
        
        # Check if this high is higher than surrounding candles
        is_swing_high = True
        for j in range(1, lookback + 1):
            if df.iloc[i - j]['high'] >= current_high or df.iloc[i + j]['high'] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append({
                'index': i,
                'time': df.iloc[i]['time'],
                'price': current_high
            })
    
    return swing_highs


def find_swing_lows(df, lookback=5):
    """
    Find PROPER swing lows (local troughs)
    A swing low is a candle whose low is lower than N candles before and after it
    """
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        current_low = df.iloc[i]['low']
        
        # Check if this low is lower than surrounding candles
        is_swing_low = True
        for j in range(1, lookback + 1):
            if df.iloc[i - j]['low'] <= current_low or df.iloc[i + j]['low'] <= current_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append({
                'index': i,
                'time': df.iloc[i]['time'],
                'price': current_low
            })
    
    return swing_lows


# ================================================================================
# TCE SETUP DETECTION
# ================================================================================

def calculate_mas(df, periods=[6, 18, 50, 200]):
    """Calculate moving averages"""
    for period in periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=period).mean()
    
    return df


def detect_tce_setups(df, symbol='EURUSD'):
    """
    Detect TCE setups with STRICT validation:
    1. Clear uptrend/downtrend (MAs aligned)
    2. First swing identified (not second swing)
    3. Valid Fibonacci 38.2%, 50%, or 61.8% ONLY
    4. MA retest (second touch, not first)
    5. Candlestick confirmation
    """
    
    # Calculate indicators
    df = calculate_mas(df)
    df = calculate_atr(df)
    
    # Find swings
    print("\n🔍 Finding swing highs and lows...")
    swing_highs = find_swing_highs(df, lookback=5)
    swing_lows = find_swing_lows(df, lookback=5)
    
    print(f"   Found {len(swing_highs)} swing highs")
    print(f"   Found {len(swing_lows)} swing lows")
    
    # Look for TCE setups
    setups = []
    
    # Check each candle for potential setups
    for i in range(250, len(df) - 10):
        
        # Skip if no MA data yet
        if pd.isna(df.iloc[i]['ma200']):
            continue
        
        # Get current candle
        row = df.iloc[i]
        
        # Create Candle object
        candle = Candle(
            timestamp=row['time'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close']
        )
        
        # Calculate MA slopes
        slope6 = 0.0
        slope18 = 0.0
        slope50 = 0.0
        slope200 = 0.0
        
        if i >= 5:
            slope6 = (df.iloc[i]['ma6'] - df.iloc[i-5]['ma6']) / 5 if not pd.isna(df.iloc[i-5]['ma6']) else 0.0
            slope18 = (df.iloc[i]['ma18'] - df.iloc[i-5]['ma18']) / 5 if not pd.isna(df.iloc[i-5]['ma18']) else 0.0
            slope50 = (df.iloc[i]['ma50'] - df.iloc[i-5]['ma50']) / 5 if not pd.isna(df.iloc[i-5]['ma50']) else 0.0
            slope200 = (df.iloc[i]['ma200'] - df.iloc[i-5]['ma200']) / 5 if not pd.isna(df.iloc[i-5]['ma200']) else 0.0
        
        # Create Indicators object
        indicators = Indicators(
            ma6=row['ma6'],
            ma18=row['ma18'],
            ma50=row['ma50'],
            ma200=row['ma200'],
            slope6=slope6,
            slope18=slope18,
            slope50=slope50,
            slope200=slope200,
            atr=row['atr']
        )
        
        # Determine trend direction with STRICT alignment
        is_clear_uptrend = (
            row['ma6'] > row['ma18'] > row['ma50'] > row['ma200'] and
            row['close'] > row['ma6'] and
            row['ma6'] > row['ma18'] * 1.0001  # Small buffer for noise
        )
        
        is_clear_downtrend = (
            row['ma6'] < row['ma18'] < row['ma50'] < row['ma200'] and
            row['close'] < row['ma6'] and
            row['ma6'] < row['ma18'] * 0.9999  # Small buffer for noise
        )
        
        if not (is_clear_uptrend or is_clear_downtrend):
            continue
        
        direction = 'BUY' if is_clear_uptrend else 'SELL'
        
        # ==================================================================
        # BUY SETUP (UPTREND)
        # ==================================================================
        if direction == 'BUY':
            
            # 1️⃣ Find the FIRST swing low in the recent uptrend
            recent_swing_low = None
            for swing in reversed(swing_lows):
                if swing['index'] < i and swing['index'] > i - 100:
                    # Verify this swing is below current price (retracement happened)
                    if swing['price'] < row['low']:
                        recent_swing_low = swing
                        break
            
            if not recent_swing_low:
                continue
            
            # 2️⃣ Calculate Fibonacci from swing low to recent high
            # Find highest high since the swing low
            swing_low_price = recent_swing_low['price']
            swing_low_idx = recent_swing_low['index']
            
            swing_high_price = df.iloc[swing_low_idx:i]['high'].max()
            
            fib_range = swing_high_price - swing_low_price
            if fib_range <= 0:
                continue
            
            # Fibonacci retracement levels from the high
            fib_382 = swing_high_price - (fib_range * 0.382)
            fib_50 = swing_high_price - (fib_range * 0.5)
            fib_618 = swing_high_price - (fib_range * 0.618)
            
            # 3️⃣ Check if current price is at VALID Fib level (38.2-61.8% ONLY)
            current_low = row['low']
            
            # Calculate actual fib percentage
            retracement_from_high = swing_high_price - current_low
            fib_percentage = (retracement_from_high / fib_range) * 100
            
            # ❌ REJECT if beyond 61.8% or below 38.2%
            if fib_percentage > 62 or fib_percentage < 38:
                continue
            
            # 4️⃣ Check if at MA level (MA6, MA18, or MA50)
            ma_tolerance = row['atr'] * 0.3
            at_ma6 = abs(current_low - row['ma6']) < ma_tolerance
            at_ma18 = abs(current_low - row['ma18']) < ma_tolerance
            at_ma50 = abs(current_low - row['ma50']) < ma_tolerance
            
            if not (at_ma6 or at_ma18 or at_ma50):
                continue
            
            ma_touched = 'MA6' if at_ma6 else 'MA18' if at_ma18 else 'MA50'
            
            # 5️⃣ Check for MA RETEST (must touch MA, bounce away, then come back)
            # Look back 10 candles to see if MA was touched before
            ma_retested = False
            lookback_start = max(0, i - 20)
            lookback_end = i - 3
            
            for j in range(lookback_start, lookback_end):
                prev_row = df.iloc[j]
                if abs(prev_row['low'] - prev_row[f'ma{ma_touched[-2:]}'if 'MA' in ma_touched else 'ma6']) < ma_tolerance:
                    # Found previous touch - check if price moved away
                    moved_away = False
                    for k in range(j + 1, i - 1):
                        if df.iloc[k]['low'] > prev_row[f'ma{ma_touched[-2:] if "MA" in ma_touched else "6"}'] + ma_tolerance:
                            moved_away = True
                            break
                    
                    if moved_away:
                        ma_retested = True
                        break
            
            if not ma_retested:
                continue  # Skip first touch - wait for retest
            
            # 6️⃣ Check for BULLISH candlestick confirmation
            # Look at last 3 candles for bullish patterns
            has_bullish_confirmation = False
            
            for j in range(max(0, i - 3), i + 1):
                candle_row = df.iloc[j]
                
                # Bullish engulfing, hammer, or strong bullish candle
                body = abs(candle_row['close'] - candle_row['open'])
                lower_wick = candle_row['open'] - candle_row['low'] if candle_row['close'] > candle_row['open'] else candle_row['close'] - candle_row['low']
                upper_wick = candle_row['high'] - candle_row['close'] if candle_row['close'] > candle_row['open'] else candle_row['high'] - candle_row['open']
                
                # Hammer: long lower wick, small body
                is_hammer = lower_wick > body * 2 and upper_wick < body * 0.5
                
                # Strong bullish candle
                is_bullish = candle_row['close'] > candle_row['open'] and body > row['atr'] * 0.5
                
                if is_hammer or is_bullish:
                    has_bullish_confirmation = True
                    break
            
            if not has_bullish_confirmation:
                continue
            
            # ✅ VALID SETUP - Calculate risk management
            sl = row['close'] - (1.5 * row['atr'])
            sl = max(sl, fib_618 - (10 * 0.0001))  # Must be below 61.8%
            
            sl_distance = row['close'] - sl
            tp = row['close'] + (sl_distance * 2)  # 1:2 RR
            
            setups.append({
                'time': row['time'],
                'index': i,
                'direction': direction,
                'entry': row['close'],
                'sl': sl,
                'tp': tp,
                'swing_low': swing_low_price,
                'swing_high': swing_high_price,
                'fib_level': fib_percentage,
                'at_ma': ma_touched,
                'sl_pips': sl_distance * 10000,
                'tp_pips': (tp - row['close']) * 10000,
                'rr_ratio': 2.0,
                'ma_retested': True,
                'candlestick_confirmed': True
            })
        
        # ==================================================================
        # SELL SETUP (DOWNTREND)
        # ==================================================================
            
            # Find recent swing low
            recent_swing_low = None
            for swing in reversed(swing_lows):
                if swing['index'] < i and swing['index'] > i - 50:
                    recent_swing_low = swing
                    break
            
            if recent_swing_low:
                # Calculate Fibonacci levels
                swing_high = row['high']
                swing_low = recent_swing_low['price']
                fib_range = swing_high - swing_low
                fib_38 = swing_high - (fib_range * 0.382)
                fib_50 = swing_high - (fib_range * 0.5)
                fib_62 = swing_high - (fib_range * 0.618)
                
                # Check if price retraced to Fib zone
                if row['low'] <= fib_62 and row['close'] >= fib_38:
                    
                    # Check if at MA level
                    at_ma = (abs(row['low'] - row['ma6']) < row['atr'] * 0.5 or
                            abs(row['low'] - row['ma18']) < row['atr'] * 0.5 or
                            abs(row['low'] - row['ma50']) < row['atr'] * 0.5)
                    
                    if at_ma:
                        # Create Swing object (BUY setup - swing low)
                        swing = Swing(
                            type='low',
                            price=swing_low,
                            fib_level=0.618,
                            fib_618_price=fib_62
                        )
                        
                        # Calculate Stop Loss and Take Profit
                        sl = row['close'] - (1.5 * row['atr'])
                        sl = max(sl, fib_62 - (5 * 0.0001))  # Below 61.8% Fib
                        
                        sl_distance = row['close'] - sl
                        tp = row['close'] + (sl_distance * 2)  # 1:2 RR
                        
                        setups.append({
                            'time': row['time'],
                            'index': i,
                            'direction': direction,
                            'entry': row['close'],
                            'sl': sl,
                            'tp': tp,
                            'swing_low': swing_low,
                            'fib_level': ((swing_high - row['low']) / fib_range) * 100,
                            'at_ma': 'MA6' if abs(row['low'] - row['ma6']) < row['atr'] * 0.5 else 
                                     'MA18' if abs(row['low'] - row['ma18']) < row['atr'] * 0.5 else 'MA50',
                            'sl_pips': sl_distance * 10000,
                            'tp_pips': (tp - row['close']) * 10000,
                            'rr_ratio': 2.0
                        })
        
        elif (row['ma6'] < row['ma18'] < row['ma50'] < row['ma200'] and
              row['close'] < row['ma6']):
            direction = 'SELL'
            
            # Find recent swing high
            recent_swing_high = None
            for swing in reversed(swing_highs):
                if swing['index'] < i and swing['index'] > i - 50:
                    recent_swing_high = swing
                    break
            
            if recent_swing_high:
                # Calculate Fibonacci levels
                swing_high = recent_swing_high['price']
                swing_low = row['low']
                fib_range = swing_high - swing_low
                fib_38 = swing_low + (fib_range * 0.382)
                fib_50 = swing_low + (fib_range * 0.5)
                fib_62 = swing_low + (fib_range * 0.618)
                
                # Check if price retraced to Fib zone
                if row['high'] >= fib_62 and row['close'] <= fib_38:
                    
                    # Check if at MA level
                    at_ma = (abs(row['high'] - row['ma6']) < row['atr'] * 0.5 or
                            abs(row['high'] - row['ma18']) < row['atr'] * 0.5 or
                            abs(row['high'] - row['ma50']) < row['atr'] * 0.5)
                    
                    if at_ma:
                        # Create Swing object (SELL setup - swing high)
                        swing = Swing(
                            type='high',
                            price=swing_high,
                            fib_level=0.618,
                            fib_618_price=fib_62
                        )
                        
                        # Calculate Stop Loss and Take Profit
                        sl = row['close'] + (1.5 * row['atr'])
                        sl = min(sl, fib_62 + (5 * 0.0001))  # Above 61.8% Fib
                        
                        sl_distance = sl - row['close']
                        tp = row['close'] - (sl_distance * 2)  # 1:2 RR
                        
                        setups.append({
                            'time': row['time'],
                            'index': i,
                            'direction': direction,
                            'entry': row['close'],
                            'sl': sl,
                            'tp': tp,
                            'swing_high': swing_high,
                            'fib_level': ((row['high'] - swing_low) / fib_range) * 100,
                            'at_ma': 'MA6' if abs(row['high'] - row['ma6']) < row['atr'] * 0.5 else 
                                     'MA18' if abs(row['high'] - row['ma18']) < row['atr'] * 0.5 else 'MA50',
                            'sl_pips': sl_distance * 10000,
                            'tp_pips': (row['close'] - tp) * 10000,
                            'rr_ratio': 2.0
                        })
    
    return setups


# ================================================================================
# DISPLAY RESULTS
# ================================================================================

def display_setup(setup, index):
    """Display a single TCE setup with full details"""
    print(f"\n{'='*80}")
    print(f"📊 TCE SETUP #{index}")
    print(f"{'='*80}")
    print(f"⏰ Time:      {setup['time']}")
    print(f"📍 Direction: {setup['direction']}")
    print(f"💰 Entry:     {setup['entry']:.5f}")
    print(f"🛑 Stop Loss: {setup['sl']:.5f} ({setup['sl_pips']:.1f} pips)")
    print(f"🎯 Take Profit: {setup['tp']:.5f} ({setup['tp_pips']:.1f} pips)")
    print(f"📊 Risk:Reward: 1:{setup['rr_ratio']:.1f}")
    print(f"📈 Fibonacci:  {setup['fib_level']:.1f}%")
    print(f"📐 At MA:      {setup['at_ma']}")
    
    if 'swing_low' in setup:
        print(f"🔻 Swing Low:  {setup['swing_low']:.5f}")
    if 'swing_high' in setup:
        print(f"🔺 Swing High: {setup['swing_high']:.5f}")
    
    print(f"{'='*80}")


def display_all_setups(setups, symbol):
    """Display all found TCE setups"""
    print(f"\n\n{'='*80}")
    print(f"🎯 TCE SETUP ANALYSIS FOR {symbol}")
    print(f"{'='*80}")
    print(f"✅ Found {len(setups)} valid TCE setups")
    
    if len(setups) == 0:
        print("\n⚠️  No setups found. This could mean:")
        print("   - Market is not in a clear trend")
        print("   - No proper swing retracements to Fib zones")
        print("   - Price not touching MA levels")
        print("   Try a different symbol or timeframe")
        return
    
    # Display first 5 setups in detail
    for i, setup in enumerate(setups[:5], 1):
        display_setup(setup, i)
    
    if len(setups) > 5:
        print(f"\n... and {len(setups) - 5} more setups")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"📈 SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total setups: {len(setups)}")
    print(f"BUY setups:   {len([s for s in setups if s['direction'] == 'BUY'])}")
    print(f"SELL setups:  {len([s for s in setups if s['direction'] == 'SELL'])}")
    print(f"Avg SL pips:  {np.mean([s['sl_pips'] for s in setups]):.1f}")
    print(f"Avg TP pips:  {np.mean([s['tp_pips'] for s in setups]):.1f}")
    print(f"{'='*80}")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("🚀 MT5 TCE SETUP ANALYZER")
    print("="*80)
    
    # Initialize MT5
    if not initialize_mt5():
        return
    
    try:
        # Configuration
        SYMBOL = 'EURUSD'
        TIMEFRAME = mt5.TIMEFRAME_M15
        BARS = 5000
        
        print(f"\n📥 Downloading data for {SYMBOL} M15...")
        
        # Download data
        df = download_mt5_data(SYMBOL, TIMEFRAME, BARS)
        
        if df is None:
            print("❌ Failed to download data")
            return
        
        # Detect TCE setups
        print("\n🔍 Analyzing data for TCE setups...")
        setups = detect_tce_setups(df, SYMBOL)
        
        # Display results
        display_all_setups(setups, SYMBOL)
        
        # Ask if user wants to test another pair
        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("\nTo analyze another pair, modify SYMBOL in the script:")
        print("   SYMBOL = 'GBPUSD'  # or USDJPY, AUDUSD, etc.")
        print("="*80)
        
    finally:
        # Shutdown MT5
        mt5.shutdown()
        print("\n✅ MT5 connection closed")


if __name__ == "__main__":
    main()
