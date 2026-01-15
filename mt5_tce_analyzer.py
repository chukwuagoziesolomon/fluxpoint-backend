"""
MT5 TCE Setup Analyzer - STRICT TCE Rules
==========================================

STRICT RULES:
1. Clear uptrend/downtrend (MA alignment)
2. Proper swing high/low detection
3. Valid Fibonacci: 38.2%, 50%, or 61.8% ONLY
4. At MA level: MA18, MA50, or MA200 ONLY (NOT MA6)
5. MA retest (2nd touch, not 1st)
6. Candlestick confirmation
7. Enter on retest of FIRST swing

Usage:
    python mt5_tce_analyzer.py
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


# ================================================================================
# MT5 CONNECTION
# ================================================================================

def initialize_mt5():
    """Initialize MetaTrader 5"""
    if not mt5.initialize():
        print(f"❌ MT5 failed: {mt5.last_error()}")
        return False
    print(f"✅ MT5 connected")
    return True


def download_data(symbol='EURUSD', timeframe=mt5.TIMEFRAME_M15, bars=5000):
    """Download MT5 data"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"❌ No data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"✅ Downloaded {len(df)} candles for {symbol}")
    return df


# ================================================================================
# INDICATORS
# ================================================================================

def calculate_indicators(df):
    """Calculate all required indicators"""
    # Moving Averages - TCE uses MA18, MA50, MA200 (NOT MA6)
    df['ma18'] = df['close'].rolling(window=18).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    # ATR for stop loss
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    return df


# ================================================================================
# SWING DETECTION
# ================================================================================

def find_swing_highs(df, lookback=5):
    """Find proper swing highs (local peaks)"""
    swings = []
    for i in range(lookback, len(df) - lookback):
        high = df.iloc[i]['high']
        is_swing = True
        
        for j in range(1, lookback + 1):
            if df.iloc[i-j]['high'] >= high or df.iloc[i+j]['high'] >= high:
                is_swing = False
                break
        
        if is_swing:
            swings.append({'index': i, 'time': df.iloc[i]['time'], 'price': high})
    
    return swings


def find_swing_lows(df, lookback=5):
    """Find proper swing lows (local troughs)"""
    swings = []
    for i in range(lookback, len(df) - lookback):
        low = df.iloc[i]['low']
        is_swing = True
        
        for j in range(1, lookback + 1):
            if df.iloc[i-j]['low'] <= low or df.iloc[i+j]['low'] <= low:
                is_swing = False
                break
        
        if is_swing:
            swings.append({'index': i, 'time': df.iloc[i]['time'], 'price': low})
    
    return swings


# ================================================================================
# CANDLESTICK PATTERNS
# ================================================================================

def has_bullish_confirmation(candles):
    """Check for bullish candlestick patterns in last 3 candles"""
    if len(candles) < 3:
        return False
    
    for i in range(-3, 0):
        c = candles.iloc[i]
        body = abs(c['close'] - c['open'])
        wick_lower = c['open'] - c['low'] if c['close'] > c['open'] else c['close'] - c['low']
        wick_upper = c['high'] - c['close'] if c['close'] > c['open'] else c['high'] - c['open']
        
        # Bullish engulfing, hammer, pin bar
        if c['close'] > c['open']:  # Green candle
            if body > (wick_lower + wick_upper):  # Strong body
                return True
        if wick_lower > body * 2:  # Hammer/pin bar
            return True
    
    return False


def has_bearish_confirmation(candles):
    """Check for bearish candlestick patterns in last 3 candles"""
    if len(candles) < 3:
        return False
    
    for i in range(-3, 0):
        c = candles.iloc[i]
        body = abs(c['close'] - c['open'])
        wick_lower = c['open'] - c['low'] if c['close'] > c['open'] else c['close'] - c['low']
        wick_upper = c['high'] - c['close'] if c['close'] > c['open'] else c['high'] - c['open']
        
        # Bearish engulfing, shooting star, pin bar
        if c['close'] < c['open']:  # Red candle
            if body > (wick_lower + wick_upper):  # Strong body
                return True
        if wick_upper > body * 2:  # Shooting star/pin bar
            return True
    
    return False


# ================================================================================
# TCE SETUP DETECTION - STRICT RULES
# ================================================================================

def detect_tce_setups(df, symbol='EURUSD'):
    """
    Detect TCE setups with STRICT validation
    """
    
    df = calculate_indicators(df)
    
    print("\n🔍 Finding swings...")
    swing_highs = find_swing_highs(df, lookback=5)
    swing_lows = find_swing_lows(df, lookback=5)
    print(f"   {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
    
    setups = []
    
    # Analyze recent candles
    for i in range(250, len(df) - 10):
        
        if pd.isna(df.iloc[i]['ma200']):
            continue
        
        row = df.iloc[i]
        tolerance = row['atr'] * 0.5
        
        # ==============================================================
        # BUY SETUP DETECTION
        # ==============================================================
        
        # 1. Check for clear UPTREND (MA18 > MA50 > MA200, price > MA18)
        if (row['ma18'] > row['ma50'] > row['ma200'] and 
            row['close'] > row['ma18']):
            
            # 2. Find the FIRST swing low before current candle
            first_swing_low = None
            for swing in swing_lows:
                if swing['index'] < i - 10 and swing['index'] > i - 50:
                    first_swing_low = swing
                    break
            
            if not first_swing_low:
                continue
            
            # 3. Calculate Fibonacci from swing low to highest high after swing
            swing_low_price = first_swing_low['price']
            swing_high_price = df.iloc[first_swing_low['index']:i]['high'].max()
            
            fib_range = swing_high_price - swing_low_price
            if fib_range <= 0:
                continue
            
            fib_38 = swing_high_price - (fib_range * 0.382)
            fib_50 = swing_high_price - (fib_range * 0.5)
            fib_62 = swing_high_price - (fib_range * 0.618)
            
            # 4. Check if price retraced to VALID Fib level (38.2%, 50%, or 61.8%)
            if not (fib_38 <= row['low'] <= swing_high_price):
                continue
            
            # Calculate exact Fib percentage
            retracement = (swing_high_price - row['low']) / fib_range
            fib_percent = retracement * 100
            
            # ✅ STRICT: Only 38.2%, 50%, or 61.8% (±5% tolerance)
            valid_fib = False
            if 33 <= fib_percent <= 43:  # 38.2%
                valid_fib = True
            elif 45 <= fib_percent <= 55:  # 50%
                valid_fib = True
            elif 57 <= fib_percent <= 67:  # 61.8%
                valid_fib = True
            
            if not valid_fib:
                continue
            
            # 5. Check if at MA level (MA18, MA50, MA200 ONLY - NOT MA6)
            at_ma_level = False
            ma_name = None
            
            if abs(row['low'] - row['ma18']) < tolerance:
                at_ma_level = True
                ma_name = 'MA18'
            elif abs(row['low'] - row['ma50']) < tolerance:
                at_ma_level = True
                ma_name = 'MA50'
            elif abs(row['low'] - row['ma200']) < tolerance:
                at_ma_level = True
                ma_name = 'MA200'
            
            if not at_ma_level:
                continue
            
            # 6. Check for MA retest (not first touch)
            # Look back 20 candles for previous MA touch
            ma_value = row[ma_name.lower()]
            has_retest = False
            
            for j in range(i-20, i-5):
                if j < 0:
                    continue
                prev_row = df.iloc[j]
                if abs(prev_row['low'] - ma_value) < tolerance:
                    has_retest = True
                    break
            
            if not has_retest:
                continue
            
            # 7. Check for bullish candlestick confirmation
            recent_candles = df.iloc[i-3:i+1]
            if not has_bullish_confirmation(recent_candles):
                continue
            
            # ✅ VALID BUY SETUP - Calculate SL/TP
            sl = row['close'] - (1.5 * row['atr'])
            sl = max(sl, fib_62 - (5 * 0.0001))  # Below 61.8%
            
            sl_distance = row['close'] - sl
            tp = row['close'] + (sl_distance * 2)  # 1:2 RR
            
            setups.append({
                'time': row['time'],
                'direction': 'BUY',
                'entry': row['close'],
                'sl': sl,
                'tp': tp,
                'swing_low': swing_low_price,
                'swing_high': swing_high_price,
                'fib_level': fib_percent,
                'ma': ma_name,
                'sl_pips': sl_distance * 10000,
                'tp_pips': (tp - row['close']) * 10000,
                'rr_ratio': 2.0
            })
        
        # ==============================================================
        # SELL SETUP DETECTION
        # ==============================================================
        
        # 1. Check for clear DOWNTREND (MA18 < MA50 < MA200, price < MA18)
        elif (row['ma18'] < row['ma50'] < row['ma200'] and 
              row['close'] < row['ma18']):
            
            # 2. Find the FIRST swing high before current candle
            first_swing_high = None
            for swing in swing_highs:
                if swing['index'] < i - 10 and swing['index'] > i - 50:
                    first_swing_high = swing
                    break
            
            if not first_swing_high:
                continue
            
            # 3. Calculate Fibonacci from swing high to lowest low after swing
            swing_high_price = first_swing_high['price']
            swing_low_price = df.iloc[first_swing_high['index']:i]['low'].min()
            
            fib_range = swing_high_price - swing_low_price
            if fib_range <= 0:
                continue
            
            fib_38 = swing_low_price + (fib_range * 0.382)
            fib_50 = swing_low_price + (fib_range * 0.5)
            fib_62 = swing_low_price + (fib_range * 0.618)
            
            # 4. Check if price retraced to VALID Fib level
            if not (swing_low_price <= row['high'] <= fib_38):
                continue
            
            retracement = (row['high'] - swing_low_price) / fib_range
            fib_percent = retracement * 100
            
            # ✅ STRICT: Only 38.2%, 50%, or 61.8%
            valid_fib = False
            if 33 <= fib_percent <= 43:  # 38.2%
                valid_fib = True
            elif 45 <= fib_percent <= 55:  # 50%
                valid_fib = True
            elif 57 <= fib_percent <= 67:  # 61.8%
                valid_fib = True
            
            if not valid_fib:
                continue
            
            # 5. Check if at MA level (MA18, MA50, MA200 ONLY - NOT MA6)
            at_ma_level = False
            ma_name = None
            
            if abs(row['high'] - row['ma18']) < tolerance:
                at_ma_level = True
                ma_name = 'MA18'
            elif abs(row['high'] - row['ma50']) < tolerance:
                at_ma_level = True
                ma_name = 'MA50'
            elif abs(row['high'] - row['ma200']) < tolerance:
                at_ma_level = True
                ma_name = 'MA200'
            
            if not at_ma_level:
                continue
            
            # 6. Check for MA retest
            ma_value = row[ma_name.lower()]
            has_retest = False
            
            for j in range(i-20, i-5):
                if j < 0:
                    continue
                prev_row = df.iloc[j]
                if abs(prev_row['high'] - ma_value) < tolerance:
                    has_retest = True
                    break
            
            if not has_retest:
                continue
            
            # 7. Check for bearish candlestick confirmation
            recent_candles = df.iloc[i-3:i+1]
            if not has_bearish_confirmation(recent_candles):
                continue
            
            # ✅ VALID SELL SETUP - Calculate SL/TP
            sl = row['close'] + (1.5 * row['atr'])
            sl = min(sl, fib_62 + (5 * 0.0001))  # Above 61.8%
            
            sl_distance = sl - row['close']
            tp = row['close'] - (sl_distance * 2)  # 1:2 RR
            
            setups.append({
                'time': row['time'],
                'direction': 'SELL',
                'entry': row['close'],
                'sl': sl,
                'tp': tp,
                'swing_low': swing_low_price,
                'swing_high': swing_high_price,
                'fib_level': fib_percent,
                'ma': ma_name,
                'sl_pips': sl_distance * 10000,
                'tp_pips': (row['close'] - tp) * 10000,
                'rr_ratio': 2.0
            })
    
    return setups


# ================================================================================
# DISPLAY RESULTS
# ================================================================================

def display_setup(setup, index):
    """Display setup details"""
    print(f"\n{'='*80}")
    print(f"Setup #{index}")
    print(f"{'='*80}")
    print(f"⏰ Time:       {setup['time']}")
    print(f"📍 Direction:  {setup['direction']}")
    print(f"💰 Entry:      {setup['entry']:.5f}")
    print(f"🛑 Stop Loss:  {setup['sl']:.5f} ({setup['sl_pips']:.1f} pips)")
    print(f"🎯 Take Profit:{setup['tp']:.5f} ({setup['tp_pips']:.1f} pips)")
    print(f"📊 Risk:Reward: 1:{setup['rr_ratio']:.1f}")
    print(f"📈 Fibonacci:  {setup['fib_level']:.1f}% ✅")
    print(f"📐 MA:         {setup['ma']} (RETESTED)")
    print(f"🔻 Swing Low:  {setup['swing_low']:.5f}")
    print(f"🔺 Swing High: {setup['swing_high']:.5f}")


def display_all_setups(setups, symbol):
    """Display all setups"""
    print(f"\n{'='*80}")
    print(f"🎯 TCE SETUPS FOR {symbol} - STRICT RULES")
    print(f"{'='*80}")
    print(f"Found {len(setups)} valid setups")
    
    if len(setups) == 0:
        print("\n⚠️  No valid setups found")
        print("   Requirements:")
        print("   - Clear trend (MA18 > MA50 > MA200)")
        print("   - Proper swing detection")
        print("   - Fib 38.2%, 50%, or 61.8% ONLY")
        print("   - At MA18, MA50, or MA200 (NOT MA6)")
        print("   - MA retest (2nd touch)")
        print("   - Candlestick confirmation")
        return
    
    for i, setup in enumerate(setups[:5], 1):
        display_setup(setup, i)
    
    if len(setups) > 5:
        print(f"\n... and {len(setups) - 5} more setups")


# ================================================================================
# MAIN
# ================================================================================

def main():
    print("="*80)
    print("🚀 MT5 TCE ANALYZER - STRICT RULES")
    print("="*80)
    
    if not initialize_mt5():
        return
    
    try:
        SYMBOL = 'EURUSD'
        df = download_data(SYMBOL, mt5.TIMEFRAME_M15, 5000)
        
        if df is None:
            return
        
        print("\n🔍 Analyzing for TCE setups...")
        setups = detect_tce_setups(df, SYMBOL)
        display_all_setups(setups, SYMBOL)
        
    finally:
        mt5.shutdown()
        print("\n✅ Complete")


if __name__ == "__main__":
    main()
