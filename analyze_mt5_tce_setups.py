"""
MT5 TCE Setup Analyzer - STRICT Validation
===========================================

Extracts real TCE setups from MT5 data with STRICT rules:
✅ Clear uptrend/downtrend (MA alignment)
✅ First swing only (not second swing)
✅ Valid Fibonacci: 38.2-61.8% ONLY
✅ MA retest required (second touch)
✅ Candlestick confirmation
✅ Risk management (SL/TP)

Usage:
    python analyze_mt5_tce_setups.py
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# ================================================================================
# MT5 CONNECTION
# ================================================================================

def initialize_mt5():
    """Initialize MT5"""
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return False
    print("✅ MT5 initialized")
    return True


def download_data(symbol='EURUSD', timeframe=mt5.TIMEFRAME_M15, bars=5000):
    """Download historical data"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"✅ Downloaded {len(df)} candles ({df['time'].min()} to {df['time'].max()})")
    return df


# ================================================================================
# SWING DETECTION
# ================================================================================

def find_swing_highs(df, lookback=5):
    """Find proper swing highs (local peaks)"""
    swings = []
    for i in range(lookback, len(df) - lookback):
        current = df.iloc[i]['high']
        is_swing = True
        
        for j in range(1, lookback + 1):
            if df.iloc[i-j]['high'] >= current or df.iloc[i+j]['high'] >= current:
                is_swing = False
                break
        
        if is_swing:
            swings.append({'index': i, 'price': current, 'time': df.iloc[i]['time']})
    
    return swings


def find_swing_lows(df, lookback=5):
    """Find proper swing lows (local troughs)"""
    swings = []
    for i in range(lookback, len(df) - lookback):
        current = df.iloc[i]['low']
        is_swing = True
        
        for j in range(1, lookback + 1):
            if df.iloc[i-j]['low'] <= current or df.iloc[i+j]['low'] <= current:
                is_swing = False
                break
        
        if is_swing:
            swings.append({'index': i, 'price': current, 'time': df.iloc[i]['time']})
    
    return swings


# ================================================================================
# TCE SETUP DETECTION
# ================================================================================

def detect_setups(df):
    """
    Detect TCE setups with STRICT validation
    """
    
    # Calculate MAs
    df['ma6'] = df['close'].rolling(6).mean()
    df['ma18'] = df['close'].rolling(18).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Find swings
    print("\n🔍 Finding swings...")
    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    print(f"   {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
    
    setups = []
    
    # Analyze each candle
    for i in range(250, len(df) - 10):
        row = df.iloc[i]
        
        if pd.isna(row['ma200']):
            continue
        
        # Check for CLEAR uptrend
        is_uptrend = (row['ma6'] > row['ma18'] > row['ma50'] > row['ma200'] and
                     row['close'] > row['ma6'])
        
        # Check for CLEAR downtrend
        is_downtrend = (row['ma6'] < row['ma18'] < row['ma50'] < row['ma200'] and
                       row['close'] < row['ma6'])
        
        if not (is_uptrend or is_downtrend):
            continue
        
        # ============================================================
        # BUY SETUP
        # ============================================================
        if is_uptrend:
            # Find FIRST swing low
            swing_low = None
            for s in reversed(swing_lows):
                if s['index'] < i and s['index'] > i - 100 and s['price'] < row['low']:
                    swing_low = s
                    break
            
            if not swing_low:
                continue
            
            # Get swing high (highest point after swing low)
            swing_high = df.iloc[swing_low['index']:i]['high'].max()
            
            # Calculate Fibonacci
            fib_range = swing_high - swing_low['price']
            if fib_range <= 0:
                continue
            
            fib_pct = ((swing_high - row['low']) / fib_range) * 100
            
            # ❌ REJECT if not 38.2-61.8%
            if fib_pct < 38 or fib_pct > 62:
                continue
            
            # Check if at MA
            ma_tol = row['atr'] * 0.3
            at_ma6 = abs(row['low'] - row['ma6']) < ma_tol
            at_ma18 = abs(row['low'] - row['ma18']) < ma_tol
            at_ma50 = abs(row['low'] - row['ma50']) < ma_tol
            
            if not (at_ma6 or at_ma18 or at_ma50):
                continue
            
            ma_name = 'MA6' if at_ma6 else 'MA18' if at_ma18 else 'MA50'
            
            # Check for MA RETEST
            ma_retested = False
            for j in range(max(0, i-20), i-3):
                prev = df.iloc[j]
                if abs(prev['low'] - prev[ma_name.lower()]) < ma_tol:
                    # Check if moved away
                    if any(df.iloc[k]['low'] > prev[ma_name.lower()] + ma_tol 
                          for k in range(j+1, i)):
                        ma_retested = True
                        break
            
            if not ma_retested:
                continue
            
            # Check candlestick confirmation
            confirmed = False
            for j in range(max(0, i-2), i+1):
                c = df.iloc[j]
                body = abs(c['close'] - c['open'])
                lower_wick = min(c['open'], c['close']) - c['low']
                upper_wick = c['high'] - max(c['open'], c['close'])
                
                is_hammer = lower_wick > body * 2 and upper_wick < body * 0.5
                is_bullish = c['close'] > c['open'] and body > row['atr'] * 0.5
                
                if is_hammer or is_bullish:
                    confirmed = True
                    break
            
            if not confirmed:
                continue
            
            # Calculate SL/TP
            fib_618 = swing_high - (fib_range * 0.618)
            sl = max(row['close'] - (1.5 * row['atr']), fib_618 - (10 * 0.0001))
            tp = row['close'] + ((row['close'] - sl) * 2)
            
            setups.append({
                'time': row['time'],
                'dir': 'BUY',
                'entry': row['close'],
                'sl': sl,
                'tp': tp,
                'sl_pips': (row['close'] - sl) * 10000,
                'tp_pips': (tp - row['close']) * 10000,
                'fib': fib_pct,
                'ma': ma_name,
                'swing_low': swing_low['price'],
                'swing_high': swing_high
            })
        
        # ============================================================
        # SELL SETUP  
        # ============================================================
        elif is_downtrend:
            # Find FIRST swing high
            swing_high = None
            for s in reversed(swing_highs):
                if s['index'] < i and s['index'] > i - 100 and s['price'] > row['high']:
                    swing_high = s
                    break
            
            if not swing_high:
                continue
            
            # Get swing low (lowest point after swing high)
            swing_low = df.iloc[swing_high['index']:i]['low'].min()
            
            # Calculate Fibonacci
            fib_range = swing_high['price'] - swing_low
            if fib_range <= 0:
                continue
            
            fib_pct = ((row['high'] - swing_low) / fib_range) * 100
            
            # ❌ REJECT if not 38.2-61.8%
            if fib_pct < 38 or fib_pct > 62:
                continue
            
            # Check if at MA
            ma_tol = row['atr'] * 0.3
            at_ma6 = abs(row['high'] - row['ma6']) < ma_tol
            at_ma18 = abs(row['high'] - row['ma18']) < ma_tol
            at_ma50 = abs(row['high'] - row['ma50']) < ma_tol
            
            if not (at_ma6 or at_ma18 or at_ma50):
                continue
            
            ma_name = 'MA6' if at_ma6 else 'MA18' if at_ma18 else 'MA50'
            
            # Check for MA RETEST
            ma_retested = False
            for j in range(max(0, i-20), i-3):
                prev = df.iloc[j]
                if abs(prev['high'] - prev[ma_name.lower()]) < ma_tol:
                    # Check if moved away
                    if any(df.iloc[k]['high'] < prev[ma_name.lower()] - ma_tol 
                          for k in range(j+1, i)):
                        ma_retested = True
                        break
            
            if not ma_retested:
                continue
            
            # Check candlestick confirmation
            confirmed = False
            for j in range(max(0, i-2), i+1):
                c = df.iloc[j]
                body = abs(c['close'] - c['open'])
                lower_wick = min(c['open'], c['close']) - c['low']
                upper_wick = c['high'] - max(c['open'], c['close'])
                
                is_shooting_star = upper_wick > body * 2 and lower_wick < body * 0.5
                is_bearish = c['close'] < c['open'] and body > row['atr'] * 0.5
                
                if is_shooting_star or is_bearish:
                    confirmed = True
                    break
            
            if not confirmed:
                continue
            
            # Calculate SL/TP
            fib_618 = swing_low + (fib_range * 0.618)
            sl = min(row['close'] + (1.5 * row['atr']), fib_618 + (10 * 0.0001))
            tp = row['close'] - ((sl - row['close']) * 2)
            
            setups.append({
                'time': row['time'],
                'dir': 'SELL',
                'entry': row['close'],
                'sl': sl,
                'tp': tp,
                'sl_pips': (sl - row['close']) * 10000,
                'tp_pips': (row['close'] - tp) * 10000,
                'fib': fib_pct,
                'ma': ma_name,
                'swing_high': swing_high['price'],
                'swing_low': swing_low
            })
    
    return setups


# ================================================================================
# DISPLAY
# ================================================================================

def display_setups(setups, symbol):
    """Display all setups"""
    print(f"\n{'='*80}")
    print(f"🎯 TCE SETUPS FOR {symbol}")
    print(f"{'='*80}")
    print(f"Found {len(setups)} VALID setups\n")
    
    if len(setups) == 0:
        print("⚠️  No valid setups found")
        print("  ❌ Either: No clear trend, invalid Fib%, no MA retest, or no confirmation")
        return
    
    for i, s in enumerate(setups[:10], 1):
        print(f"{'='*80}")
        print(f"Setup #{i}")
        print(f"{'='*80}")
        print(f"⏰ Time:       {s['time']}")
        print(f"📍 Direction:  {s['dir']}")
        print(f"💰 Entry:      {s['entry']:.5f}")
        print(f"🛑 Stop Loss:  {s['sl']:.5f} ({s['sl_pips']:.1f} pips)")
        print(f"🎯 Take Profit:{s['tp']:.5f} ({s['tp_pips']:.1f} pips)")
        print(f"📊 Risk:Reward: 1:2.0")
        print(f"📈 Fibonacci:  {s['fib']:.1f}% ✅")
        print(f"📐 MA:         {s['ma']} (RETESTED)")
        if 'swing_low' in s:
            print(f"🔻 Swing Low:  {s['swing_low']:.5f}")
        if 'swing_high' in s:
            print(f"🔺 Swing High: {s['swing_high']:.5f}")
    
    if len(setups) > 10:
        print(f"\n... and {len(setups)-10} more setups")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(setups)}")
    print(f"BUY:   {len([s for s in setups if s['dir']=='BUY'])}")
    print(f"SELL:  {len([s for s in setups if s['dir']=='SELL'])}")
    print(f"{'='*80}")


# ================================================================================
# MAIN
# ================================================================================

def main():
    print("="*80)
    print("🚀 MT5 TCE SETUP ANALYZER - STRICT VALIDATION")
    print("="*80)
    
    if not initialize_mt5():
        return
    
    try:
        SYMBOL = 'EURUSD'
        df = download_data(SYMBOL, mt5.TIMEFRAME_M15, 5000)
        
        if df is None:
            print("❌ Failed to download data")
            return
        
        setups = detect_setups(df)
        display_setups(setups, SYMBOL)
        
    finally:
        mt5.shutdown()
        print("\n✅ MT5 closed")


if __name__ == "__main__":
    main()
