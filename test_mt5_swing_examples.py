"""
Test Swing Detection with MT5 Data
Shows clear examples of uptrends and downtrends with proper swing identification
"""
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def detect_swing_highs_lows(df, left_bars=5, right_bars=5):
    """
    Detect swing highs and lows in price data
    A swing high: high[i] > high[i-left:i] and high[i] > high[i+1:i+right+1]
    A swing low: low[i] < low[i-left:i] and low[i] < low[i+1:i+right+1]
    """
    swing_highs = []
    swing_lows = []
    
    for i in range(left_bars, len(df) - right_bars):
        # Check for swing high
        is_swing_high = True
        for j in range(i - left_bars, i):
            if df['high'].iloc[i] <= df['high'].iloc[j]:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(i + 1, i + right_bars + 1):
                if df['high'].iloc[i] <= df['high'].iloc[j]:
                    is_swing_high = False
                    break
        if is_swing_high:
            swing_highs.append((i, df['high'].iloc[i], df['time'].iloc[i]))
        
        # Check for swing low
        is_swing_low = True
        for j in range(i - left_bars, i):
            if df['low'].iloc[i] >= df['low'].iloc[j]:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(i + 1, i + right_bars + 1):
                if df['low'].iloc[i] >= df['low'].iloc[j]:
                    is_swing_low = False
                    break
        if is_swing_low:
            swing_lows.append((i, df['low'].iloc[i], df['time'].iloc[i]))
    
    return swing_highs, swing_lows

def calculate_ma(df, period=20):
    """Calculate moving average"""
    return df['close'].rolling(window=period).mean()

def identify_trend(df, ma_period=20):
    """Identify if price is in uptrend or downtrend relative to MA"""
    df['ma'] = calculate_ma(df, ma_period)
    
    # Check if price is consistently above/below MA
    last_50 = df.tail(50)
    above_ma = (last_50['close'] > last_50['ma']).sum()
    below_ma = (last_50['close'] < last_50['ma']).sum()
    
    if above_ma > 35:  # 70% above MA
        return 'UPTREND'
    elif below_ma > 35:  # 70% below MA
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'

def test_pair_swings(symbol, timeframe_str, timeframe_mt5, bars=500):
    """Test swing detection on a currency pair"""
    print(f"\n{'='*80}")
    print(f"Testing: {symbol} {timeframe_str}")
    print(f"{'='*80}")
    
    # Download data
    rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"❌ Failed to get data for {symbol}")
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate MA
    df['ma20'] = calculate_ma(df, 20)
    
    # Identify trend
    trend = identify_trend(df)
    print(f"\n📊 Current Trend: {trend}")
    
    # Detect swings
    swing_highs, swing_lows = detect_swing_highs_lows(df, left_bars=5, right_bars=5)
    
    print(f"\n🔍 Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
    
    # Show last 5 swings based on trend
    if trend == 'UPTREND':
        print(f"\n📈 UPTREND - Analyzing Swing Lows (should be getting higher):")
        recent_swings = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows
        
        for i, (idx, price, time) in enumerate(recent_swings):
            ma_at_swing = df['ma20'].iloc[idx]
            above_below = "ABOVE" if price > ma_at_swing else "BELOW"
            print(f"\n  Swing Low #{i+1}:")
            print(f"    Time: {time}")
            print(f"    Price: {price:.5f}")
            print(f"    MA20: {ma_at_swing:.5f}")
            print(f"    Position: {above_below} MA (Distance: {abs(price - ma_at_swing):.5f})")
            
            if i > 0:
                prev_price = recent_swings[i-1][1]
                if price > prev_price:
                    print(f"    ✅ Higher than previous swing ({prev_price:.5f})")
                else:
                    print(f"    ❌ Lower than previous swing ({prev_price:.5f})")
        
        # Check for proper setup
        if len(recent_swings) >= 2:
            first_swing = recent_swings[-2]
            second_swing = recent_swings[-1]
            
            print(f"\n🎯 SETUP VALIDATION:")
            print(f"  First Swing Low: {first_swing[1]:.5f} at {first_swing[2]}")
            print(f"  Second Swing Low: {second_swing[1]:.5f} at {second_swing[2]}")
            
            if second_swing[1] > first_swing[1]:
                print(f"  ✅ Second swing is HIGHER - Valid uptrend setup!")
            else:
                print(f"  ❌ Second swing is LOWER - Not a valid uptrend setup")
    
    elif trend == 'DOWNTREND':
        print(f"\n📉 DOWNTREND - Analyzing Swing Highs (should be getting lower):")
        recent_swings = swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs
        
        for i, (idx, price, time) in enumerate(recent_swings):
            ma_at_swing = df['ma20'].iloc[idx]
            above_below = "ABOVE" if price > ma_at_swing else "BELOW"
            print(f"\n  Swing High #{i+1}:")
            print(f"    Time: {time}")
            print(f"    Price: {price:.5f}")
            print(f"    MA20: {ma_at_swing:.5f}")
            print(f"    Position: {above_below} MA (Distance: {abs(price - ma_at_swing):.5f})")
            
            if i > 0:
                prev_price = recent_swings[i-1][1]
                if price < prev_price:
                    print(f"    ✅ Lower than previous swing ({prev_price:.5f})")
                else:
                    print(f"    ❌ Higher than previous swing ({prev_price:.5f})")
        
        # Check for proper setup
        if len(recent_swings) >= 2:
            first_swing = recent_swings[-2]
            second_swing = recent_swings[-1]
            
            print(f"\n🎯 SETUP VALIDATION:")
            print(f"  First Swing High: {first_swing[1]:.5f} at {first_swing[2]}")
            print(f"  Second Swing High: {second_swing[1]:.5f} at {second_swing[2]}")
            
            if second_swing[1] < first_swing[1]:
                print(f"  ✅ Second swing is LOWER - Valid downtrend setup!")
            else:
                print(f"  ❌ Second swing is HIGHER - Not a valid downtrend setup")
    
    # Show most recent price action
    recent = df.tail(10)
    print(f"\n📊 Last 10 Candles:")
    for _, row in recent.iterrows():
        above_below = "↑" if row['close'] > row['ma20'] else "↓"
        print(f"  {row['time']} | Close: {row['close']:.5f} | MA20: {row['ma20']:.5f} {above_below}")

def main():
    print("="*80)
    print("MT5 SWING DETECTION TEST")
    print("="*80)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
    
    print(f"✅ MT5 initialized successfully")
    print(f"📍 MT5 version: {mt5.version()}")
    
    # Test pairs on different timeframes
    test_cases = [
        ('EURUSD', 'H1', mt5.TIMEFRAME_H1),
        ('GBPUSD', 'H1', mt5.TIMEFRAME_H1),
        ('USDJPY', 'H1', mt5.TIMEFRAME_H1),
        ('AUDUSD', 'H4', mt5.TIMEFRAME_H4),
        ('USDCAD', 'H4', mt5.TIMEFRAME_H4),
    ]
    
    for symbol, tf_str, tf_mt5 in test_cases:
        try:
            test_pair_swings(symbol, tf_str, tf_mt5, bars=500)
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
    
    # Shutdown MT5
    mt5.shutdown()
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
