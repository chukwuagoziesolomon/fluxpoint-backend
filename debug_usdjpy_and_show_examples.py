"""
Debug USDJPY (0 setups) and show concrete examples of VALID setups.
Shows: MA alignment, trend confirmation, price retest, and candlestick patterns.
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

import pandas as pd
import numpy as np
from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce
from trading.tce.utils import is_uptrend, is_downtrend


def calculate_slope(values, period=20):
    """Calculate slope of a moving average"""
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]


# Load data
data_folder = 'trading/tce/data'

# Test all pairs
pairs = [
    'AUDJPY', 'AUDUSD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD',
    'GBPCHF', 'GBPJPY', 'GBPUSD', 'NZDJPY', 'NZDUSD', 'USDCAD',
    'USDCHF', 'USDHKD', 'USDJPY'
]

pair_data = {}
for pair in pairs:
    file_path = f'{data_folder}/{pair}_DATA.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pair_data[pair] = df
    else:
        print(f"⚠️  Missing: {pair}")

print("\n" + "="*100)
print("DEBUGGING: WHY IS USDJPY RETURNING 0 SETUPS?")
print("="*100)

# ============================================================================
# 1. CHECK USDJPY DATA QUALITY
# ============================================================================

if 'USDJPY' in pair_data:
    df = pair_data['USDJPY'].copy()
    df = df.dropna()
    
    print(f"\n📊 USDJPY DATA QUALITY:")
    print(f"  Total candles: {len(df)}")
    print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"  Close price range: {df['Close'].min():.5f} to {df['Close'].max():.5f}")
    
    # Check last 100 candles for MA alignment
    print(f"\n🔍 MA ALIGNMENT CHECK (Last 100 candles):")
    
    uptrend_count = 0
    downtrend_count = 0
    neutral_count = 0
    
    for row_idx in range(250, min(350, len(df))):
        close_window = df['Close'].values[row_idx-240:row_idx+1]
        
        ma6 = np.mean(close_window[-6:]) if len(close_window) >= 6 else 0
        ma18 = np.mean(close_window[-18:]) if len(close_window) >= 18 else 0
        ma50 = np.mean(close_window[-50:]) if len(close_window) >= 50 else 0
        ma200 = np.mean(close_window[-200:]) if len(close_window) >= 200 else 0
        
        slope6 = calculate_slope(close_window, 6)
        slope18 = calculate_slope(close_window, 18)
        slope50 = calculate_slope(close_window, 50)
        
        # Check uptrend condition
        if ma6 > ma18 > ma50 > ma200:
            slopes_positive = sum([slope6 > 0, slope18 > 0, slope50 > 0])
            if slopes_positive >= 2:
                uptrend_count += 1
        # Check downtrend condition
        elif ma200 > ma50 > ma18 > ma6:
            slopes_negative = sum([slope6 < 0, slope18 < 0, slope50 < 0])
            if slopes_negative >= 2:
                downtrend_count += 1
        else:
            neutral_count += 1
    
    total_checked = 100
    print(f"  ✅ Uptrends found: {uptrend_count}/{total_checked} ({100*uptrend_count/total_checked:.1f}%)")
    print(f"  ✅ Downtrends found: {downtrend_count}/{total_checked} ({100*downtrend_count/total_checked:.1f}%)")
    print(f"  ⚠️  Neutral/No MA alignment: {neutral_count}/{total_checked} ({100*neutral_count/total_checked:.1f}%)")
    
    # If mostly neutral, that's the problem!
    if neutral_count > total_checked * 0.7:
        print(f"\n🔴 ISSUE FOUND: USDJPY has mostly neutral MA alignment!")
        print(f"   The MAs are NOT in proper order (6>18>50>200 or 200>50>18>6)")
        print(f"   This is why 0 setups are found - trend confirmation fails first!")

# ============================================================================
# 2. FIND VALID SETUPS FROM ALL PAIRS (Show at least 10 examples)
# ============================================================================

print(f"\n" + "="*100)
print("SEARCHING FOR VALID SETUPS IN ALL PAIRS (Need at least 10)")
print("="*100)

valid_setups = []

for symbol in pair_data.keys():
    df = pair_data[symbol].copy().reset_index(drop=True)
    df = df.dropna()
    
    if len(df) < 250:
        continue
    
    # Scan every 10 candles to find setups faster
    for row_idx in range(250, len(df) - 50, 10):
        try:
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            open_price = df['Open'].values
            dates = df['Date'].values
            
            close_window = close[row_idx-240:row_idx+1]
            high_window = high[row_idx-240:row_idx+1]
            low_window = low[row_idx-240:row_idx+1]
            
            ma6 = np.mean(close_window[-6:]) if len(close_window) >= 6 else 0
            ma18 = np.mean(close_window[-18:]) if len(close_window) >= 18 else 0
            ma50 = np.mean(close_window[-50:]) if len(close_window) >= 50 else 0
            ma200 = np.mean(close_window[-200:]) if len(close_window) >= 200 else 0
            
            slope6 = calculate_slope(close_window, 6)
            slope18 = calculate_slope(close_window, 18)
            slope50 = calculate_slope(close_window, 50)
            slope200 = calculate_slope(close_window, 200)
            
            tr_list = []
            for i in range(1, min(15, len(high_window))):
                tr = max(
                    high_window[i] - low_window[i],
                    abs(high_window[i] - close_window[i-1]),
                    abs(low_window[i] - close_window[i-1])
                )
                tr_list.append(tr)
            atr = np.mean(tr_list) if tr_list else 0
            
            indicators = Indicators(
                ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
                slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200,
                atr=atr
            )
            
            candle = Candle(
                open=float(open_price[row_idx]),
                high=float(high[row_idx]),
                low=float(low[row_idx]),
                close=float(close[row_idx]),
                timestamp=str(dates[row_idx])
            )
            
            ma_ref = ma18
            fib_level = 0.618
            if candle.low < ma_ref and atr > 0:
                depth = (ma_ref - candle.low) / atr
                if depth < 0.5:
                    fib_level = 0.382
                elif depth < 1.0:
                    fib_level = 0.5
            
            swing = Swing(
                type='high' if close[row_idx] > np.mean(close[row_idx-50:row_idx]) else 'low',
                price=float(close[row_idx]),
                fib_level=fib_level
            )
            
            recent_close = close[max(0, row_idx-50):row_idx+1]
            recent_high = high[max(0, row_idx-50):row_idx+1]
            recent_low = low[max(0, row_idx-50):row_idx+1]
            recent_open = open_price[max(0, row_idx-50):row_idx+1]
            
            recent_candles = [
                Candle(
                    open=float(recent_open[i]),
                    high=float(recent_high[i]),
                    low=float(recent_low[i]),
                    close=float(recent_close[i]),
                    timestamp=str(dates[max(0, row_idx-50)+i])
                )
                for i in range(len(recent_close))
            ]
            
            structure = MarketStructure(
                highs=list(recent_high[-50:]),
                lows=list(recent_low[-50:])
            )
            
            result = validate_tce(
                candle=candle,
                indicators=indicators,
                swing=swing,
                sr_levels=[],
                higher_tf_candles=[],
                correlations={},
                structure=structure,
                recent_candles=recent_candles,
                timeframe="H1",
                account_balance=10000.0,
                risk_percentage=1.0,
                symbol=symbol
            )
            
            if result["is_valid"]:
                valid_setups.append({
                    'symbol': symbol,
                    'date': dates[row_idx],
                    'index': row_idx,
                    'price': close[row_idx],
                    'direction': result['direction'],
                    'ma6': ma6, 'ma18': ma18, 'ma50': ma50, 'ma200': ma200,
                    'slope6': slope6, 'slope18': slope18, 'slope50': slope50,
                    'atr': atr,
                    'stop_loss': result['stop_loss'],
                    'take_profit': result['take_profit'],
                    'trend_ok': result['trend_ok'],
                    'fib_ok': result['fib_ok'],
                    'ma_level_ok': result['ma_level_ok'],
                    'ma_retest_ok': result['ma_retest_ok'],
                    'candlestick_ok': result['candlestick_ok'],
                })
        
        except Exception as e:
            pass
    
    if len(valid_setups) >= 15:  # Stop when we have enough examples
        break

# ============================================================================
# 3. DISPLAY VALID SETUPS WITH FULL DETAILS
# ============================================================================

print(f"\n✅ FOUND {len(valid_setups)} VALID SETUPS\n")

if len(valid_setups) > 0:
    print("📍 SHOWING FIRST 15 VALID SETUPS (with full MA alignment and retest details):\n")
    
    for idx, setup in enumerate(valid_setups[:15]):
        print(f"{'='*100}")
        print(f"SETUP #{idx+1} | {setup['symbol']} | {setup['date']} | Direction: {setup['direction'].upper()}")
        print(f"{'='*100}")
        
        print(f"\n🎯 ENTRY & RISK MANAGEMENT:")
        print(f"  Entry Price:    {setup['price']:.5f}")
        print(f"  Stop Loss:      {setup['stop_loss']:.5f}")
        print(f"  Take Profit:    {setup['take_profit']:.5f}")
        print(f"  ATR:            {setup['atr']:.5f}")
        
        print(f"\n📊 MOVING AVERAGES (Multi-MA alignment):")
        print(f"  MA6:            {setup['ma6']:.5f}")
        print(f"  MA18:           {setup['ma18']:.5f}")
        print(f"  MA50:           {setup['ma50']:.5f}")
        print(f"  MA200:          {setup['ma200']:.5f}")
        
        # Show alignment
        if setup['direction'] == 'buy':
            alignment = f"✅ {setup['ma6']:.2f} > {setup['ma18']:.2f} > {setup['ma50']:.2f} > {setup['ma200']:.2f}"
        else:
            alignment = f"✅ {setup['ma200']:.2f} > {setup['ma50']:.2f} > {setup['ma18']:.2f} > {setup['ma6']:.2f}"
        print(f"  Alignment:      {alignment}")
        
        print(f"\n📈 SLOPES (Rate of change):")
        print(f"  Slope6:         {setup['slope6']:+.6f} {'↗️' if setup['slope6'] > 0 else '↘️'}")
        print(f"  Slope18:        {setup['slope18']:+.6f} {'↗️' if setup['slope18'] > 0 else '↘️'}")
        print(f"  Slope50:        {setup['slope50']:+.6f} {'↗️' if setup['slope50'] > 0 else '↘️'}")
        
        print(f"\n✔️  VALIDATION RULES:")
        print(f"  1️⃣  Trend (MA alignment + slopes):     {'✅ PASS' if setup['trend_ok'] else '❌ FAIL'}")
        print(f"  2️⃣  Fibonacci (38.2%, 50%, 61.8%):     {'✅ PASS' if setup['fib_ok'] else '❌ FAIL'}")
        print(f"  3️⃣  MA Level (price at MA ±5%):       {'✅ PASS' if setup['ma_level_ok'] else '❌ FAIL'}")
        print(f"  3.5️⃣ MA Retest (price retested MA):   {'✅ PASS' if setup['ma_retest_ok'] else '❌ FAIL'}")
        print(f"  4️⃣  Candlestick Pattern:               {'✅ PASS' if setup['candlestick_ok'] else '❌ FAIL'}")
        print()

else:
    print("❌ NO VALID SETUPS FOUND! This indicates a serious problem with validation rules.")
