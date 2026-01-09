"""
Generate detailed report of all 97 valid TCE setups with:
- Date
- Entry price
- Stop Loss
- Take Profit
- Direction
- Reasons for entry (which rules passed)
"""

import os
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

import numpy as np
from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce

def calculate_slope(values, period=20):
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

# Load data
data_folder = 'trading/tce/data'
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

print("\n" + "="*150)
print("ALL 97 VALID TCE SETUPS - DETAILED REPORT")
print("="*150)

all_setups = []

for symbol in pair_data.keys():
    df = pair_data[symbol].copy().reset_index(drop=True)
    df = df.dropna()
    
    if len(df) < 250:
        continue
    
    # Scan every 10 candles to find setups
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
                # Determine entry reason based on which rules passed
                reasons = []
                if result['trend_ok']:
                    if result['direction'] == 'BUY':
                        reasons.append(f"Uptrend: MA6({ma6:.4f}) > MA18({ma18:.4f}) > MA50({ma50:.4f})")
                    else:
                        reasons.append(f"Downtrend: MA200({ma200:.4f}) > MA50({ma50:.4f}) > MA18({ma18:.4f})")
                
                if result['fib_ok']:
                    reasons.append(f"Fibonacci: {int(swing.fib_level*100)}% retracement")
                
                if result['ma_level_ok']:
                    reasons.append(f"Price at MA level (within 5%)")
                
                if result['ma_retest_ok']:
                    reasons.append(f"MA retest detected in last 20 candles")
                
                if result['candlestick_ok']:
                    reasons.append(f"Candlestick pattern confirmed")
                
                all_setups.append({
                    'Pair': symbol,
                    'Date': dates[row_idx],
                    'Entry': close[row_idx],
                    'Stop Loss': result['stop_loss'],
                    'Take Profit': result['take_profit'],
                    'Direction': result['direction'],
                    'Risk/Reward': f"{result['risk_reward_ratio']:.2f}:1",
                    'Reasons': ' + '.join(reasons),
                    'MA6': f"{ma6:.5f}",
                    'MA18': f"{ma18:.5f}",
                    'MA50': f"{ma50:.5f}",
                    'MA200': f"{ma200:.5f}",
                    'Fib Level': f"{int(swing.fib_level*100)}%",
                })
        
        except Exception as e:
            pass

print(f"\n✅ FOUND {len(all_setups)} VALID SETUPS\n")

# Display as formatted table
if all_setups:
    # Create DataFrame
    df_setups = pd.DataFrame(all_setups)
    
    # Save to CSV
    csv_filename = 'ALL_97_TCE_SETUPS_DETAILED.csv'
    df_setups.to_csv(csv_filename, index=False)
    print(f"✅ Saved to: {csv_filename}\n")
    
    # Display first 20 in readable format
    print("="*150)
    print(f"DISPLAYING FIRST 20 SETUPS (Total: {len(all_setups)})")
    print("="*150)
    
    for idx, setup in enumerate(all_setups[:20], 1):
        print(f"\n📍 SETUP #{idx}")
        print(f"   Pair:        {setup['Pair']}")
        print(f"   Date:        {setup['Date']}")
        print(f"   Direction:   {setup['Direction']}")
        print(f"   ├─ Entry:    {setup['Entry']:.5f}")
        print(f"   ├─ SL:       {setup['Stop Loss']:.5f}  (Risk)")
        print(f"   ├─ TP:       {setup['Take Profit']:.5f}  (Profit)")
        print(f"   └─ RR:       {setup['Risk/Reward']}")
        print(f"   Moving Averages:")
        print(f"   ├─ MA6:      {setup['MA6']}")
        print(f"   ├─ MA18:     {setup['MA18']}")
        print(f"   ├─ MA50:     {setup['MA50']}")
        print(f"   ├─ MA200:    {setup['MA200']}")
        print(f"   └─ Fib:      {setup['Fib Level']}")
        print(f"   Entry Reasons:")
        for reason in setup['Reasons'].split(' + '):
            print(f"      ✓ {reason}")
    
    print(f"\n{'='*150}")
    print(f"SUMMARY BY PAIR:")
    print(f"{'='*150}\n")
    
    pair_summary = df_setups.groupby('Pair').size().sort_values(ascending=False)
    for pair, count in pair_summary.items():
        print(f"  {pair:10s}: {count:3d} setups")
    
    print(f"\n{'='*150}")
    print(f"✅ All {len(all_setups)} setups saved to: {csv_filename}")
    print(f"{'='*150}\n")

else:
    print("❌ NO SETUPS FOUND!")
