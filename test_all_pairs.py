#!/usr/bin/env python
"""
Test validation rules on ALL pairs to verify they're working
"""

import os
import django
import sys
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce
from trading.tce.utils import is_uptrend, is_downtrend
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend

def calculate_slope(values, period=20):
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

data_dir = Path('./training_data')
csv_files = list(data_dir.glob('*.csv'))

print("="*80)
print("LOCAL TEST: VALIDATION RULES ON ALL PAIRS")
print("="*80 + "\n")

all_results = []
total_setups = 0

for csv_file in sorted(csv_files):
    try:
        df = pd.read_csv(csv_file)
        df.columns = [col.lower().strip() for col in df.columns]
        
        if 'price' in df.columns and 'close' not in df.columns:
            df['close'] = df['price']
        for col in df.columns:
            if 'open' in col.lower() and 'open' not in df.columns:
                df['open'] = df[col]
            elif 'high' in col.lower() and 'high' not in df.columns:
                df['high'] = df[col]
            elif 'low' in col.lower() and 'low' not in df.columns:
                df['low'] = df[col]
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 250:
            continue
        
        stats = {
            'pair': csv_file.stem.upper(),
            'setups_found': 0,
            'checked': 0,
            'trend_ok': 0
        }
        
        # Scan every 15 candles
        for row_idx in range(250, len(df) - 50, 15):
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            open_price = df['open'].values
            
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
            
            tr_list = []
            for i in range(1, min(15, len(high_window))):
                tr = max(
                    high_window[i] - low_window[i],
                    abs(high_window[i] - close_window[i-1]),
                    abs(low_window[i] - close_window[i-1])
                )
                tr_list.append(tr)
            atr = np.mean(tr_list) if tr_list else 0.001
            
            indicators = Indicators(
                ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
                slope6=slope6, slope18=slope18, slope50=slope50, slope200=0,
                atr=atr
            )
            
            candle = Candle(
                open=float(open_price[row_idx]),
                high=float(high[row_idx]),
                low=float(low[row_idx]),
                close=float(close[row_idx]),
                timestamp=str(row_idx)
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
                    timestamp=str(i)
                )
                for i in range(len(recent_close))
            ]
            
            structure = MarketStructure(
                highs=list(recent_high[-50:]),
                lows=list(recent_low[-50:])
            )
            
            try:
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
                    symbol=stats['pair']
                )
                
                stats['checked'] += 1
                if result.get('trend_ok'):
                    stats['trend_ok'] += 1
                if result.get('is_valid'):
                    stats['setups_found'] += 1
                    total_setups += 1
            except:
                pass
        
        if stats['checked'] > 0:
            all_results.append(stats)
            status = "PASS" if stats['setups_found'] > 0 else "FAIL"
            print(f"{stats['pair']:10} - Checked: {stats['checked']:3} | Trend: {stats['trend_ok']:3}/{stats['checked']:3} ({100*stats['trend_ok']/stats['checked']:4.1f}%) | Valid: {stats['setups_found']:2} [{status}]")
    
    except Exception as e:
        pass

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal pairs tested: {len(all_results)}")
print(f"Total valid setups found: {total_setups}")
pairs_with_setups = sum(1 for r in all_results if r['setups_found'] > 0)
print(f"Pairs with at least 1 setup: {pairs_with_setups}/{len(all_results)} ({100*pairs_with_setups/len(all_results):.0f}%)")
avg_trend_rate = np.mean([r['trend_ok']/r['checked'] for r in all_results if r['checked'] > 0]) * 100
print(f"\nAverage trend pass rate: {avg_trend_rate:.1f}%")

if total_setups > 0:
    print(f"\nSUCCESS: Found {total_setups} valid setups across {len(all_results)} pairs!")
else:
    print(f"\nWARNING: No valid setups found")
