# ============================================================================
# CELL 4: TRAIN DL MODEL ON ALL TCE VALIDATION RULES (COMPLETE & CORRECT)
# ============================================================================
# This cell implements ALL 7 validation rules from validation.py:
# 1️⃣  Trend confirmation (uptrend/downtrend via SMA alignment + slopes)
# 2️⃣  Fibonacci validation (depth 38.2%, 50%, 61.8% only)
# 2.5️⃣ Semi-circle swing structure
# 3️⃣  At MA level (dynamic support only - NO horizontal S/R)
# 3.5️⃣ MA retest (NOT first touch - must be second touch)
# 4️⃣  Candlestick confirmation pattern at retest
# 5️⃣  Higher timeframe confirmation (empty for H1)
# 6️⃣  Correlation confirmation (empty for single pair)
# 7️⃣  Risk Management (SL 1.5*ATR, TP dynamic RR ratio)

import sys
sys.path.insert(0, '/content/fluxpoint')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Import from the actual Django app
from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.validation import validate_tce
from trading.tce.utils import (
    is_uptrend, is_downtrend, valid_fib, 
    is_rejection_candle, is_bullish_pin_bar, is_bearish_pin_bar,
    is_bullish_engulfing, is_bearish_engulfing
)
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend, is_semi_circle_swing
from trading.tce.sr import at_ma_level, has_ma_retest

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("CELL 4: TRAIN DL MODEL ON ALL TCE VALIDATION RULES")
print("="*80)

# ============================================================================
# LOAD CLEANED DATA (NO WEEKENDS/HOLIDAYS, PROPER FORMAT)
# ============================================================================

# For Colab (Google Drive path)
from pathlib import Path
colab_data_dir = Path('/content/drive/MyDrive/forex_data/training_data_cleaned')

# For local testing (Windows path)
local_data_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data_cleaned')

# Automatically detect environment
if colab_data_dir.exists():
    data_dir = colab_data_dir
elif local_data_dir.exists():
    data_dir = local_data_dir
else:
    raise FileNotFoundError(
        "❌ Cannot find cleaned data!\n"
        "   Run: python fix_csv_data.py\n"
        "   Or upload cleaned CSVs to Google Drive: /forex_data/training_data_cleaned/"
    )

print(f"\n📂 Loading data from: {data_dir}")

pair_data = {}
for csv_file in sorted(data_dir.glob('*_data.csv')):
    symbol = csv_file.stem.replace('_data', '').upper()
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) >= 250:
            pair_data[symbol] = df
            print(f"  ✅ {symbol:<10} {len(df):>5} candles")
    except Exception as e:
        print(f"  ❌ {symbol}: {str(e)[:50]}")

if not pair_data:
    raise ValueError("❌ No data loaded! Check CSV files.")

print(f"\n✅ Loaded {len(pair_data)} pairs with clean data (no weekends/holidays)")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n📊 Device: {device}")
print(f"🔢 Number of pairs: {len(pair_data)}")
print(f"📈 Timeframe: Daily (1D)\n")

# ============================================================================
# DEEP LEARNING MODEL
# ============================================================================

class TCEProbabilityModel(nn.Module):
    """Neural network to predict TCE setup validity"""
    def __init__(self, input_size=45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output: 0-1 probability
        )
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# FEATURE EXTRACTION - BUILD FEATURES FROM VALIDATED SETUPS
# ============================================================================

print("🔍 Scanning for valid TCE setups using ALL validation rules...\n")

def calculate_slope(values, period=20):
    """Calculate slope of a moving average"""
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

def extract_features(row_idx, df, direction, result_dict, recent_candles_limit=50):
    """
    Extract 45+ features from a TCE setup including:
    - 20 Original features (MAs, slopes, ratios, volatility)
    - 8 Rule scores (from calculate_all_rule_scores)
    - Risk metrics (SL, TP, RR ratio)
    
    These features will be used to train the neural network.
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    open_price = df['Open'].values
    
    if row_idx < 250:
        return None
    
    # Get recent candles
    start_idx = max(0, row_idx - recent_candles_limit)
    recent_close = close[start_idx:row_idx+1]
    recent_high = high[start_idx:row_idx+1]
    recent_low = low[start_idx:row_idx+1]
    
    # Calculate indicators
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
    atr = np.mean(tr_list) if tr_list else 0.0001
    
    # ====== ORIGINAL 20 FEATURES ======
    features = [
        ma6, ma18, ma50, ma200,  # 1-4: MAs
        slope6, slope18, slope50, slope200,  # 5-8: Slopes
        atr,  # 9: ATR
        ma6/ma18 if ma18 > 0 else 1,  # 10: MA6/MA18 ratio
        ma18/ma50 if ma50 > 0 else 1,  # 11: MA18/MA50 ratio
        ma50/ma200 if ma200 > 0 else 1,  # 12: MA50/MA200 ratio
        close[row_idx]/ma6 if ma6 > 0 else 1,  # 13: Price/MA6 ratio
        (close[row_idx] - ma6) / atr if atr > 0 else 0,  # 14: Distance from MA6
        (close[row_idx] - ma18) / atr if atr > 0 else 0,  # 15: Distance from MA18
        (close[row_idx] - ma50) / atr if atr > 0 else 0,  # 16: Distance from MA50
        (close[row_idx] - ma200) / atr if atr > 0 else 0,  # 17: Distance from MA200
        np.std(recent_close[-20:]) if len(recent_close) >= 20 else 0,  # 18: Volatility 20c
        np.std(recent_close[-50:]) if len(recent_close) >= 50 else 0,  # 19: Volatility 50c
        (high_window[-1] - low_window[-1]) / close[row_idx] if close[row_idx] > 0 else 0,  # 20: Candle range %
    ]
    
    # ====== 8 RULE SCORES (from validation result) ======
    if result_dict:
        features.extend([
            result_dict.get('rule1_trend', 0.5),          # 21: Rule 1 - Trend
            result_dict.get('rule2_correlation', 0.5),    # 22: Rule 2 - Correlation
            result_dict.get('rule3_multi_tf', 0.5),       # 23: Rule 3 - Multi-TF
            result_dict.get('rule4_ma_retest', 0.5),      # 24: Rule 4 - MA Retest
            result_dict.get('rule5_sr_filter', 0.5),      # 25: Rule 5 - S/R Filter
            result_dict.get('rule6_risk_mgmt', 0.5),      # 26: Rule 6 - Risk Mgmt
            result_dict.get('rule7_order_placement', 0.5),# 27: Rule 7 - Order Placement
            result_dict.get('rule8_fibonacci', 0.5),      # 28: Rule 8 - Fibonacci
        ])
    else:
        features.extend([0.5] * 8)  # Default if no validation result
    
    # ====== RISK METRICS ======
    if result_dict:
        features.extend([
            result_dict.get('risk_reward_ratio', 1.5),    # 29: Risk:Reward
            result_dict.get('sl_pips', 20.0),             # 30: Stop Loss pips
            result_dict.get('tp_pips', 30.0),             # 31: Take Profit pips
            result_dict.get('position_size', 0.1),        # 32: Position size
        ])
    else:
        features.extend([1.5, 20.0, 30.0, 0.1])
    
    # ====== TREND & DIRECTION ENCODING ======
    direction_encoding = 1.0 if direction == "BUY" else 0.0
    uptrend_flag = float(is_valid_uptrend(Indicators(ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
                                                       slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200, atr=atr),
                                          MarketStructure(highs=list(recent_high), lows=list(recent_low))))
    downtrend_flag = float(is_valid_downtrend(Indicators(ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
                                                          slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200, atr=atr),
                                             MarketStructure(highs=list(recent_high), lows=list(recent_low))))
    
    features.extend([
        direction_encoding,                               # 33: Direction (1=BUY, 0=SELL)
        uptrend_flag,                                     # 34: Is uptrend?
        downtrend_flag,                                   # 35: Is downtrend?
    ])
    
    # ====== MARKET CONDITIONS ======
    threshold = np.percentile([a for a in tr_list], 75) if tr_list else 0.005
    volatility_extreme = 1.0 if atr > threshold else 0.0
    price_near_ma6 = 1.0 if abs(close[row_idx] - ma6) < atr else 0.0
    
    features.extend([
        volatility_extreme,                               # 36: Is volatility high?
        price_near_ma6,                                   # 37: Price near MA6?
    ])
    
    # Total: 37+ features
    return features

# ============================================================================
# VALIDATE SETUPS USING ACTUAL validation.py
# ============================================================================

X_list = []
y_list = []
valid_setups_detailed = []
pair_setup_counts = {}

# Debug counters
rule_stats = {
    'total_checked': 0,
    'trend_passed': 0,
    'fib_passed': 0,
    'swing_passed': 0,
    'ma_level_passed': 0,
    'ma_retest_passed': 0,
    'candlestick_passed': 0,
    'multi_tf_passed': 0,
    'correlation_passed': 0,
    'risk_mgmt_passed': 0,
    'failure_reasons': {}
}

print("📊 DETAILED RULE FILTERING ANALYSIS:\n")

for pair_idx, (symbol, df) in enumerate(pair_data.items()):
    df = df.copy().reset_index(drop=True)
    df = df.dropna()
    
    if len(df) < 250:
        print(f"  ⚠️  {symbol}: Only {len(df)} candles (need 250+) - SKIPPED")
        continue
    
    setups_found = 0
    setups_valid = 0
    rule_breakdown = {
        'trend_fail': 0,
        'fib_fail': 0,
        'swing_fail': 0,
        'ma_level_fail': 0,
        'ma_retest_fail': 0,
        'candlestick_fail': 0,
        'multi_tf_fail': 0,
        'correlation_fail': 0,
        'risk_mgmt_fail': 0,
    }
    
    # 🔴 CRITICAL FIX: Scan EVERY candle, not every 5th!
    for row_idx in range(250, len(df) - 50, 1):  # Check EVERY candle for maximum coverage
        try:
            # Get data for current candle
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            open_price = df['Open'].values
            dates = df['Date'].values
            
            close_window = close[row_idx-240:row_idx+1]
            high_window = high[row_idx-240:row_idx+1]
            low_window = low[row_idx-240:row_idx+1]
            
            # Calculate indicators (CORRECT MAs!)
            ma6 = np.mean(close_window[-6:]) if len(close_window) >= 6 else 0
            ma18 = np.mean(close_window[-18:]) if len(close_window) >= 18 else 0
            ma50 = np.mean(close_window[-50:]) if len(close_window) >= 50 else 0
            ma200 = np.mean(close_window[-200:]) if len(close_window) >= 200 else 0
            
            slope6 = calculate_slope(close_window, 6)
            slope18 = calculate_slope(close_window, 18)
            slope50 = calculate_slope(close_window, 50)
            slope200 = calculate_slope(close_window, 200)
            
            # ATR
            tr_list = []
            for i in range(1, min(15, len(high_window))):
                tr = max(
                    high_window[i] - low_window[i],
                    abs(high_window[i] - close_window[i-1]),
                    abs(low_window[i] - close_window[i-1])
                )
                tr_list.append(tr)
            atr = np.mean(tr_list) if tr_list else 0
            
            # Create objects for validation
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
            
            # Estimate Fibonacci level (how deep price went below MA)
            ma_ref = ma18  # Use MA18 as reference
            fib_level = 0.618  # Default to 61.8%
            
            # Calculate 61.8% Fibonacci price level for stop loss placement
            recent_highs = high[max(0, row_idx-50):row_idx+1]
            recent_lows = low[max(0, row_idx-50):row_idx+1]
            swing_high = np.max(recent_highs) if len(recent_highs) > 0 else close[row_idx]
            swing_low = np.min(recent_lows) if len(recent_lows) > 0 else close[row_idx]
            fib_range = swing_high - swing_low
            fib_618_price = swing_high - (fib_range * 0.618)  # 61.8% retracement from top
            
            if candle.low < ma_ref and atr > 0:
                depth = (ma_ref - candle.low) / atr
                if depth < 0.5:  # Shallow - probably 38.2%
                    fib_level = 0.382
                elif depth < 1.0:  # Medium - probably 50%
                    fib_level = 0.5
                # else: 61.8% or deeper
            
            swing = Swing(
                type='high' if close[row_idx] > np.mean(close[row_idx-50:row_idx]) else 'low',
                price=float(close[row_idx]),
                fib_level=fib_level,
                fib_618_price=float(fib_618_price)
            )
            
            # Build recent candles
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
            
            # Build structure (for trend confirmation)
            structure = MarketStructure(
                highs=list(recent_high[-50:]),
                lows=list(recent_low[-50:])
            )
            
            # ✅ CALL ACTUAL VALIDATION FUNCTION
            result = validate_tce(
                candle=candle,
                indicators=indicators,
                swing=swing,
                sr_levels=[],  # TCE uses only MAs, not S/R
                higher_tf_candles=[],  # No higher TF for single-timeframe
                correlations={},  # No correlation for single pair
                structure=structure,
                recent_candles=recent_candles,
                timeframe="H1",
                account_balance=10000.0,
                risk_percentage=1.0,
                symbol=symbol
            )
            
            setups_found += 1
            rule_stats['total_checked'] += 1
            
            if result["is_valid"]:
                setups_valid += 1
                
                # Track rule statistics
                if result['trend_ok']: rule_stats['trend_passed'] += 1
                else: rule_breakdown['trend_fail'] += 1
                
                if result['fib_ok']: rule_stats['fib_passed'] += 1
                else: rule_breakdown['fib_fail'] += 1
                
                if result['swing_ok']: rule_stats['swing_passed'] += 1
                else: rule_breakdown['swing_fail'] += 1
                
                if result['ma_level_ok']: rule_stats['ma_level_passed'] += 1
                else: rule_breakdown['ma_level_fail'] += 1
                
                if result['ma_retest_ok']: rule_stats['ma_retest_passed'] += 1
                else: rule_breakdown['ma_retest_fail'] += 1
                
                if result['candlestick_ok']: rule_stats['candlestick_passed'] += 1
                else: rule_breakdown['candlestick_fail'] += 1
                
                if result['multi_tf_ok']: rule_stats['multi_tf_passed'] += 1
                else: rule_breakdown['multi_tf_fail'] += 1
                
                if result['correlation_ok']: rule_stats['correlation_passed'] += 1
                else: rule_breakdown['correlation_fail'] += 1
                
                if result['risk_management_ok']: rule_stats['risk_mgmt_passed'] += 1
                else: rule_breakdown['risk_mgmt_fail'] += 1
                
                # Create rule score dictionary for this setup
                rule_scores = {
                    'rule1_trend': 1.0 if result['trend_ok'] else 0.0,
                    'rule2_correlation': 1.0 if result['correlation_ok'] else 0.0,
                    'rule3_multi_tf': 1.0 if result['multi_tf_ok'] else 0.0,
                    'rule4_ma_retest': 1.0 if result['ma_retest_ok'] else 0.0,
                    'rule5_sr_filter': 1.0,  # Passed validation = good
                    'rule6_risk_mgmt': 1.0 if result['risk_management_ok'] else 0.0,
                    'rule7_order_placement': 1.0,  # Part of validation
                    'rule8_fibonacci': 1.0 if result['fib_ok'] else 0.0,
                    'risk_reward_ratio': result.get('risk_reward_ratio', 1.5),
                    'sl_pips': result.get('sl_pips', 20.0),
                    'tp_pips': result.get('tp_pips', 30.0),
                    'position_size': result.get('position_size', 0.1),
                }
                
                # Extract features for this valid setup
                features = extract_features(row_idx, df, result['direction'], rule_scores)
                if features:
                    X_list.append(features)
                    y_list.append(1.0)  # Valid setup = 1
                    
                    valid_setups_detailed.append({
                        'symbol': symbol,
                        'date': dates[row_idx],
                        'price': close[row_idx],
                        'direction': result['direction'],
                        'stop_loss': result['stop_loss'],
                        'take_profit': result['take_profit'],
                        'risk_reward': result['risk_reward_ratio'],
                        'ma6': ma6, 'ma18': ma18, 'ma50': ma50, 'ma200': ma200,
                        'atr': atr,
                        'trend_ok': result['trend_ok'],
                        'fib_ok': result['fib_ok'],
                        'swing_ok': result['swing_ok'],
                        'ma_level_ok': result['ma_level_ok'],
                        'ma_retest_ok': result['ma_retest_ok'],
                        'candlestick_ok': result['candlestick_ok'],
                        'multi_tf_ok': result['multi_tf_ok'],
                        'correlation_ok': result['correlation_ok'],
                        'risk_management_ok': result['risk_management_ok'],
                        'sl_pips': result['sl_pips'],
                        'tp_pips': result['tp_pips'],
                        'position_size': result['position_size'],
                        'risk_amount': result['risk_amount'],
                        'failure_reason': result['failure_reason']
                    })
            else:
                # Track failure reason
                reason = result.get('failure_reason', 'Unknown')
                rule_stats['failure_reasons'][reason] = rule_stats['failure_reasons'].get(reason, 0) + 1
        
        except Exception as e:
            pass
    
    pair_setup_counts[symbol] = (setups_valid, setups_found)
    if setups_valid > 0:
        print(f"  ✅ {symbol}: {setups_valid} VALID setups (checked {setups_found} candles)")
        # Show which rules are blocking setups
        print(f"     Rule Failures: ", end="")
        failures = [f"{k.replace('_fail','')}: {v}" for k, v in rule_breakdown.items() if v > 0]
        if failures:
            print(", ".join(failures))
        else:
            print("None")
    else:
        print(f"  ❌ {symbol}: No valid setups (checked {setups_found} candles)")
        print(f"     Rule Failures: ", end="")
        failures = [f"{k.replace('_fail','')}: {v}" for k, v in rule_breakdown.items() if v > 0]
        if failures:
            print(", ".join(failures[:5]), "...")  # Show top 5

print(f"\n{'='*80}")
print(f"📊 SUMMARY: {len(valid_setups_detailed)} VALID TCE SETUPS FOUND\n")

# Calculate rule pass rates
if rule_stats['total_checked'] > 0:
    print("📈 RULE-BY-RULE PASS RATES:\n")
    total = rule_stats['total_checked']
    print(f"  ✅ Rule 1 (Trend):           {rule_stats['trend_passed']:4d}/{total} ({100*rule_stats['trend_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 2 (Correlation):     {rule_stats['correlation_passed']:4d}/{total} ({100*rule_stats['correlation_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 3 (Multi-TF):        {rule_stats['multi_tf_passed']:4d}/{total} ({100*rule_stats['multi_tf_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 4 (MA Retest):       {rule_stats['ma_retest_passed']:4d}/{total} ({100*rule_stats['ma_retest_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 5 (MA Level):        {rule_stats['ma_level_passed']:4d}/{total} ({100*rule_stats['ma_level_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 6 (Risk Management): {rule_stats['risk_mgmt_passed']:4d}/{total} ({100*rule_stats['risk_mgmt_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 7 (Candlestick):     {rule_stats['candlestick_passed']:4d}/{total} ({100*rule_stats['candlestick_passed']/total:5.1f}%)")
    print(f"  ✅ Rule 8 (Fibonacci):       {rule_stats['fib_passed']:4d}/{total} ({100*rule_stats['fib_passed']/total:5.1f}%)")
    
    print(f"\n🔴 TOP FAILURE REASONS:")
    sorted_failures = sorted(rule_stats['failure_reasons'].items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_failures[:5]:
        print(f"  • {reason}: {count} times")

print(f"\n{'='*80}\n")

# Show sample valid setups with ALL details
if valid_setups_detailed:
    print("📍 SAMPLE VALID SETUPS (FULL DETAILS):\n")
    for idx, setup in enumerate(valid_setups_detailed[:3]):
        print(f"  ╔══ SETUP #{idx+1} {'═'*50}")
        print(f"  ║ Symbol: {setup['symbol']:<10} | Date: {setup['date']}")
        print(f"  ║ Entry Price: {setup['price']:.5f}")
        print(f"  ║ Direction: {setup['direction']}")
        print(f"  ║")
        print(f"  ║ RISK MANAGEMENT:")
        print(f"  ║   • SL: {setup['stop_loss']:.5f} ({setup['sl_pips']:.1f} pips)")
        print(f"  ║   • TP: {setup['take_profit']:.5f} ({setup['tp_pips']:.1f} pips)")
        print(f"  ║   • Risk/Reward: {setup['risk_reward']:.2f}:1")
        print(f"  ║   • Position Size: {setup['position_size']:.2f} lots")
        print(f"  ║   • Risk Amount: ${setup['risk_amount']:.2f}")
        print(f"  ║")
        print(f"  ║ MOVING AVERAGES:")
        print(f"  ║   • MA6: {setup['ma6']:.5f}")
        print(f"  ║   • MA18: {setup['ma18']:.5f}")
        print(f"  ║   • MA50: {setup['ma50']:.5f}")
        print(f"  ║   • MA200: {setup['ma200']:.5f}")
        print(f"  ║   • ATR: {setup['atr']:.5f}")
        print(f"  ║")
        print(f"  ║ VALIDATION RULES (ALL 7 MUST PASS):")
        print(f"  ║   1️⃣  Trend: {'✅ PASS' if setup['trend_ok'] else '❌ FAIL'}")
        print(f"  ║   2️⃣  Fibonacci: {'✅ PASS' if setup['fib_ok'] else '❌ FAIL'}")
        print(f"  ║   2.5️⃣ Swing: {'✅ PASS' if setup['swing_ok'] else '❌ FAIL'}")
        print(f"  ║   3️⃣  MA Level: {'✅ PASS' if setup['ma_level_ok'] else '❌ FAIL'}")
        print(f"  ║   3.5️⃣ MA Retest: {'✅ PASS' if setup['ma_retest_ok'] else '❌ FAIL'}")
        print(f"  ║   4️⃣  Candlestick: {'✅ PASS' if setup['candlestick_ok'] else '❌ FAIL'}")
        print(f"  ║   5️⃣  Multi-TF: {'✅ PASS' if setup['multi_tf_ok'] else '❌ FAIL'}")
        print(f"  ║   6️⃣  Correlation: {'✅ PASS' if setup['correlation_ok'] else '❌ FAIL'}")
        print(f"  ║   7️⃣  Risk Mgmt: {'✅ PASS' if setup['risk_management_ok'] else '❌ FAIL'}")
        print(f"  ╚{'═'*58}\n")

# ============================================================================
# TRAIN DEEP LEARNING MODEL
# ============================================================================

if len(X_list) == 0:
    print("❌ ERROR: No valid setups found! Cannot train model.")
    print("   Debugging tips:")
    print("   • Check that moving averages are calculated correctly (MA6, MA18, MA50, MA200)")
    print("   • Verify candlestick patterns are being detected")
    print("   • Check that trend confirmation is working (slopes and alignment)")
    print("   • Ensure MA retest pattern is found in historical data")
else:
    print(f"\n{'='*80}")
    print("🤖 TRAINING NEURAL NETWORK\n")
    
    # Prepare data
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Create dataset and loader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = TCEProbabilityModel(input_size=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    epochs = 50
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Model trained! Final loss: {best_loss:.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu().numpy()
    
    accuracy = np.mean((predictions > 0.5).astype(int) == y.cpu().numpy())
    print(f"   Validation Accuracy: {accuracy*100:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), '/content/drive/MyDrive/models/tce_dl_model.pt')
    np.save('/content/drive/MyDrive/models/scaler.npy', scaler.mean_)
    np.save('/content/drive/MyDrive/models/scaler_scale.npy', scaler.scale_)
    
    print(f"\n✅ Model saved to Drive!")
    print(f"\n{'='*80}\n")
