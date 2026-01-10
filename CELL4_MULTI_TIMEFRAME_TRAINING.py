# ============================================================================
# CELL 4: MULTI-TIMEFRAME TCE TRAINING (100X MORE DATA!)
# ============================================================================
# Extract TCE setups from MULTIPLE timeframes to maximize training data:
# - 1M (1-Minute)
# - 5M (5-Minute)
# - 15M (15-Minute)
# - 30M (30-Minute)
# - 1H (Hourly)
# - 4H (4-Hour)
# - 1D (Daily)
# - 1W (Weekly)
#
# This gives us 100X+ more training examples from the same historical data!

import sys
sys.path.insert(0, '/content/fluxpoint')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
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
print("CELL 4: MULTI-TIMEFRAME TCE TRAINING")
print("="*80)

# ============================================================================
# LOAD AND RESAMPLE DATA TO MULTIPLE TIMEFRAMES
# ============================================================================

# For Colab (Google Drive path)
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

def resample_to_timeframe(df, timeframe):
    """
    Resample daily data to different timeframes
    
    Args:
        df: DataFrame with Date, Open, High, Low, Close columns
        timeframe: '1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W'
    
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df.set_index('Date', inplace=True)
    
    # Helper function to create intraday candles with realistic price movement
    def create_intraday_candles(df, num_candles_per_day, time_delta_minutes):
        """Create synthetic intraday candles from daily data"""
        intraday_data = []
        for idx, row in df.iterrows():
            daily_range = row['High'] - row['Low']
            daily_open = row['Open']
            daily_close = row['Close']
            
            # Create realistic price path using random walk
            price_path = np.linspace(daily_open, daily_close, num_candles_per_day + 1)
            # Add volatility
            noise = np.random.normal(0, daily_range * 0.05, num_candles_per_day + 1)
            price_path = price_path + noise
            
            for i in range(num_candles_per_day):
                candle_range = daily_range / num_candles_per_day * np.random.uniform(0.5, 1.5)
                mid_price = price_path[i]
                
                intraday_data.append({
                    'Date': idx + pd.Timedelta(minutes=i * time_delta_minutes),
                    'Open': price_path[i],
                    'High': min(row['High'], mid_price + candle_range * 0.6),
                    'Low': max(row['Low'], mid_price - candle_range * 0.6),
                    'Close': price_path[i + 1],
                    'Volume': row.get('Volume', 0) / num_candles_per_day
                })
        
        resampled = pd.DataFrame(intraday_data)
        resampled.set_index('Date', inplace=True)
        return resampled
    
    # For daily data, simulate intraday candles
    if timeframe == '1M':
        # 1440 minutes in a day (24 * 60)
        # Sample every 10 minutes to avoid too much data (144 candles/day)
        resampled = create_intraday_candles(df, 144, 10)
    
    elif timeframe == '5M':
        # 288 5-minute candles per day (24 * 60 / 5)
        resampled = create_intraday_candles(df, 288, 5)
    
    elif timeframe == '15M':
        # 96 15-minute candles per day (24 * 60 / 15)
        resampled = create_intraday_candles(df, 96, 15)
    
    elif timeframe == '30M':
        # 48 30-minute candles per day (24 * 60 / 30)
        resampled = create_intraday_candles(df, 48, 30)
    
    elif timeframe == '1H':
        # 24 hourly candles per day
        resampled = create_intraday_candles(df, 24, 60)
    
    elif timeframe == '4H':
        # 6 4-hour candles per day
        resampled = create_intraday_candles(df, 6, 240)
    
    elif timeframe == '1D':
        # Already daily, just return
        resampled = df
    
    elif timeframe == '1W':
        # Resample to weekly
        resampled = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    resampled.reset_index(inplace=True)
    return resampled

# Load all pairs and create multi-timeframe datasets
print("\n🔄 Creating multi-timeframe datasets...")

# All supported timeframes (ordered from smallest to largest)
timeframes = ['1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W']
multi_tf_data = {}  # {(symbol, timeframe): DataFrame}

print(f"⏱️  Timeframes to generate: {', '.join(timeframes)}")
print(f"⚠️  WARNING: Minute-level data will create HUGE datasets (100K+ candles per pair)")
print(f"   Consider using only 15M, 30M, 1H, 4H, 1D, 1W for faster training\n")

csv_files = sorted(data_dir.glob('*_data.csv'))
for csv_file in csv_files:
    symbol = csv_file.stem.replace('_data', '').upper()
    try:
        # Load daily data
        df_daily = pd.read_csv(csv_file)
        df_daily['Date'] = pd.to_datetime(df_daily['Date'])
        df_daily = df_daily.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'])
        df_daily = df_daily.sort_values('Date').reset_index(drop=True)
        
        if len(df_daily) < 250:
            continue
        
        # Create datasets for each timeframe
        for tf in timeframes:
            df_tf = resample_to_timeframe(df_daily, tf)
            if len(df_tf) >= 250:
                multi_tf_data[(symbol, tf)] = df_tf
                print(f"  ✅ {symbol:<10} {tf:<4} {len(df_tf):>6} candles")
    
    except Exception as e:
        print(f"  ❌ {symbol}: {str(e)[:50]}")

print(f"\n✅ Created {len(multi_tf_data)} timeframe datasets from {len(set(k[0] for k in multi_tf_data.keys()))} pairs")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📊 Device: {device}")
print(f"📈 Timeframes: {', '.join(timeframes)}\n")

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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def calculate_slope(values, period=20):
    """Calculate slope of a moving average"""
    if len(values) < period:
        return 0
    x = np.arange(period)
    y = np.array(values[-period:])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

def extract_features(row_idx, df, direction, result_dict, timeframe, recent_candles_limit=50):
    """Extract features from a TCE setup (adapted for multiple timeframes)"""
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
        ma6, ma18, ma50, ma200,
        slope6, slope18, slope50, slope200,
        atr,
        ma6/ma18 if ma18 > 0 else 1,
        ma18/ma50 if ma50 > 0 else 1,
        ma50/ma200 if ma200 > 0 else 1,
        close[row_idx]/ma6 if ma6 > 0 else 1,
        (close[row_idx] - ma6) / atr if atr > 0 else 0,
        (close[row_idx] - ma18) / atr if atr > 0 else 0,
        (close[row_idx] - ma50) / atr if atr > 0 else 0,
        (close[row_idx] - ma200) / atr if atr > 0 else 0,
        np.std(recent_close[-20:]) if len(recent_close) >= 20 else 0,
        np.std(recent_close[-50:]) if len(recent_close) >= 50 else 0,
        (high_window[-1] - low_window[-1]) / close[row_idx] if close[row_idx] > 0 else 0,
    ]
    
    # ====== 8 RULE SCORES ======
    if result_dict:
        features.extend([
            result_dict.get('rule1_trend', 0.5),
            result_dict.get('rule2_correlation', 0.5),
            result_dict.get('rule3_multi_tf', 0.5),
            result_dict.get('rule4_ma_retest', 0.5),
            result_dict.get('rule5_sr_filter', 0.5),
            result_dict.get('rule6_risk_mgmt', 0.5),
            result_dict.get('rule7_order_placement', 0.5),
            result_dict.get('rule8_fibonacci', 0.5),
        ])
    else:
        features.extend([0.5] * 8)
    
    # ====== RISK METRICS ======
    if result_dict:
        features.extend([
            result_dict.get('risk_reward_ratio', 1.5),
            result_dict.get('sl_pips', 20.0),
            result_dict.get('tp_pips', 30.0),
            result_dict.get('position_size', 0.1),
        ])
    else:
        features.extend([1.5, 20.0, 30.0, 0.1])
    
    # ====== TREND & DIRECTION ======
    direction_encoding = 1.0 if direction == "BUY" else 0.0
    uptrend_flag = float(is_valid_uptrend(Indicators(ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
                                                       slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200, atr=atr),
                                          MarketStructure(highs=list(recent_high), lows=list(recent_low))))
    downtrend_flag = float(is_valid_downtrend(Indicators(ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
                                                          slope6=slope6, slope18=slope18, slope50=slope50, slope200=slope200, atr=atr),
                                             MarketStructure(highs=list(recent_high), lows=list(recent_low))))
    
    features.extend([
        direction_encoding,
        uptrend_flag,
        downtrend_flag,
    ])
    
    # ====== MARKET CONDITIONS ======
    threshold = np.percentile([a for a in tr_list], 75) if tr_list else 0.005
    volatility_extreme = 1.0 if atr > threshold else 0.0
    price_near_ma6 = 1.0 if abs(close[row_idx] - ma6) < atr else 0.0
    
    features.extend([
        volatility_
        '1M': 0.0,
        '5M': 0.125,
        '15M': 0.25,
        '30M': 0.375,
        '1H': 0.5,
        '4H': 0.625,
        '1D': 0.75,
        '1W': 1.0
    
        price_near_ma6,
    ])
    
    # ====== TIMEFRAME ENCODING (NEW!) ======
    tf_encoding = {'1H': 0.0, '4H': 0.33, '1D': 0.67, '1W': 1.0}
    features.append(tf_encoding.get(timeframe, 0.5))  # Feature 38: Timeframe
    
    return features

# ============================================================================
# VALIDATE SETUPS ACROSS ALL TIMEFRAMES
# ============================================================================

print("🔍 Scanning for valid TCE setups across ALL timeframes...\n")

X_list = []
y_list = []
valid_setups_detailed = []

# Statistics per timeframe
tf_stats = {tf: {'checked': 0, 'valid': 0} for tf in timeframes}

for (symbol, timefrNth candle (to balance speed vs coverage)
    # Minute-level: sample heavily to avoid overwhelming dataset
    step_sizes = {
        '1M': 100,   # Every 100th minute candle (~ every 10 hours)
        '5M': 50,    # Every 50th 5-min candle (~ every 4 hours)
        '15M': 20,   # Every 20th 15-min candle (~ every 5 hours)
        '30M': 10,   # Every 10th 30-min candle (~ every 5 hours)
        '1H': 5,     # Every 5th hourly candle
        '4H': 3,     # Every 3rd 4-hour candle
        '1D': 1,     # Every daily candle
        '1W': 1      # Every weekly candle
    }
    step = step_sizes.get(timeframe, 1)
    df = df.dropna()
    
    if len(df) < 250:
        continue
    
    setups_valid = 0
    
    # Sample every 10th candle (to balance speed vs coverage)
    step = 10 if timeframe == '1H' else 5 if timeframe == '4H' else 1
    
    for row_idx in range(250, len(df) - 50, step):
        try:
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            open_price = df['Open'].values
            dates = df['Date'].values
            
            close_window = close[row_idx-240:row_idx+1]
            high_window = high[row_idx-240:row_idx+1]
            low_window = low[row_idx-240:row_idx+1]
            
            # Calculate indicators
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
            
            # Fibonacci calculation
            ma_ref = ma18
            fib_level = 0.618
            
            recent_highs = high[max(0, row_idx-50):row_idx+1]
            recent_lows = low[max(0, row_idx-50):row_idx+1]
            swing_high = np.max(recent_highs) if len(recent_highs) > 0 else close[row_idx]
            swing_low = np.min(recent_lows) if len(recent_lows) > 0 else close[row_idx]
            fib_range = swing_high - swing_low
            fib_618_price = swing_high - (fib_range * 0.618)
            
            if candle.low < ma_ref and atr > 0:
                depth = (ma_ref - candle.low) / atr
                if depth < 0.5:
                    fib_level = 0.382
                elif depth < 1.0:
                    fib_level = 0.5
            
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
            
            structure = MarketStructure(
                highs=list(recent_high[-50:]),
                lows=list(recent_low[-50:])
            )
            
            # Validate TCE setup
            result = validate_tce(
                candle=candle,
                indicators=indicators,
                swing=swing,
                sr_levels=[],
                higher_tf_candles=[],
                correlations={},
                structure=structure,
                recent_candles=recent_candles,
                timeframe=timeframe,
                account_balance=10000.0,
                risk_percentage=1.0,
                symbol=symbol
            )
            
            tf_stats[timeframe]['checked'] += 1
            
            if result["is_valid"]:
                setups_valid += 1
                tf_stats[timeframe]['valid'] += 1
                
                rule_scores = {
                    'rule1_trend': 1.0 if result['trend_ok'] else 0.0,
                    'rule2_correlation': 1.0 if result['correlation_ok'] else 0.0,
                    'rule3_multi_tf': 1.0 if result['multi_tf_ok'] else 0.0,
                    'rule4_ma_retest': 1.0 if result['ma_retest_ok'] else 0.0,
                    'rule5_sr_filter': 1.0,
                    'rule6_risk_mgmt': 1.0 if result['risk_management_ok'] else 0.0,
                    'rule7_order_placement': 1.0,
                    'rule8_fibonacci': 1.0 if result['fib_ok'] else 0.0,
                    'risk_reward_ratio': result.get('risk_reward_ratio', 1.5),
                    'sl_pips': result.get('sl_pips', 20.0),
                    'tp_pips': result.get('tp_pips', 30.0),
                    'position_size': result.get('position_size', 0.1),
                }
                
                features = extract_features(row_idx, df, result['direction'], rule_scores, timeframe)
                if features:
                    X_list.append(features)
                    y_list.append(1.0)
                    
                    valid_setups_detailed.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'date': dates[row_idx],
                        'price': close[row_idx],
                        'direction': result['direction'],
                        'stop_loss': result['stop_loss'],
                        'take_profit': result['take_profit'],
                        'risk_reward': result['risk_reward_ratio'],
                        'sl_pips': result['sl_pips'],
                        'tp_pips': result['tp_pips'],
                    })
        
        except Exception as e:
            pass
    
    if setups_valid > 0:
        print(f"  ✅ {symbol:<10} {timeframe:<4} {setups_valid:>4} valid setups")

print(f"\n{'='*80}")
print(f"📊 MULTI-TIMEFRAME SUMMARY:\n")

total_valid = sum(s['valid'] for s in tf_stats.values())
total_checked = sum(s['checked'] for s in tf_stats.values())

for tf in timeframes:
    checked = tf_stats[tf]['checked']
    valid = tf_stats[tf]['valid']
    if checked > 0:
        print(f"  {tf:<4} {valid:>6} valid / {checked:>8} checked ({100*valid/checked:5.2f}%)")

print(f"\n  TOTAL: {total_valid} VALID TCE SETUPS ({total_checked} checked)")
print(f"{'='*80}\n")

# ============================================================================
# TRAIN DEEP LEARNING MODEL
# ============================================================================

if len(X_list) == 0:
    print("❌ ERROR: No valid setups found!")
else:
    print(f"🤖 TRAINING NEURAL NETWORK ON {len(X_list)} MULTI-TIMEFRAME SETUPS\n")
    
    # Prepare data
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Create dataset
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
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
    print(f"   Training Accuracy: {accuracy*100:.1f}%")
    
    # Save model
    save_dir = Path('/content/drive/MyDrive/models') if colab_data_dir.exists() else Path('models')
    save_dir.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / 'tce_multi_tf_model.pt')
    np.save(save_dir / 'scaler_mean.npy', scaler.mean_)
    np.save(save_dir / 'scaler_scale.npy', scaler.scale_)
    
    print(f"\n✅ Model saved to: {save_dir}")
    print(f"   • tce_multi_tf_model.pt")
    print(f"   • scaler_mean.npy")
    print(f"   • scaler_scale.npy")
    
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   • Total setups: {len(X_list)}")
    print(f"   • Feature dimensions: {X.shape[1]}")
    print(f"   • Timeframes used: {', '.join(timeframes)}")
    print(f"   • Model accuracy: {accuracy*100:.1f}%")
    print(f"\n{'='*80}\n")
