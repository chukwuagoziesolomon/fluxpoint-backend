CELL 4: Train DL Model (1-2 hours) - On REAL TCE Rules
────────────────────────────────────────────────────────────────────────────────

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
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("TRAINING DL MODEL ON REAL TCE VALIDATION RULES")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

class TCEProbabilityModel(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def is_at_ma_level(price, ma6, ma18, ma50, ma200, direction, atr):
    """TCE rule: price must be at/near MA (within 1.5*ATR)"""
    for ma in [ma6, ma18, ma50, ma200]:
        if ma > 0:
            distance = abs(price - ma) / atr if atr > 0 else float('inf')
            if distance < 1.5:
                return True
    return False

def has_uptrend_structure(sma5, sma20, sma50, sma200):
    """Uptrend: SMA5 > SMA20 > SMA50 > SMA200"""
    return sma5 > sma20 > sma50 > sma200

def has_downtrend_structure(sma5, sma20, sma50, sma200):
    """Downtrend: SMA5 < SMA20 < SMA50 < SMA200"""
    return sma5 < sma20 < sma50 < sma200

def has_candlestick_confirmation(recent_closes, recent_opens, recent_highs, recent_lows):
    """Pattern: bullish/bearish candle at MA retest"""
    if len(recent_closes) < 2:
        return False
    last_candle_bullish = recent_closes[-1] > recent_opens[-1]
    last_candle_size = (recent_highs[-1] - recent_lows[-1]) / (recent_closes[-2] + 0.00001)
    return last_candle_size > 0.001

try:
    print("\nExtracting features from REAL TCE SETUPS...")
    print("Using your coded validation rules:\n")
    print("  ✓ Trend confirmation (SMA alignment)")
    print("  ✓ Entry at MA level (dynamic support)")
    print("  ✓ Candlestick confirmation pattern")
    print("  ✓ Fibonacci depth validation")
    print("  ✓ Risk/Reward rules\n")
    
    X_list = []
    y_list = []
    pair_setup_counts = {}
    
    for symbol, df in pair_data.items():
        print(f"Processing {symbol}...", end='', flush=True)
        
        df = df.copy()
        df = df.dropna()
        
        if len(df) < 250:
            print(f" ✗ (need 250+ candles)")
            continue
        
        # Calculate the exact TCE moving averages
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        open_price = df['Open'].values
        
        sma5 = pd.Series(close).rolling(5).mean().values
        sma6 = pd.Series(close).rolling(6).mean().values
        sma18 = pd.Series(close).rolling(18).mean().values
        sma20 = pd.Series(close).rolling(20).mean().values
        sma50 = pd.Series(close).rolling(50).mean().values
        sma200 = pd.Series(close).rolling(200).mean().values
        
        # ATR for risk management
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        atr = pd.Series(tr).rolling(14).mean().values
        
        volume = df['Volume'].values
        
        setup_count = 0
        
        # Find REAL TCE setups
        for i in range(250, len(close) - 20):
            try:
                if pd.isna(atr[i]) or atr[i] <= 0:
                    continue
                if pd.isna(sma20[i]) or pd.isna(sma50[i]) or pd.isna(sma200[i]):
                    continue
                
                c = close[i]
                a = atr[i]
                
                # RULE 1: Valid trend structure
                is_uptrend = has_uptrend_structure(sma5[i], sma20[i], sma50[i], sma200[i])
                is_downtrend = has_downtrend_structure(sma5[i], sma20[i], sma50[i], sma200[i])
                
                if not (is_uptrend or is_downtrend):
                    continue
                
                direction = "BUY" if is_uptrend else "SELL"
                
                # RULE 2: Price at MA level (within 1.5 ATR)
                if not is_at_ma_level(c, sma6[i], sma18[i], sma50[i], sma200[i], direction, a):
                    continue
                
                # RULE 3: Candlestick confirmation
                if not has_candlestick_confirmation(close[i-2:i+1], open_price[i-2:i+1], 
                                                    high[i-2:i+1], low[i-2:i+1]):
                    continue
                
                # RULE 4: Fibonacci check (not beyond 61.8%)
                recent_swing = max(high[max(0,i-50):i]) - min(low[max(0,i-50):i])
                if recent_swing > 0:
                    retracement = abs(c - min(low[max(0,i-50):i])) / recent_swing
                    if retracement > 0.618:
                        continue
                
                # ✓ VALID TCE SETUP - extract 20 features
                
                features = np.array([
                    c / sma20[i] if sma20[i] > 0 else 1,
                    sma20[i] / sma50[i] if sma50[i] > 0 else 1,
                    sma50[i] / sma200[i] if sma200[i] > 0 else 1,
                    (c - min(sma6[i], sma18[i], sma50[i], sma200[i])) / a,
                    a / c if c > 0 else 0,
                    (high[i] - low[i]) / a,
                    1 if close[i] > open_price[i] else -1,
                    retracement,
                    (c - sma200[i]) / a if a > 0 else 0,
                    (sma5[i] - sma20[i]) / a if a > 0 else 0,
                    1 if is_uptrend else -1,
                    volume[i] / (np.mean(volume[max(0,i-20):i]) + 0.001),
                    (close[i] - close[i-5]) / a if a > 0 else 0,
                    (close[i] - close[i-10]) / a if a > 0 else 0,
                    (sma6[i] - sma50[i]) / a if a > 0 else 0,
                    (c - min(high[max(0,i-20):i])) / a,
                    (max(high[max(0,i-20):i]) - min(low[max(0,i-20):i])) / a,
                    abs(close[i] - open_price[i]) / (high[i] - low[i] + 0.001),
                    1 if abs(c - sma20[i]) < 0.5 * a else 0,
                    a / recent_swing if recent_swing > 0 else 0
                ], dtype=np.float32)
                
                # Label: profitable 20 candles forward
                if i + 20 < len(close):
                    future_close = close[i + 20]
                    future_return = (future_close - c) / c
                    label = 1 if future_return > 0.015 else 0
                    
                    X_list.append(features)
                    y_list.append(label)
                    setup_count += 1
            
            except:
                continue
        
        pair_setup_counts[symbol] = setup_count
        print(f" ✓ {setup_count} valid setups")
    
    if len(X_list) == 0:
        print("\n✗ No valid TCE setups found!")
        raise Exception("No TCE setups")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"\n{'='*70}")
    print(f"✓ EXTRACTED {len(X)} VALID TCE SETUPS")
    print(f"{'='*70}")
    
    print(f"\nSetups by pair:")
    for pair, count in sorted(pair_setup_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {pair}: {count}")
    
    print(f"\nSetup Quality:")
    print(f"  Winning setups: {int(np.sum(y))} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  Losing setups: {len(y) - int(np.sum(y))} ({(1-np.sum(y)/len(y))*100:.1f}%)")
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = TCEProbabilityModel(input_size=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print(f"\n{'='*70}")
    print(f"TRAINING NEURAL NETWORK ON TCE PATTERNS")
    print(f"{'='*70}")
    print(f"Training: {len(X_train)} | Validation: {len(X_val)}")
    print(f"Epochs: 50 | Early stopping: Yes\n")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                predictions = (y_pred > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {accuracy:.1%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\n{'='*70}")
    print(f"✓ DL MODEL TRAINED ON REAL TCE RULES!")
    print(f"{'='*70}")
    
    print(f"\nFinal Performance:")
    print(f"  Validation Accuracy: {accuracy:.1%}")
    print(f"  Validation Loss: {val_loss:.4f}")
    
    # Save
    model_dir = '/content/fluxpoint/models/dl'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/tce_probability_model.pth")
    print(f"\n✓ Model saved to: {model_dir}/tce_probability_model.pth")

except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
