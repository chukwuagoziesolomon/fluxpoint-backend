"""
Multi-Pair RL Training Script

Example usage of the multi-pair trainer to train on multiple currency pairs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Adjust import path as needed
from trading.rl.multi_pair_training import MultiPairRLTrainer, train_rl_multipair
from trading.tce.validation import validate_tce_setups
from trading.mt5_integration import get_historical_data


# ============================================================================
# EXAMPLE 1: Using convenience function (simplest)
# ============================================================================

def train_multipair_simple():
    """
    Simplest way to train on multiple pairs.
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Multi-Pair Training")
    print("="*70 + "\n")
    
    # Define pairs to trade
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Load data for each pair
    pair_data = {}
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        
        # Load historical data (adjust date range as needed)
        candles = get_historical_data(
            symbol=symbol,
            start_date="2023-01-01",
            end_date="2023-12-31",
            timeframe="H1"
        )
        
        # Validate TCE setups
        valid_setups = validate_tce_setups(
            candles=candles,
            symbol=symbol
        )
        
        print(f"  ✓ {len(candles)} candles, {len(valid_setups)} setups\n")
        
        pair_data[symbol] = (candles, valid_setups)
    
    # Train on all pairs simultaneously
    metrics = train_rl_multipair(
        pair_data=pair_data,
        symbols=symbols,
        model_name="tce_execution_eur_gbp_jpy",
        initial_balance=10000,
        risk_percentage=1.0,
        total_timesteps=200000
    )
    
    print("\nTraining complete!")
    print(f"Results: {metrics}")


# ============================================================================
# EXAMPLE 2: Using trainer class directly (more control)
# ============================================================================

def train_multipair_advanced():
    """
    More advanced usage with custom configuration.
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Advanced Multi-Pair Training")
    print("="*70 + "\n")
    
    # Define all major pairs
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD']
    
    # Create trainer
    trainer = MultiPairRLTrainer(
        model_name="tce_execution_5pairs_v1",
        initial_balance=10000,
        risk_percentage=1.0,
        symbols=symbols,
        require_candlestick_pattern=True,
        enforce_risk_management=True
    )
    
    # Load data
    pair_data = {}
    
    for symbol in symbols:
        print(f"📥 Loading {symbol}...")
        
        try:
            candles = get_historical_data(
                symbol=symbol,
                start_date="2023-01-01",
                end_date="2023-12-31",
                timeframe="H1"
            )
            
            valid_setups = validate_tce_setups(
                candles=candles,
                symbol=symbol
            )
            
            if len(valid_setups) == 0:
                print(f"  ⚠️  No valid setups for {symbol}, skipping...")
                continue
            
            print(f"  ✓ {len(valid_setups)} setups")
            pair_data[symbol] = (candles, valid_setups)
        
        except Exception as e:
            print(f"  ❌ Error loading {symbol}: {e}")
            continue
    
    print(f"\n✅ Loaded {len(pair_data)} pairs\n")
    
    # Prepare data
    train_env, eval_env, stats = trainer.prepare_training_data(pair_data)
    
    print(f"\n📊 Data Preparation Summary:")
    print(f"   Total setups: {stats['total_setups']}")
    print(f"   Training: {stats['train_setups']}")
    print(f"   Evaluation: {stats['eval_setups']}")
    print(f"   Per-pair breakdown: {stats['setups_per_pair']}\n")
    
    # Train with custom hyperparameters
    metrics = trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=300000,  # More timesteps for multiple pairs
        eval_freq=10000,
        save_freq=20000
    )
    
    # Save model
    model_path = "models/rl/tce_execution_5pairs_v1"
    trainer.save_model(model_path)
    
    return metrics


# ============================================================================
# EXAMPLE 3: Staged training (start small, add pairs)
# ============================================================================

def train_multipair_staged():
    """
    Progressive training: start with 2 pairs, then add more.
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Staged Multi-Pair Training")
    print("="*70 + "\n")
    
    # Stage 1: Train on 2 major pairs
    print("STAGE 1: Training on EURUSD + GBPUSD\n")
    
    pair_data_stage1 = {}
    
    for symbol in ['EURUSD', 'GBPUSD']:
        candles = get_historical_data(
            symbol=symbol,
            start_date="2023-01-01",
            end_date="2023-12-31",
            timeframe="H1"
        )
        valid_setups = validate_tce_setups(candles, symbol)
        pair_data_stage1[symbol] = (candles, valid_setups)
    
    trainer1 = MultiPairRLTrainer(
        model_name="tce_stage1_eur_gbp",
        initial_balance=10000,
        symbols=['EURUSD', 'GBPUSD']
    )
    
    train_env, eval_env, stats = trainer1.prepare_training_data(pair_data_stage1)
    
    metrics1 = trainer1.train(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=150000
    )
    
    trainer1.save_model("models/rl/stage1_eur_gbp")
    
    # Stage 2: Add USDJPY
    print("\n" + "="*70)
    print("STAGE 2: Adding USDJPY\n")
    
    pair_data_stage2 = pair_data_stage1.copy()
    
    candles_jpy = get_historical_data(
        symbol='USDJPY',
        start_date="2023-01-01",
        end_date="2023-12-31",
        timeframe="H1"
    )
    valid_setups_jpy = validate_tce_setups(candles_jpy, 'USDJPY')
    pair_data_stage2['USDJPY'] = (candles_jpy, valid_setups_jpy)
    
    trainer2 = MultiPairRLTrainer(
        model_name="tce_stage2_3pairs",
        initial_balance=10000,
        symbols=['EURUSD', 'GBPUSD', 'USDJPY']
    )
    
    train_env, eval_env, stats = trainer2.prepare_training_data(pair_data_stage2)
    
    metrics2 = trainer2.train(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=200000
    )
    
    trainer2.save_model("models/rl/stage2_3pairs")
    
    return {
        'stage1': metrics1,
        'stage2': metrics2
    }


# ============================================================================
# EXAMPLE 4: Custom data loading (if not using MT5)
# ============================================================================

def train_multipair_custom_data():
    """
    Training with custom CSV data instead of MT5 API.
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Data Loading")
    print("="*70 + "\n")
    
    # Load from CSV files
    data_dir = Path("data/forex")  # Adjust path as needed
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    pair_data = {}
    
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_H1.csv"
        
        if not csv_path.exists():
            print(f"⚠️  {csv_path} not found, skipping...")
            continue
        
        print(f"📂 Loading {csv_path}...")
        
        # Load CSV (adjust columns as needed)
        df = pd.read_csv(
            csv_path,
            index_col='timestamp',
            parse_dates=True
        )
        
        # Ensure columns: open, high, low, close, volume
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"  ❌ Missing required columns: {required_cols}")
            continue
        
        # Validate setups
        valid_setups = validate_tce_setups(df, symbol)
        
        print(f"  ✓ {len(df)} candles, {len(valid_setups)} setups\n")
        
        pair_data[symbol] = (df, valid_setups)
    
    if not pair_data:
        print("❌ No data loaded!")
        return
    
    # Train
    metrics = train_rl_multipair(
        pair_data=pair_data,
        model_name="tce_custom_data_v1",
        total_timesteps=200000
    )
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Choose which example to run
    example = 1  # Change to 1, 2, 3, or 4
    
    try:
        if example == 1:
            train_multipair_simple()
        
        elif example == 2:
            train_multipair_advanced()
        
        elif example == 3:
            train_multipair_staged()
        
        elif example == 4:
            train_multipair_custom_data()
        
        else:
            print(f"Unknown example: {example}")
    
    except Exception as e:
        print(f"\n❌ Error during training:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
