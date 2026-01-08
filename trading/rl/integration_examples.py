"""
Integration Examples: Using Multi-Pair Trainer with Your Existing Code

Shows how to connect multi-pair training with your existing:
- MT5 data integration
- TCE validation
- Risk management
"""

# ============================================================================
# EXAMPLE 1: Integration with MT5 Integration Module
# ============================================================================

from pathlib import Path
from trading.mt5_integration import get_historical_data, MT5Manager
from trading.tce.validation import validate_tce_setups
from trading.rl.multi_pair_training import train_rl_multipair


def train_on_mt5_data():
    """
    Train RL agent using data from MT5 (your existing integration).
    """
    
    # Initialize MT5 connection (from your existing code)
    mt5_manager = MT5Manager()
    
    # Define symbols
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD']
    
    # Collect data
    pair_data = {}
    
    for symbol in symbols:
        print(f"\n📥 Downloading {symbol}...")
        
        # Get data from MT5
        candles = get_historical_data(
            symbol=symbol,
            start_date="2023-01-01",
            end_date="2023-12-31",
            timeframe="H1"
        )
        
        if candles is None or len(candles) == 0:
            print(f"❌ No data for {symbol}")
            continue
        
        print(f"   ✓ Got {len(candles)} candles")
        
        # Validate TCE setups
        print(f"   📊 Validating TCE setups...")
        setups = validate_tce_setups(
            candles=candles,
            symbol=symbol
        )
        
        if len(setups) == 0:
            print(f"   ⚠️  No valid setups for {symbol}")
            continue
        
        print(f"   ✓ Found {len(setups)} valid setups")
        
        pair_data[symbol] = (candles, setups)
    
    if not pair_data:
        print("❌ No data collected!")
        return
    
    print(f"\n✅ Data collection complete: {len(pair_data)} pairs")
    
    # Train
    print("\n" + "="*70)
    print("Starting multi-pair training...")
    print("="*70)
    
    metrics = train_rl_multipair(
        pair_data=pair_data,
        model_name="tce_mt5_v1",
        initial_balance=10000,
        risk_percentage=1.0,
        total_timesteps=250000
    )
    
    return metrics


# ============================================================================
# EXAMPLE 2: Integration with Trading Models
# ============================================================================

from trading.models import Trade, Candle, Symbol
from django.utils import timezone
from trading.tce.data_collection import TCETrainingData, get_training_dataset
import pandas as pd
import numpy as np


def train_on_historical_trades():
    """
    Train RL agent using real historical trades from Django models.
    """
    
    # Get all symbols we've traded
    symbols = Symbol.objects.filter(active=True).values_list('symbol', flat=True)
    
    print(f"\n📊 Training on {len(symbols)} symbols")
    
    pair_data = {}
    
    for symbol in symbols:
        print(f"\n📥 Processing {symbol}...")
        
        # Get historical candles
        candles_qs = Candle.objects.filter(
            symbol=symbol,
            timestamp__gte=timezone.make_aware(timezone.datetime(2023, 1, 1)),
            timestamp__lte=timezone.make_aware(timezone.datetime(2023, 12, 31))
        ).order_by('timestamp')
        
        if candles_qs.count() == 0:
            print(f"   ⚠️  No candles for {symbol}")
            continue
        
        # Convert to DataFrame
        candles_data = []
        for candle in candles_qs:
            candles_data.append({
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'timestamp': candle.timestamp
            })
        
        candles_df = pd.DataFrame(candles_data)
        candles_df.set_index('timestamp', inplace=True)
        
        print(f"   ✓ Got {len(candles_df)} candles")
        
        # Get training data (setups with outcomes)
        training_data = TCETrainingData.objects.filter(
            symbol=symbol,
            label__isnull=False  # Only labeled trades
        )
        
        # Convert to setup format
        setups = []
        for td in training_data:
            setup = {
                'symbol': symbol,
                'features': td.features,
                'timestamp': td.entry_timestamp,
                'direction': td.direction,
                'entry_price': td.entry_price,
                'stop_loss': td.stop_loss,
                'take_profit': td.take_profit,
                'position_size': 0.1,  # Default
                'label': td.label,
                'r_multiple': td.r_multiple
            }
            setups.append(setup)
        
        if len(setups) == 0:
            print(f"   ⚠️  No training data for {symbol}")
            continue
        
        print(f"   ✓ Found {len(setups)} labeled trades")
        
        pair_data[symbol] = (candles_df, setups)
    
    if not pair_data:
        print("❌ No data found in models!")
        return
    
    # Train
    metrics = train_rl_multipair(
        pair_data=pair_data,
        symbols=list(pair_data.keys()),
        model_name="tce_historical_v1",
        total_timesteps=200000
    )
    
    return metrics


# ============================================================================
# EXAMPLE 3: Integration with Risk Management Module
# ============================================================================

from trading.tce.risk_management import RiskManager


def train_with_custom_risk_params():
    """
    Train using custom risk management parameters.
    """
    
    # Initialize risk manager (from your existing code)
    risk_mgr = RiskManager(
        account_balance=10000,
        risk_percent=1.0,
        max_daily_loss=500,  # $500 max daily loss
        correlation_limit=0.7
    )
    
    # Load data
    pair_data = {}
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    for symbol in symbols:
        candles = get_historical_data(symbol, "2023-01-01", "2023-12-31")
        setups = validate_tce_setups(candles, symbol)
        
        # Enhance setups with risk management calculations
        for setup in setups:
            entry_price = setup['entry_price']
            sl_price = setup['stop_loss']
            
            # Calculate using risk manager
            sl_distance = abs(entry_price - sl_price)
            position_size = risk_mgr.calculate_position_size(
                entry_price=entry_price,
                stop_loss_price=sl_price,
                risk_amount=None  # Use default from risk_mgr
            )
            
            setup['position_size'] = position_size
            setup['sl_distance_pips'] = sl_distance * 10000  # For EURUSD
        
        pair_data[symbol] = (candles, setups)
    
    # Train with risk params in trainer
    from trading.rl.multi_pair_training import MultiPairRLTrainer
    
    trainer = MultiPairRLTrainer(
        model_name="tce_with_risk_mgmt",
        initial_balance=risk_mgr.account_balance,
        risk_percentage=risk_mgr.risk_percent,
        symbols=symbols,
        enforce_risk_management=True
    )
    
    train_env, eval_env, stats = trainer.prepare_training_data(pair_data)
    
    metrics = trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=200000
    )
    
    return metrics


# ============================================================================
# EXAMPLE 4: Automated Training Pipeline
# ============================================================================

from datetime import datetime, timedelta
import json


def automated_multipair_training_pipeline():
    """
    Full automated pipeline:
    1. Download latest data
    2. Validate setups
    3. Train model
    4. Evaluate and save
    5. Log metrics
    """
    
    print("\n" + "="*70)
    print("AUTOMATED MULTI-PAIR TRAINING PIPELINE")
    print("="*70)
    
    run_timestamp = datetime.now()
    print(f"\nRun ID: {run_timestamp.isoformat()}\n")
    
    # ---- STAGE 1: DATA COLLECTION ----
    print("STAGE 1: Data Collection")
    print("-" * 70)
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD']
    pair_data = {}
    collection_stats = {}
    
    for symbol in symbols:
        try:
            # Download 1 year of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            candles = get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe="H1"
            )
            
            setups = validate_tce_setups(candles, symbol)
            
            pair_data[symbol] = (candles, setups)
            collection_stats[symbol] = {
                'candles': len(candles),
                'setups': len(setups),
                'date_range': f"{start_date.date()} to {end_date.date()}"
            }
            
            print(f"✓ {symbol}: {len(candles)} candles, {len(setups)} setups")
        
        except Exception as e:
            print(f"✗ {symbol}: {type(e).__name__}: {e}")
            continue
    
    total_setups = sum(s['setups'] for s in collection_stats.values())
    print(f"\n✓ Total: {total_setups} setups across {len(pair_data)} pairs\n")
    
    # ---- STAGE 2: TRAINING ----
    print("STAGE 2: RL Training")
    print("-" * 70)
    
    try:
        metrics = train_rl_multipair(
            pair_data=pair_data,
            symbols=list(pair_data.keys()),
            model_name=f"tce_auto_{run_timestamp.strftime('%Y%m%d_%H%M%S')}",
            total_timesteps=200000
        )
        
        print("\n✓ Training completed\n")
    
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return None
    
    # ---- STAGE 3: EVALUATION & LOGGING ----
    print("STAGE 3: Evaluation & Logging")
    print("-" * 70)
    
    eval_metrics = metrics.get('eval', {})
    
    summary = {
        'timestamp': run_timestamp.isoformat(),
        'symbols': list(pair_data.keys()),
        'collection_stats': collection_stats,
        'total_setups': total_setups,
        'eval_metrics': {
            'mean_reward': eval_metrics.get('mean_reward'),
            'mean_r_multiple': eval_metrics.get('mean_r_multiple'),
            'mean_win_rate': eval_metrics.get('mean_win_rate'),
            'n_episodes': eval_metrics.get('n_episodes')
        }
    }
    
    # Log to file
    log_dir = Path("logs/rl_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"run_{run_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {log_file}")
    print(f"\nFinal Metrics:")
    print(f"  R-Multiple: {eval_metrics.get('mean_r_multiple', 'N/A'):.2f}R")
    print(f"  Win Rate: {eval_metrics.get('mean_win_rate', 'N/A'):.1%}")
    print(f"  Mean Reward: {eval_metrics.get('mean_reward', 'N/A'):.2f}")
    
    return summary


# ============================================================================
# EXAMPLE 5: Scheduled Training (runs periodically)
# ============================================================================

from celery import shared_task
import os
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    """
    Django management command to run multi-pair training.
    
    Usage:
        python manage.py train_rl_multipair --pairs EURUSD,GBPUSD,USDJPY
    """
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--pairs',
            type=str,
            help='Comma-separated list of pairs (e.g., EURUSD,GBPUSD,USDJPY)'
        )
        parser.add_argument(
            '--timesteps',
            type=int,
            default=200000,
            help='Total training timesteps'
        )
        parser.add_argument(
            '--risk',
            type=float,
            default=1.0,
            help='Risk percentage per trade'
        )
    
    def handle(self, *args, **options):
        pairs = options.get('pairs', '').split(',')
        pairs = [p.strip().upper() for p in pairs if p.strip()]
        
        if not pairs:
            self.stdout.write("Error: No pairs specified")
            return
        
        self.stdout.write(f"\n📊 Training on: {', '.join(pairs)}")
        
        # Load data
        pair_data = {}
        for symbol in pairs:
            candles = get_historical_data(symbol, "2023-01-01", "2023-12-31")
            setups = validate_tce_setups(candles, symbol)
            pair_data[symbol] = (candles, setups)
        
        # Train
        metrics = train_rl_multipair(
            pair_data=pair_data,
            symbols=pairs,
            total_timesteps=options['timesteps']
        )
        
        self.stdout.write("\n✅ Training complete!")
        self.stdout.write(f"Results: {metrics}")


# ============================================================================
# MAIN: Choose which example to run
# ============================================================================

if __name__ == "__main__":
    import sys
    
    example = 1  # Change to 1, 2, 3, 4, or 5
    
    print(f"\nRunning example {example}...\n")
    
    try:
        if example == 1:
            metrics = train_on_mt5_data()
        
        elif example == 2:
            metrics = train_on_historical_trades()
        
        elif example == 3:
            metrics = train_with_custom_risk_params()
        
        elif example == 4:
            metrics = automated_multipair_training_pipeline()
        
        elif example == 5:
            print("Django command - run via:")
            print("  python manage.py train_rl_multipair --pairs EURUSD,GBPUSD,USDJPY --timesteps 200000")
        
        else:
            print(f"Unknown example: {example}")
    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
