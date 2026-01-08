"""
Minimal Working Example - Multi-Pair RL Training

Copy-paste this to get started in 5 minutes!
"""

# ============================================================================
# OPTION 1: If you have MT5 data available
# ============================================================================

def minimal_example_mt5():
    """
    Minimal example using MT5 data.
    Copy-paste and run this.
    """
    from trading.rl.multi_pair_training import train_rl_multipair
    from trading.tce.validation import validate_tce_setups
    from trading.mt5_integration import get_historical_data
    
    # 1. Load data
    pair_data = {}
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        print(f"Loading {symbol}...")
        candles = get_historical_data(symbol, "2023-01-01", "2023-12-31", "H1")
        setups = validate_tce_setups(candles, symbol)
        pair_data[symbol] = (candles, setups)
        print(f"  {len(setups)} setups\n")
    
    # 2. Train (takes 8-12 hours)
    print("\nTraining... (this takes a while)")
    metrics = train_rl_multipair(pair_data, total_timesteps=200000)
    
    # 3. Check results
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    eval_results = metrics.get('eval', {})
    print(f"R-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
    print(f"Win Rate: {eval_results.get('mean_win_rate', 0):.1%}")
    print("="*50)


# ============================================================================
# OPTION 2: If you have CSV files
# ============================================================================

def minimal_example_csv():
    """
    Minimal example using CSV files.
    Requires CSV files in data/forex/ folder.
    """
    from trading.rl.multi_pair_training import train_rl_multipair
    from trading.tce.validation import validate_tce_setups
    import pandas as pd
    from pathlib import Path
    
    # 1. Load from CSV
    data_dir = Path("data/forex")
    pair_data = {}
    
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        csv_file = data_dir / f"{symbol}_H1.csv"
        
        if not csv_file.exists():
            print(f"⚠️  {csv_file} not found!")
            continue
        
        print(f"Loading {symbol}...")
        
        # Read CSV (adjust columns if needed)
        df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
        
        # Validate setups
        setups = validate_tce_setups(df, symbol)
        pair_data[symbol] = (df, setups)
        print(f"  {len(setups)} setups\n")
    
    if not pair_data:
        print("❌ No CSV files found!")
        return
    
    # 2. Train
    print("\nTraining...")
    metrics = train_rl_multipair(pair_data, total_timesteps=200000)
    
    # 3. Results
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    eval_results = metrics.get('eval', {})
    print(f"R-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
    print(f"Win Rate: {eval_results.get('mean_win_rate', 0):.1%}")
    print("="*50)


# ============================================================================
# OPTION 3: If you have Django models
# ============================================================================

def minimal_example_django():
    """
    Minimal example using Django models.
    """
    from trading.rl.multi_pair_training import train_rl_multipair
    from trading.models import Candle, Symbol, Trade
    from trading.tce.data_collection import TCETrainingData
    from trading.tce.validation import validate_tce_setups
    import pandas as pd
    from django.utils import timezone
    
    # 1. Load from models
    pair_data = {}
    symbols = Symbol.objects.filter(active=True)[:5]  # First 5 symbols
    
    for sym_obj in symbols:
        symbol = sym_obj.symbol
        print(f"Loading {symbol}...")
        
        # Get candles
        candles_qs = Candle.objects.filter(
            symbol=symbol,
            timeframe='H1'
        ).order_by('timestamp')[:100000]  # Limit to recent data
        
        if candles_qs.count() == 0:
            print(f"  No candles found!")
            continue
        
        # Convert to DataFrame
        candles_data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
        }
        timestamps = []
        
        for candle in candles_qs:
            candles_data['open'].append(candle.open)
            candles_data['high'].append(candle.high)
            candles_data['low'].append(candle.low)
            candles_data['close'].append(candle.close)
            candles_data['volume'].append(candle.volume or 0)
            timestamps.append(candle.timestamp)
        
        df = pd.DataFrame(candles_data, index=pd.DatetimeIndex(timestamps))
        
        # Validate setups
        setups = validate_tce_setups(df, symbol)
        pair_data[symbol] = (df, setups)
        print(f"  {len(setups)} setups\n")
    
    if not pair_data:
        print("❌ No data found in models!")
        return
    
    # 2. Train
    print("\nTraining...")
    metrics = train_rl_multipair(pair_data, total_timesteps=200000)
    
    # 3. Results
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    eval_results = metrics.get('eval', {})
    print(f"R-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
    print(f"Win Rate: {eval_results.get('mean_win_rate', 0):.1%}")
    print("="*50)


# ============================================================================
# CHOOSE YOUR OPTION AND RUN
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Set this to the option you want to run
    option = 1  # 1=MT5, 2=CSV, 3=Django
    
    print("\n" + "="*50)
    print("MINIMAL MULTI-PAIR RL TRAINING EXAMPLE")
    print("="*50 + "\n")
    
    try:
        if option == 1:
            print("Option 1: MT5 Data\n")
            minimal_example_mt5()
        
        elif option == 2:
            print("Option 2: CSV Files\n")
            minimal_example_csv()
        
        elif option == 3:
            print("Option 3: Django Models\n")
            # Need to setup Django first
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
            import django
            django.setup()
            minimal_example_django()
        
        else:
            print(f"Invalid option: {option}")
    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
