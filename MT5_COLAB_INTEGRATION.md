MT5 DATA INTEGRATION FOR COLAB
═══════════════════════════════════════════════════════════════════════════════

Perfect! You can get data directly from MT5 using your existing integration code.

REPLACE CELL 3 with this:
═══════════════════════════════════════════════════════════════════════════════

CELL 3 (UPDATED): Load data directly from MT5
────────────────────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, '/content/fluxpoint')

from trading.mt5_integration import get_historical_data
from datetime import datetime, timedelta
import pandas as pd

print("\nLoading data directly from MT5...\n")

# Define pairs to train on
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY']

pair_data = {}

# Calculate date range (1 year of data)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Fetching data from {start_date.date()} to {end_date.date()}\n")

for symbol in pairs:
    try:
        print(f"  {symbol}...", end='', flush=True)
        
        # Get historical data from MT5
        df = get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='H1'  # 1-hour candles
        )
        
        if df is not None and len(df) > 0:
            # Validate TCE setups
            from trading.tce.validation import validate_tce_setups
            setups = validate_tce_setups(df, symbol)
            
            pair_data[symbol] = (df, setups)
            print(f" ✓ ({len(df)} candles, {len(setups)} setups)")
        else:
            print(f" ✗ No data returned")
    
    except Exception as e:
        print(f" ✗ Error: {str(e)[:50]}")

print(f"\n✓ Successfully loaded {len(pair_data)} pairs")

if len(pair_data) == 0:
    print("\n⚠️  WARNING: No data loaded from MT5")
    print("\nTroubleshooting:")
    print("  1. Is MT5 terminal running?")
    print("  2. Are symbols correct? (EURUSD not eur/usd)")
    print("  3. Check MT5 connection in get_historical_data()")


════════════════════════════════════════════════════════════════════════════════
IF MT5 NOT AVAILABLE (Cloud environment issue):
════════════════════════════════════════════════════════════════════════════════

If MT5 API doesn't work in Colab, use this FALLBACK instead:

CELL 3 (FALLBACK): Use CSV from Google Drive + MT5 backup
────────────────────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, '/content/fluxpoint')

from pathlib import Path
import pandas as pd

print("\nAttempting to load data from multiple sources...\n")

pair_data = {}

# First, try MT5
try:
    print("1. Trying MT5 API...")
    from trading.mt5_integration import get_historical_data
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    pairs = ['EURUSD', 'GBPUSD', 'AUDUSD']
    
    for symbol in pairs:
        df = get_historical_data(symbol, start_date, end_date, 'H1')
        if df is not None and len(df) > 0:
            from trading.tce.validation import validate_tce_setups
            setups = validate_tce_setups(df, symbol)
            pair_data[symbol] = (df, setups)
            print(f"  ✓ {symbol}: {len(df)} candles from MT5")
    
    if len(pair_data) > 0:
        print("✓ MT5 data loaded successfully!\n")

except Exception as e:
    print(f"✗ MT5 not available: {e}\n")

# If MT5 didn't work, try Google Drive
if len(pair_data) == 0:
    print("2. Trying Google Drive CSV files...")
    
    csv_dir = Path('/content/drive/MyDrive/forex_data')
    
    if csv_dir.exists():
        csv_files = list(csv_dir.glob('*.csv'))
        
        if csv_files:
            print(f"   Found {len(csv_files)} CSV files\n")
            
            for csv_file in csv_files:
                symbol = csv_file.stem.split('_')[0].upper()
                
                try:
                    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    from trading.tce.validation import validate_tce_setups
                    setups = validate_tce_setups(df, symbol)
                    pair_data[symbol] = (df, setups)
                    print(f"  ✓ {symbol}: {len(df)} candles from CSV")
                
                except Exception as e:
                    print(f"  ✗ {symbol}: {e}")
        else:
            print("   No CSV files in forex_data/")
    else:
        print("   No forex_data folder found")

# If still nothing, generate sample data
if len(pair_data) == 0:
    print("\n3. Generating sample data for testing...")
    
    import numpy as np
    from datetime import datetime, timedelta
    
    pairs = ['EURUSD', 'GBPUSD', 'AUDUSD']
    
    for pair in pairs:
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1h')
        df = pd.DataFrame({
            'Open': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 101,
            'Low': np.random.randn(len(dates)).cumsum() + 99,
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        from trading.tce.validation import validate_tce_setups
        setups = validate_tce_setups(df, pair)
        pair_data[pair] = (df, setups)
        print(f"  ✓ {pair}: {len(df)} sample candles")

print(f"\n✓ Loaded {len(pair_data)} pairs total")


════════════════════════════════════════════════════════════════════════════════
SETUP REQUIREMENTS FOR MT5 IN COLAB:
════════════════════════════════════════════════════════════════════════════════

For MT5 to work in Colab, you need:

1. Install MT5 Python package (add to CELL 2):
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "MetaTrader5"])

2. Check your mt5_integration.py requires these functions:
   ├─ get_historical_data(symbol, start_date, end_date, timeframe)
   ├─ Should handle connection internally
   └─ Return pandas DataFrame with OHLC data

3. MT5 Terminal needs to be running:
   └─ Run on your local machine (not Colab)
   └─ Leave it open while Colab is training
   └─ This acts as the data server

4. Network access:
   └─ Colab can connect to MT5 on your local machine
   └─ Use VPN or port forwarding if needed


════════════════════════════════════════════════════════════════════════════════
BEST APPROACH: Hybrid Method
════════════════════════════════════════════════════════════════════════════════

Recommended workflow:

STEP 1: Download data locally (once)
────────────────────────────────────
# Run this on your LOCAL machine (VS Code), NOT in Colab

import sys
sys.path.insert(0, r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint')

from trading.mt5_integration import get_historical_data
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Create output folder
output_dir = Path('C:/Users/USER-PC/fluxpointai-backend/fluxpoint/training_data')
output_dir.mkdir(exist_ok=True)

pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

for symbol in pairs:
    print(f"Downloading {symbol}...")
    df = get_historical_data(symbol, start_date, end_date, 'H1')
    
    if df is not None:
        df.to_csv(output_dir / f"{symbol.lower()}_data.csv")
        print(f"  ✓ Saved {len(df)} candles")

print("\nNow upload these CSV files to Google Drive/forex_data/")


STEP 2: Upload to Google Drive
──────────────────────────────
1. Go to https://drive.google.com/
2. Create folder: forex_data
3. Upload the CSV files you just saved


STEP 3: In Colab, use the CSV backup
──────────────────────────────────────
Run the "FALLBACK" cell above - it will:
  1. Try MT5 first (if available)
  2. Fall back to CSV files on Drive
  3. Generate sample data if needed


════════════════════════════════════════════════════════════════════════════════
SUMMARY: Your Options
════════════════════════════════════════════════════════════════════════════════

OPTION A: Pure MT5 (simplest if MT5 available)
  └─ CELL 3: Use MT5 directly
  └─ Requires MT5 terminal running
  └─ Real-time data

OPTION B: MT5 + CSV Backup (most reliable)
  └─ CELL 3: Use FALLBACK version
  └─ Download once locally with MT5
  └─ Upload to Google Drive
  └─ Colab uses whichever available

OPTION C: CSV Only (works anywhere)
  └─ Download data locally with MT5
  └─ Upload to Google Drive
  └─ Use CSV files in Colab


════════════════════════════════════════════════════════════════════════════════
WHICH OPTION DO YOU WANT TO USE?
════════════════════════════════════════════════════════════════════════════════

Option A: Pure MT5 in Colab
   └─ Best: Real-time, automatic, no upload needed
   └─ Requirement: MT5 terminal open while training

Option B: Hybrid MT5 + CSV
   └─ Best: Reliable, flexible
   └─ Steps:
      1. Download with local MT5 (one time)
      2. Upload to Drive
      3. Train in Colab

Option C: CSV Files Only
   └─ Best: Simple, works anywhere
   └─ Same as Option B but skip MT5 step

═══════════════════════════════════════════════════════════════════════════════
