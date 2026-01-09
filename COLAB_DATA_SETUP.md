COLAB DATA UPLOAD GUIDE
═══════════════════════════════════════════════════════════════════════════════

Problem: "✓ Loaded 0 pairs"
─────────────────────────────────────────────────────────────────────────────

This means NO CSV files were found in Google Drive.

SOLUTION: Upload your CSV files to Google Drive


OPTION 1: Upload CSV Files (Recommended)
═══════════════════════════════════════════════════════════════════════════════

Step 1: Create folder in Google Drive
  1. Go to: https://drive.google.com/
  2. Right-click > New Folder
  3. Name it: forex_data
  4. Keep it open

Step 2: Add your CSV files to the folder
  1. Upload your EURUSD, GBPUSD, etc. CSV files
  2. Files should be named like:
     - eurusd_data.csv
     - gbpusd_data.csv
     - audusd_data.csv
     - etc.

Step 3: Run diagnostic cell in Colab (below)
  Copy and paste this cell to check if files are found:

─────────────────────────────────────────────────────────────────────────────
DIAGNOSTIC CELL: Check what's in Google Drive
─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import os

print("Checking Google Drive contents...\n")

# List everything in Drive root
drive_root = Path('/content/drive/MyDrive')
print(f"Contents of {drive_root}:\n")

for item in sorted(drive_root.iterdir()):
    if item.is_dir():
        print(f"  📁 {item.name}/")
        # List items inside
        for subitem in list(item.iterdir())[:5]:
            print(f"      └─ {subitem.name}")
        if len(list(item.iterdir())) > 5:
            print(f"      └─ ... and {len(list(item.iterdir()))-5} more")
    else:
        print(f"  📄 {item.name}")

# Check for forex_data folder
print("\n" + "="*70)
csv_dir = Path('/content/drive/MyDrive/forex_data')

if csv_dir.exists():
    print(f"✓ Found forex_data folder\n")
    csv_files = list(csv_dir.glob('*.csv'))
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV files:\n")
        for csv_file in csv_files:
            print(f"  ✓ {csv_file.name}")
    else:
        print("✗ Folder exists but NO CSV files inside!")
        print("  → Upload CSV files to: My Drive/forex_data/")
else:
    print(f"✗ No forex_data folder found!")
    print("\nCreate it manually:")
    print("  1. Go to https://drive.google.com/")
    print("  2. Right-click > New Folder")
    print("  3. Name: forex_data")
    print("  4. Upload CSV files inside")


═══════════════════════════════════════════════════════════════════════════════
OPTION 2: Generate Sample Data (If you don't have CSV files)
═══════════════════════════════════════════════════════════════════════════════

If you don't have CSV files yet, run this cell to CREATE SAMPLE DATA:

─────────────────────────────────────────────────────────────────────────────
SAMPLE DATA GENERATOR CELL
─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

print("Generating sample forex data for training...\n")

# Create folder
output_dir = Path('/content/drive/MyDrive/forex_data')
output_dir.mkdir(parents=True, exist_ok=True)

# Generate data for multiple pairs
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY']

for pair in pairs:
    # Create date range (1 year of data)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    
    # Generate OHLC data
    n = len(dates)
    open_prices = np.random.randn(n).cumsum() + 100
    high_prices = open_prices + np.abs(np.random.randn(n) * 0.5)
    low_prices = open_prices - np.abs(np.random.randn(n) * 0.5)
    close_prices = open_prices + np.random.randn(n) * 0.3
    volumes = np.random.randint(1000, 5000, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    # Save to CSV
    filename = f"{pair.lower()}_data.csv"
    filepath = output_dir / filename
    df.to_csv(filepath)
    
    print(f"  ✓ {filename}: {len(df)} candles")

print(f"\n✓ Sample data created in: My Drive/forex_data/")
print("Ready to train!")


═══════════════════════════════════════════════════════════════════════════════
OPTION 3: Use Sample Data Without Upload (Quick Test)
═══════════════════════════════════════════════════════════════════════════════

If you want to test training WITHOUT uploading files, replace CELL 3 with:

─────────────────────────────────────────────────────────────────────────────
CELL 3 REPLACEMENT: Generate data locally
─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, '/content/fluxpoint')
from trading.tce.validation import validate_tce_setups

print("\nGenerating sample training data...\n")

pair_data = {}
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD']

for pair in pairs:
    try:
        # Generate synthetic OHLC data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
        n = len(dates)
        
        df = pd.DataFrame({
            'Open': np.random.randn(n).cumsum() + 100,
            'High': np.random.randn(n).cumsum() + 101,
            'Low': np.random.randn(n).cumsum() + 99,
            'Close': np.random.randn(n).cumsum() + 100,
            'Volume': np.random.randint(1000, 5000, n)
        }, index=dates)
        
        # Validate setups
        setups = validate_tce_setups(df, pair)
        pair_data[pair] = (df, setups)
        
        print(f"✓ {pair}: {len(df)} candles, {len(setups)} setups")
    
    except Exception as e:
        print(f"✗ {pair}: {e}")

print(f"\n✓ Generated {len(pair_data)} pairs")


═══════════════════════════════════════════════════════════════════════════════
STEP-BY-STEP: Upload Real CSV Files
═══════════════════════════════════════════════════════════════════════════════

You need REAL forex data to train a GOOD model. Here's how:

1. EXPORT FROM MT5 (if you have it)
   ├─ Open MetaTrader 5
   ├─ Select symbol (EURUSD)
   ├─ Right-click > Export > CSV
   └─ Save to: forex_data/ folder

2. USE FREE DATA SOURCES
   ├─ Yahoo Finance: yfinance package
   ├─ OANDA: oandapyV20 API
   ├─ Forex Factory
   └─ Investing.com

3. FORMAT REQUIRED
   Your CSV needs columns:
   ├─ Date/Time (index)
   ├─ Open
   ├─ High
   ├─ Low
   ├─ Close
   └─ Volume (optional)

Example format:
   DateTime,Open,High,Low,Close,Volume
   2023-01-01 00:00,1.0500,1.0505,1.0495,1.0503,50000
   2023-01-01 01:00,1.0503,1.0508,1.0500,1.0505,55000


═══════════════════════════════════════════════════════════════════════════════
RECOMMENDED: Download Sample Data with Python
═══════════════════════════════════════════════════════════════════════════════

Use this cell to download real data programmatically:

─────────────────────────────────────────────────────────────────────────────
DOWNLOAD REAL DATA CELL
─────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
from pathlib import Path

# Install yfinance
print("Installing yfinance...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])

import yfinance as yf
import pandas as pd

print("\nDownloading forex data from Yahoo Finance...\n")

# Create folder
output_dir = Path('/content/drive/MyDrive/forex_data')
output_dir.mkdir(parents=True, exist_ok=True)

# Download data for multiple pairs
# Note: Yahoo Finance uses ^EURUSD format
pairs_yahoo = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDJPY': 'JPY=X'
}

for pair_name, yahoo_symbol in pairs_yahoo.items():
    try:
        print(f"  Downloading {pair_name}...", end='', flush=True)
        
        # Download 1 year of data
        df = yf.download(yahoo_symbol, start='2023-01-01', end='2023-12-31', progress=False)
        
        # Rename columns if needed
        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)
        
        # Save
        filename = f"{pair_name.lower()}_data.csv"
        df.to_csv(output_dir / filename)
        
        print(f" ✓ ({len(df)} candles)")
    
    except Exception as e:
        print(f" ✗ ({e})")

print(f"\n✓ Data saved to: My Drive/forex_data/")


═══════════════════════════════════════════════════════════════════════════════
QUICK FIX: Next Steps
═══════════════════════════════════════════════════════════════════════════════

1. Choose ONE of these options:
   ☐ Option 1: Upload CSV files manually (need files ready)
   ☐ Option 2: Generate sample data (quick test)
   ☐ Option 3: Download real data (best results)

2. Run appropriate cell above

3. Re-run CELL 3 (Load data)
   Should now see: ✓ Loaded X pairs

4. Continue with CELL 4 (Train DL)


═══════════════════════════════════════════════════════════════════════════════
