import sys
import os
sys.path.insert(0, r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint')

# Configure Django before importing models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
import django
django.setup()

from trading.mt5_integration import MT5DataIngestion
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

print("="*70)
print("DOWNLOADING FOREX DATA FROM MT5")
print("="*70)

# Create output folder
output_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data')
output_dir.mkdir(exist_ok=True)

print(f"\nOutput folder: {output_dir}\n")

# Initialize MT5 connection
print("Connecting to MT5...")
mt5_ingestion = MT5DataIngestion()

# Define pairs to download
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY']

# Date range (1 year of data)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Downloading data from {start_date.date()} to {end_date.date()}\n")
print("Pairs to download:")
for pair in pairs:
    print(f"  - {pair}")

print("\n" + "-"*70)

downloaded = 0

for symbol in pairs:
    print(f"\n{symbol}...", end='', flush=True)
    
    try:
        # Download data from MT5 using the class method
        df = mt5_ingestion.fetch_historical_data(
            symbol=symbol,
            timeframe='H1',
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None and len(df) > 0:
            # Save to CSV
            filename = f"{symbol.lower()}_data.csv"
            filepath = output_dir / filename
            
            df.to_csv(filepath)
            
            print(f" ✓")
            print(f"  Saved: {filepath}")
            print(f"  Candles: {len(df)}")
            print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
            
            downloaded += 1
        
        else:
            print(f" ✗ No data returned")
    
    except Exception as e:
        print(f" ✗ Error: {str(e)[:50]}")

print("\n" + "="*70)
print(f"✓ DOWNLOAD COMPLETE: {downloaded}/{len(pairs)} pairs")
print("="*70)

if downloaded > 0:
    print(f"\nNext steps:")
    print(f"1. Go to: {output_dir}")
    print(f"2. Check CSV files were created")
    print(f"3. Upload to Google Drive:")
    print(f"   - Create folder: My Drive/forex_data/")
    print(f"   - Upload all CSV files there")
    print(f"4. Run Colab training!")
else:
    print("\n✗ No data downloaded!")
    print("\nTroubleshooting:")
    print("1. Is MT5 terminal running?")
    print("2. Are symbols correct? (EURUSD not eur/usd)")
    print("3. Check MT5 API connection")

