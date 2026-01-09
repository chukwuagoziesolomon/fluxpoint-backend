"""
ALTERNATIVE: Download Forex Data from Yahoo Finance (No Broker Needed)

This script downloads REAL forex data without needing MT5 connection.
Faster and simpler than waiting for MT5 setup.
"""

import sys
import os
sys.path.insert(0, r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint')

from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

print("="*70)
print("DOWNLOADING FOREX DATA FROM YAHOO FINANCE")
print("="*70)

# Create output folder
output_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data')
output_dir.mkdir(exist_ok=True)

print(f"\nOutput folder: {output_dir}\n")

# Install yfinance if not already installed
print("Installing yfinance...")
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])

import yfinance as yf

# Forex pairs (Yahoo Finance format)
pairs = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDJPY': 'JPY=X',
    'USDCHF': 'CHF=X',
    'USDHKD': 'HKD=X',
    'USDCAD': 'CAD=X',
    'EURGBP': 'EURGBP=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'AUDJPY': 'AUDJPY=X',
    'NZDJPY': 'NZDJPY=X',
    'GBPCHF': 'GBPCHF=X',
    'EURCHF': 'EURCHF=X'
}

# Date range: 2020 to present (5+ years of data for better model training)
start_date = datetime(2020, 1, 1)
end_date = datetime.now()

print(f"Downloading from {start_date.date()} to {end_date.date()}\n")
print("Pairs to download:")
for pair in pairs.keys():
    print(f"  - {pair}")

print("\n" + "-"*70 + "\n")

downloaded = 0

for pair_name, yahoo_symbol in pairs.items():
    print(f"{pair_name}...", end='', flush=True)
    
    try:
        # Download data
        df = yf.download(
            yahoo_symbol,
            start=start_date.date(),
            end=end_date.date(),
            progress=False
        )
        
        if df is not None and len(df) > 0:
            # Keep only needed columns and handle different column names
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna()  # Remove any NaN rows
            
            # Save to CSV
            filename = f"{pair_name.lower()}_data.csv"
            filepath = output_dir / filename
            df.to_csv(filepath)
            
            print(f" ✓")
            print(f"  Saved: {filepath}")
            print(f"  Candles: {len(df)}")
            print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
            
            downloaded += 1
        
        else:
            print(f" ✗ No data")
    
    except Exception as e:
        print(f" ✗ Error: {str(e)[:50]}")

print("\n" + "="*70)
print(f"✓ DOWNLOAD COMPLETE: {downloaded}/{len(pairs)} pairs")
print("="*70)

if downloaded > 0:
    print(f"\n✓ SUCCESS! CSV files created:\n")
    for csv_file in sorted(output_dir.glob('*.csv')):
        size_kb = csv_file.stat().st_size / 1024
        print(f"  ✓ {csv_file.name} ({size_kb:.1f} KB)")
    
    print(f"\n" + "="*70)
    print(f"NEXT STEPS:")
    print(f"="*70)
    print(f"\n1. Upload to Google Drive:")
    print(f"   - Go to: https://drive.google.com/")
    print(f"   - Create folder: forex_data")
    print(f"   - Upload all CSV files")
    print(f"\n2. Run Colab training:")
    print(f"   - Open: https://colab.research.google.com/")
    print(f"   - Create new notebook")
    print(f"   - Runtime > GPU")
    print(f"   - Run CELL 1-7 from COLAB_COMPLETE_PIPELINE.py")
    print(f"\n3. Wait 12-15 hours for training")
    print(f"\n4. Download trained models from Google Drive")
else:
    print("\n✗ No data downloaded!")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Yahoo Finance might be down (try again later)")
