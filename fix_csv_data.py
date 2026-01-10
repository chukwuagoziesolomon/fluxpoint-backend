"""
Fix CSV data issues:
1. Remove broken header rows (Ticker, EURUSD=X)
2. Clean up data format
3. Filter out market closed days (weekends, holidays)
4. Add proper Date column
5. Validate OHLC data

Run this BEFORE training!
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# Market closed days (holidays)
HOLIDAYS_2020_2024 = {
    # 2020
    '2020-01-01', '2020-04-10', '2020-04-13', '2020-05-08', '2020-05-25',
    '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
    # 2021
    '2021-01-01', '2021-04-02', '2021-04-05', '2021-05-31', '2021-07-05',
    '2021-09-06', '2021-11-25', '2021-12-24', '2021-12-25',
    # 2022
    '2022-01-01', '2022-04-15', '2022-04-18', '2022-05-30', '2022-07-04',
    '2022-09-05', '2022-11-24', '2022-12-26',
    # 2023
    '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-29', '2023-07-04',
    '2023-09-04', '2023-11-23', '2023-12-25',
    # 2024
    '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-27', '2024-07-04',
    '2024-09-02', '2024-11-28', '2024-12-25',
}

def is_trading_day(date):
    """Check if date is a trading day (not weekend or holiday)"""
    if date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    date_str = date.strftime('%Y-%m-%d')
    return date_str not in HOLIDAYS_2020_2024

def fix_csv_file(input_file, output_file):
    """Fix a single CSV file"""
    print(f"\n📝 Processing: {input_file.name}")
    
    try:
        # Read CSV, skip the broken header rows
        df = pd.read_csv(input_file, skiprows=2)  # Skip "Price,Open,..." and "Ticker,EURUSD=X,..."
        
        # First column should be Date
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Filter out non-trading days (weekends & holidays)
        df = df[df['Date'].apply(is_trading_day)]
        
        # Convert OHLC to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid OHLC
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Validate OHLC logic (High >= Low, Close between High/Low)
        valid_rows = (
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Open']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) &
            (df['Low'] <= df['Close'])
        )
        df = df[valid_rows]
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"   ✅ Saved: {len(df)} valid trading days")
        print(f"   📅 Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        return df
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def main():
    """Fix all CSV files in training_data folder"""
    print("="*80)
    print("FIX CSV DATA - Remove bad headers, filter holidays/weekends")
    print("="*80)
    
    # Input/output directories
    input_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data')
    output_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data_cleaned')
    output_dir.mkdir(exist_ok=True)
    
    csv_files = sorted(input_dir.glob('*_data.csv'))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in: {input_dir}")
        return
    
    print(f"\n📂 Found {len(csv_files)} CSV files\n")
    
    fixed_count = 0
    total_candles = 0
    
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        df = fix_csv_file(csv_file, output_file)
        
        if df is not None:
            fixed_count += 1
            total_candles += len(df)
    
    print(f"\n{'='*80}")
    print(f"✅ FIXED {fixed_count}/{len(csv_files)} files")
    print(f"📊 Total valid trading days: {total_candles}")
    print(f"💾 Cleaned files saved to: {output_dir}")
    print(f"\n📌 NEXT STEPS:")
    print(f"   1. Review cleaned data in: {output_dir}")
    print(f"   2. Update your training script to load from: training_data_cleaned/")
    print(f"   3. Run CELL4_COMPLETE_TCE_VALIDATION.py again")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
