"""
Data loader for CELL4 training - loads cleaned CSV data with proper validation
USE THIS instead of manually loading pair_data!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

def load_forex_data(data_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load cleaned forex data from CSV files
    
    Returns:
        dict: {symbol: DataFrame} with columns [Date, Open, High, Low, Close, Volume]
    """
    if data_dir is None:
        data_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data_cleaned')
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"❌ Data directory not found: {data_dir}\n"
            f"   Run 'python fix_csv_data.py' first to clean your data!"
        )
    
    pair_data = {}
    csv_files = sorted(data_dir.glob('*_data.csv'))
    
    if not csv_files:
        raise FileNotFoundError(
            f"❌ No CSV files found in: {data_dir}\n"
            f"   Expected files like: eurusd_data.csv, gbpusd_data.csv, etc."
        )
    
    print(f"\n{'='*80}")
    print(f"📂 LOADING FOREX DATA FROM: {data_dir}")
    print(f"{'='*80}\n")
    
    for csv_file in csv_files:
        # Extract symbol from filename (e.g., "eurusd_data.csv" -> "EURUSD")
        symbol = csv_file.stem.replace('_data', '').upper()
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Validate required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                print(f"  ❌ {symbol}: Missing required columns!")
                continue
            
            # Convert OHLC to numeric
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any remaining NaN values
            df = df.dropna(subset=required_cols)
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Validate minimum data
            if len(df) < 250:
                print(f"  ⚠️  {symbol}: Only {len(df)} candles (need 250+) - SKIPPED")
                continue
            
            pair_data[symbol] = df
            
            print(f"  ✅ {symbol:<10} {len(df):>5} candles | {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        except Exception as e:
            print(f"  ❌ {symbol}: Error loading - {str(e)[:60]}")
    
    print(f"\n{'='*80}")
    print(f"✅ Loaded {len(pair_data)} pairs successfully")
    print(f"{'='*80}\n")
    
    if not pair_data:
        raise ValueError("No valid data loaded! Check your CSV files.")
    
    return pair_data


def validate_data_quality(pair_data: Dict[str, pd.DataFrame]) -> None:
    """Validate data quality before training"""
    print("\n🔍 VALIDATING DATA QUALITY...\n")
    
    issues = []
    
    for symbol, df in pair_data.items():
        # Check for duplicates
        duplicates = df['Date'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"{symbol}: {duplicates} duplicate dates")
        
        # Check for gaps (missing days)
        date_diffs = df['Date'].diff().dt.days
        large_gaps = (date_diffs > 7).sum()  # More than 1 week gap
        if large_gaps > 0:
            issues.append(f"{symbol}: {large_gaps} large date gaps")
        
        # Check OHLC validity
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        ).sum()
        if invalid_ohlc > 0:
            issues.append(f"{symbol}: {invalid_ohlc} invalid OHLC rows")
        
        # Check for weekend/holiday data (basic check)
        weekend_data = df[df['Date'].dt.weekday >= 5]
        if len(weekend_data) > 0:
            issues.append(f"{symbol}: {len(weekend_data)} weekend dates found")
    
    if issues:
        print("⚠️  DATA QUALITY ISSUES FOUND:\n")
        for issue in issues:
            print(f"  • {issue}")
        print("\n   Consider running fix_csv_data.py again\n")
    else:
        print("✅ All data passed quality checks!\n")


# Example usage
if __name__ == '__main__':
    # Load data
    pair_data = load_forex_data()
    
    # Validate quality
    validate_data_quality(pair_data)
    
    # Show sample
    print("📊 SAMPLE DATA (EURUSD first 5 rows):\n")
    if 'EURUSD' in pair_data:
        print(pair_data['EURUSD'].head())
    
    print("\n" + "="*80)
    print("✅ Data ready for training!")
    print("   Use: from load_clean_data import load_forex_data")
    print("   Then: pair_data = load_forex_data()")
    print("="*80 + "\n")
