"""
Download Multi-Timeframe Forex Data from MT5
==============================================

This script downloads REAL historical price data from MetaTrader 5 for:
- 15 forex pairs (EURUSD, GBPUSD, AUDJPY, etc.)
- 8 timeframes (M1, M5, M15, M30, H1, H4, D1, W1)
- Automatically removes weekends and holidays
- Saves to training_data_mt5/ folder

Requirements:
1. MT5 terminal installed and running
2. Demo or real account logged in
3. pip install MetaTrader5 pandas

Usage:
    python download_mt5_multi_timeframe.py
"""

import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

# ================================================================================
# CONFIGURATION
# ================================================================================

# Forex pairs to download (28 major pairs)
FOREX_PAIRS = [
    # Major pairs (USD base or quote)
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'USDCHF', 'USDHKD', 'USDSGD', 'USDZAR',
    # Euro cross pairs
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURNZD', 'EURCHF',
    # GBP cross pairs
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'GBPCHF',
    # Other major crosses
    'AUDJPY', 'AUDCAD', 'AUDNZD', 'AUDCHF',
    'NZDJPY', 'CADJPY', 'CHFJPY'
]

# Timeframes configuration
# Brokers store different amounts of history per timeframe
TIMEFRAMES = {
    'M1': {
        'mt5_code': mt5.TIMEFRAME_M1,
        'label': '1-minute',
        'months_back': 1  # Conservative: 1 month (most brokers limit M1)
    },
    'M5': {
        'mt5_code': mt5.TIMEFRAME_M5,
        'label': '5-minute',
        'months_back': 8  # Extended to 8 months
    },
    'M15': {
        'mt5_code': mt5.TIMEFRAME_M15,
        'label': '15-minute',
        'months_back': 18  # Extended to 18 months
    },
    'M30': {
        'mt5_code': mt5.TIMEFRAME_M30,
        'label': '30-minute',
        'months_back': 24  # Extended to 24 months (2 years)
    },
    'H1': {
        'mt5_code': mt5.TIMEFRAME_H1,
        'label': '1-hour',
        'bars_back': 30000  # Conservative: ~3.4 years (720 bars/month * 47 months)
    },
    'H4': {
        'mt5_code': mt5.TIMEFRAME_H4,
        'label': '4-hour',
        'bars_back': 40000  # ~16 years of H4 data
    },
    'D1': {
        'mt5_code': mt5.TIMEFRAME_D1,
        'label': 'Daily',
        'bars_back': 10000  # ~27 years of daily data
    },
    'W1': {
        'mt5_code': mt5.TIMEFRAME_W1,
        'label': 'Weekly',
        'bars_back': 2000   # ~38 years of weekly data
    }
}

# Output directory
OUTPUT_DIR = Path('training_data_mt5')

# Major market holidays (add more as needed)
HOLIDAYS_2020_2026 = {
    # 2020
    datetime(2020, 1, 1), datetime(2020, 12, 25), datetime(2020, 12, 26),
    # 2021
    datetime(2021, 1, 1), datetime(2021, 12, 25), datetime(2021, 12, 26),
    # 2022
    datetime(2022, 1, 1), datetime(2022, 12, 25), datetime(2022, 12, 26),
    # 2023
    datetime(2023, 1, 1), datetime(2023, 12, 25), datetime(2023, 12, 26),
    # 2024
    datetime(2024, 1, 1), datetime(2024, 12, 25), datetime(2024, 12, 26),
    # 2025
    datetime(2025, 1, 1), datetime(2025, 12, 25), datetime(2025, 12, 26),
    # 2026
    datetime(2026, 1, 1), datetime(2026, 12, 25), datetime(2026, 12, 26),
}

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def is_trading_day(date: datetime) -> bool:
    """Check if a date is a valid trading day (not weekend or holiday)"""
    # Check if weekend (Saturday=5, Sunday=6)
    if date.weekday() >= 5:
        return False
    
    # Check if holiday (compare only date, not time)
    if date.replace(hour=0, minute=0, second=0, microsecond=0) in HOLIDAYS_2020_2026:
        return False
    
    return True

def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove weekend and holiday data from DataFrame"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to trading days only
    df = df[df['Date'].apply(is_trading_day)]
    
    return df.reset_index(drop=True)

def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

# ================================================================================
# MT5 CONNECTION
# ================================================================================

def initialize_mt5():
    """Initialize connection to MT5"""
    print("="*80)
    print("🔌 CONNECTING TO MT5")
    print("="*80 + "\n")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed!")
        print("\n📝 TROUBLESHOOTING:")
        print("   1. Make sure MT5 terminal is installed")
        print("   2. Open MT5 and log into your account (demo or real)")
        print("   3. Keep MT5 running in the background")
        print("   4. Run this script again")
        return False
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print("⚠️  No account logged in!")
        print("   Please log into MT5 terminal first")
        mt5.shutdown()
        return False
    
    print(f"✅ Connected to MT5")
    print(f"   Account: {account_info.login}")
    print(f"   Broker: {account_info.company}")
    print(f"   Server: {account_info.server}")
    print(f"   Currency: {account_info.currency}\n")
    
    return True

def verify_symbol(symbol: str) -> str:
    """
    Verify symbol exists and return correct format
    Returns None if symbol not available
    """
    # Try exact symbol name
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is not None:
        return symbol
    
    # Try common variations
    variations = [
        symbol,
        symbol + '.raw',
        symbol + 'm',
        symbol + '#',
        symbol.lower(),
        symbol.upper()
    ]
    
    for variant in variations:
        symbol_info = mt5.symbol_info(variant)
        if symbol_info is not None:
            return variant
    
    return None

# ================================================================================
# DATA DOWNLOAD
# ================================================================================

def download_symbol_timeframe(symbol: str, timeframe_key: str, 
                               timeframe_config: dict) -> tuple:
    """
    Download data for a single symbol and timeframe
    
    Returns:
        (success: bool, candles_count: int, info: dict)
    """
    try:
        # Verify symbol exists
        verified_symbol = verify_symbol(symbol)
        if verified_symbol is None:
            return False, 0, {'error': f'Symbol not available'}
        
        # Enable symbol in Market Watch if needed
        if not mt5.symbol_select(verified_symbol, True):
            return False, 0, {'error': 'Could not enable symbol'}
        
        # Calculate parameters based on timeframe configuration
        if 'months_back' in timeframe_config:
            # Intraday timeframes: calculate bar count from months
            months = timeframe_config['months_back']
            days = months * 30
            
            # Estimate bars based on timeframe
            if timeframe_key == 'M1':
                bars_needed = days * 24 * 60      # 1440 bars/day
            elif timeframe_key == 'M5':
                bars_needed = days * 24 * 12      # 288 bars/day
            elif timeframe_key == 'M15':
                bars_needed = days * 24 * 4       # 96 bars/day
            elif timeframe_key == 'M30':
                bars_needed = days * 24 * 2       # 48 bars/day
            else:
                bars_needed = days * 24           # Default: 24 bars/day
        else:
            # Higher timeframes: use bars_back directly
            bars_needed = timeframe_config['bars_back']
        
        # Use copy_rates_from_pos (most reliable - counts from current bar backwards)
        # Position 0 = current bar, position 1 = previous bar, etc.
        rates = mt5.copy_rates_from_pos(
            verified_symbol,
            timeframe_config['mt5_code'],
            0,  # Start from current bar
            bars_needed  # Number of bars to retrieve
        )
        
        # Check if data was returned
        if rates is None:
            error_code, error_msg = mt5.last_error()
            return False, 0, {'error': f'MT5 error {error_code}: {error_msg}'}
        
        if len(rates) == 0:
            return False, 0, {'error': 'No data for this date range'}
        
        # Convert structured array to DataFrame
        # MT5 returns numpy structured array with lowercase field names:
        # 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'
        try:
            # Create DataFrame directly from structured array
            df = pd.DataFrame(rates)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            })
            
            # Convert Unix timestamp to datetime
            df['Date'] = pd.to_datetime(df['Date'], unit='s')
            
            # Select only needed columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            return False, 0, {'error': f'Data conversion failed: {str(e)}'}
        
        # Filter trading days
        original_count = len(df)
        df = filter_trading_days(df)
        filtered_count = len(df)
        
        # Check if we have any data left after filtering
        if filtered_count == 0:
            return False, 0, {'error': 'No trading days in date range'}
        
        # Sort by date (oldest first)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create timeframe directory
        timeframe_dir = OUTPUT_DIR / timeframe_key
        timeframe_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        file_path = timeframe_dir / f"{symbol.lower()}_{timeframe_key.lower()}.csv"
        df.to_csv(file_path, index=False)
        
        # Get date range (handle NaT values for W1 strftime error)
        date_min = df['Date'].min()
        date_max = df['Date'].max()
        date_from = date_min.strftime('%Y-%m-%d') if pd.notna(date_min) else 'Unknown'
        date_to = date_max.strftime('%Y-%m-%d') if pd.notna(date_max) else 'Unknown'
        
        return True, filtered_count, {
            'path': file_path,
            'original': original_count,
            'filtered': filtered_count,
            'removed': original_count - filtered_count,
            'date_from': date_from,
            'date_to': date_to
        }
        
    except Exception as e:
        return False, 0, {'error': str(e)}

def download_all_data():
    """Download all pairs and timeframes"""
    print("="*80)
    print("📥 DOWNLOADING MULTI-TIMEFRAME DATA")
    print("="*80 + "\n")
    
    # Check symbol availability first
    print("🔍 Verifying symbol availability...\n")
    available_symbols = {}
    unavailable_symbols = []
    
    for symbol in FOREX_PAIRS:
        verified = verify_symbol(symbol)
        if verified:
            available_symbols[symbol] = verified
            status = f"✅ {symbol:<10} → {verified}" if verified != symbol else f"✅ {symbol}"
            print(f"   {status}")
        else:
            unavailable_symbols.append(symbol)
            print(f"   ❌ {symbol:<10} → Not available")
    
    print(f"\n   Available: {len(available_symbols)}/{len(FOREX_PAIRS)} pairs")
    
    if len(available_symbols) == 0:
        print("\n❌ No symbols available! Check your broker's symbol list.")
        return
    
    if unavailable_symbols:
        print(f"   ⚠️  Skipping: {', '.join(unavailable_symbols)}\n")
    else:
        print()
    
    total_pairs = len(available_symbols)
    total_timeframes = len(TIMEFRAMES)
    
    print(f"📊 Configuration:")
    print(f"   • Pairs: {total_pairs}")
    print(f"   • Timeframes: {total_timeframes}")
    print(f"   • Total downloads: {total_pairs * total_timeframes}")
    print(f"   • Output: {OUTPUT_DIR}/\n")
    
    # Statistics
    stats = {
        'successful': 0,
        'failed': 0,
        'total_candles': 0,
        'total_removed': 0,
        'start_time': time.time()
    }
    
    # Download each pair and timeframe
    for pair_idx, (symbol, verified_symbol) in enumerate(available_symbols.items(), 1):
        display_name = symbol if symbol == verified_symbol else f"{symbol} ({verified_symbol})"
        print(f"[{pair_idx}/{total_pairs}] {display_name}")
        
        for tf_idx, (tf_key, tf_config) in enumerate(TIMEFRAMES.items(), 1):
            print(f"   {tf_key:<4} ({tf_config['label']:<10})", end=" ", flush=True)
            
            success, candles, info = download_symbol_timeframe(
                symbol, tf_key, tf_config
            )
            
            if success:
                stats['successful'] += 1
                stats['total_candles'] += candles
                stats['total_removed'] += info['removed']
                
                print(f"✅ {candles:>6,} candles | {info['date_from']} to {info['date_to']}")
            else:
                stats['failed'] += 1
                error_msg = info.get('error', 'Unknown error') if info else 'No data'
                print(f"❌ Failed: {error_msg}")
            
            # Small delay to avoid overwhelming MT5
            time.sleep(0.1)
        
        print()  # Blank line between pairs
    
    # Calculate elapsed time
    elapsed = time.time() - stats['start_time']
    
    # Print summary
    print("="*80)
    print("📊 DOWNLOAD COMPLETE")
    print("="*80)
    print(f"\n✅ Results:")
    print(f"   • Successful: {stats['successful']}/{total_pairs * total_timeframes}")
    print(f"   • Failed: {stats['failed']}")
    print(f"   • Total candles: {stats['total_candles']:,}")
    print(f"   • Removed (weekends/holidays): {stats['total_removed']:,}")
    print(f"   • Time elapsed: {elapsed:.1f}s")
    print(f"   • Data location: {OUTPUT_DIR.absolute()}/")
    
    # Get directory size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*.csv'))
    print(f"   • Total size: {format_size(total_size)}")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("1. Verify data files in training_data_mt5/ folder")
    print("2. Update COLAB_COMPLETE_PIPELINE.py to use this data")
    print("3. Upload to Google Drive for Colab training")
    print("4. Run training pipeline!\n")

# ================================================================================
# MAIN
# ================================================================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🚀 MT5 MULTI-TIMEFRAME DATA DOWNLOADER")
    print("="*80 + "\n")
    
    # Initialize MT5
    if not initialize_mt5():
        return
    
    try:
        # Download all data
        download_all_data()
        
    finally:
        # Always shutdown MT5 connection
        mt5.shutdown()
        print("✅ MT5 connection closed\n")

if __name__ == "__main__":
    main()
