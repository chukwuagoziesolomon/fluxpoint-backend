# ============================================================================
# COLAB SETUP CELL - Run this FIRST in Google Colab
# ============================================================================
# This cell mounts Google Drive and verifies your cleaned data is ready

from google.colab import drive
from pathlib import Path
import pandas as pd

print("="*80)
print("🔧 GOOGLE COLAB SETUP")
print("="*80)

# 1. Mount Google Drive
print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
print("✅ Google Drive mounted!\n")

# 2. Check for cleaned data
data_dir = Path('/content/drive/MyDrive/forex_data/training_data_cleaned')

print(f"🔍 Checking for cleaned data...")
print(f"   Location: {data_dir}\n")

if not data_dir.exists():
    print("❌ ERROR: Cleaned data folder not found!")
    print("\n📝 TO FIX THIS:")
    print("   1. On your local computer, run: python prepare_for_colab.py")
    print("   2. Follow the upload instructions")
    print("   3. Upload the cleaned data to Google Drive")
    print("   4. Re-run this cell")
    raise FileNotFoundError(f"Data folder not found: {data_dir}")

# 3. List CSV files
csv_files = sorted(data_dir.glob('*_data.csv'))

if not csv_files:
    print("❌ ERROR: No CSV files found in the folder!")
    print("\n📝 TO FIX THIS:")
    print("   1. Make sure you uploaded the files to the correct folder")
    print("   2. Check that files are named: eurusd_data.csv, gbpusd_data.csv, etc.")
    raise FileNotFoundError(f"No CSV files in: {data_dir}")

print(f"✅ Found {len(csv_files)} CSV files:\n")

# 4. Validate each file
total_candles = 0
valid_files = 0

for csv_file in csv_files:
    symbol = csv_file.stem.replace('_data', '').upper()
    try:
        # Quick read to check format
        df = pd.read_csv(csv_file, nrows=5)
        
        # Check columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            print(f"   ⚠️  {symbol:<10} Missing required columns!")
            continue
        
        # Count total rows
        df_full = pd.read_csv(csv_file)
        candles = len(df_full)
        total_candles += candles
        valid_files += 1
        
        # Get date range
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        date_range = f"{df_full['Date'].min().date()} to {df_full['Date'].max().date()}"
        
        print(f"   ✅ {symbol:<10} {candles:>5} candles | {date_range}")
    
    except Exception as e:
        print(f"   ❌ {symbol:<10} Error: {str(e)[:50]}")

print(f"\n{'='*80}")
print(f"📊 SUMMARY:")
print(f"   • Valid files: {valid_files}/{len(csv_files)}")
print(f"   • Total candles: {total_candles:,}")
print(f"   • Average per pair: {total_candles // valid_files if valid_files > 0 else 0:,}")
print(f"{'='*80}\n")

if valid_files == 0:
    raise ValueError("No valid CSV files found! Check your data format.")

# 5. Check GPU availability
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {device}")

if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  No GPU available - training will be SLOW")
    print("   💡 Enable GPU: Runtime → Change runtime type → GPU")

print(f"\n{'='*80}")
print("✅ SETUP COMPLETE - Ready to train!")
print("="*80)
print("\n📝 NEXT STEP: Run CELL4_MULTI_TIMEFRAME_TRAINING.py")
print("   Or paste the training code in the next cell\n")
