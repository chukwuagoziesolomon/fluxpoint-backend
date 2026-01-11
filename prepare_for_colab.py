# ============================================================================
# UPLOAD CLEANED DATA TO GOOGLE DRIVE FOR COLAB
# ============================================================================
# This script helps you upload your cleaned CSV files to Google Drive
# so they can be used in Google Colab notebooks

import shutil
from pathlib import Path
import os

print("\n" + "="*80)
print("PREPARE CLEANED DATA FOR GOOGLE COLAB")
print("="*80)

# Source: Your cleaned data
source_dir = Path(r'C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data_cleaned')

# Check if cleaned data exists
if not source_dir.exists():
    print("\n❌ ERROR: Cleaned data not found!")
    print(f"   Expected location: {source_dir}")
    print("\n📝 Run this first:")
    print("   python fix_csv_data.py")
    exit(1)

# Count files
csv_files = list(source_dir.glob('*_data.csv'))
if not csv_files:
    print(f"\n❌ ERROR: No CSV files found in {source_dir}")
    exit(1)

print(f"\n✅ Found {len(csv_files)} cleaned CSV files")
print(f"📂 Source: {source_dir}\n")

# Calculate total size
total_size = sum(f.stat().st_size for f in csv_files)
total_size_mb = total_size / (1024 * 1024)

print(f"📊 Total data size: {total_size_mb:.2f} MB\n")

# Show files
print("📄 Files to upload:")
for csv_file in sorted(csv_files):
    size_kb = csv_file.stat().st_size / 1024
    print(f"   • {csv_file.name:<25} ({size_kb:>6.1f} KB)")

print("\n" + "="*80)
print("📤 UPLOAD TO GOOGLE DRIVE - INSTRUCTIONS")
print("="*80)

print("""
OPTION 1: Manual Upload (Recommended for beginners)
────────────────────────────────────────────────────────────────

1. Open Google Drive in your browser:
   https://drive.google.com

2. Create this folder structure:
   My Drive/
   └── forex_data/
       └── training_data_cleaned/

3. Upload all CSV files from:
   {source}
   
   Into Google Drive folder:
   My Drive/forex_data/training_data_cleaned/

4. In Google Colab, the path will be:
   /content/drive/MyDrive/forex_data/training_data_cleaned/


OPTION 2: Use Google Drive Desktop App (Faster)
────────────────────────────────────────────────────────────────

1. Install Google Drive for Desktop:
   https://www.google.com/drive/download/

2. Sign in and sync your Google Drive

3. Copy the cleaned data folder:
   From: {source}
   To:   G:\\My Drive\\forex_data\\training_data_cleaned\\
   
   (Your G: drive letter may vary)

4. Wait for sync to complete (check system tray icon)


OPTION 3: Zip and Upload (For slow connections)
────────────────────────────────────────────────────────────────

1. Create a ZIP file of cleaned data:
   
""".format(source=source_dir))

# Create zip file
zip_path = Path('training_data_cleaned.zip')
print(f"   Creating ZIP file: {zip_path}")
shutil.make_archive('training_data_cleaned', 'zip', source_dir)
print(f"   ✅ Created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

print(f"""
2. Upload {zip_path} to Google Drive

3. In Google Colab, run this code to extract:

   ```python
   from google.colab import drive
   import zipfile
   
   # Mount Drive
   drive.mount('/content/drive')
   
   # Extract ZIP
   with zipfile.ZipFile('/content/drive/MyDrive/training_data_cleaned.zip', 'r') as zip_ref:
       zip_ref.extractall('/content/drive/MyDrive/forex_data/')
   
   print("✅ Data extracted!")
   ```


VERIFY IN COLAB
────────────────────────────────────────────────────────────────

Run this in a Colab cell to verify your upload:

```python
from google.colab import drive
from pathlib import Path

# Mount Drive
drive.mount('/content/drive')

# Check for cleaned data
data_dir = Path('/content/drive/MyDrive/forex_data/training_data_cleaned')

if data_dir.exists():
    csv_files = list(data_dir.glob('*.csv'))
    print(f"✅ Found {{len(csv_files)}} CSV files!")
    for f in sorted(csv_files)[:5]:
        print(f"   • {{f.name}}")
else:
    print("❌ Data folder not found!")
    print(f"   Expected: {{data_dir}}")
```


NEXT STEPS
────────────────────────────────────────────────────────────────

After uploading to Google Drive:

1. Open your Colab notebook

2. Run CELL4_MULTI_TIMEFRAME_TRAINING.py
   - It will auto-detect the cleaned data in Drive
   - It will generate multi-timeframe training data
   - It will train the neural network

3. The script will save trained models back to Drive:
   /content/drive/MyDrive/models/
   ├── tce_multi_tf_model.pt
   ├── scaler_mean.npy
   └── scaler_scale.npy

""")

print("="*80)
print(f"✅ ZIP file created: {zip_path.absolute()}")
print("="*80 + "\n")
