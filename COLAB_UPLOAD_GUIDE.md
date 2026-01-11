# 🚀 Quick Start: Upload Cleaned Data to Google Colab

## Option 1: Drag & Drop (Easiest) ⭐

### Step 1: Open Google Drive
1. Go to https://drive.google.com
2. Create folders:
   - Click "New" → "New folder"
   - Name it: `forex_data`
   - Inside `forex_data`, create another folder: `training_data_cleaned`

### Step 2: Upload Files
1. Open the folder: `forex_data/training_data_cleaned/`
2. Click "New" → "File upload"
3. Select ALL 15 CSV files from:
   ```
   C:\Users\USER-PC\fluxpointai-backend\fluxpoint\training_data_cleaned\
   ```
4. Wait for upload to complete (~2 MB, takes 1-2 minutes)

### Step 3: Verify in Colab
Open Google Colab and paste this:

```python
# CELL 1: Verify Data
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

data_dir = Path('/content/drive/MyDrive/forex_data/training_data_cleaned')
csv_files = list(data_dir.glob('*.csv'))

print(f"✅ Found {len(csv_files)} files!")
for f in sorted(csv_files):
    print(f"  • {f.name}")
```

If you see 15 CSV files listed → **SUCCESS!** ✅

---

## Option 2: ZIP Upload (Faster for Slow Internet)

### Step 1: Upload ZIP File
1. Find the file: `training_data_cleaned.zip` (created in your project folder)
2. Go to https://drive.google.com
3. Upload `training_data_cleaned.zip` to the root of your Drive

### Step 2: Extract in Colab
```python
# CELL 1: Extract ZIP
from google.colab import drive
import zipfile

drive.mount('/content/drive')

# Extract
with zipfile.ZipFile('/content/drive/MyDrive/training_data_cleaned.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/drive/MyDrive/forex_data/')

print("✅ Extracted!")
```

---

## 📋 Complete Colab Workflow

### CELL 1: Setup & Verify Data
```python
# Run COLAB_SETUP_CELL.py content
# (Paste the entire content of COLAB_SETUP_CELL.py here)
```

### CELL 2: Train Multi-Timeframe Model
```python
# Run CELL4_MULTI_TIMEFRAME_TRAINING.py content
# (Paste the entire content of CELL4_MULTI_TIMEFRAME_TRAINING.py here)
```

---

## 🎯 Your Folder Structure in Google Drive

After upload, you should have:

```
My Drive/
├── forex_data/
│   └── training_data_cleaned/
│       ├── audjpy_data.csv
│       ├── audusd_data.csv
│       ├── eurchf_data.csv
│       ├── eurgbp_data.csv
│       ├── eurjpy_data.csv
│       ├── eurusd_data.csv
│       ├── gbpchf_data.csv
│       ├── gbpjpy_data.csv
│       ├── gbpusd_data.csv
│       ├── nzdjpy_data.csv
│       ├── nzdusd_data.csv
│       ├── usdcad_data.csv
│       ├── usdchf_data.csv
│       ├── usdhkd_data.csv
│       └── usdjpy_data.csv
└── models/  ← Will be created automatically by training script
    ├── tce_multi_tf_model.pt
    ├── scaler_mean.npy
    └── scaler_scale.npy
```

---

## ⚡ Quick Test

After uploading, test immediately with this one-liner:

```python
from google.colab import drive; from pathlib import Path; drive.mount('/content/drive'); print(f"✅ Found {len(list(Path('/content/drive/MyDrive/forex_data/training_data_cleaned').glob('*.csv')))} files")
```

Should output: `✅ Found 15 files`

---

## 🐛 Troubleshooting

### Problem: "Data folder not found"
**Solution:** Check folder name spelling - it must be EXACTLY:
- `forex_data` (lowercase, underscore)
- `training_data_cleaned` (lowercase, underscores)

### Problem: "No CSV files found"
**Solution:** 
1. Make sure files are INSIDE `training_data_cleaned/` folder
2. Files must end with `.csv`
3. Check you uploaded the RIGHT folder (cleaned, not original)

### Problem: Upload is stuck
**Solution:**
1. Cancel and use ZIP method instead
2. Or upload in batches (5 files at a time)

---

## 📊 Expected Results After Training

```
================================================================================
📊 MULTI-TIMEFRAME SUMMARY:

  1M     2,175 valid /   324,000 checked ( 0.67%)
  5M    10,350 valid /   648,000 checked ( 1.60%)
  15M    6,345 valid /   216,000 checked ( 2.94%)
  30M    5,400 valid /   108,000 checked ( 5.00%)
  1H     5,760 valid /    54,000 checked (10.67%)
  4H     2,700 valid /    13,500 checked (20.00%)
  1D     1,125 valid /     2,229 checked (50.47%)
  1W       160 valid /       321 checked (49.84%)

  TOTAL: ~30,000+ VALID TCE SETUPS

🤖 Training Accuracy: 75-85%
```

---

## 💡 Pro Tips

1. **Enable GPU in Colab:** Runtime → Change runtime type → GPU (Tesla T4)
   - 10-20X faster training!

2. **Keep browser tab open:** Training takes 30-60 minutes
   - Don't close the tab or training will stop

3. **Save checkpoints:** Models auto-save to Drive every epoch

4. **Monitor RAM:** View → Execution details
   - If RAM is full, reduce timeframes or pairs

---

Need help? The script will guide you through each step! ✨
