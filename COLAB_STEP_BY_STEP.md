# Google Colab Training - Complete Step-by-Step Guide

## Why Use Google Colab?

✅ **Free GPU** - 3-4x faster training
✅ **No setup** - Everything pre-installed
✅ **Easy data sharing** - Google Drive integration
✅ **Run overnight** - Start training, come back tomorrow
✅ **Free backup** - Models saved in Google Drive

---

## 📋 Complete Checklist

### Pre-Training (30 minutes)

- [ ] Open https://colab.research.google.com/
- [ ] Create new notebook
- [ ] Check GPU available (Runtime > Change runtime type > GPU)
- [ ] Prepare data files or APIs

### Step 1: Mount Google Drive (2 minutes)

**Colab Cell 1:**
```python
from google.colab import drive
drive.mount('/content/drive')
print("✓ Drive mounted")
```

**What it does:**
- Connects Colab to your Google Drive
- Allows reading/writing files

**Expected output:**
```
Go to this URL in a browser and authorize...
✓ Drive mounted
```

### Step 2: Install Dependencies (5-10 minutes)

**Colab Cell 2:**
```python
import subprocess
import sys

packages = [
    'pandas', 'numpy', 'torch', 
    'stable-baselines3', 'gymnasium',
    'scikit-learn', 'matplotlib'
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✓ All packages installed")
```

**What it does:**
- Installs all required Python packages
- Takes 5-10 minutes (be patient!)

**Expected output:**
```
Installing pandas...
Installing numpy...
... (lots of text)
✓ All packages installed
```

### Step 3: Prepare Your Data (10 minutes)

**Option A: Upload CSV Files (Simplest)**

1. Create folder in Google Drive: `My Drive/forex_data/`
2. Upload your CSV files there
3. Files should be named: `EURUSD_H1.csv`, `GBPUSD_H1.csv`, etc.

**Option B: Use Mt5 API**
- Have your credentials ready
- We'll download in next cell

**Colab Cell 3:**
```python
import os
from pathlib import Path

# Create directories
os.makedirs('/content/data', exist_ok=True)
os.makedirs('/content/models', exist_ok=True)

# Check Drive data
data_dir = '/content/drive/MyDrive/forex_data'
if os.path.exists(data_dir):
    csv_files = list(Path(data_dir).glob('*.csv'))
    print(f"✓ Found {len(csv_files)} CSV files:")
    for f in csv_files[:5]:  # Show first 5
        print(f"  - {f.name}")
else:
    print(f"⚠️  Create folder: My Drive/forex_data/")
    print(f"⚠️  Upload CSV files there")
```

### Step 4: Copy Project Files (5 minutes)

**Before this step:**
- Upload your `trading/rl/` folder to Google Drive
- Or sync your project to Drive

**Colab Cell 4:**
```python
import os
import sys

# Copy from Drive to Colab
project_path = '/content/drive/MyDrive/fluxpoint'
os.system(f"cp -r {project_path}/trading /content/")

# Verify
sys.path.insert(0, '/content')

try:
    from trading.rl.multi_pair_training import train_rl_multipair
    print("✓ Project files copied successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Make sure trading/rl/ files are uploaded to Drive")
```

### Step 5: Load & Validate Data (5 minutes)

**Colab Cell 5:**
```python
import pandas as pd
from pathlib import Path

# Load CSV files
pair_data = {}
csv_dir = Path('/content/drive/MyDrive/forex_data')

for csv_file in csv_dir.glob('*.csv'):
    symbol = csv_file.stem.split('_')[0].upper()
    
    try:
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        
        # Validate TCE setups
        from trading.tce.validation import validate_tce_setups
        setups = validate_tce_setups(df, symbol)
        
        pair_data[symbol] = (df, setups)
        print(f"✓ {symbol}: {len(setups)} valid setups")
    
    except Exception as e:
        print(f"✗ {symbol}: {e}")

print(f"\n✓ Total: {len(pair_data)} pairs ready to train")
```

**Expected output:**
```
✓ EURUSD: 1245 valid setups
✓ GBPUSD: 987 valid setups
✓ USDJPY: 1150 valid setups

✓ Total: 3 pairs ready to train
```

### Step 6: Train RL Model (8-12 hours) ⭐ MAIN STEP

**Colab Cell 6:**
```python
from trading.rl.multi_pair_training import train_rl_multipair

# Training parameters
MODEL_NAME = "tce_colab_v1"
TOTAL_TIMESTEPS = 200000

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Pairs: {list(pair_data.keys())}")
print(f"Total setups: {sum(len(s) for _, (_, s) in pair_data.items())}")
print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
print(f"GPU: ✓ ENABLED")
print("="*70 + "\n")

# Train!
metrics = train_rl_multipair(
    pair_data=pair_data,
    model_name=MODEL_NAME,
    total_timesteps=TOTAL_TIMESTEPS
)

# Show results
print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)

eval_results = metrics.get('eval', {})
print(f"\nR-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
print(f"Win Rate: {eval_results.get('mean_win_rate', 0):.1%}")
print(f"Mean Reward: {eval_results.get('mean_reward', 0):.2f}")
```

**What happens:**
- ✓ Training starts (takes 8-12 hours)
- ✓ Checkpoints auto-save every 20K timesteps
- ✓ Metrics logged to TensorBoard
- ✓ Model saved when done

**How to monitor:**
- Watch the output (loss decreasing = good)
- Check TensorBoard if available
- Colab won't disconnect for 12 hours (good enough!)

### Step 7: Save to Google Drive (1 minute)

**Colab Cell 7:**
```python
import os
import shutil

# Save location in Drive
drive_model_dir = '/content/drive/MyDrive/rl_models'
os.makedirs(drive_model_dir, exist_ok=True)

# Copy model
local_model = f'./models/rl/{MODEL_NAME}'
drive_model = f'{drive_model_dir}/{MODEL_NAME}'

os.system(f"cp -r {local_model}* {drive_model_dir}/")

print(f"✓ Model saved to Google Drive")
print(f"  Location: {drive_model}")
print(f"  You can download it now")
```

**What it does:**
- Copies model from Colab to Drive
- Keeps it even if Colab session ends

### Step 8: Evaluate Results (5 minutes)

**Colab Cell 8:**
```python
eval_results = metrics.get('eval', {})

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\nMean Reward:       {eval_results.get('mean_reward', 0):.2f}")
print(f"R-Multiple:        {eval_results.get('mean_r_multiple', 0):.2f}R")
print(f"Win Rate:          {eval_results.get('mean_win_rate', 0):.1%}")
print(f"Episodes:          {eval_results.get('n_episodes', 0)}")

# Compare to baseline
print("\n" + "-"*70)
print("Comparison to Single-Pair Training:")
print("-"*70)

r_multiple = eval_results.get('mean_r_multiple', 0)
improvement = ((r_multiple - 1.2) / 1.2) * 100

print(f"Single-Pair R:     1.2R")
print(f"Multi-Pair R:      {r_multiple:.2f}R")
print(f"Improvement:       {improvement:+.1f}%")

if r_multiple > 1.3:
    print("\n✓ EXCELLENT RESULTS!")
elif r_multiple > 1.0:
    print("\n✓ GOOD RESULTS")
else:
    print("\n⚠️  NEEDS IMPROVEMENT - Check data quality")
```

### Step 9: Download Model (2 minutes)

**You're done! Your model is in Google Drive:**

1. Go to Google Drive
2. Find `My Drive/rl_models/{MODEL_NAME}/`
3. Download the .zip file
4. Extract on your computer
5. Place in `trading/rl/models/`

**Now you can:**
- Backtest on your computer
- Paper trade on demo
- Deploy to live trading

### Step 10: What's Next?

```
Week 1: Evaluation
  [ ] Backtest on 2024 data
  [ ] Compare metrics
  [ ] Test on unseen pairs

Week 2-3: Paper Trading
  [ ] Deploy to demo
  [ ] Run 50+ trades
  [ ] Monitor performance

Week 4+: Live Trading
  [ ] Deploy to live
  [ ] Monitor daily
  [ ] Plan retraining
```

---

## ⏱️ Colab Training Timeline

| Step | Time | What | Notes |
|------|------|------|-------|
| 1. Mount Drive | 2 min | Connect to Drive | Automatic |
| 2. Install deps | 5-10 min | Packages | One-time |
| 3. Data prep | 5 min | Load files | From Drive |
| 4. Copy files | 2 min | Project code | One-time |
| 5. Validate data | 5 min | Check setups | Quick check |
| 6. **TRAIN** | **8-12 hrs** | **Main step** | **Can close tab!** |
| 7. Save model | 1 min | To Drive | Automatic |
| 8. Evaluate | 5 min | Check results | Quick look |
| 9. Download | 2 min | Get model | To computer |
| **Total Setup** | **30 min** | - | - |
| **Total Training** | **8-12 hrs** | - | Overnight! |

---

## 🖥️ Hardware Expectations

### Colab GPU (Free)
- **Type**: NVIDIA Tesla T4 or V100
- **Speed**: 8-12 hours for 200K timesteps
- **Cost**: Free!
- **Limit**: 12 hours continuous session

### Colab CPU (Fallback)
- **Speed**: 20-30 hours for 200K timesteps
- **Cost**: Free
- **Best for**: If GPU unavailable

### Your Local GPU (Optional)
- **Speed**: 2-4 hours (depends on GPU)
- **Cost**: Electricity only
- **Best for**: Frequent training

**Recommendation**: Use Colab GPU (free and fast!)

---

## ✅ Complete Checklist Before Training

### Data Ready
- [ ] Have 3+ CSV files with OHLCV data
- [ ] Each file has 1000+ candles
- [ ] Each pair has 1000+ valid setups
- [ ] Files uploaded to Drive

### Colab Preparation
- [ ] Opened https://colab.research.google.com/
- [ ] Created new notebook
- [ ] Enabled GPU (Runtime > Change runtime type > GPU)
- [ ] Project files uploaded to Drive

### Code Ready
- [ ] Copied COLAB_SETUP_GUIDE.py cells
- [ ] Adjusted paths for your Drive
- [ ] Ran cells 1-5 without errors
- [ ] Verified data loads in cell 5

### Ready to Train
- [ ] All dependencies installed
- [ ] Data validated (3+ pairs)
- [ ] Model name set (e.g., "tce_colab_v1")
- [ ] Cell 6 ready to run

---

## 🚨 Common Colab Issues & Fixes

### Issue: "GPU not available"
**Solution:**
```
1. Runtime > Change runtime type
2. Hardware accelerator > GPU
3. Save
4. Re-run all cells from top
```

### Issue: "File not found"
**Solution:**
```
1. Check file is in Google Drive
2. Verify path is correct
3. Print os.path.exists(path) to debug
```

### Issue: "ModuleNotFoundError: trading"
**Solution:**
```python
# Make sure this works
import sys
sys.path.insert(0, '/content')
from trading.rl.multi_pair_training import train_rl_multipair
```

### Issue: "Out of memory"
**Solution:**
```python
# Reduce dataset size
TOTAL_TIMESTEPS = 100000  # Instead of 200000
# Or use fewer pairs (2 instead of 5)
```

### Issue: "Timeout / Session disconnected"
**Solution:**
```
- GPU sessions stay connected longer
- Enable background execution
- Or train overnight (takes 8 hrs anyway)
```

### Issue: "Can't save to Drive"
**Solution:**
```
1. Check Drive mount successful (Cell 1)
2. Check folder exists in Drive
3. Check permissions (Drive > Settings > Share)
```

---

## 📊 Monitoring Training in Colab

### Watch the Output
```
Step 50000/200000
  Loss: 0.45 → 0.42 (improving ✓)
  Reward: 2.1 (increasing ✓)

Step 100000/200000
  Loss: 0.38
  Reward: 3.2

Step 200000/200000 DONE
```

If loss INCREASES → Something wrong
If reward DECREASES → Data issue

### Check TensorBoard (Advanced)
```python
# In Colab cell:
%load_ext tensorboard
%tensorboard --logdir /content/logs/tensorboard/
```

### Save Output
```python
# Copy logs to Drive
import shutil
shutil.copytree('./logs', '/content/drive/MyDrive/training_logs')
```

---

## 🎯 After Training

### Download Model
1. Go to Google Drive
2. Find `rl_models/{MODEL_NAME}/`
3. Download .zip
4. Extract on computer
5. Use in trading system

### Save Metrics
```python
import json

results = {
    'model': MODEL_NAME,
    'timesteps': TOTAL_TIMESTEPS,
    'metrics': metrics['eval']
}

with open('/content/drive/MyDrive/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Schedule Retraining
- Add new data to Drive
- Run notebook again next month
- Compare results (should improve)

---

## 💡 Pro Tips

1. **Run overnight** - Start training at 9 PM, check results at 9 AM
2. **Keep Drive organized** - Create folders: /forex_data, /rl_models, /training_logs
3. **Save intermediate** - Models auto-save every 20K timesteps
4. **Monitor GPU** - Watch for memory warnings
5. **Test before training** - Run cells 1-5 without errors first
6. **Save results** - Copy logs and metrics to Drive
7. **Retrain monthly** - Keep model fresh with new data

---

## ❓ Need Help?

**Issue with Colab?** → Check "Common Colab Issues" above
**Issue with training?** → See MULTIPAIR_QUICK_CHECKLIST.md
**Issue with data?** → See MULTIPAIR_TRAINING_GUIDE.md
**Code not working?** → See train_multipair_example.py

---

## 🎉 Summary

**Google Colab Training in 3 Steps:**

1. **Setup** (30 min)
   - Mount Drive
   - Install packages
   - Copy project files

2. **Train** (8-12 hours)
   - Load data
   - Run training
   - Monitor output

3. **Download** (2 min)
   - Save to Drive
   - Download model
   - Use for trading

**Total time investment: 30 minutes + waiting overnight**

**Result: Multi-pair RL model trained on GPU, ready for trading!** 🚀
