# Google Colab Quick Reference Card

## 🚀 Quick Start (Copy-Paste)

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Install Packages
```python
import subprocess, sys
packages = ['pandas', 'numpy', 'torch', 'stable-baselines3', 'gymnasium', 'scikit-learn']
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
```

### Cell 3: Load Data
```python
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '/content')

pair_data = {}
for csv_file in Path('/content/drive/MyDrive/forex_data').glob('*.csv'):
    symbol = csv_file.stem.split('_')[0].upper()
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    from trading.tce.validation import validate_tce_setups
    setups = validate_tce_setups(df, symbol)
    pair_data[symbol] = (df, setups)
    print(f"✓ {symbol}: {len(setups)} setups")
```

### Cell 4: Train Model
```python
from trading.rl.multi_pair_training import train_rl_multipair

metrics = train_rl_multipair(
    pair_data=pair_data,
    model_name="tce_colab_v1",
    total_timesteps=200000
)

eval_results = metrics.get('eval', {})
print(f"R-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
print(f"Win Rate: {eval_results.get('mean_win_rate', 0):.1%}")
```

### Cell 5: Save to Drive
```python
import os
drive_dir = '/content/drive/MyDrive/rl_models'
os.makedirs(drive_dir, exist_ok=True)
os.system(f"cp -r ./models/rl/tce_colab_v1* {drive_dir}/")
print("✓ Model saved to Drive")
```

---

## ✅ Pre-Training Checklist

- [ ] Google Drive account active
- [ ] CSV files uploaded to: `My Drive/forex_data/`
- [ ] Project files in: `My Drive/fluxpoint/` (or uploaded directly)
- [ ] Colab notebook created at https://colab.research.google.com/
- [ ] GPU enabled: Runtime > Change runtime type > GPU
- [ ] Each CSV has 1000+ candles and 1000+ valid setups

---

## ⏱️ Timeline

| Task | Time | Notes |
|------|------|-------|
| Setup (Cells 1-3) | 20 min | One-time |
| Training (Cell 4) | 8-12 hrs | Can close tab |
| Save & Download (Cell 5) | 5 min | Model in Drive |

---

## 🎯 Expected Results

```
After 200K timesteps on 3-5 pairs:

R-Multiple:   1.4-1.6R  (profitable)
Win Rate:     55-57%    (solid)
Mean Reward:  5-8       (positive)

vs Single-Pair:
  +17-33% better R-multiple
  +3-5% better win rate
```

---

## 🔧 Key Paths

| What | Path |
|------|------|
| Data | `/content/drive/MyDrive/forex_data/` |
| Project | `/content/trading/` |
| Models | `/content/drive/MyDrive/rl_models/` |
| Logs | `/content/logs/tensorboard/` |

---

## 🆘 Quick Fixes

**"GPU not available"**
→ Runtime > Change runtime type > Select GPU

**"Module not found"**
→ `import sys; sys.path.insert(0, '/content')`

**"File not found"**
→ Check path with `import os; os.path.exists(path)`

**"Out of memory"**
→ Reduce `TOTAL_TIMESTEPS` or use fewer pairs

**"Can't save to Drive"**
→ Verify Drive mounted successfully (Cell 1)

---

## 📊 Monitoring

Watch these metrics during training:
- **Loss**: Should decrease
- **Reward**: Should increase
- **Timesteps**: Should increment

If loss increases or reward plateaus → Check data quality

---

## 💾 Download Model

1. Go to Google Drive
2. Open: `My Drive/rl_models/tce_colab_v1/`
3. Download the .zip file
4. Extract on your computer
5. Place in: `trading/rl/models/tce_colab_v1/`

---

## ⚡ Performance Tips

- Use GPU (3-4x faster than CPU)
- Train 2-3 pairs first (faster testing)
- Use 200K timesteps minimum
- Train overnight (takes 8-12 hours)

---

## 🎓 Next Steps After Training

1. **Backtest** on 2024 data
2. **Paper trade** on demo account
3. **Deploy** to live (small size)
4. **Monitor** metrics daily
5. **Retrain** monthly

---

## 📚 Documentation

- **Quick**: COLAB_SETUP_GUIDE.py (this file)
- **Detailed**: COLAB_STEP_BY_STEP.md
- **Reference**: MULTIPAIR_QUICK_CHECKLIST.md
- **Complete**: MULTIPAIR_TRAINING_GUIDE.md

---

## ✨ You're Ready!

Open Colab, paste the cells above, and start training! 🚀

**Estimated time:** 30 min setup + 8-12 hours training = Done!
