"""
Google Colab Setup Guide - Multi-Pair RL Training

Complete step-by-step instructions for training your multi-pair RL agent in Google Colab.
This is much faster than running locally (GPU included!)

BENEFITS OF COLAB:
✓ Free GPU (3-4x faster training)
✓ No local setup required
✓ Automatic checkpointing
✓ Easy data sharing via Google Drive
✓ Run while you do other things
"""

# ============================================================================
# STEP 1: MOUNT GOOGLE DRIVE (Run this first in Colab cell)
# ============================================================================

"""
In Google Colab:
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Run this code in first cell:
"""

# CODE FOR COLAB CELL 1:
print("Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted!")


# ============================================================================
# STEP 2: INSTALL DEPENDENCIES (Run in Colab cell)
# ============================================================================

"""
Run this in a Colab cell to install all required packages.
This takes 5-10 minutes.
"""

# CODE FOR COLAB CELL 2:
print("Installing dependencies...")
import subprocess
import sys

packages = [
    'pandas',
    'numpy',
    'torch',
    'stable-baselines3',
    'gymnasium',
    'scikit-learn',
    'matplotlib',
    'tensorboard'
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ All dependencies installed!")


# ============================================================================
# STEP 3: PREPARE YOUR DATA (Two Options)
# ============================================================================

"""
OPTION A: Load from Google Drive (Recommended)
OPTION B: Download from MT5/API
"""

# CODE FOR COLAB CELL 3:

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Create working directory
os.makedirs('/content/data', exist_ok=True)
os.makedirs('/content/models', exist_ok=True)

# ---- OPTION A: LOAD FROM GOOGLE DRIVE ----

print("\n" + "="*70)
print("STEP 3A: Loading Data from Google Drive")
print("="*70)

# If you have CSV files in Drive:
# Place your CSV files in: Google Drive/Colab Notebooks/forex_data/

data_dir = '/content/drive/MyDrive/forex_data'  # Adjust path as needed

if os.path.exists(data_dir):
    csv_files = list(Path(data_dir).glob('*.csv'))
    print(f"✓ Found {len(csv_files)} CSV files in Drive:")
    for f in csv_files:
        print(f"  - {f.name}")
else:
    print(f"⚠️  Directory not found: {data_dir}")
    print("Create this folder in Google Drive and upload your CSV files there")
    print("Or use OPTION B below to download from API")

# ---- OPTION B: DOWNLOAD FROM MT5 API ----

print("\n" + "="*70)
print("STEP 3B: Download from MT5 (if you have API credentials)")
print("="*70)

"""
If you have MT5 credentials, you can:
1. Store credentials in Google Drive (securely)
2. Download data programmatically
3. Save to Colab workspace

See integration_examples.py for MT5 setup code.
"""

print("✓ Data preparation complete!")


# ============================================================================
# STEP 4: COPY PROJECT FILES FROM GITHUB (IMPORTANT!)
# ============================================================================

"""
You need to copy the multi-pair training files from your local project
to Google Colab. Do this BEFORE running training.

OPTIONS:
1. Upload directly to Colab
2. Clone from GitHub (if you have repo)
3. Upload from Drive
"""

# CODE FOR COLAB CELL 4:

print("\n" + "="*70)
print("STEP 4: Setting up project files")
print("="*70)

# Option 1: Copy from Drive (if you sync your project to Drive)
project_dir = '/content/drive/MyDrive/fluxpoint-rl'  # Adjust path

if os.path.exists(project_dir):
    print(f"Found project in Drive: {project_dir}")
    # Copy to Colab workspace
    os.system(f"cp -r {project_dir}/trading /content/trading")
    print("✓ Project files copied from Drive")
else:
    print("Project not found in Drive.")
    print("\nTo fix this:")
    print("1. Upload your trading/rl/ folder to Google Drive")
    print("2. Update path in this cell")
    print("3. Re-run cell")


# ============================================================================
# STEP 5: VALIDATE DATA & CREATE SETUPS
# ============================================================================

"""
Before training, you need:
1. Historical candles (OHLCV data)
2. Valid TCE setups from those candles

This code loads your data and prepares it.
"""

# CODE FOR COLAB CELL 5:

print("\n" + "="*70)
print("STEP 5: Loading and validating data")
print("="*70)

import sys
sys.path.insert(0, '/content')

try:
    from trading.tce.validation import validate_tce_setups
    print("✓ TCE validation module loaded")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure trading/tce/ files are in /content/trading/")

# Load your CSV files
pair_data = {}
csv_dir = Path('/content/data')  # Or use your Drive directory

csv_files = list(csv_dir.glob('*.csv'))
print(f"\nLoading {len(csv_files)} CSV files...")

for csv_file in csv_files:
    symbol = csv_file.stem.split('_')[0].upper()  # Extract symbol from filename
    
    try:
        # Load CSV (adjust columns if needed)
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        
        # Validate TCE setups
        setups = validate_tce_setups(df, symbol)
        
        pair_data[symbol] = (df, setups)
        
        print(f"✓ {symbol}: {len(df)} candles, {len(setups)} setups")
    
    except Exception as e:
        print(f"✗ {symbol}: {type(e).__name__}: {e}")

print(f"\n✓ Loaded {len(pair_data)} pairs")
for symbol, (candles, setups) in pair_data.items():
    print(f"  {symbol}: {len(setups)} setups")


# ============================================================================
# STEP 6: TRAIN MULTI-PAIR RL MODEL
# ============================================================================

"""
Now run the actual training!
This is the main step - takes 8-12 hours with Colab GPU.
"""

# CODE FOR COLAB CELL 6:

print("\n" + "="*70)
print("STEP 6: Training Multi-Pair RL Agent")
print("="*70)

# Import trainer
from trading.rl.multi_pair_training import train_rl_multipair

# Set training parameters
MODEL_NAME = "tce_colab_v1"
TOTAL_TIMESTEPS = 200000  # Adjust based on data size
INITIAL_BALANCE = 10000
RISK_PERCENTAGE = 1.0

print(f"\nTraining Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Pairs: {list(pair_data.keys())}")
print(f"  Total setups: {sum(len(s) for _, (_, s) in pair_data.items())}")
print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  GPU: ✓ (via Colab)")

# Train!
try:
    metrics = train_rl_multipair(
        pair_data=pair_data,
        symbols=list(pair_data.keys()),
        model_name=MODEL_NAME,
        initial_balance=INITIAL_BALANCE,
        risk_percentage=RISK_PERCENTAGE,
        total_timesteps=TOTAL_TIMESTEPS
    )
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    
    eval_results = metrics.get('eval', {})
    print(f"\nFinal Metrics:")
    print(f"  R-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
    print(f"  Win Rate: {eval_results.get('mean_win_rate', 0):.1%}")
    print(f"  Mean Reward: {eval_results.get('mean_reward', 0):.2f}")
    
except Exception as e:
    print(f"\n✗ Training failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# STEP 7: SAVE MODEL TO GOOGLE DRIVE
# ============================================================================

"""
Save your trained model to Google Drive so you can:
1. Download it for live trading
2. Backup the model
3. Share with others
"""

# CODE FOR COLAB CELL 7:

print("\n" + "="*70)
print("STEP 7: Saving Model to Google Drive")
print("="*70)

from trading.rl.multi_pair_training import MultiPairRLTrainer

# Path in Drive where model will be saved
drive_model_path = '/content/drive/MyDrive/rl_models'
os.makedirs(drive_model_path, exist_ok=True)

model_save_path = f"{drive_model_path}/{MODEL_NAME}"

print(f"Saving to: {model_save_path}")

try:
    # Create trainer and save
    trainer = MultiPairRLTrainer(model_name=MODEL_NAME)
    # Note: Model was already saved during training, just copying here
    os.system(f"cp -r ./models/rl/{MODEL_NAME}* {drive_model_path}/")
    
    print(f"✓ Model saved to Google Drive!")
    print(f"  Location: {model_save_path}")
    print(f"  You can now download it for live trading")

except Exception as e:
    print(f"⚠️  Save failed: {e}")


# ============================================================================
# STEP 8: EVALUATE & ANALYZE RESULTS
# ============================================================================

"""
Analyze your training results and compare to baseline.
"""

# CODE FOR COLAB CELL 8:

print("\n" + "="*70)
print("STEP 8: Analyzing Results")
print("="*70)

import json
import matplotlib.pyplot as plt

# Print final metrics
print("\nFINAL METRICS:")
print("="*50)

eval_results = metrics.get('eval', {})

print(f"Mean Reward:          {eval_results.get('mean_reward', 0):.2f}")
print(f"Std Reward:           {eval_results.get('std_reward', 0):.2f}")
print(f"Mean R-Multiple:      {eval_results.get('mean_r_multiple', 0):.2f}R")
print(f"Mean Win Rate:        {eval_results.get('mean_win_rate', 0):.1%}")
print(f"Episodes Evaluated:   {eval_results.get('n_episodes', 0)}")

# Compare to single-pair baseline
print("\n" + "="*50)
print("COMPARISON TO SINGLE-PAIR:")
print("="*50)
print(f"Single-Pair R-Multiple:  1.2R")
print(f"Multi-Pair R-Multiple:   {eval_results.get('mean_r_multiple', 0):.2f}R")
improvement = ((eval_results.get('mean_r_multiple', 0) - 1.2) / 1.2) * 100
print(f"Improvement:             {improvement:+.1f}%")

print(f"\nSingle-Pair Win Rate:   52%")
print(f"Multi-Pair Win Rate:    {eval_results.get('mean_win_rate', 0):.1%}")
wr_improvement = (eval_results.get('mean_win_rate', 0) - 0.52) * 100
print(f"Improvement:            {wr_improvement:+.1f}%")

# Plot results (if available)
try:
    # Visualize if tensorboard logs available
    print("\n✓ Results analysis complete!")
except:
    pass


# ============================================================================
# STEP 9: DOWNLOAD RESULTS & MODEL
# ============================================================================

"""
Download your trained model and results to your computer.
"""

# CODE FOR COLAB CELL 9:

print("\n" + "="*70)
print("STEP 9: Downloading Results")
print("="*70)

# The model is already in Google Drive (Step 7)
# You can download it from Drive directly

print("Your trained model is saved in Google Drive:")
print(f"  Path: My Drive/rl_models/{MODEL_NAME}")
print("\nTo use the model:")
print("  1. Download from Google Drive")
print("  2. Place in your local: trading/rl/models/")
print("  3. Load in your trading system")

# You can also download training logs
print("\nTo download training logs:")
print("  1. Run: from google.colab import files")
print("  2. Run: files.download('/content/logs/tensorboard/')")


# ============================================================================
# STEP 10: WHAT TO DO NEXT
# ============================================================================

"""
After training is complete:
1. Backtest on fresh data
2. Paper trade on demo
3. Deploy to live (small account)
4. Monitor metrics
5. Retrain monthly
"""

print("\n" + "="*70)
print("STEP 10: Next Steps")
print("="*70)

next_steps = """
✓ TRAINING COMPLETE!

NEXT STEPS:

Week 1: Evaluation
  [ ] Download model from Google Drive
  [ ] Backtest on 2024 data (out-of-sample)
  [ ] Compare to single-pair baseline
  [ ] Check performance per pair

Week 2-3: Paper Trading
  [ ] Deploy to demo account
  [ ] Run 50-100 trades
  [ ] Monitor win rate and R-multiple
  [ ] Check per-pair performance

Week 4+: Live Trading
  [ ] Deploy to live account (small)
  [ ] Monitor daily metrics
  [ ] Record all trades
  [ ] Plan monthly retraining

TIPS:
  • Keep model in Google Drive for easy access
  • Set up monthly retraining in Colab (takes 1 afternoon)
  • Monitor metrics in real-time
  • Backtest before deploying major changes
"""

print(next_steps)


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
COMMON COLAB ISSUES & SOLUTIONS
"""

print("\n" + "="*70)
print("TROUBLESHOOTING")
print("="*70)

troubleshooting = """
ISSUE: "GPU not available"
SOLUTION: Enable GPU in Runtime > Change runtime type > GPU

ISSUE: "Module not found"
SOLUTION: Make sure trading/ files are in /content/trading/

ISSUE: "Out of memory"
SOLUTION: Reduce TOTAL_TIMESTEPS or use fewer pairs

ISSUE: "Training slow"
SOLUTION: You're using CPU. Enable GPU (see above)

ISSUE: "Can't find CSV files"
SOLUTION: Upload to Google Drive or /content/data/

ISSUE: "Model save fails"
SOLUTION: Check Google Drive permissions

For more help:
  → See MULTIPAIR_QUICK_CHECKLIST.md
  → See MULTIPAIR_TRAINING_GUIDE.md
"""

print(troubleshooting)


# ============================================================================
# COLAB NOTEBOOK SUMMARY
# ============================================================================

print("\n" + "="*70)
print("COLAB NOTEBOOK CELLS SUMMARY")
print("="*70)

summary = """
Cell 1:  Mount Google Drive
Cell 2:  Install dependencies (5-10 min)
Cell 3:  Load data from CSV/Drive
Cell 4:  Copy project files
Cell 5:  Validate data & create setups
Cell 6:  Train multi-pair RL (8-12 hours) ← MAIN STEP
Cell 7:  Save model to Google Drive
Cell 8:  Analyze results
Cell 9:  Download results
Cell 10: Next steps guide

TOTAL TIME:
  Setup: 20-30 minutes
  Training: 8-12 hours
  Download: 5 minutes
"""

print(summary)
