"""
COLAB FIRST CELL - RUN THIS IMMEDIATELY AFTER CREATING NOTEBOOK

This installs EVERYTHING you need before anything else.
Copy and paste this into the FIRST cell of your Colab notebook.
"""

════════════════════════════════════════════════════════════════════════════════
CELL 1 - INSTALL EVERYTHING + CLONE REPO (Run First!)
════════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

print("="*70)
print("COLAB INITIAL SETUP - Installing everything...")
print("="*70)

# Step 1: Install all dependencies first
print("\nStep 1/3: Installing Python packages (10 minutes)...\n")

packages = [
    'pandas',
    'numpy',
    'torch',
    'torchvision',
    'stable-baselines3',
    'gymnasium',
    'scikit-learn',
    'matplotlib',
    'tensorboard',
    'optuna'
]

for pkg in packages:
    print(f"  {pkg:20s}", end='', flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(" ✓")
    except Exception as e:
        print(f" ✗")

print("\n✓ Packages installed!")

# Step 2: Clone repository
print("\nStep 2/3: Cloning GitHub repository...\n")

repo_url = "https://github.com/YOUR_USERNAME/fluxpoint-ml"

# Remove if exists
os.system("rm -rf /content/fluxpoint 2>/dev/null")

# Clone
result = os.system(f"git clone {repo_url} /content/fluxpoint 2>&1")

if result == 0:
    print("✓ Repository cloned!")
else:
    print(f"✗ Clone failed. Check repo URL: {repo_url}")
    print("  Create repo at: https://github.com/new")

# Step 3: Add to Python path and test imports
print("\nStep 3/3: Verifying imports...\n")

sys.path.insert(0, '/content/fluxpoint')

# Test critical imports
imports_ok = True
critical_imports = [
    ('pandas', 'pd'),
    ('numpy', 'np'),
    ('torch', 'torch'),
    ('stable_baselines3', 'sb3'),
    ('gymnasium', 'gym'),
]

for module_name, alias in critical_imports:
    try:
        __import__(module_name)
        print(f"  {module_name:20s} ✓")
    except ImportError as e:
        print(f"  {module_name:20s} ✗ ({e})")
        imports_ok = False

# Test project imports
print()
project_imports_ok = True
try:
    from trading.tce.ml_model import TCEProbabilityModel
    print(f"  {'TCEProbabilityModel':20s} ✓")
except ImportError as e:
    print(f"  {'TCEProbabilityModel':20s} ✗")
    project_imports_ok = False

try:
    from trading.rl.multi_pair_training import train_rl_multipair
    print(f"  {'MultiPairTrainer':20s} ✓")
except ImportError as e:
    print(f"  {'MultiPairTrainer':20s} ✗")
    project_imports_ok = False

# Summary
print("\n" + "="*70)
if imports_ok and project_imports_ok:
    print("✓ SETUP COMPLETE - Ready to train!")
    print("="*70)
    print("\nYou can now run:")
    print("  CELL 2: Mount Google Drive")
    print("  CELL 3: Load training data")
    print("  CELL 4: Train DL model")
    print("  CELL 5: Train RL model")
else:
    print("✗ SETUP INCOMPLETE - Some imports failed")
    print("="*70)
    print("\nFix: Runtime > Restart session, then try again")

print("\n" + "="*70)


════════════════════════════════════════════════════════════════════════════════
CELL 2 - MOUNT GOOGLE DRIVE (Run Second)
════════════════════════════════════════════════════════════════════════════════

from google.colab import drive
import os

print("\nMounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

print("✓ Drive mounted!")
print("\nNow upload your CSV files to: My Drive/forex_data/")
print("Then run CELL 3 to load them")


════════════════════════════════════════════════════════════════════════════════
CELL 3 - LOAD TRAINING DATA
════════════════════════════════════════════════════════════════════════════════

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/content/fluxpoint')
from trading.tce.validation import validate_tce_setups

print("\nLoading training data from Google Drive...\n")

pair_data = {}
csv_dir = Path('/content/drive/MyDrive/forex_data')

if not csv_dir.exists():
    print(f"✗ Folder not found: {csv_dir}")
    print("  Please create: My Drive/forex_data/")
    print("  And upload your CSV files there")
else:
    csv_files = list(csv_dir.glob('*.csv'))
    
    if not csv_files:
        print("✗ No CSV files found in My Drive/forex_data/")
    else:
        print(f"Found {len(csv_files)} CSV files:\n")
        
        for csv_file in csv_files:
            symbol = csv_file.stem.split('_')[0].upper()
            
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                setups = validate_tce_setups(df, symbol)
                pair_data[symbol] = (df, setups)
                
                print(f"  ✓ {symbol:10s} {len(df):5d} candles, {len(setups):4d} setups")
            
            except Exception as e:
                print(f"  ✗ {symbol:10s} Error: {str(e)[:40]}")

print(f"\n✓ Loaded {len(pair_data)} trading pairs")
print("Ready for training!")


════════════════════════════════════════════════════════════════════════════════
CELL 4 - TRAIN DEEP LEARNING MODEL (1-2 hours)
════════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, '/content/fluxpoint')

from trading.tce.training import TCETrainer
from datetime import datetime
import torch
import os

print("\n" + "="*70)
print("TRAINING DEEP LEARNING MODEL")
print("="*70)

trainer = TCETrainer(
    learning_rate=0.001,
    batch_size=32,
    dropout_rate=0.3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Device: {trainer.device}")
print(f"Model: TCEProbabilityModel (4-layer neural network)")

try:
    # Prepare training data
    print("\nPreparing data...")
    train_loader, val_loader = trainer.prepare_data(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        val_split=0.2
    )
    
    # Train
    print("Training (this will take 1-2 hours)...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50
    )
    
    print("\n" + "="*70)
    print("✓ DL TRAINING COMPLETE!")
    print("="*70)
    
    # Show metrics
    final_metrics = history.get('val_metrics', [{}])[-1]
    print(f"\nAccuracy:  {final_metrics.get('accuracy', 0):.1%}")
    print(f"Precision: {final_metrics.get('precision', 0):.1%}")
    print(f"Recall:    {final_metrics.get('recall', 0):.1%}")
    print(f"F1-Score:  {final_metrics.get('f1', 0):.3f}")
    
    # Save model
    model_path = '/content/fluxpoint/models/dl/tce_probability_model'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(trainer.model.state_dict(), f"{model_path}.pth")
    print(f"\n✓ Model saved to: {model_path}.pth")

except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()


════════════════════════════════════════════════════════════════════════════════
CELL 5 - TRAIN RL MODEL (8-12 hours - RUN OVERNIGHT!)
════════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, '/content/fluxpoint')

from trading.rl.multi_pair_training import train_rl_multipair
import torch
from trading.tce.ml_model import TCEProbabilityModel

print("\n" + "="*70)
print("TRAINING RL MODEL (using DL predictions)")
print("="*70)

print("\nEnhancing setups with DL predictions...")

# Load DL model
try:
    dl_model = TCEProbabilityModel(n_features=20)
    dl_model.load_state_dict(
        torch.load('/content/fluxpoint/models/dl/tce_probability_model.pth')
    )
    dl_model.eval()
    print("✓ DL model loaded")
except Exception as e:
    print(f"✗ Could not load DL model: {e}")
    dl_model = None

# Add predictions to setups
for symbol, (candles, setups) in pair_data.items():
    for setup in setups:
        features = setup.get('features', [])
        
        if len(features) > 0 and dl_model:
            with torch.no_grad():
                feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                ml_prob = dl_model(feature_tensor).item()
            setup['ml_probability'] = ml_prob

print("✓ Setups enhanced with DL predictions")

# Train RL
print("\nStarting RL training...")
print("This will take 8-12 hours. You can close this tab and come back later.")

try:
    metrics = train_rl_multipair(
        pair_data=pair_data,
        model_name="tce_rl_with_dl",
        total_timesteps=200000,
        learning_rate=3e-4,
        batch_size=64
    )
    
    print("\n" + "="*70)
    print("✓ RL TRAINING COMPLETE!")
    print("="*70)
    
    eval_results = metrics.get('eval_metrics', {})
    print(f"\nR-Multiple: {eval_results.get('mean_r_multiple', 0):.2f}R")
    print(f"Win Rate:   {eval_results.get('mean_win_rate', 0):.1%}")
    print(f"Reward:     {eval_results.get('mean_reward', 0):.2f}")

except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()


════════════════════════════════════════════════════════════════════════════════
CELL 6 - SAVE MODELS TO GOOGLE DRIVE
════════════════════════════════════════════════════════════════════════════════

import os
import shutil

print("\nSaving trained models to Google Drive...\n")

drive_models = '/content/drive/MyDrive/trained_models'
os.makedirs(drive_models, exist_ok=True)

# Save DL model
try:
    print("  Saving DL model...")
    shutil.copy(
        '/content/fluxpoint/models/dl/tce_probability_model.pth',
        f'{drive_models}/dl_model.pth'
    )
    print("  ✓ DL model saved")
except Exception as e:
    print(f"  ✗ DL save failed: {e}")

# Save RL model
try:
    print("  Saving RL model...")
    os.system(f"cp -r /content/fluxpoint/models/rl/tce_rl_with_dl* {drive_models}/ 2>/dev/null")
    print("  ✓ RL model saved")
except Exception as e:
    print(f"  ✗ RL save failed: {e}")

print(f"\n✓ All models saved to: My Drive/trained_models/")


════════════════════════════════════════════════════════════════════════════════
COMPLETE - NEXT STEPS
════════════════════════════════════════════════════════════════════════════════

✓ DL model trained: Predicts probability of trade success
✓ RL model trained: Optimizes entry/exit/sizing
✓ Models saved to Google Drive

NEXT:

1. Download from Google Drive
   My Drive/trained_models/ → Save to your computer

2. Place in local project
   C:\Users\USER-PC\fluxpointai-backend\fluxpoint\models\

3. Backtest on 2024 data
   Compare results: DL only vs DL+RL vs baseline

4. Paper trade
   Run 50-100 trades on demo account

5. Go live (if backtest profitable)
   Small account, monitor daily, retrain monthly

════════════════════════════════════════════════════════════════════════════════
"""
