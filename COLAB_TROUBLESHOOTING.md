COLAB TROUBLESHOOTING GUIDE
═══════════════════════════════════════════════════════════════════════════════

Problem: ImportError: No module named 'stable_baselines3'
─────────────────────────────────────────────────────────────────────────────

SOLUTION:

You skipped CELL 2 (Install Dependencies). Run it NOW:

CELL 2: Install all packages (Copy and paste this)
────────────────────────────────────────────────────────────────────────────────

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

print("Installing dependencies (10 minutes)...")

for pkg in packages:
    print(f"  Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✓ All dependencies installed!")


IMPORTANT: RUN CELLS IN ORDER

  CELL 1: Clone from GitHub
  CELL 2: Install dependencies ← RUN THIS NOW
  CELL 3: Load data
  CELL 4: Train DL
  CELL 5: Train RL
  CELL 6: Save models
  CELL 7: Evaluate


═══════════════════════════════════════════════════════════════════════════════
OTHER COMMON ERRORS
═══════════════════════════════════════════════════════════════════════════════

ERROR: "No module named 'trading'"
─────────────────────────────────
Make sure CELL 1 ran successfully and you see:
  ✓ Repository cloned!

If not:
  1. Check repo URL is correct
  2. Internet connection is working
  3. Run CELL 1 again


ERROR: "ModuleNotFoundError: No module named 'django'"
──────────────────────────────────────────────────────
This is OK! Django isn't needed for Colab training.
Remove this line if you see it:
  from trading.models import TCESetup  # Only works with Django

Use CSV files instead (CELL 3)


ERROR: "CUDA out of memory"
───────────────────────────
Solutions:
  1. Reduce batch size: batch_size=16 instead of 32
  2. Reduce timesteps: 100000 instead of 200000
  3. Train fewer pairs: 2-3 instead of 5-10
  4. Restart runtime: Runtime > Restart session


ERROR: "Port 6006 already in use" (TensorBoard)
────────────────────────────────────────────────
Just ignore it. TensorBoard will still work.
Or kill the process:

  import subprocess
  subprocess.run(["fuser", "-k", "6006/tcp"], capture_output=True)


═══════════════════════════════════════════════════════════════════════════════
QUICK FIX: Run This Cell First (If you have errors)
═══════════════════════════════════════════════════════════════════════════════

CELL 0: Initialize Colab + Install Everything
────────────────────────────────────────────────────────────────────────────────

# Fix sys path
import sys
sys.path.insert(0, '/content')

# Install everything at once
import subprocess

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

print("Installing all dependencies (15 minutes)...\n")

for pkg in packages:
    print(f"Installing {pkg}...", end='', flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(" ✓")
    except Exception as e:
        print(f" ✗ ({e})")

print("\n✓ All dependencies ready!")

# Test imports
try:
    import pandas as pd
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    import gymnasium as gym
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")


═══════════════════════════════════════════════════════════════════════════════
STEP-BY-STEP RECOVERY
═══════════════════════════════════════════════════════════════════════════════

If you're getting multiple errors:

1. RESTART RUNTIME
   Runtime > Restart session
   (Clears all cache, starts fresh)

2. RUN CELL 0 ABOVE
   (Installs everything cleanly)

3. RUN CELL 1
   (Clone from GitHub)

4. SKIP to CELL 3
   (Data loading)

5. CELL 4, 5, 6, 7...
   (Continue pipeline)


═══════════════════════════════════════════════════════════════════════════════
VERIFY EACH STEP
═══════════════════════════════════════════════════════════════════════════════

After each cell, you should see:

CELL 1 Complete:
  ✓ Repository cloned!
  ✓ All modules imported successfully!

CELL 2 Complete (or CELL 0):
  ✓ All dependencies installed!

CELL 3 Complete:
  ✓ Loaded X pairs
  (e.g., "✓ Loaded 5 pairs")

CELL 4 Complete:
  ✓ DL TRAINING COMPLETE!
  Accuracy: 68%
  ...

CELL 5 Complete:
  ✓ RL TRAINING COMPLETE!
  R-Multiple: 1.5R
  Win Rate: 56%
  ...

If you see ✓ for each step, you're on track!


═══════════════════════════════════════════════════════════════════════════════
STILL STUCK?
═══════════════════════════════════════════════════════════════════════════════

Try this nuclear option:

1. Click: Runtime > Factory reset runtime
2. Click: Yes
3. Wait 30 seconds
4. Run the CELL 0 above first
5. Then CELL 1 (clone)
6. Then rest of pipeline


═══════════════════════════════════════════════════════════════════════════════
EXPECTED TIMELINE
═══════════════════════════════════════════════════════════════════════════════

Step          Time        Status
─────────────────────────────────────
CELL 0        15 min      Installing deps
CELL 1        2 min       Cloning repo
CELL 2        5 min       Loading data
CELL 3        2 hours     Training DL (can go overnight)
CELL 4        12 hours    Training RL (sleep!)
CELL 5        5 min       Saving models
CELL 6        2 min       Evaluating
─────────────────────────────────────
TOTAL:        ~15 hours   (mostly automated)

✓ This is NORMAL. Go to bed, wake up to trained models!


═══════════════════════════════════════════════════════════════════════════════
