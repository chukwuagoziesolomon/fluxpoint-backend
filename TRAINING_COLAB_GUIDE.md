# Training TCE Strategy on Google Colab

## Prerequisites

- ✅ Code pushed to GitHub
- ✅ Google account (for Colab & Drive)
- ✅ Historical market data (OHLCV candles)

## Quick Start Guide

### Step 1: Get Google Colab Pro

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with Google account
3. **Recommended**: Upgrade to Colab Pro ($10/month)
   - Go to "Upgrade" → "Subscribe to Colab Pro"
   - Benefits: GPU access, 24h runtime, faster CPUs

### Step 2: Prepare Training Data

**Option A: Upload to Google Drive**

```python
# Create folder structure in Google Drive
Google Drive/
└── fluxpoint/
    └── data/
        ├── EURUSD_H1.csv  # or .parquet
        ├── GBPUSD_H1.csv
        └── USDJPY_H1.csv
```

**Data Format (CSV):**
```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0900,1.0920,1.0895,1.0910,1000
2024-01-01 01:00:00,1.0910,1.0935,1.0905,1.0925,1200
...
```

**Option B: Download from MT5 in Colab**

We'll show you how to fetch data directly in the notebook.

### Step 3: Create Training Notebook

1. Open [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click "File" → "New notebook"
3. Copy the code from `colab/train_tce_complete.ipynb` (see below)
4. Save as "Train_TCE_FluxPoint.ipynb"

### Step 4: Run Training

1. Click "Runtime" → "Change runtime type"
2. Select "T4 GPU" or "V100 GPU" (if Pro)
3. Click "Runtime" → "Run all"
4. Wait for training to complete (~10-30 minutes)

### Step 5: Download Trained Models

Models are automatically saved to:
```
Google Drive/fluxpoint/models/ml/tce_ml_model.pt
Google Drive/fluxpoint/models/rl/tce_execution_ppo.zip
```

## Detailed Walkthrough

### 1. Setup Authentication

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# You'll see a prompt - click the link, authorize, copy code
```

### 2. Clone Your Repository

```python
# Clone from GitHub (replace with your repo)
!git clone https://github.com/YOUR_USERNAME/fluxpointai-backend.git
%cd fluxpointai-backend/fluxpoint
```

### 3. Install Dependencies

```python
# Install required packages
!pip install -q torch pandas numpy scikit-learn django stable-baselines3 gymnasium
```

### 4. Prepare Data

**If data in Drive:**
```python
import pandas as pd

# Load from Drive
candles = pd.read_csv('/content/drive/MyDrive/fluxpoint/data/EURUSD_H1.csv')
candles['timestamp'] = pd.to_datetime(candles['timestamp'])
candles.set_index('timestamp', inplace=True)
```

**If downloading sample data:**
```python
# Download sample data (for testing)
!wget https://github.com/FluxPointAI/sample-data/raw/main/EURUSD_H1_2020_2024.csv -O /tmp/EURUSD_H1.csv
candles = pd.read_csv('/tmp/EURUSD_H1.csv')
```

### 5. Train ML Model

```python
# Import training function
import sys
sys.path.append('/content/fluxpointai-backend/fluxpoint')

from trading.tce.training import train_tce_model

# Train
metrics = train_tce_model(
    symbol='EURUSD',
    timeframe='H1',
    candles=candles,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)

# Print results
print("\n" + "="*60)
print("ML TRAINING COMPLETE")
print("="*60)
print(f"Final Loss: {metrics['final_loss']:.4f}")
print(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
print(f"Training Time: {metrics['training_time']:.1f}s")
print("="*60)
```

### 6. Train RL Model

```python
from trading.rl.training import train_rl_execution

# First, get valid setups from ML model
# (This requires running TCE validation on historical data)

# Train RL
rl_metrics = train_rl_execution(
    candles=candles,
    valid_setups=valid_setups,  # From TCE validation
    total_timesteps=100000
)

print("\n" + "="*60)
print("RL TRAINING COMPLETE")
print("="*60)
print(f"Mean Reward: {rl_metrics['eval']['mean_reward']:.2f}")
print(f"Mean R-Multiple: {rl_metrics['eval']['mean_r_multiple']:.2f}R")
print(f"Win Rate: {rl_metrics['eval']['mean_win_rate']:.2%}")
print("="*60)
```

### 7. Save to Drive

```python
# Create directories
!mkdir -p /content/drive/MyDrive/fluxpoint/models/ml/
!mkdir -p /content/drive/MyDrive/fluxpoint/models/rl/

# Copy ML model
!cp models/ml/tce_ml_model.pt /content/drive/MyDrive/fluxpoint/models/ml/

# Copy RL model
!cp -r models/rl/tce_execution_ppo* /content/drive/MyDrive/fluxpoint/models/rl/

print("✅ Models saved to Google Drive")
```

### 8. Save Training Logs

```python
import json
from datetime import datetime

# Save metrics
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_data = {
    'timestamp': timestamp,
    'symbol': 'EURUSD',
    'timeframe': 'H1',
    'ml_metrics': metrics,
    'rl_metrics': rl_metrics
}

log_path = f'/content/drive/MyDrive/fluxpoint/logs/training_{timestamp}.json'
with open(log_path, 'w') as f:
    json.dump(log_data, f, indent=2)

print(f"✅ Training log saved to {log_path}")
```

## Troubleshooting

### Issue: "No module named 'trading'"

**Solution:**
```python
import sys
sys.path.insert(0, '/content/fluxpointai-backend/fluxpoint')
```

### Issue: "CUDA out of memory"

**Solution:**
```python
# Reduce batch size
batch_size=16  # instead of 32

# Or use CPU
import torch
device = 'cpu'
```

### Issue: "Runtime disconnected"

**Solution:**
- Upgrade to Colab Pro for 24h runtime
- Or save checkpoints every hour:
```python
# In training loop
if epoch % 10 == 0:
    torch.save(model.state_dict(), f'/content/drive/MyDrive/checkpoint_epoch_{epoch}.pt')
```

### Issue: "Not enough training data"

**Solution:**
- Need at least 1000 historical setups
- Use 2+ years of H1 data
- Or train on multiple pairs

## Monitoring Training

### TensorBoard (for RL)

```python
# Load TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/tensorboard/

# Training will log metrics automatically
```

### Manual Logging

```python
# Check progress
print(f"Epoch {epoch}/{total_epochs}")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.2%}")
```

## Scheduling Automated Training

### Option 1: Manual Re-run

- Re-run notebook weekly
- Check for updated data
- Retrain models

### Option 2: GitHub Actions + Colab API

```yaml
# .github/workflows/train_weekly.yml
name: Weekly Training

on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight

jobs:
  trigger_colab:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Colab Training
        run: |
          # Use Colab API or webhook
          echo "Training triggered"
```

### Option 3: Google Cloud Functions

```python
# Cloud Function to trigger Colab
def trigger_training(request):
    # Use Colab API
    # Run training notebook
    return 'Training started'
```

## After Training: Deploy to VPS

### 1. Sync Models from Drive to VPS

```bash
# On VPS
rclone sync gdrive:fluxpoint/models/ ~/fluxpointai-backend/fluxpoint/models/
```

### 2. Restart Django

```bash
sudo supervisorctl restart fluxpoint_django
```

### 3. Verify Models Loaded

```python
# Test API endpoint
curl http://localhost:8000/api/tce/validate/
```

## Expected Training Time

| Task | Time | GPU |
|------|------|-----|
| ML Training (TCE) | 10-20 min | T4 |
| RL Training (Execution) | 20-40 min | T4 |
| Data Preparation | 5 min | CPU |
| **Total** | **35-65 min** | T4 |

## Cost Breakdown

| Service | Cost |
|---------|------|
| Colab (Free) | $0 - but limited runtime |
| Colab Pro | $10/month - 24h runtime, faster GPUs |
| **Recommended** | **Colab Pro ($10/month)** |

## Next Steps

1. ✅ Train TCE ML model on Colab
2. ✅ Train TCE RL model on Colab
3. ✅ Save models to Drive
4. ⏳ Sync models to VPS
5. ⏳ Test inference on VPS
6. ⏳ Deploy to production

## Tips for Success

1. **Start Small**: Train on 1 pair first (EURUSD)
2. **Use Checkpoints**: Save every 10 epochs
3. **Monitor Progress**: Use TensorBoard
4. **Validate Results**: Check metrics before deploying
5. **Keep Logs**: Save all training logs to Drive

## Common Questions

**Q: How often should I retrain?**  
A: Weekly or monthly, depending on market conditions.

**Q: Can I train multiple strategies at once?**  
A: Yes, but train sequentially to avoid memory issues.

**Q: What if training fails?**  
A: Check logs, reduce batch size, or use smaller model.

**Q: How do I know if model is good?**  
A: Check validation accuracy (>60% is good), win rate, R-multiple.

**Q: Can I use free Colab?**  
A: Yes, but limited to 12h runtime. Pro recommended for serious use.
