# Deployment Flow - VS Code → Colab → VPS

## Overview

This document describes the complete deployment workflow for FluxPoint AI, from local development to production deployment.

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  VS Code      │────▶│   GitHub      │────▶│  Colab Pro    │────▶│  Google Drive │
│  (Local Dev)  │     │  (Code Sync)  │     │  (Training)   │     │  (Models)     │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                                                            │
                                                                            ▼
                                                                    ┌───────────────┐
                                                                    │  VPS (Django) │
                                                                    │  (Production) │
                                                                    └───────────────┘
```

## Phase 1: Local Development (VS Code)

### 1.1 Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fluxpointai-backend.git
cd fluxpointai-backend

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run migrations
cd fluxpoint
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Start development server
python manage.py runserver
```

### 1.2 Development Workflow

```python
# 1. Write code in VS Code
# Example: Create new trading strategy

# fluxpoint/trading/strategies/my_strategy.py
def validate_my_strategy(symbol, timeframe):
    # Strategy logic here
    pass

# 2. Test locally
python manage.py test trading.tests

# 3. Check for errors
python manage.py check

# 4. Format code
black fluxpoint/
flake8 fluxpoint/

# 5. Git workflow
git add .
git commit -m "Add my_strategy"
git push origin feature/my-strategy
```

### 1.3 Local Testing

```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test trading
python manage.py test strategy_builder

# Run with coverage
pip install coverage
coverage run manage.py test
coverage report

# Test ML components
python -m pytest trading/tce/tests/

# Test RL components
python -m pytest trading/rl/tests/
```

## Phase 2: GitHub (Code Sync)

### 2.1 Repository Structure

```
fluxpointai-backend/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration
│       ├── deploy.yml          # Auto-deployment
│       └── train_models.yml    # Trigger Colab training
├── fluxpoint/
│   ├── trading/                # TCE strategy
│   ├── strategy_builder/       # No-code builder
│   ├── api/                    # REST API
│   └── manage.py
├── colab/
│   ├── train_tce_ml.ipynb      # TCE ML training
│   ├── train_tce_rl.ipynb      # TCE RL training
│   ├── train_user_strategy.ipynb  # User strategy training
│   └── backtest.ipynb          # Backtesting
├── scripts/
│   ├── deploy_vps.sh           # VPS deployment script
│   ├── sync_models.sh          # Sync models from Drive
│   └── backup_db.sh            # Database backup
├── requirements.txt            # Production dependencies
├── requirements_training.txt   # Training dependencies (PyTorch, etc.)
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── INFRASTRUCTURE.md           # Infrastructure guide
```

### 2.2 Branching Strategy

```
main                 # Production code
├── develop          # Development branch
│   ├── feature/tce-improvements
│   ├── feature/user-strategy-builder
│   └── fix/rl-training-bug
```

### 2.3 CI/CD Pipeline

**`.github/workflows/ci.yml`:**

```yaml
name: CI Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest coverage black flake8
    
    - name: Run linting
      run: |
        black --check fluxpoint/
        flake8 fluxpoint/
    
    - name: Run tests
      run: |
        cd fluxpoint
        coverage run manage.py test
        coverage report
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

**`.github/workflows/deploy.yml`:**

```yaml
name: Deploy to VPS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Deploy to Production VPS
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.VPS_HOST }}
        username: ${{ secrets.VPS_USER }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          cd ~/fluxpointai-backend
          git pull origin main
          source venv/bin/activate
          pip install -r requirements.txt
          cd fluxpoint
          python manage.py migrate
          python manage.py collectstatic --no-input
          sudo supervisorctl restart fluxpoint_django
          echo "✅ Deployment complete"
```

## Phase 3: Colab Pro (Training)

### 3.1 Setup Colab for Training

**Create `colab/setup_colab.ipynb`:**

```python
# Install dependencies
!pip install -q stable-baselines3 gymnasium torch pandas numpy

# Authenticate with Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository (or pull latest)
!git clone https://github.com/yourusername/fluxpointai-backend.git
# OR
%cd fluxpointai-backend
!git pull origin main

# Install project
%cd fluxpoint
!pip install -e .

# Verify setup
print("✅ Colab environment ready")
```

### 3.2 TCE ML Training

**`colab/train_tce_ml.ipynb`:**

```python
# Setup
!pip install -q torch pandas numpy django
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/fluxpointai-backend/fluxpoint')

# Load data from Drive
import pandas as pd
candles = pd.read_parquet('/content/drive/MyDrive/fluxpoint/data/EURUSD_H1.parquet')
print(f"Loaded {len(candles)} candles")

# Train TCE ML model
from trading.tce.training import train_tce_model

metrics = train_tce_model(
    symbol='EURUSD',
    timeframe='H1',
    candles=candles,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)

print("\n" + "="*60)
print("Training Results:")
print(f"Final Loss: {metrics['final_loss']:.4f}")
print(f"Val Accuracy: {metrics['val_accuracy']:.2%}")
print(f"Training Time: {metrics['training_time']:.1f}s")
print("="*60)

# Save model to Drive
!mkdir -p /content/drive/MyDrive/fluxpoint/models/ml/
!cp models/ml/tce_ml_model.pt /content/drive/MyDrive/fluxpoint/models/ml/

# Save metrics
import json
with open('/content/drive/MyDrive/fluxpoint/logs/tce_ml_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Model saved to Google Drive")
```

### 3.3 TCE RL Training

**`colab/train_tce_rl.ipynb`:**

```python
# Setup
!pip install -q stable-baselines3 gymnasium
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/fluxpointai-backend/fluxpoint')

# Load data
import pandas as pd
candles = pd.read_parquet('/content/drive/MyDrive/fluxpoint/data/EURUSD_H1.parquet')

# Load valid setups (from ML model)
import torch
from trading.tce.ml_integration import TCEMLPredictor

predictor = TCEMLPredictor()
predictor.load_model('/content/drive/MyDrive/fluxpoint/models/ml/tce_ml_model.pt')

# Get valid setups
valid_setups = []
# ... extract valid setups from data ...

# Train RL
from trading.rl.training import train_rl_execution

metrics = train_rl_execution(
    candles=candles,
    valid_setups=valid_setups,
    model_name='tce_execution_ppo',
    total_timesteps=100000
)

print("\n" + "="*60)
print("RL Training Results:")
print(f"Mean Reward: {metrics['eval']['mean_reward']:.2f}")
print(f"Mean R-Multiple: {metrics['eval']['mean_r_multiple']:.2f}R")
print(f"Win Rate: {metrics['eval']['mean_win_rate']:.2%}")
print("="*60)

# Save to Drive
!mkdir -p /content/drive/MyDrive/fluxpoint/models/rl/
!cp -r models/rl/tce_execution_ppo* /content/drive/MyDrive/fluxpoint/models/rl/

print("✅ RL model saved to Google Drive")
```

### 3.4 User Strategy Training

**`colab/train_user_strategy.ipynb`:**

```python
# Setup
!pip install -q stable-baselines3 torch
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/fluxpointai-backend/fluxpoint')

# Load user strategy
strategy_id = 123  # From database
user_id = 456

from strategy_builder.workflow import NoCodeStrategyBuilder
builder = NoCodeStrategyBuilder(user_id=user_id)

# Get strategy configuration
# In production, this would fetch from database
strategy_config = {...}

# Train ML model
from strategy_builder.ml.training import train_user_strategy_ml

ml_metrics = train_user_strategy_ml(
    strategy_config=strategy_config,
    data_path='/content/drive/MyDrive/fluxpoint/data/'
)

# Train RL model
from strategy_builder.rl.training import train_user_strategy_rl

rl_metrics = train_user_strategy_rl(
    strategy_config=strategy_config,
    ml_model_path=f'/content/drive/MyDrive/fluxpoint/models/ml/user_{user_id}_strategy_{strategy_id}.pt'
)

# Save to user-specific folder
!mkdir -p /content/drive/MyDrive/fluxpoint/models/users/{user_id}/
!cp models/users/{user_id}/* /content/drive/MyDrive/fluxpoint/models/users/{user_id}/

print(f"✅ User {user_id} Strategy {strategy_id} trained and saved")
```

### 3.5 Automated Training Schedule

```python
# Run weekly on Colab
# Schedule with GitHub Actions + Colab API

# .github/workflows/train_models.yml
name: Train Models Weekly

on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight
  workflow_dispatch:  # Manual trigger

jobs:
  trigger_colab_training:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Colab Notebook
        run: |
          # Use Colab API to run training notebooks
          # Or use Google Cloud Functions to trigger
          echo "Training triggered"
```

## Phase 4: Google Drive (Model Storage)

### 4.1 Drive Structure

```
Google Drive/fluxpoint/
├── models/
│   ├── ml/
│   │   ├── tce_ml_model.pt                    # TCE ML model
│   │   ├── tce_ml_model_backup_2026_01_06.pt  # Backup
│   │   └── users/
│   │       ├── 1/                             # User 1 models
│   │       │   ├── strategy_1.pt
│   │       │   └── strategy_2.pt
│   │       └── 2/                             # User 2 models
│   └── rl/
│       ├── tce_execution_ppo.zip              # TCE RL model
│       └── users/
│           ├── 1/
│           │   ├── strategy_1_rl.zip
│           │   └── strategy_2_rl.zip
│           └── 2/
├── data/
│   ├── EURUSD_H1.parquet
│   ├── GBPUSD_H1.parquet
│   ├── USDJPY_H1.parquet
│   └── ...
├── logs/
│   ├── training_2026_01_06.log
│   ├── backtest_results_2026_01_06.json
│   └── ...
└── backups/
    ├── db_backup_2026_01_06.sql
    └── ...
```

### 4.2 Model Versioning

```python
# Save with timestamp
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'/content/drive/MyDrive/fluxpoint/models/ml/tce_ml_model_{timestamp}.pt'

torch.save(model.state_dict(), model_path)

# Keep latest as main
!cp {model_path} /content/drive/MyDrive/fluxpoint/models/ml/tce_ml_model.pt

print(f"✅ Model saved: {model_path}")
```

## Phase 5: VPS (Production Deployment)

### 5.1 Sync Models from Drive to VPS

**`scripts/sync_models.sh`:**

```bash
#!/bin/bash

# Sync models from Google Drive to VPS
# Uses rclone configured with service account

echo "Syncing models from Google Drive..."

# Sync ML models
rclone sync gdrive:fluxpoint/models/ml/ ~/fluxpointai-backend/fluxpoint/models/ml/ --progress

# Sync RL models
rclone sync gdrive:fluxpoint/models/rl/ ~/fluxpointai-backend/fluxpoint/models/rl/ --progress

# Sync user models (only active users)
rclone sync gdrive:fluxpoint/models/users/ ~/fluxpointai-backend/fluxpoint/models/users/ --progress

echo "✅ Model sync complete"

# Restart Django to reload models
sudo supervisorctl restart fluxpoint_django

echo "✅ Django restarted"
```

### 5.2 Automated Deployment

**`scripts/deploy_vps.sh`:**

```bash
#!/bin/bash

# Complete deployment script

set -e  # Exit on error

echo "Starting deployment..."

# 1. Pull latest code
cd ~/fluxpointai-backend
git pull origin main

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt --quiet

# 4. Run migrations
cd fluxpoint
python manage.py migrate --no-input

# 5. Collect static files
python manage.py collectstatic --no-input

# 6. Sync models from Drive
cd ..
bash scripts/sync_models.sh

# 7. Run tests (optional)
# python manage.py test --parallel

# 8. Restart services
sudo supervisorctl restart fluxpoint_django
sudo systemctl reload nginx

# 9. Health check
sleep 5
curl -f http://localhost:8000/api/health/ || exit 1

echo "✅ Deployment complete and healthy"
```

### 5.3 Production Settings

**`fluxpoint/settings_production.py`:**

```python
from .settings import *

DEBUG = False

ALLOWED_HOSTS = ['api.fluxpointai.com', 'www.fluxpointai.com']

# Database (PostgreSQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'fluxpoint'),
        'USER': os.getenv('DB_USER', 'fluxpoint'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# Security
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000

# Logging
LOGGING = {
    'version': 1,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/var/log/fluxpoint/django.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
        },
    },
}
```

### 5.4 Cron Jobs

```bash
# Add to crontab: crontab -e

# Sync models from Drive every hour
0 * * * * /home/fluxpoint/fluxpointai-backend/scripts/sync_models.sh >> /var/log/fluxpoint/sync.log 2>&1

# Backup database daily at 2 AM
0 2 * * * /home/fluxpoint/fluxpointai-backend/scripts/backup_db.sh >> /var/log/fluxpoint/backup.log 2>&1

# Health check every 5 minutes
*/5 * * * * curl -f http://localhost:8000/api/health/ || echo "Health check failed" | mail -s "FluxPoint Health Alert" admin@fluxpointai.com
```

## Complete Deployment Workflow

### 1. Developer pushes code

```bash
# Local (VS Code)
git add .
git commit -m "Add new feature"
git push origin main
```

### 2. GitHub CI runs tests

```
GitHub Actions:
- Run linting (black, flake8)
- Run unit tests
- Check coverage
- If tests pass → proceed
- If tests fail → notify developer
```

### 3. GitHub triggers deployment

```
GitHub Actions:
- SSH into VPS
- Pull latest code
- Install dependencies
- Run migrations
- Restart Django
- Health check
```

### 4. Training on Colab (weekly)

```
Sunday Midnight:
- GitHub Action triggers Colab notebook
- Colab pulls latest code from GitHub
- Loads data from Google Drive
- Trains ML models (TCE, user strategies)
- Trains RL models (execution optimization)
- Saves models to Google Drive
```

### 5. VPS syncs models (hourly)

```
Every Hour:
- rclone syncs models from Drive to VPS
- Django reloads models
- New models available for inference
```

## Rollback Procedure

### If deployment fails:

```bash
# 1. SSH into VPS
ssh fluxpoint@your-vps-ip

# 2. Rollback code
cd ~/fluxpointai-backend
git log --oneline -10  # Find last good commit
git reset --hard COMMIT_HASH
git push origin main --force

# 3. Rollback database (if needed)
# Restore from backup
pg_restore -d fluxpoint /backups/db_backup_latest.sql

# 4. Rollback models (if needed)
rclone copy gdrive:fluxpoint/models/backups/BACKUP_DATE/ models/

# 5. Restart services
sudo supervisorctl restart fluxpoint_django

# 6. Verify
curl http://localhost:8000/api/health/
```

## Monitoring & Alerts

### 1. Uptime Monitoring

- UptimeRobot: Ping `/api/health/` every 5 minutes
- Alert via email/SMS if down

### 2. Error Tracking

- Sentry: Captures Django errors
- Email notifications for critical errors

### 3. Performance Monitoring

- Netdata: Server metrics (CPU, RAM, disk)
- Custom metrics: API latency, model inference time

### 4. Trading Alerts

- Email on trade execution
- SMS on critical errors
- Telegram bot for real-time updates

## Summary

This deployment flow ensures:
✅ **Automated**: Push code → Auto-deploy  
✅ **Tested**: CI runs tests before deployment  
✅ **Scalable**: Easy to add more VPS instances  
✅ **Reliable**: Rollback procedures in place  
✅ **Cost-Effective**: Uses Colab Pro for training ($10/month)  
✅ **Maintainable**: Clear structure, good documentation

Total monthly cost: **$45-80/month** for complete infrastructure!
