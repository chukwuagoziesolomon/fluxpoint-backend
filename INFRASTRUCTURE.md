# Infrastructure Setup - Nigeria-Realistic

## Overview

FluxPoint AI uses a cost-effective, scalable infrastructure suitable for Nigeria:
- **Training**: Google Colab Pro ($10/month)
- **Production**: VPS for Django inference (no GPU needed)
- **MT5**: Separate Windows VPS for MT5 bridge
- **Code Sync**: GitHub for version control

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT                              │
│  VS Code (Local) → GitHub → Colab Pro (Training)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Google Drive (Models)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     PRODUCTION                               │
│  VPS (Django) ← Models ← Google Drive                       │
│  VPS (Django) ← Data ← MT5 VPS (Windows)                    │
└─────────────────────────────────────────────────────────────┘
```

## Cost Breakdown (Monthly)

| Service | Purpose | Cost | Notes |
|---------|---------|------|-------|
| Google Colab Pro | ML/RL Training | $10/month | GPU access, 24h runtime |
| VPS (8-16GB RAM) | Django + PostgreSQL | $20-40/month | Linode, DigitalOcean, Vultr |
| MT5 VPS (Windows) | MT5 Bridge | $15-30/month | Contabo, Kamatera |
| GitHub | Code hosting | Free | Public/private repos |
| Google Drive | Model storage | Free | 15GB free tier |
| **Total** | | **$45-80/month** | Scales with users |

## 1. Development Environment (Local)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fluxpointai-backend.git
cd fluxpointai-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver
```

### Daily Workflow

1. Write code in VS Code
2. Test locally with SQLite
3. Commit to GitHub
4. Training happens on Colab
5. Models sync to Drive
6. Deploy to VPS when ready

## 2. Training (Google Colab Pro)

### Why Colab Pro?

✅ **GPU Access**: Free T4/P100 GPUs for training  
✅ **24h Runtime**: Enough for most training sessions  
✅ **No Setup**: Pre-installed ML libraries  
✅ **Drive Integration**: Easy model storage  
✅ **Cost**: Only $10/month (vs $200+/month for dedicated GPU)

### Setup Colab Training

**Create `train_colab.ipynb`:**

```python
# Install dependencies
!pip install -q stable-baselines3 gymnasium torch

# Clone repository
!git clone https://github.com/yourusername/fluxpointai-backend.git
%cd fluxpointai-backend/fluxpoint

# Authenticate with Drive
from google.colab import drive
drive.mount('/content/drive')

# Train TCE model
from trading.tce.training import train_tce_model

# Get data from Drive
import pandas as pd
candles = pd.read_parquet('/content/drive/MyDrive/fluxpoint/data/EURUSD_H1.parquet')

# Train
metrics = train_tce_model(
    symbol='EURUSD',
    timeframe='H1',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Save model to Drive
!cp models/ml/tce_ml_model.pt /content/drive/MyDrive/fluxpoint/models/
```

**Train RL on Colab:**

```python
# Train RL execution optimizer
from trading.rl.training import train_rl_execution

metrics = train_rl_execution(
    candles=candles,
    valid_setups=setups,
    total_timesteps=100000
)

# Save to Drive
!cp models/rl/tce_execution_ppo.zip /content/drive/MyDrive/fluxpoint/models/
```

### Automation

Schedule training with Colab Pro:
- Weekly: Retrain ML models
- Monthly: Retrain RL models
- On-demand: New user strategies

## 3. Production VPS (Django)

### Recommended Providers (Nigeria-Friendly)

1. **Linode** ($20-40/month, 8-16GB RAM)
   - Good latency to Nigeria
   - Excellent documentation
   - Easy setup

2. **DigitalOcean** ($24-48/month, 8-16GB RAM)
   - Popular, reliable
   - 1-click apps
   - Good support

3. **Vultr** ($18-36/month, 8-16GB RAM)
   - Multiple African locations
   - Cheaper than DO
   - Good performance

### VPS Specifications

**Minimum (Dev/Testing):**
- CPU: 2 cores
- RAM: 8GB
- Storage: 50GB SSD
- Bandwidth: 4TB/month
- Cost: ~$20/month

**Recommended (Production):**
- CPU: 4 cores
- RAM: 16GB
- Storage: 100GB SSD
- Bandwidth: 5TB/month
- Cost: ~$40/month

### VPS Setup (Ubuntu 22.04)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install Nginx
sudo apt install nginx -y

# Install Supervisor (for Django process management)
sudo apt install supervisor -y

# Install Git
sudo apt install git -y

# Create app user
sudo useradd -m -s /bin/bash fluxpoint
sudo su - fluxpoint

# Clone repository
git clone https://github.com/yourusername/fluxpointai-backend.git
cd fluxpointai-backend

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies (NO PyTorch - inference only)
pip install django djangorestframework psycopg2-binary gunicorn requests python-dotenv django-apscheduler

# Configure environment
cp .env.example .env
nano .env  # Add production settings

# Download models from Drive
mkdir -p models/ml models/rl
# Use rclone or gdown to sync from Drive

# Run migrations
python manage.py migrate --settings=fluxpoint.settings_production

# Collect static files
python manage.py collectstatic --no-input

# Test server
gunicorn fluxpoint.wsgi:application --bind 0.0.0.0:8000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name api.fluxpointai.com;

    location /static/ {
        alias /home/fluxpoint/fluxpointai-backend/staticfiles/;
    }

    location /media/ {
        alias /home/fluxpoint/fluxpointai-backend/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Supervisor Configuration

```ini
[program:fluxpoint_django]
command=/home/fluxpoint/fluxpointai-backend/venv/bin/gunicorn fluxpoint.wsgi:application --bind 127.0.0.1:8000 --workers 4
directory=/home/fluxpoint/fluxpointai-backend/fluxpoint
user=fluxpoint
autostart=true
autorestart=true
stderr_logfile=/var/log/fluxpoint/django.err.log
stdout_logfile=/var/log/fluxpoint/django.out.log
```

## 4. MT5 VPS (Windows)

### Why Separate MT5 VPS?

- MT5 runs best on Windows
- 24/7 connectivity to brokers
- Isolated from Django (security)
- No GPU needed
- Cheap ($15-30/month)

### Recommended Providers

1. **Contabo** ($10-20/month)
   - Very cheap
   - Windows Server
   - Good for MT5

2. **Kamatera** ($15-25/month)
   - Multiple locations
   - Windows VPS
   - Free trial

### MT5 VPS Setup

1. **Install MT5**
   - Download from broker
   - Login to trading account
   - Enable Expert Advisors

2. **Install Python**
   ```powershell
   # Download Python 3.12 for Windows
   # Install with "Add to PATH" checked
   
   # Install MT5 package
   pip install MetaTrader5 pyzmq
   ```

3. **Create MT5 Bridge Script**
   ```python
   # mt5_bridge.py
   import MetaTrader5 as mt5
   import zmq
   import json
   
   # Connect to MT5
   mt5.initialize()
   mt5.login(account=YOUR_ACCOUNT, password="PASSWORD", server="BROKER_SERVER")
   
   # ZMQ server (receives commands from Django VPS)
   context = zmq.Context()
   socket = context.socket(zmq.REP)
   socket.bind("tcp://*:5555")
   
   while True:
       message = socket.recv_json()
       
       if message['action'] == 'get_candles':
           # Fetch candles
           candles = mt5.copy_rates_range(...)
           socket.send_json({'candles': candles.tolist()})
       
       elif message['action'] == 'place_order':
           # Place trade
           request = {
               "action": mt5.TRADE_ACTION_DEAL,
               "symbol": message['symbol'],
               "volume": message['volume'],
               "type": mt5.ORDER_TYPE_BUY,
               "price": mt5.symbol_info_tick(message['symbol']).ask,
               "sl": message['sl'],
               "tp": message['tp'],
           }
           result = mt5.order_send(request)
           socket.send_json({'result': result._asdict()})
   ```

4. **Run as Windows Service**
   - Use NSSM (Non-Sucking Service Manager)
   - Auto-start on reboot
   - Logs to file

## 5. GitHub for Code Sync

### Repository Structure

```
fluxpointai-backend/
├── .github/
│   └── workflows/
│       ├── test.yml          # CI tests
│       └── deploy.yml        # Auto-deploy to VPS
├── fluxpoint/
│   ├── trading/              # TCE strategy
│   ├── strategy_builder/     # No-code builder
│   └── ...
├── colab/
│   ├── train_tce.ipynb       # TCE training notebook
│   ├── train_rl.ipynb        # RL training notebook
│   └── backtest.ipynb        # Backtesting notebook
├── requirements.txt          # Python dependencies
├── requirements_training.txt # Training-only deps (PyTorch, etc.)
├── .env.example              # Environment template
└── README.md
```

### GitHub Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to VPS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to VPS
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.VPS_HOST }}
          username: fluxpoint
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd ~/fluxpointai-backend
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            python manage.py migrate
            python manage.py collectstatic --no-input
            sudo supervisorctl restart fluxpoint_django
```

## 6. Model Storage (Google Drive)

### Structure

```
Google Drive/fluxpoint/
├── models/
│   ├── ml/
│   │   ├── tce_ml_model.pt
│   │   └── user_123_strategy_1.pt
│   └── rl/
│       ├── tce_execution_ppo.zip
│       └── user_123_strategy_1_rl.zip
├── data/
│   ├── EURUSD_H1.parquet
│   ├── GBPUSD_H1.parquet
│   └── ...
└── logs/
    └── training_2026_01_06.log
```

### Sync to VPS

**Using rclone:**

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Drive
rclone config

# Sync models from Drive to VPS
rclone sync gdrive:fluxpoint/models/ ~/fluxpointai-backend/fluxpoint/models/

# Add to cron (hourly sync)
0 * * * * rclone sync gdrive:fluxpoint/models/ ~/fluxpointai-backend/fluxpoint/models/
```

## 7. Network Architecture

```
                    ┌──────────────────────┐
                    │   Users (Web/App)    │
                    └──────────────────────┘
                              │
                              │ HTTPS
                              ↓
                    ┌──────────────────────┐
                    │  Nginx (VPS)         │
                    │  SSL Certificate     │
                    └──────────────────────┘
                              │
                              │ HTTP
                              ↓
                    ┌──────────────────────┐
                    │  Django (Gunicorn)   │
                    │  - API Endpoints     │
                    │  - ML Inference      │
                    │  - User Management   │
                    └──────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              PostgreSQL        ZMQ→ MT5 VPS (Windows)
              (User Data)            - Get candles
              (Strategies)           - Place trades
              (Performance)          - Monitor positions
```

## 8. Security

### VPS Hardening

```bash
# Firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# SSH hardening
sudo nano /etc/ssh/sshd_config
# Change: PermitRootLogin no
# Change: PasswordAuthentication no
sudo systemctl restart sshd

# Install Fail2Ban
sudo apt install fail2ban -y
```

### Environment Variables

Never commit to Git:
- API keys
- Database passwords
- MT5 credentials
- Broker details

Use `.env` file (in `.gitignore`).

## 9. Monitoring

### Tools

1. **Uptime monitoring**: UptimeRobot (free)
2. **Error tracking**: Sentry (free tier)
3. **Server metrics**: Netdata (free, self-hosted)
4. **Logs**: Papertrail (free tier)

### Django Logging

```python
# settings.py
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

## 10. Scaling Strategy

### Phase 1 (0-100 users)
- Single VPS (16GB RAM)
- Colab Pro for training
- Cost: ~$50/month

### Phase 2 (100-1000 users)
- Upgrade VPS to 32GB RAM
- Add load balancer
- Colab Pro + occasional GPU cloud
- Cost: ~$150/month

### Phase 3 (1000+ users)
- Multiple VPS instances
- Dedicated ML server (if needed)
- Professional broker integration
- Cost: ~$500+/month

## Summary

This infrastructure is:
✅ **Affordable**: $45-80/month starting cost  
✅ **Scalable**: Can grow with users  
✅ **Nigeria-Friendly**: Works with local payment methods  
✅ **Reliable**: Industry-standard tools  
✅ **Maintainable**: Simple architecture  

No expensive GPU servers needed - training happens on Colab Pro, production VPS only runs inference (CPU is fine).
