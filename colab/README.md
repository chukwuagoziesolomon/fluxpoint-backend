# Quick Start: Train TCE on Colab

## 3-Minute Setup

### 1. Open Colab Notebook

Click this link: [Open in Colab](https://colab.research.google.com/)

Or:
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Choose `colab/train_tce_complete.ipynb` from this repository

### 2. Enable GPU

1. Click "Runtime" → "Change runtime type"
2. Select "T4 GPU" from dropdown
3. Click "Save"

### 3. Run All Cells

1. Click "Runtime" → "Run all"
2. When prompted, authorize Google Drive access
3. Wait 30-60 minutes for training to complete

### 4. Models Auto-Saved to Drive

After training, models are automatically saved to:
```
Google Drive/fluxpoint/models/ml/tce_ml_model.pt
Google Drive/fluxpoint/models/rl/tce_execution_ppo.zip
```

## That's It!

Models are now ready to deploy to your VPS.

## What You Need Before Starting

✅ Google account (for Colab)  
✅ GitHub account (code already pushed)  
✅ Historical data uploaded to Drive (or use sample data in notebook)

## Detailed Guide

See [TRAINING_COLAB_GUIDE.md](../TRAINING_COLAB_GUIDE.md) for:
- Step-by-step walkthrough
- Data preparation
- Troubleshooting
- Advanced options
- Deployment instructions

## Cost

**Free Colab**: Limited to 12h runtime (may disconnect)  
**Colab Pro ($10/month)**: 24h runtime, faster GPUs ⭐ Recommended

## Training Time

| Task | Time |
|------|------|
| ML Training | 10-20 min |
| RL Training | 20-40 min |
| Total | 30-60 min |

## Next Steps After Training

```bash
# On your VPS, sync models from Drive
rclone sync gdrive:fluxpoint/models/ ~/fluxpointai-backend/fluxpoint/models/

# Restart Django
sudo supervisorctl restart fluxpoint_django

# Test
curl http://localhost:8000/api/tce/validate/
```

## Troubleshooting

**Issue**: "Runtime disconnected"  
**Solution**: Upgrade to Colab Pro or save checkpoints every 10 epochs

**Issue**: "No training data"  
**Solution**: Upload historical data to Drive at `/fluxpoint/data/EURUSD_H1.csv`

**Issue**: "Out of memory"  
**Solution**: Reduce batch_size from 32 to 16

## Support

Questions? Check [TRAINING_COLAB_GUIDE.md](../TRAINING_COLAB_GUIDE.md) for detailed troubleshooting.
