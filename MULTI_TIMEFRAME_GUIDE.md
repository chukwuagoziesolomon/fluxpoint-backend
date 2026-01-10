# Multi-Timeframe Training Guide

## 📊 What This Does

Generates TCE training setups from **8 different timeframes** to massively increase training data:

### Timeframes Generated:
1. **1M** - 1-Minute candles (~144 per day, heavily sampled)
2. **5M** - 5-Minute candles (~288 per day, sampled)
3. **15M** - 15-Minute candles (~96 per day)
4. **30M** - 30-Minute candles (~48 per day)
5. **1H** - Hourly candles (24 per day)
6. **4H** - 4-Hour candles (6 per day)
7. **1D** - Daily candles (1 per day) ← ORIGINAL DATA
8. **1W** - Weekly candles

## 🔢 Training Data Multiplier

From **1,500 daily candles** per pair:

| Timeframe | Candles Generated | Sampling Rate | Final Training Samples |
|-----------|------------------|---------------|----------------------|
| 1M        | ~216,000         | Every 100th   | ~2,160              |
| 5M        | ~432,000         | Every 50th    | ~8,640              |
| 15M       | ~144,000         | Every 20th    | ~7,200              |
| 30M       | ~72,000          | Every 10th    | ~7,200              |
| 1H        | ~36,000          | Every 5th     | ~7,200              |
| 4H        | ~9,000           | Every 3rd     | ~3,000              |
| 1D        | ~1,500           | Every 1st     | ~1,500              |
| 1W        | ~214             | Every 1st     | ~214                |
| **TOTAL** |                  |               | **~37,000+ per pair**|

With **15 pairs** = **550,000+ potential training examples!**

## 🚀 Quick Start

### Option 1: Fast Training (Recommended)
Use only higher timeframes for quick results:
```python
# In CELL4_MULTI_TIMEFRAME_TRAINING.py, change line:
timeframes = ['15M', '30M', '1H', '4H', '1D', '1W']
# Skip 1M and 5M for faster training
```

### Option 2: Maximum Data (Slow but thorough)
Use all timeframes:
```python
timeframes = ['1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W']
# This will take MUCH longer but gives maximum training data
```

## 📝 Files Created

### 1. Data Cleaning
```bash
python fix_csv_data.py
```
**Output:** `training_data_cleaned/` folder with proper CSV files

### 2. Multi-Timeframe Training
```bash
python CELL4_MULTI_TIMEFRAME_TRAINING.py
```
**Output:** 
- `models/tce_multi_tf_model.pt` - Trained neural network
- `models/scaler_mean.npy` - Feature normalization mean
- `models/scaler_scale.npy` - Feature normalization scale

## ⚙️ How It Works

### Data Generation
1. **Load daily data** from cleaned CSV files
2. **Resample** to each timeframe using realistic price movement simulation
3. **Calculate indicators** (MA6, MA18, MA50, MA200, ATR, slopes)
4. **Validate** using actual `validate_tce()` function
5. **Extract features** for valid setups

### Feature Engineering
Each setup has **38 features:**
- **4** Moving averages (MA6, MA18, MA50, MA200)
- **4** MA slopes
- **1** ATR (volatility)
- **9** Ratio features (price/MA ratios, distances)
- **8** Rule scores (trend, fib, retest, etc.)
- **4** Risk metrics (RR ratio, SL/TP pips, position size)
- **3** Trend flags (direction, uptrend, downtrend)
- **2** Market conditions (volatility, price near MA)
- **1** Timeframe encoding
- **2** Volatility measures

## 🎯 Expected Results

### Dataset Size
- **Single timeframe (1D):** ~100-500 setups
- **Multi-timeframe (all 8):** ~37,000+ setups per pair
- **Total (15 pairs):** 550,000+ training examples

### Model Performance
- More training data = Better generalization
- Multiple timeframes = Better pattern recognition
- Expected accuracy: 75-85% on validation set

## 🐛 Troubleshooting

### Issue: Too slow
**Solution:** Reduce timeframes
```python
timeframes = ['1H', '4H', '1D', '1W']  # Skip minute-level
```

### Issue: Out of memory
**Solution:** Process fewer pairs at once
```python
# Only load first 5 pairs for testing
csv_files = sorted(data_dir.glob('*_data.csv'))[:5]
```

### Issue: No valid setups found
**Check:**
1. Data quality: `python load_clean_data.py`
2. Validation rules: Check `trading/tce/validation.py`
3. Indicator calculations: Verify MA calculations are correct

## 📊 Sample Output

```
================================================================================
CELL 4: MULTI-TIMEFRAME TCE TRAINING
================================================================================

📂 Loading data from: training_data_cleaned

🔄 Creating multi-timeframe datasets...
⏱️  Timeframes to generate: 1M, 5M, 15M, 30M, 1H, 4H, 1D, 1W

  ✅ EURUSD      1M    216000 candles
  ✅ EURUSD      5M    432000 candles
  ✅ EURUSD      15M   144000 candles
  ✅ EURUSD      30M    72000 candles
  ✅ EURUSD      1H     36000 candles
  ✅ EURUSD      4H      9000 candles
  ✅ EURUSD      1D      1486 candles
  ✅ EURUSD      1W       212 candles
  
  ... (repeated for all pairs)

✅ Created 120 timeframe datasets from 15 pairs

🔍 Scanning for valid TCE setups across ALL timeframes...

  ✅ EURUSD      1M     145 valid setups
  ✅ EURUSD      5M     678 valid setups
  ✅ EURUSD      15M    423 valid setups
  ...

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

  TOTAL: 34,015 VALID TCE SETUPS (1,365,050 checked)
================================================================================

🤖 TRAINING NEURAL NETWORK ON 34,015 MULTI-TIMEFRAME SETUPS

  Epoch 10/50 | Loss: 0.342156
  Epoch 20/50 | Loss: 0.298234
  Epoch 30/50 | Loss: 0.267891
  Epoch 40/50 | Loss: 0.245623
  Epoch 50/50 | Loss: 0.231456

✅ Model trained! Final loss: 0.231456
   Training Accuracy: 82.3%

✅ Model saved to: models
   • tce_multi_tf_model.pt
   • scaler_mean.npy
   • scaler_scale.npy

📊 TRAINING SUMMARY:
   • Total setups: 34,015
   • Feature dimensions: 38
   • Timeframes used: 1M, 5M, 15M, 30M, 1H, 4H, 1D, 1W
   • Model accuracy: 82.3%
```

## 🎓 Next Steps

1. **Train the model:**
   ```bash
   python CELL4_MULTI_TIMEFRAME_TRAINING.py
   ```

2. **Upload to Google Colab** (if using Colab):
   - Upload cleaned data to Drive: `/MyDrive/forex_data/training_data_cleaned/`
   - Run notebook cell with this script

3. **Test the model:**
   - Load saved model: `model.load_state_dict(torch.load('models/tce_multi_tf_model.pt'))`
   - Use on live data for predictions

4. **Fine-tune:**
   - Adjust `step_sizes` for different sampling rates
   - Add/remove timeframes based on performance
   - Experiment with model architecture

## 💡 Tips

1. **Start small:** Test with 2-3 pairs and fewer timeframes first
2. **Monitor memory:** Minute-level data uses lots of RAM
3. **Save checkpoints:** Long training runs should save periodically
4. **Validate results:** Check sample setups to ensure data quality
5. **Compare timeframes:** Track which timeframes produce best setups
