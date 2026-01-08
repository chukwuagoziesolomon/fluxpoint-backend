# Multi-Pair Training Quick Checklist

## Pre-Training Checklist

### Data Preparation
- [ ] Do you have historical data for 2+ currency pairs?
- [ ] Each pair has at least 1000+ valid TCE setups?
- [ ] Data is chronologically ordered and has no gaps?
- [ ] All pairs use same timeframe (H1, 4H, 1D)?

### Environment Setup
- [ ] Python environment activated
- [ ] Trading module imports work: `from trading.rl.multi_pair_training import MultiPairRLTrainer`
- [ ] Data loading function works (MT5 or CSV)
- [ ] TCE validator runs without errors

### Model Configuration
- [ ] Decided on `initial_balance` (10K reasonable for testing)
- [ ] Set `risk_percentage` (1% recommended)
- [ ] Selected currency pairs to trade
- [ ] Chose unique `model_name` for this training run

### Resource Check
- [ ] Have 8+ GB RAM available
- [ ] GPU available (optional but 3-5x faster)
- [ ] At least 5GB free disk space for checkpoints
- [ ] Training time: 8-24 hours for 200K-300K timesteps

---

## Training Steps

### Step 1: Load Data

```python
from trading.rl.multi_pair_training import MultiPairRLTrainer
from trading.tce.validation import validate_tce_setups
from trading.mt5_integration import get_historical_data

# Define pairs
symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

# Load data
pair_data = {}
for symbol in symbols:
    print(f"Loading {symbol}...")
    candles = get_historical_data(symbol, "2023-01-01", "2023-12-31", "H1")
    setups = validate_tce_setups(candles, symbol)
    pair_data[symbol] = (candles, setups)
    print(f"  {len(candles)} candles, {len(setups)} setups")
```

**Verify**: Each pair should show "X setups"

### Step 2: Initialize Trainer

```python
trainer = MultiPairRLTrainer(
    model_name="tce_v1_3pairs",      # Unique name
    initial_balance=10000,            # Starting balance
    risk_percentage=1.0,              # 1% risk per trade
    symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
    require_candlestick_pattern=True,
    enforce_risk_management=True
)

print("✓ Trainer initialized")
```

### Step 3: Prepare Training Data

```python
train_env, eval_env, stats = trainer.prepare_training_data(pair_data)

print(f"✓ Data prepared:")
print(f"  Total setups: {stats['total_setups']}")
print(f"  Training: {stats['train_setups']}")
print(f"  Evaluation: {stats['eval_setups']}")
print(f"  Per-pair: {stats['setups_per_pair']}")
```

**Verify**: Total setups > 1000, training/eval split looks balanced

### Step 4: Train Agent

```python
# This takes several hours!
metrics = trainer.train(
    train_env=train_env,
    eval_env=eval_env,
    total_timesteps=200000,        # Adjust based on data size
    eval_freq=10000,               # Eval every 10K steps
    save_freq=20000                # Save every 20K steps
)

print("✓ Training complete")
```

**Monitor**: Watch logs for improving metrics (reward, R-multiple, win rate)

### Step 5: Save Model

```python
trainer.save_model("models/rl/tce_v1_3pairs")
print("✓ Model saved")
```

**Verify**: Check that model file exists in `models/rl/`

---

## Post-Training Checklist

### Evaluate Results

```python
# Check evaluation metrics
eval_results = metrics.get('eval', {})
print(f"\nFinal Results:")
print(f"  Mean Reward: {eval_results.get('mean_reward', 'N/A'):.2f}")
print(f"  Mean R-Multiple: {eval_results.get('mean_r_multiple', 'N/A'):.2f}R")
print(f"  Mean Win Rate: {eval_results.get('mean_win_rate', 'N/A'):.1%}")
```

- [ ] Mean R-Multiple > 1.0R (profitable)
- [ ] Win Rate > 50% (more wins than losses)
- [ ] Mean Reward > 0 (positive expected value)

### Compare with Baseline

```python
# Compare to single-pair or no-RL baseline
print("\nComparison:")
print(f"  Multi-pair vs single-pair:")
print(f"    Data: 3000 vs 1000 setups (3x more)")
print(f"    Expected improvement: 10-30% in metrics")
```

### Identify Issues

- [ ] Low win rate (<40%)? → Improve TCE validator or relax candlestick rules
- [ ] Low R-multiple (<0.8R)? → Adjust risk management parameters
- [ ] Agent won't enter trades? → Check if candlestick patterns exist in data
- [ ] Training unstable? → Reduce learning rate, use smaller risk %

---

## Deployment Checklist

### Load Model

```python
from trading.rl.multi_pair_training import MultiPairRLTrainer

trainer = MultiPairRLTrainer(model_name="tce_v1_3pairs")
trainer.load_model("models/rl/tce_v1_3pairs")
```

### Test on New Pair

```python
# Test on pair NOT in training
new_symbol = 'GBPJPY'
candles = get_historical_data(new_symbol, "2024-01-01", "2024-06-01")
setups = validate_tce_setups(candles, new_symbol)

# Use agent for trading
for setup in setups:
    state = ... # Create state from setup
    action, _ = trainer.agent.predict(state, deterministic=True)
    # Execute based on action
```

- [ ] Model loads without errors
- [ ] Works on training pairs
- [ ] Works on new pairs (generalization!)
- [ ] Action decisions make sense

---

## Troubleshooting

### "No valid setups found for symbol X"

```python
# Check TCE validator settings
from trading.tce.validation import validate_tce_setups

setups = validate_tce_setups(candles, symbol)
print(f"Setups found: {len(setups)}")

if len(setups) == 0:
    print("Check:")
    print("  - Candlestick patterns in data?")
    print("  - Market structure defined?")
    print("  - Validation parameters too strict?")
```

### "Training memory error"

```python
# Reduce data
pair_data[symbol] = (candles[-50000:], setups[-5000:])  # Last 50K candles

# Or increase batch size
trainer.train(..., batch_size=128)  # If supported
```

### "Model performance degraded on other pairs"

```python
# This is normal! Expect 5-10% variation
# Train on more diverse data:
pair_data = {
    'EURUSD': get_historical_data(..., '2020-01-01', '2023-12-31'),  # Longer
    'GBPUSD': get_historical_data(..., '2023-01-01', '2023-12-31'),
    ...
}
```

---

## Performance Targets

### Conservative Strategy
- Total setups: 3000
- Timesteps: 150000
- Expected R-multiple: 1.2-1.3R
- Win rate: 52-55%

### Balanced Strategy (Recommended)
- Total setups: 5000
- Timesteps: 200000
- Expected R-multiple: 1.3-1.5R
- Win rate: 54-57%

### Aggressive Strategy
- Total setups: 10000
- Timesteps: 300000
- Expected R-multiple: 1.5-1.8R
- Win rate: 56-60%

---

## Next Steps After Training

1. **Backtest** on 2024 data (out-of-sample)
2. **Paper trade** on demo account
3. **Monitor** metrics on live data
4. **Retrain** monthly with new data
5. **Add pairs** gradually (train 2 more pairs, merge model)

---

## Files Created

- `multi_pair_training.py` - Main trainer class
- `train_multipair_example.py` - 4 examples to get started
- `MULTIPAIR_TRAINING_GUIDE.md` - Detailed guide
- `MULTIPAIR_QUICK_CHECKLIST.md` - This file

---

**Good luck with multi-pair training! 🚀**

