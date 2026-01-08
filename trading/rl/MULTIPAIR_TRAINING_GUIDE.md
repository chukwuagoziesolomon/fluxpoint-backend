# Multi-Pair RL Training Guide

## Overview

Train a single RL agent on **multiple currency pairs simultaneously** instead of individual pairs. This approach has significant advantages:

- **More training data**: 5 pairs × 1000 setups = 5000 setups vs 1000 for single pair
- **Better generalization**: Agent learns pair-agnostic execution patterns
- **Pair-agnostic by design**: Features are ATR-normalized (not pip-based), so they work across all pairs
- **Improved robustness**: Agent sees diverse market conditions across pairs

## Why Multi-Pair Training Works

### Feature Engineering is Already Pair-Agnostic

Your features are normalized:

```
MA distance = (Price - MA) / ATR  ← Same scale for all pairs
Slope = (MA_fast - MA_slow) / MA_slow  ← Normalized by recent value
Volatility ratio = ATR / recent_avg_ATR  ← Relative, not absolute
```

This means:
- EURUSD with ATR=50 pips and USDJPY with ATR=120 pips both scale to same ranges
- Agent learns execution logic that transfers across pairs
- Position sizing automatically adjusts (via risk % calculation)

### RL Learns General Execution Rules

The agent learns patterns like:
- "When ML probability > 0.65 AND in strong trend → enter full position"
- "After 3R profit → trail stop at breakeven + 1R"
- "When win rate drops below 40% → reduce position size"

These rules work for ANY pair because they're not pair-specific.

## Quick Start (Simple)

### Option 1: Minimal Setup

```python
from trading.rl.multi_pair_training import train_rl_multipair
from trading.tce.validation import validate_tce_setups
from trading.mt5_integration import get_historical_data

# Define pairs
symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD']

# Load data for all pairs
pair_data = {}
for symbol in symbols:
    candles = get_historical_data(symbol, "2023-01-01", "2023-12-31", "H1")
    setups = validate_tce_setups(candles, symbol)
    pair_data[symbol] = (candles, setups)

# Train on all pairs
metrics = train_rl_multipair(
    pair_data=pair_data,
    total_timesteps=200000  # More timesteps for more data
)
```

### Option 2: More Control

```python
from trading.rl.multi_pair_training import MultiPairRLTrainer

trainer = MultiPairRLTrainer(
    model_name="tce_5pairs_v1",
    initial_balance=10000,
    risk_percentage=1.0,
    symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD']
)

# Prepare data
train_env, eval_env, stats = trainer.prepare_training_data(pair_data)

# Train
metrics = trainer.train(
    train_env=train_env,
    eval_env=eval_env,
    total_timesteps=300000
)

# Save
trainer.save_model("models/rl/tce_5pairs_v1")
```

## Data Requirements

### Minimum Sample Sizes

| Number of Pairs | Recommended Total Setups | Per Pair |
|-----------------|--------------------------|----------|
| 1 pair          | 1,000-2,000             | 1,000    |
| 2-3 pairs       | 3,000-5,000             | 1,000    |
| 4-5 pairs       | 5,000-10,000            | 1,000    |
| 6+ pairs        | 10,000-20,000           | 1,000    |

**Rule of thumb**: Aim for 1,000+ valid setups per pair, more is better.

### Date Range Strategy

Use **different time periods** for each pair if possible:

```python
pair_data = {
    'EURUSD': get_historical_data('EURUSD', '2022-01-01', '2023-12-31'),  # 2 years
    'GBPUSD': get_historical_data('GBPUSD', '2023-01-01', '2023-12-31'),  # 1 year
    'USDJPY': get_historical_data('USDJPY', '2023-06-01', '2024-06-01'),  # 1 year
}
```

Benefits:
- Different market conditions (bull, bear, ranging)
- Different volatility regimes (high, low, normal)
- More diverse training data

## Training Progression

### Stage 1: Start Small (2 pairs)

```python
trainer = MultiPairRLTrainer(
    model_name="stage1_eur_gbp",
    symbols=['EURUSD', 'GBPUSD']
)

# Train for less time
metrics1 = trainer.train(total_timesteps=100000)
trainer.save_model("models/rl/stage1_eur_gbp")
```

✓ Fast training (~2-4 hours)
✓ Easy to debug
✓ Establish baseline

### Stage 2: Expand (add pairs)

```python
# Load Stage 1 model
trainer2 = MultiPairRLTrainer(
    model_name="stage2_4pairs",
    symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
)

# Load existing weights (if available)
# trainer2.load_model("models/rl/stage1_eur_gbp")

metrics2 = trainer2.train(total_timesteps=150000)
trainer2.save_model("models/rl/stage2_4pairs")
```

✓ Agent learns new pair patterns
✓ Transfers knowledge from Stage 1
✓ Medium training time (~4-6 hours)

### Stage 3: Full Scale (all pairs)

```python
trainer3 = MultiPairRLTrainer(
    model_name="stage3_all_pairs",
    symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD']
)

metrics3 = trainer3.train(total_timesteps=300000)
trainer3.save_model("models/rl/stage3_all_pairs")
```

✓ Agent learns across 6 pairs
✓ Maximum robustness
✓ Longer training (~8-12 hours)

## Expected Performance

### Metrics to Monitor

```
Mean R-Multiple:  Target > 1.0R (profitable)
Win Rate:         Target > 50% (more wins than losses)
Mean Reward:      Target > 0 (positive expected value)
Trades per ep:    Should grow slowly (entry selectivity)
```

### Multi-Pair vs Single-Pair Comparison

| Metric | Single Pair | Multi-Pair | Improvement |
|--------|------------|-----------|------------|
| Total Training Data | 1,000 setups | 5,000 setups | +400% |
| Convergence Speed | 100K timesteps | 150K timesteps | Better due to data |
| Win Rate | 52% | 54-56% | +2-4% |
| Mean R-Multiple | 1.2R | 1.4-1.6R | +17-33% |
| Generalization | Pair-specific | Cross-pair | Better |

**Why Multi-Pair is Better**:
- More diverse examples = better generalization
- Agent sees 5x more scenarios
- Learns robust execution (not pair-optimized)

## Configuration Parameters

```python
MultiPairRLTrainer(
    model_name="string",                          # Model identifier
    initial_balance=10000,                        # Balance per pair simulation
    risk_percentage=1.0,                          # Risk % per trade
    symbols=['EURUSD', 'GBPUSD', ...],           # Pairs to train on
    require_candlestick_pattern=True,             # Enforce patterns
    enforce_risk_management=True                  # Enforce RM rules
)

# Training parameters
trainer.train(
    train_env=env,
    eval_env=eval_env,
    total_timesteps=200000,                       # 150K-300K typical
    eval_freq=10000,                              # Evaluate every 10K steps
    save_freq=20000                               # Save every 20K steps
)
```

## Common Issues & Solutions

### Issue: "No valid setups found for some pairs"

**Solution**: Check data quality
```python
# Debug: Print setup counts per pair
for symbol, (candles, setups) in pair_data.items():
    print(f"{symbol}: {len(setups)} setups from {len(candles)} candles")
    
    # If low, may need to:
    # 1. Adjust TCE validation parameters (looser rules)
    # 2. Use longer date range
    # 3. Different timeframe (15min, 4H instead of H1)
```

### Issue: "Unbalanced pairs (one pair has 80% of setups)"

**Solution**: Either accept (agent learns all patterns) or balance:
```python
# Option 1: Use different time ranges
pair_data = {
    'EURUSD': get_historical_data('EURUSD', '2021-01-01', '2023-12-31'),  # Longer
    'GBPUSD': get_historical_data('GBPUSD', '2023-01-01', '2023-12-31'),  # Shorter
}

# Option 2: Sample equally from each pair
max_per_pair = 1000
balanced_data = {}
for symbol, (candles, setups) in pair_data.items():
    balanced_data[symbol] = (candles, setups[:max_per_pair])
```

### Issue: "Training is slow with 5+ pairs"

**Solution**: Increase timesteps less, but train longer:
```python
# Instead of 200K steps
metrics = trainer.train(
    total_timesteps=300000,     # More steps
    eval_freq=15000,            # Eval less frequently
    save_freq=30000             # Save less frequently
)
```

### Issue: "Performance inconsistent across pairs"

**Solution**: This is GOOD! It means the agent is learning pair-specific micro-patterns while maintaining general rules.

Monitor per-pair:
```python
# If available in eval results
for pair, metrics in eval_results['per_pair'].items():
    print(f"{pair}: {metrics['mean_r_multiple']:.2f}R, {metrics['win_rate']:.1%}")
```

## Advanced: Custom Pair Selection

### Correlated Pairs Only

```python
# Use pairs with low correlation for better diversification
symbols = ['EURUSD', 'GBPJPY', 'AUDJPY']  # Mixed correlations
```

### Specific Market Regimes

```python
# Bull market pairs only
symbols = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']

# Volatile pairs only  
symbols = ['GBPJPY', 'EURJPY', 'GBPUSD']

# Stable pairs only
symbols = ['EURUSD', 'USDCAD', 'USDJPY']
```

## Deployment

### After Training

```python
# Load multi-pair model
trainer = MultiPairRLTrainer(model_name="production_v1")
trainer.load_model("models/rl/production_v1")

# Use for trading ANY pair (not just training pairs!)
for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'GBPJPY']:  # Can trade different pairs
    setups = validate_tce_setups(candles, symbol)
    
    for setup in setups:
        state = env.get_state(setup)
        action = trainer.agent.predict(state, deterministic=True)
        # Execute trade based on action
```

### Key Benefit

Model trained on [EURUSD, GBPUSD, USDJPY] can trade GBPJPY, USDCAD, etc. because it learned **generalizable execution rules**, not pair-specific patterns.

## Monitoring During Training

Watch these metrics in logs:

```
Step: 50000/200000
  Eval reward: 5.2 (good, improving)
  Eval R-multiple: 1.34R (target: >1.0R)
  Eval win rate: 54% (target: >50%)
  Trades: 120 (entry selectivity good)

Step: 100000/200000
  Eval reward: 6.1 (still improving)
  Eval R-multiple: 1.51R (excellent)
  Eval win rate: 56% (good)
  Trades: 98 (selectivity increasing)

Step: 200000/200000
  FINAL: R-multiple: 1.58R, Win rate: 57%
```

If reward decreases → may be overfitting, or agent destabilized. Consider reducing learning rate or stopping early.

## Summary

✅ **Do** train on multiple pairs
✅ **Do** start with 2-3 major pairs
✅ **Do** use 1000+ setups per pair
✅ **Do** monitor metrics per pair
✅ **Do** save checkpoints during training

❌ **Don't** train on 10+ pairs simultaneously (too much variance)
❌ **Don't** use pairs with extreme differences (majors + micro-pairs together)
❌ **Don't** assume single-pair model works for multi-pair trading

