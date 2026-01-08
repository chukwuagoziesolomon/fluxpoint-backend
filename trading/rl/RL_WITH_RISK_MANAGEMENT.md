# RL Agent with Risk Management & Candlestick Pattern Validation

## Overview

The RL agent now integrates:
1. **Candlestick Pattern Validation** - Enforces specific patterns before entry
2. **Risk Management Rules** - Position sizing, SL/TP constraints, RR ratio validation
3. **Dynamic Position Sizing** - Calculated from account risk % and stop loss distance

## Key Features

### 1. Candlestick Pattern Validation

Before the agent can execute an entry action, the environment validates that one of these patterns is confirmed:

**BUY Patterns:**
- Bullish Pin Bar (long lower wick, small upper wick)
- Bullish Engulfing (prior bearish candle, current bullish candle)
- One White Soldier (strong bullish after bearish)
- Tweezer Bottom (two candles with same low)
- Morning Star (3-candle reversal pattern)

**SELL Patterns:**
- Bearish Pin Bar (long upper wick, small lower wick)
- Bearish Engulfing (prior bullish candle, current bearish candle)
- One Black Crow (strong bearish after bullish)
- Tweezer Top (two candles with same high)
- Evening Star (3-candle reversal pattern)

**Penalty for Invalid Entry:**
- If candlestick pattern is missing: -0.5 reward (large penalty)

### 2. Risk Management Rules

The environment enforces professional risk management:

#### Stop Loss Calculation
```
1.5 × ATR from entry price
Minimum: 12 pips
Additional constraint: Must be below 61.8% Fibonacci level (BUY) or above (SELL)
```

#### Take Profit Calculation
Based on Risk:Reward ratio:
- SL distance ≤ 20 pips: Use 1:2 ratio
- SL distance 20-40 pips: Use 1:1.5 ratio  
- SL distance > 40 pips: Use 1:1.5 ratio

#### Position Sizing Formula
```
Position Size (lots) = (Account Balance × Risk%) / (SL Distance (pips) × $Value-per-pip)
```

Example: Account $10k, 1% risk, SL=20 pips, pip value=$10
- Risk amount = $10k × 1% = $100
- Position size = $100 / (20 × $10) = 0.5 lots

#### Risk/Reward Ratio Validation
```
RR Ratio = Take Profit Distance / Stop Loss Distance
Minimum RR Ratio: 1.0 (TP distance ≥ SL distance)
Penalty if RR < 1.0: -0.3 reward
```

### 3. Reward Structure

The agent learns from the R-multiple reward signal:

```python
R-multiple = Profit/Loss ÷ Risk Amount

Example 1 (Winning Trade):
- Entry: 1.2500
- Exit: 1.2550 (50 pips gain)
- SL: 1.2480 (20 pips risk)
- Risk amount: $100
- Profit: $500 (50 pips × $10)
- R-multiple: 500 / 100 = 5.0R ✅ (strong reward)

Example 2 (Losing Trade):
- Entry: 1.2500
- Exit: 1.2480 (hit stop loss)
- Risk amount: $100
- Loss: -$100
- R-multiple: -100 / 100 = -1.0R ❌ (penalty)

Example 3 (Partial Loss):
- Entry: 1.2500
- Exit: 1.2490 (10 pips loss)
- Risk amount: $100
- Loss: -$100
- R-multiple: -100 / 100 = -1.0R
```

### 4. Entry Actions with Constraints

| Action | Effect | Constraint |
|--------|--------|-----------|
| 0: Enter Full | 100% position size | Candlestick pattern required |
| 1: Enter Half | 50% position size | Candlestick pattern required |
| 2: Wait | Skip setup (opportunity cost) | Always allowed |
| 3: Exit | Close trade | Only if in position |
| 4: Trail Stop | Tighten SL to lock profit | Only if in position |

## Configuration

### Initialize Environment with Risk Management

```python
from trading.rl.environment import create_execution_env

env = create_execution_env(
    candles=df_candles,
    valid_setups=setups,
    initial_balance=10000,
    risk_percentage=1.0,  # Risk 1% per trade
    symbol="EURUSD",
    require_candlestick_pattern=True,  # Enforce patterns
    enforce_risk_management=True       # Enforce SL/TP/sizing
)
```

### Initialize Trainer with Risk Management

```python
from trading.rl.training import RLExecutionTrainer

trainer = RLExecutionTrainer(
    model_name="tce_execution_ppo",
    initial_balance=10000,
    risk_percentage=1.0,              # 1% risk per trade
    symbol="EURUSD",
    require_candlestick_pattern=True, # Must see candlestick patterns
    enforce_risk_management=True      # Enforce all RM rules
)

train_env, eval_env = trainer.prepare_training_data(candles, valid_setups)
```

## How RL Learns

### Training Loop

1. **Agent receives state** (20 TCE features + ML probability + 9 context features)
2. **Agent selects action** (Enter/Wait/Exit/Trail Stop)
3. **Action is validated** against constraints:
   - ✓ Candlestick pattern confirmed?
   - ✓ Risk management rules respected?
   - ✓ Position size properly calculated?
4. **Trade executes** in backtesting environment
5. **Reward calculated** as R-multiple
6. **Agent learns** to maximize long-term R-multiples

### What the Agent Learns to Optimize

- **Entry selectivity**: Wait for high-probability setups (high ML probability)
- **Position sizing**: Size positions based on SL distance and account risk
- **Exit timing**: Take profits at optimal time vs let TP/SL hit
- **Stop loss trailing**: Lock in profits when in winning trades

### What the Agent DOESN'T Learn

- **Strategy discovery**: Finding valid setups (TCE validator does this)
- **Ignoring candlestick patterns**: Can't enter without pattern confirmation
- **Excessive risk**: Position sizes are capped by risk % rule
- **Bad R:R ratios**: Can't take trades with TP < SL distance

## State Space

The 30-dimensional state includes:

**TCE Features (20 dims):**
- MA distances (normalized by ATR)
- MA slopes (trend strength)
- Fibonacci level
- Market structure (trend consistency)
- Volatility
- Correlation alignment

**ML Probability (1 dim):**
- P(success) from deep learning model
- Ranges [0, 1]

**Context Features (9 dims):**
- In-trade flag
- Position size
- Unrealized R-multiple
- Account % change
- Trade count
- Win rate (recent)
- Average R-multiple (recent)
- Risk/Reward ratio
- Data progress

## Example: Training with Risk Management

```python
# Prepare data
candles_df = get_historical_data("EURUSD", "2023-01-01", "2023-12-31")
valid_setups = validate_tce_setups(candles_df)

# Create trainer with RM enabled
trainer = RLExecutionTrainer(
    model_name="tce_ppo_with_rm",
    initial_balance=10000,
    risk_percentage=1.0,
    symbol="EURUSD",
    require_candlestick_pattern=True,
    enforce_risk_management=True
)

# Prepare environments
train_env, eval_env = trainer.prepare_training_data(candles_df, valid_setups)

# Train agent
metrics = trainer.train(
    train_env,
    eval_env=eval_env,
    total_timesteps=100000,
    eval_freq=5000
)

# Evaluate
eval_results = trainer.agent.evaluate(eval_env, n_episodes=50)
print(f"Mean R-multiple: {eval_results['mean_r_multiple']:.2f}R")
print(f"Win rate: {eval_results['mean_win_rate']:.2%}")
print(f"Mean reward: {eval_results['mean_reward']:.2f}")
```

## Key Metrics to Monitor

During training, watch these metrics:

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| Mean R-multiple | > 1.0R | Profitable trading |
| Win rate | > 50% | More wins than losses |
| Mean reward | > 0 | Positive expected value |
| Avg drawdown | < 10-15% | Risk control working |
| Trades per episode | Varies | Entry selectivity (not too many trades) |

## Common Issues & Solutions

### Issue: Agent takes many trades with low win rate
**Solution:** Increase `require_candlestick_pattern=True` strictness or increase ML probability threshold in setup selection.

### Issue: Agent positions too large
**Solution:** Decrease `risk_percentage` (e.g., 0.5% instead of 1%) or ensure `enforce_risk_management=True`.

### Issue: Many negative R-multiples
**Solution:** Improve TCE setup validation (strategy validator), or increase ML model accuracy.

### Issue: Agent won't enter trades
**Solution:** Ensure candlestick patterns are actually appearing in setups, or set `require_candlestick_pattern=False` for debugging.

## Integration with Other Components

```
TCE Validator (finds valid setups)
         ↓
Feature Engineering (20 features)
         ↓
Deep Learning Model (predicts P(success))
         ↓
RL Environment (executes with RM)
         ↓
RL Agent (optimizes entry/exit/sizing)
```

The RL agent receives already-validated setups from the TCE validator and focuses purely on execution optimization.
