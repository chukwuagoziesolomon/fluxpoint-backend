# 🧠 ML/RL Architecture for No-Code Strategy Builder

## 🎯 Core Concept

The ML/RL pipeline for user-defined strategies works **exactly like TCE**, but dynamically adapts to user indicators.

### **Two-Layer AI System:**

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Deep Learning (Probability Filter)        │
│  ─────────────────────────────────────────────────  │
│  ✅ Trains on: Historical setups from user strategy │
│  ✅ Input: Features from user's indicators          │
│  ✅ Output: P(success) for each setup               │
│  ✅ Purpose: Filter out low-probability setups      │
│  ✅ Threshold: Only trade if P >= 0.65              │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Reinforcement Learning (Execution)        │
│  ─────────────────────────────────────────────────  │
│  ✅ Trains on: Valid setups (from Layer 1)          │
│  ✅ Actions: Enter full, enter half, wait, exit     │
│  ✅ State: ML probability + market context          │
│  ✅ Reward: R-multiple (TP/SL ratio)                │
│  ✅ Purpose: Optimize when/how to execute           │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Layer 1: Deep Learning (Probability Prediction)

### **Purpose:**
Learn which setups from the user's strategy are likely to succeed.

### **Training Data Generation:**

```python
# Example: User describes strategy
description = """
Buy when:
- Price crosses above 50 EMA
- RSI is between 40-60
- MACD histogram is positive
- Higher timeframe is bullish

Exit when:
- 2:1 risk-reward hit
- OR RSI goes above 80

Stop loss: 1.5 ATR below entry
"""

# System workflow:
# 1. Parse description → structured rules
# 2. Fetch historical data (MT5)
# 3. Calculate user's indicators (EMA50, RSI, MACD, ATR)
# 4. Scan history for valid setups
# 5. Label each setup: 1 if TP hit first, 0 if SL hit first
# 6. Extract features for ML training
```

### **Feature Engineering (Auto-Generated):**

Features are **automatically created** from the user's indicators:

```python
# If user uses EMA50:
features.extend([
    'ema50',
    'ema50_slope',
    'distance_to_ema50',
    'price_above_ema50'
])

# If user uses RSI14:
features.extend([
    'rsi14',
    'rsi14_slope',
    'rsi_oversold',
    'rsi_overbought'
])

# If user uses MACD:
features.extend([
    'macd_line',
    'macd_signal',
    'macd_histogram',
    'macd_cross'
])

# Always add market context:
features.extend([
    'volatility',
    'trend_strength',
    'candle_body_size',
    'volume_ratio'
])
```

### **Model Architecture (Same as TCE):**

```python
class UserStrategyMLModel(nn.Module):
    """
    Generic DNN for any user strategy.
    Adapts input size based on user's indicators.
    """
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output: probability [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)
```

### **Training Process:**

```python
# Same as TCE training (CELL4):
1. Extract valid setups from historical data
2. Calculate features from user's indicators
3. Normalize features (StandardScaler)
4. Train DNN with BCE loss
5. Validate on holdout set
6. Save model for inference
```

### **Inference (Live Trading):**

```python
# When user strategy signals a trade:
def should_take_trade(setup_features):
    """
    Filter trade with ML model.
    
    Args:
        setup_features: Features from user's strategy
    
    Returns:
        True if probability >= 0.65
    """
    # Load user's trained model
    model = load_user_model(strategy_id)
    
    # Normalize features
    features_normalized = scaler.transform([setup_features])
    
    # Get probability
    probability = model.predict(features_normalized)[0]
    
    # Only trade if high confidence
    if probability >= 0.65:
        print(f"✅ ML Filter: PASS (P={probability:.2%})")
        return True
    else:
        print(f"❌ ML Filter: SKIP (P={probability:.2%})")
        return False
```

---

## 🎮 Layer 2: Reinforcement Learning (Execution Optimization)

### **Purpose:**
Learn **when** and **how** to execute trades that passed the ML filter.

**RL does NOT find strategies** - it optimizes execution of VALID setups.

### **State Space (What RL Sees):**

```python
state = [
    # 1. ML Probability (from Layer 1)
    ml_probability,                  # 0.65 - 1.0
    
    # 2. User Strategy Signals
    entry_signal_strength,           # How strong is the setup?
    time_since_last_trade,           # Avoid overtrading
    open_positions_count,            # Risk management
    
    # 3. Market Context
    volatility,                      # Current ATR
    trend_strength,                  # ADX or MA alignment
    time_of_day,                     # Session timing
    spread,                          # Trading cost
    
    # 4. Account State
    account_balance,
    current_drawdown,
    win_streak,
    loss_streak,
    
    # 5. Trade Context (if in position)
    current_profit,                  # Unrealized P&L
    time_in_trade,                   # Holding duration
    distance_to_tp,                  # How close to TP?
    distance_to_sl,                  # How close to SL?
]

# Total: ~25-30 state dimensions
```

### **Action Space:**

```python
actions = {
    0: "Enter full position",        # Risk 1% of account
    1: "Enter half position",        # Risk 0.5% (conservative)
    2: "Wait (skip setup)",          # Don't trade this time
    3: "Exit trade",                 # Close position early
    4: "Trail stop loss",            # Move SL to breakeven/profit
}
```

### **Reward Function:**

```python
def calculate_reward(action, outcome, risk_multiple):
    """
    Reward based on R-multiples (not raw dollars).
    
    R-multiple = Profit/Loss relative to initial risk.
    Example: 
    - SL hit = -1R
    - TP hit (2:1) = +2R
    """
    
    if action == "wait" and outcome == "would_have_lost":
        # Avoided a losing trade
        return +0.5
    
    elif action == "wait" and outcome == "would_have_won":
        # Missed a winning trade
        return -0.3
    
    elif action == "enter_full":
        if outcome == "win":
            return +risk_multiple  # e.g., +2.0 for 2:1 RR
        else:
            return -1.0  # Lost 1R
    
    elif action == "enter_half":
        if outcome == "win":
            return +(risk_multiple * 0.5)  # Half reward
        else:
            return -0.5  # Half loss
    
    elif action == "trail_stop":
        if outcome == "breakeven_exit":
            return +0.3  # Protected capital
        elif outcome == "stopped_out_in_profit":
            return +0.5  # Locked in profit
    
    elif action == "exit_early":
        if outcome == "avoided_loss":
            return +0.8  # Smart exit
        elif outcome == "exited_winning_trade":
            return -0.2  # Premature exit
    
    return 0.0
```

### **RL Algorithm: PPO (Proximal Policy Optimization)**

```python
from stable_baselines3 import PPO

# Create custom gym environment
env = UserStrategyTradingEnv(
    strategy_id=strategy.id,
    ml_model=user_ml_model,
    data=historical_data
)

# Train PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,  # Discount factor
    verbose=1
)

# Train for 100k steps
model.learn(total_timesteps=100_000)

# Save trained agent
model.save(f"rl_agent_strategy_{strategy.id}.zip")
```

### **RL Training Process:**

```python
# Episode loop:
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # 1. Get ML probability for current setup
        ml_prob = ml_model.predict(current_features)
        
        # 2. RL agent decides action
        action = rl_agent.predict(state, ml_prob)
        
        # 3. Execute action in environment
        next_state, reward, done, info = env.step(action)
        
        # 4. Store experience
        episode_reward += reward
        state = next_state
    
    # 5. Update policy after episode
    rl_agent.update()
```

### **Inference (Live Trading with RL):**

```python
# Real-time execution:
def execute_trade_with_rl(user_strategy, current_candle):
    """
    Use ML + RL to decide if/how to trade.
    """
    
    # 1. Check if user's strategy signals entry
    entry_signal = evaluate_entry_conditions(
        current_candle,
        user_strategy.parsed_rules
    )
    
    if not entry_signal:
        return None  # No setup
    
    # 2. Extract features from user's indicators
    features = extract_features(
        current_candle,
        user_strategy.indicators
    )
    
    # 3. Get ML probability
    ml_prob = ml_model.predict([features])[0]
    
    if ml_prob < 0.65:
        print(f"❌ ML Filter: Skip (P={ml_prob:.2%})")
        return None
    
    # 4. Build state for RL agent
    state = build_rl_state(
        ml_prob=ml_prob,
        features=features,
        account_state=get_account_state(),
        market_context=get_market_context()
    )
    
    # 5. RL agent decides action
    action = rl_agent.predict(state)
    
    # 6. Execute based on RL decision
    if action == 0:  # Enter full
        position_size = calculate_position_size(
            risk_pct=user_strategy.risk_percentage,
            stop_loss=calculate_stop_loss(features)
        )
        place_order(symbol, position_size, stop_loss, take_profit)
        print(f"✅ Entered FULL position (ML: {ml_prob:.2%})")
    
    elif action == 1:  # Enter half
        position_size = calculate_position_size(
            risk_pct=user_strategy.risk_percentage * 0.5
        )
        place_order(symbol, position_size, stop_loss, take_profit)
        print(f"✅ Entered HALF position (ML: {ml_prob:.2%})")
    
    elif action == 2:  # Wait
        print(f"⏸️  RL: Wait (setup not optimal)")
    
    # ... handle other actions
```

---

## 🔄 Complete Flow: User Strategy → Live Trading

```
1. USER INPUT
   "Buy when price crosses 50 EMA and RSI < 30..."
   
2. NLP PARSING
   → Structured rules + indicators
   
3. DATA COLLECTION
   → Fetch historical data from MT5
   → Calculate user's indicators
   
4. LABEL GENERATION
   → Scan history for valid setups
   → Label: 1 if TP hit first, 0 if SL hit
   
5. FEATURE ENGINEERING
   → Auto-generate features from user indicators
   → Normalize features
   
6. ML TRAINING (Layer 1)
   → Train DNN to predict P(success)
   → Save model
   
7. RL TRAINING (Layer 2)
   → Train PPO agent on valid setups
   → Optimize entry timing, position sizing
   → Save agent
   
8. BACKTESTING
   → Test ML + RL on holdout data
   → Show performance metrics
   
9. LIVE TRADING
   → Monitor market for user's signals
   → Filter with ML (P >= 0.65)
   → Execute with RL optimization
   → Track performance
```

---

## 📊 Key Differences: TCE vs User Strategy

| Aspect | TCE Strategy | User Strategy |
|--------|-------------|---------------|
| **Rules** | Hardcoded (validation.py) | Dynamic (parsed from NLP) |
| **Indicators** | Fixed (MA6/18/50/200) | User-defined (any combo) |
| **Features** | 20 hardcoded features | Auto-generated from user indicators |
| **ML Model** | Single model for TCE | One model per user strategy |
| **RL Agent** | Single agent for TCE | One agent per user strategy |
| **Training Data** | TCE setups only | User strategy setups only |
| **Complexity** | Lower (1 strategy) | Higher (N strategies) |

---

## 🛠️ Implementation Plan

### **Phase 1: ML Training for User Strategies**

1. **Data Collection Module**
   ```python
   def collect_training_data(strategy):
       # Fetch historical data
       # Calculate user's indicators
       # Scan for valid setups
       # Label outcomes
       return X_train, y_train
   ```

2. **Feature Extractor**
   ```python
   def extract_features_from_user_indicators(strategy, candle):
       # Auto-generate features based on user's indicators
       # Normalize features
       return feature_vector
   ```

3. **ML Trainer**
   ```python
   def train_user_ml_model(strategy):
       # Build model architecture
       # Train with BCE loss
       # Validate performance
       # Save model
       return trained_model
   ```

### **Phase 2: RL Training for User Strategies**

1. **Gym Environment**
   ```python
   class UserStrategyEnv(gym.Env):
       def __init__(self, strategy, ml_model):
           self.strategy = strategy
           self.ml_model = ml_model
       
       def step(self, action):
           # Execute action
           # Calculate reward
           # Return next state
       
       def reset(self):
           # Reset to start of episode
   ```

2. **RL Trainer**
   ```python
   def train_user_rl_agent(strategy, ml_model):
       # Create environment
       # Initialize PPO agent
       # Train for N episodes
       # Save agent
       return trained_agent
   ```

### **Phase 3: Live Execution**

1. **Signal Generator**
   ```python
   def generate_signals(strategy, current_data):
       # Evaluate entry conditions
       # Extract features
       # Filter with ML
       # Execute with RL
   ```

2. **Trade Executor**
   ```python
   def execute_trade(strategy, signal):
       # Calculate position size
       # Place order on MT5
       # Log trade
       # Monitor position
   ```

---

## 💡 Key Insights

### **Why This Works:**

1. **ML Layer** = Pattern recognition from history
   - "This setup has 70% win rate historically"

2. **RL Layer** = Adaptive execution
   - "Even good setups need proper timing"
   - "Sometimes enter smaller position"
   - "Sometimes wait for better confirmation"

3. **User Flexibility** = Any strategy works
   - System adapts to user's indicators
   - Auto-generates appropriate features
   - Trains custom ML + RL models

### **Resource Optimization:**

- **Training:** Google Colab Pro ($10/month)
- **Inference:** CPU-based VPS ($40/month)
- **Per-User Models:** Isolated, no interference
- **Scalability:** Queue-based training (Celery)

---

## 🎯 Next Steps

1. ✅ **Rule execution engine** (done - see rule_engine/)
2. ⬜ **Data collection pipeline** (fetch MT5 data for user strategy)
3. ⬜ **Feature engineering** (auto-generate from user indicators)
4. ⬜ **ML training pipeline** (train DNN for user strategy)
5. ⬜ **RL training pipeline** (train PPO for user strategy)
6. ⬜ **Live execution** (integrate ML + RL)

---

**The architecture is EXACTLY like TCE, just made generic to support any user-defined strategy!** 🚀
