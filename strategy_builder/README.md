# No-Code Strategy Builder App

Complete system for users to create, train, and automate trading strategies using natural language.

## 🎯 Overview

The **Strategy Builder** app allows users to:
1. Describe their trading strategy in plain English
2. System automatically parses and validates the strategy
3. Converts to rule-based logic
4. Trains ML/RL models on historical data
5. Backtests the strategy
6. Deploys for live/demo trading

---

## 📁 Structure

```
strategy_builder/
├── models.py              # Database models
├── nlp/
│   └── parser.py          # Natural language parser
├── rule_engine/           # Rule execution engine
├── ml_training/           # ML model training
├── rl_training/           # Reinforcement learning
├── backtest/              # Backtesting engine
└── executor/              # Live trade execution
```

---

## 📊 Database Models

### 1. **UserStrategy**
- User's trading strategy with natural language description
- Status tracking (draft → parsing → validated → trained → live)
- Performance metrics

### 2. **StrategyComponent**
- Individual rules (entry, exit, filters)
- Parsed rule representation
- Logic operators (AND/OR)

### 3. **StrategyIndicator**
- Technical indicators used
- Parameters and settings

### 4. **StrategyMLModel**
- Trained ML model metadata
- Feature engineering details
- Performance metrics

### 5. **StrategyBacktest**
- Backtest results
- Performance statistics
- Equity curves

### 6. **StrategyTrade**
- Executed trades
- Entry/exit details
- P&L tracking

### 7. **ParsedCondition**
- Library of reusable trading logic patterns
- Pattern matching rules
- Implementation templates

---

## 🧠 NLP Parser

Converts natural language to structured rules.

### Example Input:
```
"Buy when price crosses above the 20-period MA and RSI is below 30. 
Exit when price hits 2:1 risk-reward or RSI goes above 70. 
Use 1.5 ATR for stop loss. Only trade in uptrends on higher timeframe."
```

### Parsed Output:
```python
{
    'indicators': [
        {'name': 'MA', 'parameters': {'period': 20}},
        {'name': 'RSI', 'parameters': {'period': 14}},
        {'name': 'ATR', 'parameters': {'period': 14}}
    ],
    'entry_conditions': [
        {'type': 'cross_above', 'variables': {'indicator': 'MA20'}},
        {'type': 'rsi_oversold', 'variables': {'threshold': 30}}
    ],
    'exit_conditions': [
        {'type': 'risk_reward', 'variables': {'risk': 1, 'reward': 2}},
        {'type': 'rsi_overbought', 'variables': {'threshold': 70}}
    ],
    'filters': [
        {'type': 'trend', 'direction': 'up'},
        {'type': 'higher_timeframe', 'check': 'alignment'}
    ],
    'risk_management': {
        'stop_loss': {'value': '1.5', 'unit': 'ATR'}
    }
}
```

---

## 🔄 Workflow

### Phase 1: Strategy Creation
1. User describes strategy in natural language
2. NLP parser extracts:
   - Indicators
   - Entry conditions
   - Exit conditions
   - Filters
   - Risk management
3. System validates completeness
4. Generates rule-based logic

### Phase 2: Data Collection
1. Identify required indicators
2. Fetch historical data from MT5
3. Calculate indicators
4. Label outcomes (win/loss)

### Phase 3: ML Training
1. Extract features from setups
2. Train Deep Neural Network
3. Optimize with Reinforcement Learning
4. Validate on out-of-sample data

### Phase 4: Backtesting
1. Run strategy on historical data
2. Calculate performance metrics
3. Generate equity curve
4. Compare with/without ML filter

### Phase 5: Deployment
1. Activate strategy for live trading
2. Monitor performance
3. Continuous learning/adaptation

---

## 🚀 Usage Example

```python
from strategy_builder.nlp.parser import parse_strategy_description

# User describes strategy
description = """
Buy when the 50 EMA crosses above the 200 EMA and 
RSI is between 40 and 60. Use 20-pip stop loss and 
40-pip take profit. Only trade EURUSD on H1 timeframe 
during uptrends.
"""

# Parse description
parsed = parse_strategy_description(description)

# Create user strategy
strategy = UserStrategy.objects.create(
    user=user,
    name="EMA Crossover Strategy",
    description=description,
    parsed_rules=parsed,
    symbols=['EURUSD'],
    timeframes=['H1']
)

# Train ML model
train_strategy_ml(strategy)

# Backtest
backtest_results = backtest_strategy(strategy)

# Deploy if profitable
if backtest_results['win_rate'] > 55:
    strategy.status = 'live'
    strategy.is_active = True
    strategy.save()
```

---

## 🎯 Supported Trading Concepts

### Indicators
- Moving Averages (MA, EMA, SMA)
- RSI, MACD, Stochastic
- Bollinger Bands
- ATR, ADX
- Volume indicators

### Conditions
- Price crosses above/below indicator
- Indicator crosses
- RSI overbought/oversold
- MACD crossovers
- Bollinger Band touches
- Trend confirmation
- Higher timeframe alignment

### Risk Management
- Fixed pip stop loss/take profit
- ATR-based stops
- Risk:Reward ratios
- Trailing stops
- Position sizing

### Filters
- Trend direction
- Time of day
- Day of week
- Volatility filters
- Higher timeframe confirmation

---

## 🔧 Next Steps

1. Build rule execution engine
2. Create feature engineering module
3. Implement ML training pipeline
4. Build RL optimization
5. Create backtesting engine
6. Build live execution system

Ready to handle diverse trading strategies! 🚀
