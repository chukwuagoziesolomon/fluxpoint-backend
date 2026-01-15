"""
Complete Example: User Strategy from Description to Live Trading

This shows the ENTIRE flow:
1. User describes strategy
2. System parses it
3. Generates executable code
4. Trains ML model
5. Trains RL agent
6. Executes live trades
"""

from strategy_builder.workflow import NoCodeStrategyBuilder
from strategy_builder.rule_engine.indicators import IndicatorCalculator
from strategy_builder.rule_engine.evaluator import RuleEvaluator
import pandas as pd
import numpy as np


# ============================================================================
# EXAMPLE 1: Simple RSI Strategy
# ============================================================================

def example_simple_rsi_strategy():
    """
    User describes: "Buy when RSI < 30, sell when RSI > 70"
    """
    
    print("="*80)
    print("EXAMPLE 1: SIMPLE RSI STRATEGY")
    print("="*80)
    
    # User input
    description = """
    Buy when RSI is below 30 (oversold).
    Sell when RSI goes above 70 (overbought).
    Use 20-pip stop loss and 40-pip take profit.
    Trade EURUSD on H1 timeframe.
    """
    
    # Step 1: Create strategy
    builder = NoCodeStrategyBuilder(user_id=1)
    result = builder.create_strategy(
        description=description,
        name="Simple RSI Strategy"
    )
    
    print(f"\n✅ Strategy created: {result['strategy_id']}")
    print(f"Status: {result['status']}")
    
    # Step 2: Parse result shows structured rules
    parsed_rules = result['parsed_rules']
    print(f"\n📋 Parsed Indicators:")
    for ind in parsed_rules['indicators']:
        print(f"  - {ind['name']}: {ind['parameters']}")
    
    print(f"\n📋 Entry Conditions:")
    for cond in parsed_rules['entry_conditions']:
        print(f"  - {cond['type']}: {cond.get('variables', {})}")
    
    # Step 3: How to execute these rules on live data
    print(f"\n🔄 EXECUTION EXAMPLE:")
    print("─" * 80)
    
    # Simulate historical data
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 1.1000,
        'high': np.random.randn(500).cumsum() + 1.1010,
        'low': np.random.randn(500).cumsum() + 1.0990,
        'close': np.random.randn(500).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    # Calculate indicators
    indicator_calc = IndicatorCalculator()
    df = indicator_calc.calculate_all(df, parsed_rules['indicators'])
    
    print(f"✅ Indicators calculated: {df.columns.tolist()}")
    
    # Evaluate rules on each candle
    evaluator = RuleEvaluator()
    
    valid_setups = []
    for i in range(50, len(df)):  # Need history for indicators
        is_valid, reason = evaluator.evaluate_entry_conditions(
            df=df,
            row_idx=i,
            entry_conditions=parsed_rules['entry_conditions'],
            operator='AND'
        )
        
        if is_valid:
            valid_setups.append({
                'timestamp': df.iloc[i]['timestamp'],
                'price': df.iloc[i]['close'],
                'rsi': df.iloc[i]['rsi14'],
                'reason': reason
            })
    
    print(f"\n✅ Found {len(valid_setups)} valid entry setups")
    
    if valid_setups:
        print(f"\n📊 First 3 setups:")
        for setup in valid_setups[:3]:
            print(f"  {setup['timestamp']}: Price={setup['price']:.5f}, "
                  f"RSI={setup['rsi']:.1f} - {setup['reason']}")
    
    return result


# ============================================================================
# EXAMPLE 2: Complex Multi-Indicator Strategy
# ============================================================================

def example_complex_ma_rsi_macd_strategy():
    """
    User describes complex strategy with multiple indicators
    """
    
    print("\n" + "="*80)
    print("EXAMPLE 2: COMPLEX MA + RSI + MACD STRATEGY")
    print("="*80)
    
    description = """
    Buy when:
    - Price crosses above the 50-period EMA
    - RSI is between 40 and 60 (not oversold, but recovering)
    - MACD histogram turns positive
    - Higher timeframe (H4) is in uptrend (price above 200 MA)
    
    Exit when:
    - Price hits 2:1 risk-reward ratio
    - OR RSI goes above 80 (overbought)
    
    Stop loss: 1.5 times ATR below entry
    Position size: Risk 1% of account per trade
    
    Trade GBPUSD on H1 timeframe only during London and New York sessions.
    """
    
    builder = NoCodeStrategyBuilder(user_id=1)
    result = builder.create_strategy(
        description=description,
        name="EMA RSI MACD Combo Strategy"
    )
    
    print(f"\n✅ Strategy created: {result['strategy_id']}")
    
    # Show parsed structure
    parsed = result['parsed_rules']
    
    print(f"\n📋 Indicators Required:")
    for ind in parsed['indicators']:
        print(f"  - {ind['name']}{ind['parameters'].get('period', '')}")
    
    print(f"\n📋 Entry Conditions (must ALL be true):")
    for i, cond in enumerate(parsed['entry_conditions'], 1):
        print(f"  {i}. {cond['type']}")
    
    print(f"\n📋 Exit Conditions (ANY can trigger):")
    for i, cond in enumerate(parsed['exit_conditions'], 1):
        print(f"  {i}. {cond['type']}")
    
    print(f"\n📋 Risk Management:")
    for key, value in parsed.get('risk_management', {}).items():
        print(f"  - {key}: {value}")
    
    print(f"\n📋 Filters:")
    for filt in parsed.get('filters', []):
        print(f"  - {filt['type']}: {filt.get('direction', filt.get('check', ''))}")
    
    # Show auto-generated features for ML
    print(f"\n🧠 Auto-Generated ML Features:")
    features = result.get('feature_config', {}).get('features', [])
    print(f"  Total features: {len(features)}")
    print(f"  Sample features: {features[:10]}")
    
    return result


# ============================================================================
# EXAMPLE 3: How ML Training Works
# ============================================================================

def example_ml_training_pipeline():
    """
    Show how ML model is trained on user's strategy
    """
    
    print("\n" + "="*80)
    print("EXAMPLE 3: ML TRAINING PIPELINE")
    print("="*80)
    
    # Assume we have a strategy
    strategy_id = 1
    
    print("\n📊 STEP 1: Collect Historical Setups")
    print("─" * 80)
    print("1. Fetch historical data from MT5 for user's symbols/timeframes")
    print("2. Calculate user's indicators (EMA50, RSI14, MACD, etc.)")
    print("3. Scan each candle to find valid entry signals")
    print("4. For each setup, track outcome:")
    print("   - Label = 1 if TP hit first")
    print("   - Label = 0 if SL hit first")
    
    # Simulate collected data
    print("\n📊 Example Training Data:")
    print("─" * 80)
    print("Setup #1: 2024-01-15 10:00")
    print("  Price: 1.0950, EMA50: 1.0940, RSI: 45, MACD: +0.0003")
    print("  Outcome: TP hit after 12 hours → Label = 1 (WIN)")
    print()
    print("Setup #2: 2024-01-15 18:00")
    print("  Price: 1.0970, EMA50: 1.0960, RSI: 52, MACD: +0.0001")
    print("  Outcome: SL hit after 3 hours → Label = 0 (LOSS)")
    print()
    print("... (collect 500-1000+ setups)")
    
    print("\n🧠 STEP 2: Extract Features")
    print("─" * 80)
    print("Auto-generate features from user's indicators:")
    features = [
        'ema50', 'ema50_slope', 'distance_to_ema50',
        'rsi14', 'rsi14_slope', 'rsi_oversold', 'rsi_overbought',
        'macd_line', 'macd_signal', 'macd_histogram',
        'atr14', 'volatility', 'trend_strength',
        'candle_body_size', 'candle_wick_ratio'
    ]
    print(f"Features: {features}")
    print(f"Total: {len(features)} features")
    
    print("\n🏋️ STEP 3: Train Neural Network")
    print("─" * 80)
    print("Architecture:")
    print("  Input: 15 features (normalized)")
    print("  Hidden: [128 → 64 → 32] with dropout")
    print("  Output: 1 (probability [0,1])")
    print()
    print("Training:")
    print("  Loss: Binary Cross Entropy")
    print("  Optimizer: Adam (lr=0.001)")
    print("  Epochs: 50 (with early stopping)")
    print("  Validation: 20% holdout")
    print()
    print("Expected output:")
    print("  Epoch 50/50: Loss=0.42, Accuracy=68%, Val Loss=0.45")
    print("  ✅ Model saved: user_strategy_1_ml.pt")
    
    print("\n🎯 STEP 4: Inference (Live Trading)")
    print("─" * 80)
    print("When new setup appears:")
    print("  1. Extract features from current candle")
    print("  2. Feed to ML model")
    print("  3. Get probability: P(success)")
    print("  4. If P >= 0.65 → TRADE")
    print("  5. If P < 0.65 → SKIP")
    print()
    print("Example:")
    print("  Setup at 2024-01-20 14:00")
    print("  Features: [1.0955, 0.0002, 0.0015, 48.5, -0.3, ...]")
    print("  ML Prediction: P = 0.72 (72% win probability)")
    print("  Decision: ✅ TAKE TRADE")


# ============================================================================
# EXAMPLE 4: How RL Optimizes Execution
# ============================================================================

def example_rl_execution_optimization():
    """
    Show how RL agent optimizes trade execution
    """
    
    print("\n" + "="*80)
    print("EXAMPLE 4: RL EXECUTION OPTIMIZATION")
    print("="*80)
    
    print("\n🎮 What RL Does:")
    print("─" * 80)
    print("RL does NOT find strategies (that's the user's job)")
    print("RL optimizes EXECUTION of valid setups")
    print()
    print("Key Questions RL Answers:")
    print("  1. Should I enter this trade NOW or wait?")
    print("  2. Should I enter full position or partial?")
    print("  3. When should I trail my stop loss?")
    print("  4. Should I exit early based on market context?")
    
    print("\n📊 RL State (What it sees):")
    print("─" * 80)
    state_components = [
        "ML probability (0.65-1.0)",
        "Entry signal strength",
        "Time since last trade",
        "Current open positions",
        "Market volatility (ATR)",
        "Trend strength (ADX)",
        "Time of day / session",
        "Current spread",
        "Account balance",
        "Current drawdown",
        "Win/loss streak",
        "Unrealized P&L (if in trade)",
        "Time in trade",
        "Distance to TP/SL"
    ]
    for i, comp in enumerate(state_components, 1):
        print(f"  {i}. {comp}")
    
    print("\n🎯 RL Actions:")
    print("─" * 80)
    actions = [
        "Enter full position (1% risk)",
        "Enter half position (0.5% risk)",
        "Wait / Skip setup",
        "Exit trade early",
        "Trail stop loss to breakeven"
    ]
    for i, action in enumerate(actions):
        print(f"  {i}. {action}")
    
    print("\n💰 RL Reward (R-multiples):")
    print("─" * 80)
    print("Scenario 1: Enter full, trade wins 2:1")
    print("  Reward: +2.0R")
    print()
    print("Scenario 2: Enter full, trade loses")
    print("  Reward: -1.0R")
    print()
    print("Scenario 3: Wait, trade would have won")
    print("  Reward: -0.3R (missed opportunity)")
    print()
    print("Scenario 4: Wait, trade would have lost")
    print("  Reward: +0.5R (avoided loss!)")
    print()
    print("Scenario 5: Enter half, trade wins")
    print("  Reward: +1.0R (half of 2:1)")
    print()
    print("Scenario 6: Trail SL, hit at breakeven")
    print("  Reward: +0.3R (protected capital)")
    
    print("\n🏋️ RL Training Process:")
    print("─" * 80)
    print("1. Start with random policy")
    print("2. For each valid setup (ML prob >= 0.65):")
    print("   - Observe state")
    print("   - Choose action (exploration vs exploitation)")
    print("   - Execute trade or wait")
    print("   - Get reward (R-multiple)")
    print("   - Update policy with PPO")
    print("3. After 100k steps:")
    print("   - Agent learns optimal execution timing")
    print("   - Learns when to be aggressive vs conservative")
    print("   - Learns to avoid overtrading")
    print("   - Learns proper stop loss management")
    
    print("\n🎯 Example: RL in Action")
    print("─" * 80)
    print("Setup #1: ML prob = 0.68, volatility HIGH, 3 positions open")
    print("  RL Decision: WAIT (too many open positions)")
    print()
    print("Setup #2: ML prob = 0.78, volatility NORMAL, 0 positions, fresh trend")
    print("  RL Decision: ENTER FULL (optimal conditions)")
    print()
    print("Setup #3: ML prob = 0.66, volatility HIGH, end of session")
    print("  RL Decision: ENTER HALF (borderline setup)")
    print()
    print("In Trade: 30 pips profit, volatility increasing, late in trade")
    print("  RL Decision: TRAIL STOP to +10 pips (lock profit)")


# ============================================================================
# EXAMPLE 5: Complete End-to-End Example
# ============================================================================

def example_complete_end_to_end():
    """
    Show complete flow from user input to live trade
    """
    
    print("\n" + "="*80)
    print("EXAMPLE 5: COMPLETE END-TO-END WORKFLOW")
    print("="*80)
    
    print("\n" + "─" * 80)
    print("USER ACTION:")
    print("─" * 80)
    description = """
    Buy when 20 EMA crosses above 50 EMA and RSI is between 40-60.
    Sell when RSI > 75 or 2:1 profit target hit.
    Stop loss: 1.5 ATR.
    """
    print(f'User types: "{description}"')
    
    print("\n" + "─" * 80)
    print("SYSTEM RESPONSE (Automatic):")
    print("─" * 80)
    
    print("\n[1/8] 🧠 NLP Parsing...")
    print("  ✅ Identified indicators: EMA20, EMA50, RSI14, ATR14")
    print("  ✅ Identified entry: EMA cross + RSI range")
    print("  ✅ Identified exit: RSI > 75 OR 2:1 RR")
    print("  ✅ Identified risk: 1.5 ATR stop loss")
    
    print("\n[2/8] 💾 Saving to Database...")
    print("  ✅ Created UserStrategy record (ID: 42)")
    print("  ✅ Created StrategyIndicator records (4 indicators)")
    print("  ✅ Created StrategyComponent records (entry/exit rules)")
    print("  ✅ Status: 'ready_for_training'")
    
    print("\n[3/8] 📊 Fetching Historical Data...")
    print("  ✅ Downloading EURUSD H1 data (2023-2024)")
    print("  ✅ Retrieved 8,760 candles")
    print("  ✅ Calculating EMA20, EMA50, RSI14, ATR14...")
    
    print("\n[4/8] 🔍 Scanning for Valid Setups...")
    print("  ✅ Found 287 valid entry signals")
    print("  ✅ Tracked outcomes (TP vs SL)")
    print("  ✅ Win rate: 62% (178 wins, 109 losses)")
    
    print("\n[5/8] 🧠 Training ML Model...")
    print("  ✅ Extracted 12 features per setup")
    print("  ✅ Training DNN [128→64→32→1]")
    print("  ✅ Epoch 50/50: Val Accuracy 67%")
    print("  ✅ Model saved: strategy_42_ml.pt")
    
    print("\n[6/8] 🎮 Training RL Agent...")
    print("  ✅ Created trading environment")
    print("  ✅ Training PPO agent (100k steps)")
    print("  ✅ Final reward: +24.5R over 150 trades")
    print("  ✅ Agent saved: strategy_42_rl.zip")
    
    print("\n[7/8] 📈 Backtesting...")
    print("  ✅ Testing on 2024 data (holdout)")
    print("  ✅ Win rate: 64% (87 trades)")
    print("  ✅ Profit factor: 1.82")
    print("  ✅ Max drawdown: 4.2%")
    print("  ✅ Sharpe ratio: 1.45")
    
    print("\n[8/8] 🚀 Ready for Live Trading!")
    print("  ✅ Strategy validated")
    print("  ✅ ML model loaded")
    print("  ✅ RL agent loaded")
    print("  ✅ Monitoring EURUSD H1...")
    
    print("\n" + "─" * 80)
    print("LIVE TRADING EXAMPLE:")
    print("─" * 80)
    
    print("\n⏰ 2024-01-20 14:00 - New H1 candle closed")
    print("  📊 Price: 1.0965")
    print("  📊 EMA20: 1.0962, EMA50: 1.0955")
    print("  📊 RSI: 48.5")
    print("  📊 ATR: 0.0015 (15 pips)")
    
    print("\n  🔍 Checking entry conditions...")
    print("  ✅ EMA20 > EMA50 (recent cross)")
    print("  ✅ RSI in range [40, 60]")
    print("  ✅ Entry conditions MET!")
    
    print("\n  🧠 ML Filter...")
    print("  📊 Extracted features: [1.0965, 0.0003, 0.001, 48.5, -0.2, ...]")
    print("  🤖 ML Prediction: P(success) = 0.74 (74%)")
    print("  ✅ Probability >= 0.65 → PASS ML FILTER")
    
    print("\n  🎮 RL Execution Decision...")
    print("  📊 State: [ml_prob=0.74, volatility=normal, positions=0, ...]")
    print("  🤖 RL Action: ENTER FULL POSITION")
    print("  ✅ Optimal conditions for aggressive entry")
    
    print("\n  💰 Position Sizing...")
    print("  💵 Account: $10,000")
    print("  ⚠️  Risk: 1% = $100")
    print("  🎯 Stop Loss: 15 pips * 1.5 = 22.5 pips")
    print("  📏 Position Size: 0.44 lots")
    
    print("\n  📤 Placing Order...")
    print("  ✅ BUY EURUSD")
    print("  ✅ Entry: 1.0965")
    print("  ✅ Stop Loss: 1.0942 (22.5 pips)")
    print("  ✅ Take Profit: 1.1010 (45 pips, 2:1 RR)")
    print("  ✅ Position Size: 0.44 lots")
    
    print("\n  🎉 TRADE EXECUTED!")
    print("  📝 Trade logged to database")
    print("  📊 Monitoring position...")
    
    print("\n⏰ 2024-01-20 18:00 - 4 hours later")
    print("  💰 Current Price: 1.1005 (+40 pips)")
    print("  💰 Unrealized P&L: +$176")
    
    print("\n  🎮 RL Monitoring...")
    print("  📊 Trade is 88% to TP target")
    print("  📊 Volatility increasing")
    print("  🤖 RL Action: TRAIL STOP LOSS")
    print("  ✅ Moving SL to 1.0985 (+20 pips, locked)")
    
    print("\n⏰ 2024-01-20 22:00 - 8 hours later")
    print("  🎯 Take Profit HIT at 1.1010")
    print("  💰 Profit: +$200 (+2R)")
    print("  📝 Trade closed and logged")
    print("  📊 Updated strategy performance")
    
    print("\n" + "="*80)
    print("✅ COMPLETE WORKFLOW DEMONSTRATED!")
    print("="*80)


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Run examples
    example_simple_rsi_strategy()
    example_complex_ma_rsi_macd_strategy()
    example_ml_training_pipeline()
    example_rl_execution_optimization()
    example_complete_end_to_end()
    
    print("\n" + "="*80)
    print("🎓 KEY TAKEAWAYS")
    print("="*80)
    print("""
1. USER describes strategy in plain English
2. SYSTEM parses to structured rules (NLP)
3. INDICATORS calculated automatically
4. RULE ENGINE evaluates conditions dynamically
5. ML MODEL trained on historical setups (probability)
6. RL AGENT optimizes execution (timing, sizing)
7. LIVE TRADING combines rule + ML + RL
    
The system is GENERIC - works for ANY strategy!
    """)
