# ============================================================================
# COMPLETE TCE VALIDATION RULES - FULL REFERENCE
# ============================================================================
# These are ALL 7 rules that MUST be satisfied for a valid TCE setup
# This document explains each rule and how Cell 4 implements it

# ============================================================================
# 1️⃣ TREND CONFIRMATION (via Structure Analysis)
# ============================================================================
# UPTREND:
#   - MA6 > MA18 > MA50 > MA200 (alignment)
#   - slope6 > 0
#   - slope18 > 0  
#   - slope50 > 0
#   - Higher highs and higher lows in recent 50 candles
#
# DOWNTREND:
#   - MA200 > MA50 > MA18 > MA6 (reverse alignment)
#   - slope6 < 0
#   - slope18 < 0
#   - slope50 < 0
#   - Lower highs and lower lows in recent 50 candles
#
# WHY: Entry only when there's a confirmed directional bias
# LOCATION: utils.py -> is_uptrend() / is_downtrend()
# LOCATION: structure.py -> is_valid_uptrend() / is_valid_downtrend()
#
# Cell 4 Implementation:
# - Calculates MA6, MA18, MA50, MA200
# - Calculates slopes via polyfit
# - Builds MarketStructure with recent highs/lows
# - Calls is_valid_uptrend() or is_valid_downtrend()

# ============================================================================
# 2️⃣ FIBONACCI VALIDATION (Depth Check)
# ============================================================================
# Price retraces from MA by certain percentages:
#   - 38.2% retracement: Shallow pullback (VALID)
#   - 50.0% retracement: Medium pullback (VALID)
#   - 61.8% retracement: Deep pullback (VALID - the limit)
#   - Beyond 61.8%: Price went TOO DEEP - INVALID (too much risk)
#
# WHY: Fibonacci shows how deeply price pulled back before reversing
#      Too deep = risk is too high, missed the setup
# LOCATION: utils.py -> valid_fib()
#
# Cell 4 Implementation:
# - Estimates which fib level based on depth (price vs MA)
# - Only allows 0.382, 0.50, 0.618
# - Rejects anything deeper (78.6%, 88.6% etc)

# ============================================================================
# 2.5️⃣ SEMI-CIRCLE SWING STRUCTURE (Curved Retracement)
# ============================================================================
# The pullback should form a smooth curved pattern (like ∿)
# NOT a sharp V or sharp triangle
#
# WHY: Curved pullbacks show healthy, controlled reversals
#      Sharp angles suggest uncertainty/more volatility
# LOCATION: structure.py -> is_semi_circle_swing()
#
# Cell 4 Implementation:
# - Checks that swing highs/lows form smooth pattern
# - Typically validated by the retest + candlestick rules

# ============================================================================
# 3️⃣ AT MA LEVEL (Dynamic Support/Resistance)
# ============================================================================
# Price MUST be at/near one of the moving averages:
#   - MA6, MA18, MA50, or MA200
#   - Within 2% tolerance (0.02)
#   - TCE uses ONLY MAs, NOT horizontal S/R levels
#
# For BUY: Price can be AT or BELOW MA (measures depth)
# For SELL: Price can be AT or ABOVE MA
#
# WHY: MAs are dynamic support/resistance that move with price
#      They adapt to market conditions better than static levels
# LOCATION: sr.py -> at_ma_level()
#
# Cell 4 Implementation:
# - Checks if current candle.low (BUY) or candle.high (SELL) is within 2% of a MA
# - Allows price slightly beyond MA (that's measured by Fibonacci)

# ============================================================================
# 3.5️⃣ MA RETEST (Second Touch, Not First)
# ============================================================================
# Entry requires a RETEST pattern:
#   1. First touch: Price hits MA
#   2. Move away: Price moves away from MA (at least 20+ pips)
#   3. Retest: Price comes BACK to same MA
#   4. Enter: You enter on the RETEST candle
#
# Lookback varies by timeframe:
#   - M15/M30: 50 candles
#   - H1/H4: 100 candles
#   - D1/W1: 200 candles
#
# WHY: Entering on first bounce often fails (false reversal)
#      Retest shows market respects the MA + testing it again = confirmation
# LOCATION: sr.py -> has_ma_retest()
#
# Cell 4 Implementation:
# - Checks last 100 candles (for H1)
# - Finds evidence that price hit MA before
# - Verifies price moved away
# - Confirms current candle is retesting

# ============================================================================
# 4️⃣ CANDLESTICK CONFIRMATION (Pattern at Retest)
# ============================================================================
# At the MA retest, a specific candlestick pattern must appear:
#
# BUY patterns:
#   - Bullish Pin Bar (long lower wick, small body)
#   - Rejection Candle (rejects lower, closes high)
#   - Bullish Engulfing
#   - Morning Star
#
# SELL patterns:
#   - Bearish Pin Bar (long upper wick, small body)
#   - Rejection Candle (rejects higher, closes low)
#   - Bearish Engulfing
#   - Evening Star
#
# WHY: Candlestick patterns show market psychology (rejection of lower prices for BUY)
#      They confirm that the MA retest is a real reversal point, not luck
# LOCATION: utils.py -> is_bullish_pin_bar(), is_rejection_candle(), etc
#
# Cell 4 Implementation:
# - Checks recent_candles[-1] (the retest candle)
# - Detects pattern based on wicks and body
# - Must match the direction (bullish for BUY, bearish for SELL)

# ============================================================================
# 5️⃣ HIGHER TIMEFRAME CONFIRMATION (Multi-Timeframe)
# ============================================================================
# If you have a higher timeframe candle:
#   - For BUY: Higher TF must also be in uptrend
#   - For SELL: Higher TF must also be in downtrend
#
# ALSO: MA Hit Rule
#   - If entry hits MA18, higher TF must hit MA6
#   - If entry hits MA50, higher TF must hit MA18
#   - This ensures alignment across timeframes
#
# WHY: Multi-timeframe confirmation avoids counter-trend entries
#      You want all timeframes agreeing on direction
# LOCATION: validation.py -> higher_timeframe_confirmed()
# LOCATION: validation.py -> ma_hit_confirmed()
#
# Cell 4 Implementation:
# - For single-pair H1 training: higher_tf_candles = []
# - Rule is skipped (empty list means "no HTF data available")
# - This is NORMAL for single-timeframe training
# - Later when you add HTF data, this rule activates

# ============================================================================
# 6️⃣ CORRELATION CONFIRMATION (Multi-Pair)
# ============================================================================
# If trading multiple pairs:
#   - Correlated pairs (like EURUSD and EURGBP) should move together
#   - Correlation threshold: 0.6 (60% correlation minimum)
#   - All pairs in your trade must have >60% correlation
#
# WHY: If two correlated pairs show opposite signals, something is wrong
#      High correlation means they move together = cleaner signals
# LOCATION: validation.py -> correlation_confirmed()
#
# Cell 4 Implementation:
# - For single-pair H1 training: correlations = {}
# - Rule is skipped (empty dict means "no correlation data available")
# - This is NORMAL for single-pair training
# - Later when you add multi-pair correlation, this rule activates

# ============================================================================
# 7️⃣ RISK MANAGEMENT (Position Sizing & SL/TP)
# ============================================================================
# Every valid setup MUST have proper risk management:
#
# A) STOP LOSS CALCULATION:
#    - Base: 1.5 × ATR (Average True Range)
#    - Minimum: 12 pips
#    - Must be below the 61.8% Fibonacci level
#    - For BUY: SL = lowest point in pullback
#    - For SELL: SL = highest point in pullback
#
# B) TAKE PROFIT CALCULATION:
#    - Dynamic based on SL distance
#    - If SL < 20 pips: RR ratio 1:2 (TP = Entry + 2×SL)
#    - If SL >= 20 pips: RR ratio 1:1.5 (TP = Entry + 1.5×SL)
#    - This adapts to market volatility
#
# C) POSITION SIZING:
#    - Risk amount = Account × Risk % (e.g., 10,000 × 1% = $100)
#    - Lots = Risk Amount / (SL pips × $/pip)
#    - Example: Risk $100, SL=30 pips, $5/pip = 100/(30×5) = 0.67 lots
#
# WHY: Risk management is NON-NEGOTIABLE
#      Without it, one bad trade ruins your account
# LOCATION: risk_management.py
#
# Cell 4 Implementation:
# - Calculates ATR from last 15 candles
# - Calls calculate_stop_loss()
# - Calls calculate_take_profit()
# - Calls validate_risk_management()
# - Calls calculate_position_size()
# - All values returned in result dict

# ============================================================================
# COMPLETE VALIDATION CALL (Cell 4)
# ============================================================================
# This is the actual call made in Cell 4:
#
# result = validate_tce(
#     candle=candle,                    # Current candle (open, high, low, close, timestamp)
#     indicators=indicators,            # All MAs and slopes (ma6, ma18, ma50, ma200, slope6, slope18, slope50, slope200, atr)
#     swing=swing,                      # Swing info (type, price, fib_level)
#     sr_levels=[],                     # Empty - TCE doesn't use static S/R
#     higher_tf_candles=[],             # Empty for H1-only training
#     correlations={},                  # Empty for single-pair training
#     structure=structure,              # Market structure (highs, lows)
#     recent_candles=recent_candles,    # Last 50 candles for pattern analysis
#     timeframe="H1",                   # Hourly timeframe
#     account_balance=10000.0,          # Demo account size
#     risk_percentage=1.0,              # 1% risk per trade
#     symbol=symbol                     # Currency pair name
# )
#
# Returns dict with:
# - is_valid: True/False
# - direction: "BUY" or "SELL"
# - All 7 individual rule results (trend_ok, fib_ok, ma_level_ok, etc)
# - stop_loss, take_profit, risk_reward_ratio
# - position_size in lots

# ============================================================================
# FEATURE EXTRACTION (Cell 4)
# ============================================================================
# For each VALID setup found, extract 20 features to train the neural network:
#
#  1-4:   MA values (ma6, ma18, ma50, ma200)
#  5-8:   MA slopes (slope6, slope18, slope50, slope200)
#  9:     ATR (volatility)
# 10-12:  MA ratios (ma6/ma18, ma18/ma50, ma50/ma200)
# 13:     Price/MA6 ratio
# 14-17:  Distance from each MA (in ATR units)
# 18-19:  Volatility metrics (20-candle and 50-candle standard deviation)
# 20:     Current candle range as % of close
#
# These 20 features capture all aspects of a valid TCE setup
# Neural network learns to recognize these patterns

# ============================================================================
# WHAT CELL 4 OUTPUTS
# ============================================================================
# 1. Scans all pairs for valid setups
# 2. Shows count of valid setups per pair
# 3. Displays 3 example setups with all details:
#    - Symbol, date, price, direction
#    - Stop loss, take profit, risk/reward ratio
#    - Which validation rules passed
# 4. Trains neural network on valid setups
# 5. Shows training progress and final accuracy
# 6. Saves model to Google Drive
#
# SUCCESS = Find many valid setups (100+) and train network with >70% accuracy

