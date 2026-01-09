# ============================================================================
# CELL 4 COMPREHENSIVE VALIDATION CHECKLIST
# ============================================================================
# Complete list of ALL validation rules that Cell 4 uses to find TCE setups

## 🎯 THE 7 MANDATORY VALIDATION RULES (ALL MUST PASS)

### 1️⃣ TREND CONFIRMATION
**What it checks:**
- Moving Average alignment (MA6 > MA18 > MA50 > MA200 for BUY)
- All MA slopes positive (slope6, slope18, slope50 > 0 for BUY)
- Market structure (higher highs AND higher lows for BUY)
- Same checks reversed for SELL (downtrend)

**Why it matters:**
- Confirms directional bias before entry
- Prevents counter-trend entries
- Ensures market structure supports the trade

**Code reference:**
```python
result["trend_ok"] = is_valid_uptrend(indicators, structure)  # or downtrend
```

**Output:**
```
1️⃣ Trend: ✅ PASS (Direction: BUY)
```

---

### 2️⃣ FIBONACCI VALIDATION
**What it checks:**
- Price retraced exactly 38.2%, 50%, OR 61.8% from the MA
- Retracement NOT deeper than 61.8% (78.6% and deeper = INVALID)

**Why it matters:**
- Fibonacci levels are natural retracement points
- Too-deep retracements indicate the setup failed
- Shallow retrace = low-risk entry

**Code reference:**
```python
result["fib_ok"] = valid_fib(swing)  # swing.fib_level must be 0.382, 0.5, or 0.618
```

**Output:**
```
2️⃣ Fibonacci: ✅ PASS (Level: 38.2%)
```

---

### 2.5️⃣ SEMI-CIRCLE SWING STRUCTURE
**What it checks:**
- Pullback forms smooth curved pattern (not sharp/jagged)
- Multiple swing points showing controlled retracement
- Price bounces smoothly back to MA

**Why it matters:**
- Curved pullbacks = healthy market reversals
- Sharp angles = uncertainty/volatility spikes
- Smooth structure = more reliable entry

**Code reference:**
```python
result["swing_ok"] = is_semi_circle_swing(structure.highs, structure.lows)
```

**Output:**
```
2.5️⃣ Swing: ✅ PASS (Semi-circle pattern detected)
```

---

### 3️⃣ AT MA LEVEL (DYNAMIC SUPPORT/RESISTANCE)
**What it checks:**
- Price is AT or NEAR one of: MA6, MA18, MA50, or MA200
- Within 2% tolerance of the MA
- Price allowed to be below MA (for BUY) - that's measured by Fibonacci

**Why it matters:**
- MAs are dynamic support that moves with price
- TCE uses ONLY moving averages (NO static S/R levels)
- Ensures entry at the correct support point

**Code reference:**
```python
result["ma_level_ok"] = at_ma_level(candle, indicators, direction)
```

**Output:**
```
3️⃣ MA Level: ✅ PASS (At MA18: 1.1040)
```

---

### 3.5️⃣ MA RETEST (SECOND TOUCH, NOT FIRST)
**What it checks:**
- Price FIRST touched this MA 50-100 candles ago
- Price moved AWAY from the MA (at least 20+ pips)
- Current candle is RETURNING to test the same MA again
- This is the RETEST, not the first bounce

**Lookback by timeframe:**
- M15/M30: 50 candles minimum
- H1/H4: 100 candles minimum
- D1/W1: 200 candles minimum

**Why it matters:**
- First bounces often fail (false reversals)
- Retest = market tested the MA, moved away, and came back = CONFIRMATION
- Entering on retest = much higher win rate

**Code reference:**
```python
result["ma_retest_ok"] = has_ma_retest(recent_candles, indicators, direction, timeframe)
```

**Output:**
```
3.5️⃣ MA Retest: ✅ PASS (2nd touch at MA18 after moving away)
```

---

### 4️⃣ CANDLESTICK CONFIRMATION (PATTERN AT RETEST)
**What it checks:**
- At the MA retest point, a specific candlestick pattern appears:

**BUY Patterns Required:**
- **Bullish Pin Bar**: Long lower wick, small body, closes high
  ```
     ┃ (small body)
  ──╋──
  ║ ║  (long lower wick)
  ─┴─
  ```

- **Rejection Candle**: Body at top, long lower wick
  ```
  ────  (open/close at top)
     ║
  ─┴─┘  (long wick down)
  ```

- **Bullish Engulfing**: Small bear candle, then larger bull candle
  ```
  First:  ─┃─  (small bearish)
  Second: ═══  (large bullish, engulfs first)
  ```

- **Morning Star**: 3-candle pattern, V-shape reversal
  ```
  ───
    ├─ (gap down)
    │  ├─ (small body)
    │  │  ├─ (gap up)
    └──┘   └── (bullish)
  ```

**SELL Patterns Required:**
- Bearish equivalents (pin bar with upper wick, rejection above, engulfing down, evening star)

**Why it matters:**
- Candlestick patterns show MARKET PSYCHOLOGY
- Rejection candle shows market rejected lower prices (for BUY)
- Pin bar shows price touched low but buyers stepped in
- Candlestick confirmation = proven reversal at that exact price level

**Code reference:**
```python
result["candlestick_ok"] = has_candlestick_confirmation(recent_candles, direction)
```

**Output:**
```
4️⃣ Candlestick: ✅ PASS (Bullish Pin Bar detected at MA18)
```

---

### 5️⃣ HIGHER TIMEFRAME CONFIRMATION
**What it checks:**
- If you have a higher timeframe candle (D1, H4, etc):
  - For BUY: Higher TF must also be in uptrend (higher highs/lows)
  - For SELL: Higher TF must also be in downtrend (lower highs/lows)
  
- ALSO validates "MA Hit Rule":
  - If entry hits MA18, higher TF must hit MA6
  - If entry hits MA50, higher TF must hit MA18

**Why it matters:**
- Prevents trading against higher timeframe trend
- All timeframes agreeing = much stronger signal
- Aligns across different timeframes

**Code reference:**
```python
result["multi_tf_ok"] = higher_timeframe_confirmed(higher_tf_candles, direction)
result["ma_hit_ok"] = ma_hit_confirmed(candle, indicators, higher_tf_candles, direction)
```

**Output:**
```
5️⃣ Multi-TF: ✅ PASS (H4 also in uptrend)
```

**For single timeframe training:**
- `higher_tf_candles = []` (empty list)
- This rule is SKIPPED (not checked)
- This is NORMAL for H1-only training

---

### 6️⃣ CORRELATION CONFIRMATION
**What it checks:**
- If trading multiple correlated pairs (like EURUSD and EURGBP):
  - Correlation must be >0.6 (60% or higher)
  - All pairs in portfolio must move together
  - Prevents one pair contradicting another

**Why it matters:**
- Correlated pairs should give same signal
- Conflicting signals = something is wrong
- Higher correlation = cleaner setups

**Code reference:**
```python
result["correlation_ok"] = correlation_confirmed(correlations, threshold=0.6)
```

**Output:**
```
6️⃣ Correlation: ✅ PASS (EURGBP: 0.72 correlation)
```

**For single pair training:**
- `correlations = {}` (empty dict)
- This rule is SKIPPED (not checked)
- This is NORMAL for single-pair training

---

### 7️⃣ RISK MANAGEMENT (POSITION SIZING & SL/TP)
**What it checks:**

#### A) STOP LOSS CALCULATION:
- Base calculation: 1.5 × ATR (Average True Range)
- Minimum: 12 pips
- Must be BELOW the 61.8% Fibonacci level
- For BUY: SL at lowest point of pullback
- For SELL: SL at highest point of pullback

Formula:
```
SL = Entry - (1.5 × ATR) for BUY
SL = Entry + (1.5 × ATR) for SELL
```

Example: Entry 1.1050, ATR 0.0010 (10 pips) = SL 1.1035 (15 pips below)

#### B) TAKE PROFIT CALCULATION:
- Dynamic based on stop loss distance
- If SL < 20 pips: Risk:Reward = 1:2 (TP = Entry + 2×SL)
- If SL ≥ 20 pips: Risk:Reward = 1:1.5 (TP = Entry + 1.5×SL)

Adapts to volatility:
```
Low volatility (SL < 20 pips) → Higher RR ratio (1:2)
High volatility (SL ≥ 20 pips) → Lower RR ratio (1:1.5)
```

#### C) POSITION SIZING:
- Risk amount = Account Balance × Risk %
- Lots = Risk Amount / (SL pips × $/pip)

Example:
```
Account: $10,000
Risk: 1% = $100
SL: 30 pips
Pip value: $5 per pip per lot
Lots = $100 / (30 × $5) = 0.67 lots
```

**Why it matters:**
- Risk management is THE most important rule
- Controls position size based on account and risk tolerance
- One bad trade without proper SL can wipe your account
- Proper TP ensures you lock in profits with positive RR ratio

**Code reference:**
```python
result["risk_management_ok"] = validate_risk_management(entry, SL, TP, direction, swing)
result["stop_loss"] = calculate_stop_loss(entry, direction, atr, swing)
result["take_profit"], rr_ratio = calculate_take_profit(entry, SL, direction)
result["position_size"] = calculate_position_size(account, risk%, sl_pips, pip_value, symbol)
```

**Output:**
```
7️⃣ Risk Mgmt: ✅ PASS
   • SL: 1.1035 (15 pips)
   • TP: 1.1080 (30 pips)
   • Risk/Reward: 1:2
   • Position Size: 0.67 lots
   • Risk Amount: $100.00
```

---

## 📊 WHAT CELL 4 OUTPUTS

When a valid setup is found, Cell 4 displays:

```
╔══ SETUP #1 ═════════════════════════════════════════════════════════════════
║ Symbol: EURUSD      | Date: 2025-06-15
║ Entry Price: 1.10150
║ Direction: BUY
║
║ RISK MANAGEMENT:
║   • SL: 1.10350 (20.0 pips)
║   • TP: 1.10750 (40.0 pips)
║   • Risk/Reward: 1:2.0
║   • Position Size: 0.50 lots
║   • Risk Amount: $100.00
║
║ MOVING AVERAGES:
║   • MA6: 1.10100
║   • MA18: 1.10080
║   • MA50: 1.10050
║   • MA200: 1.09900
║   • ATR: 0.00100
║
║ VALIDATION RULES (ALL 7 MUST PASS):
║   1️⃣  Trend: ✅ PASS
║   2️⃣  Fibonacci: ✅ PASS
║   2.5️⃣ Swing: ✅ PASS
║   3️⃣  MA Level: ✅ PASS
║   3.5️⃣ MA Retest: ✅ PASS
║   4️⃣  Candlestick: ✅ PASS
║   5️⃣  Multi-TF: ✅ PASS (H1 only)
║   6️⃣  Correlation: ✅ PASS (single pair)
║   7️⃣  Risk Mgmt: ✅ PASS
╚════════════════════════════════════════════════════════════════════════════════
```

---

## ❌ COMMON FAILURE REASONS (Why a Setup is Rejected)

If a setup doesn't pass, Cell 4 shows why:

```
❌ INVALID SETUP
Failure Reason: "Not at Moving Average - TCE requires MA bounce (dynamic support only)"
```

Possible reasons:
1. "Trend not confirmed by structure" → MA alignment or structure broken
2. "Invalid Fibonacci - price retraced beyond 61.8%" → Price went too deep
3. "No proper swing structure" → No curved pullback pattern
4. "Not at Moving Average" → Price not near any MA
5. "Not a retest" → First bounce, not a retest
6. "No candlestick confirmation pattern at MA retest" → No pattern at retest
7. "Higher timeframe disagreement" → HTF trend opposite to entry
8. "MA hit rule not satisfied" → MA alignment across timeframes broken
9. "Correlation not aligned" → Pairs moving opposite
10. Various "Risk management" failures → SL/TP/sizing issues

---

## 🎓 HOW THE NEURAL NETWORK USES THIS

Cell 4 extracts 20 features from each VALID setup:

**Features 1-4:** Moving averages (MA6, MA18, MA50, MA200)
**Features 5-8:** MA slopes (rate of change)
**Feature 9:** ATR (volatility)
**Features 10-12:** MA ratios (MA6/MA18, MA18/MA50, MA50/MA200)
**Features 13:** Price/MA6 relationship
**Features 14-17:** Distance from each MA (in ATR units)
**Features 18-19:** Price volatility (20 and 50 candle standard deviation)
**Feature 20:** Current candle range

These 20 features capture the ESSENCE of what makes a valid TCE setup.

The neural network learns:
- "When these 20 values look like this → 72% chance this is a valid TCE setup"
- Later, it can predict setups WITHOUT waiting for candlestick confirmation

---

## ✅ SUCCESS CRITERIA FOR CELL 4

Cell 4 is working correctly when:

1. **Finds valid setups:** 50+ setups from all 15 pairs (not 0!)
2. **Shows all 7 rules:** Each setup shows all 7 rules passing
3. **Risk management** calculated: SL, TP, position size, risk amount
4. **Neural network trains:** Final accuracy >70%
5. **No error messages:** Completes without Python errors

If Cell 4 finds **0 setups**, check:
- Are moving averages calculated? (MA6, MA18, MA50, MA200)
- Is trend confirmation working? (slopes and alignment)
- Are candlestick patterns being detected?
- Is MA retest pattern present in historical data?

---

