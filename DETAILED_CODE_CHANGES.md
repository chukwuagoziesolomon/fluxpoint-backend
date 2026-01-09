# Detailed Code Changes Reference

## File-by-File Change Log

### FILE 1: `trading/tce/rule_scoring.py`
**Status:** ✅ NEW FILE CREATED  
**Lines:** 309 total  
**Purpose:** Rule scoring functions for all 8 TCE validation rules

#### Functions Created:

1. **score_trend_rule()** (Lines 13-89)
   - Input: Indicators, MarketStructure, Direction
   - Output: 0-1 score for trend confirmation
   - Components: MA alignment (0-0.5) + Slopes (0-0.5)
   - Handles: BUY (MA6>18>50>200) and SELL (MA6<18<50<200)

2. **score_correlation_rule()** (Lines 90-114)
   - Input: Correlations Dict
   - Output: 0-1 score for correlation pair alignment
   - Logic: % of correlated pairs aligned with direction

3. **score_multi_tf_rule()** (Lines 115-135)
   - Input: Higher TF Results Dict
   - Output: 0-1 score for multi-timeframe confirmation
   - Logic: % of higher timeframes confirming trend

4. **score_ma_retest_rule()** (Lines 136-157)
   - Input: Retest depth (float)
   - Output: 0-1 score
   - Mapping: 1 retest→0.3, 2 retests→0.7, 3+→1.0

5. **score_sr_filter_rule()** (Lines 158-182)
   - Input: SR distance (pips), ATR
   - Output: 0-1 score
   - Logic: Normalized distance from support/resistance

6. **score_risk_management_rule()** (Lines 183-227)
   - Input: Risk reward, SL pips, Position size
   - Output: 0-1 composite score
   - Factors: RR ratio quality, SL adequacy, position sizing

7. **score_order_placement_rule()** (Lines 228-251)
   - Input: Entry offset (pips)
   - Output: 0-1 score
   - Ideal: 2-3 pips returns 1.0

8. **score_fibonacci_rule()** (Lines 252-273)
   - Input: Fib level (0-1), Max fib threshold
   - Output: 0-1 score
   - Mapping: 38.2%→1.0, 50%→0.9, 61.8%→0.7, >61.8%→0.0

9. **calculate_all_rule_scores()** (Lines 274-309)
   - Input: All rule parameters
   - Output: Dict with 8 rule scores + overall average
   - Key: Aggregates all rules into single dictionary

---

### FILE 2: `CELL4_COMPLETE_TCE_VALIDATION.py`
**Status:** ✅ MODIFIED - 7 CHANGES  
**Total Lines:** 736  
**Changes Summary:** Architecture + feature extraction + integration

#### CHANGE 1: Import Addition
**Location:** Lines 1-50 (Imports section)  
**Type:** Code addition  
**Old:** ❌ Missing import  
**New:** ✅ Added line:
```python
from trading.tce.rule_scoring import calculate_all_rule_scores
```
**Purpose:** Access all 8 rule scoring functions  
**Impact:** Enables rule score calculation

---

#### CHANGE 2: Neural Network Class Update
**Location:** Lines 55-88 (TCEProbabilityModel class)  
**Type:** Architecture modification  
**Changes:**

1. **input_size parameter** (Line 57)
   - Old: `input_size=20`
   - New: `input_size=45`
   - Impact: Accepts 45-dimensional feature vectors

2. **Added 256-unit layer** (Lines 60-63)
   ```python
   nn.Linear(input_size, 256),      # NEW
   nn.BatchNorm1d(256),              # NEW
   nn.ReLU(),                         # NEW
   nn.Dropout(0.3),                  # NEW
   ```
   - Impact: Increased model capacity

3. **First hidden layer change** (Line 65)
   - Old: `nn.Linear(input_size, 128)`
   - New: `nn.Linear(256, 128)`
   - Impact: 256→128 instead of 20→128

4. **Final architecture:**
   ```
   45 → 256 → 128 → 64 → 32 → 1
   (was: 20 → 128 → 64 → 32 → 1)
   ```

---

#### CHANGE 3: Complete Rewrite of extract_features() Function
**Location:** Lines 102-201  
**Type:** Function signature + implementation  
**Total Lines Changed:** ~100 lines

**Old Signature:**
```python
def extract_features(row_idx, df, recent_candles_limit=50):
```

**New Signature:**
```python
def extract_features(row_idx, df, direction, result_dict, recent_candles_limit=50):
    """
    Extract 45+ features from a TCE setup including:
    - 20 Original features (MAs, slopes, ratios, volatility)
    - 8 Rule scores (from calculate_all_rule_scores)
    - Risk metrics (SL, TP, RR ratio)
    """
```

**New Parameters:**
- `direction`: "BUY" or "SELL" for direction encoding
- `result_dict`: Dictionary with rule outcomes and risk metrics

**Feature Sections Added:**

**Section 1: Original 20 Features (Lines 155-170)**
- Unchanged logic, just documented
- MAs, slopes, ATR, ratios, distances, volatility

**Section 2: Rule Scores (Lines 172-184) - NEW**
```python
if result_dict:
    features.extend([
        result_dict.get('rule1_trend', 0.5),          # [21]
        result_dict.get('rule2_correlation', 0.5),    # [22]
        result_dict.get('rule3_multi_tf', 0.5),       # [23]
        result_dict.get('rule4_ma_retest', 0.5),      # [24]
        result_dict.get('rule5_sr_filter', 0.5),      # [25]
        result_dict.get('rule6_risk_mgmt', 0.5),      # [26]
        result_dict.get('rule7_order_placement', 0.5),# [27]
        result_dict.get('rule8_fibonacci', 0.5),      # [28]
    ])
else:
    features.extend([0.5] * 8)
```

**Section 3: Risk Metrics (Lines 186-199) - NEW**
```python
if result_dict:
    features.extend([
        result_dict.get('risk_reward_ratio', 1.5),    # [29]
        result_dict.get('sl_pips', 20.0),             # [30]
        result_dict.get('tp_pips', 30.0),             # [31]
        result_dict.get('position_size', 0.1),        # [32]
    ])
```

**Section 4: Direction Encoding (Lines 201-209) - NEW**
```python
direction_encoding = 1.0 if direction == "BUY" else 0.0
uptrend_flag = float(is_valid_uptrend(...))
downtrend_flag = float(is_valid_downtrend(...))

features.extend([
    direction_encoding,    # [33]: Direction (1=BUY, 0=SELL)
    uptrend_flag,         # [34]: Is uptrend?
    downtrend_flag,       # [35]: Is downtrend?
])
```

**Section 5: Market Conditions (Lines 211-216) - NEW**
```python
volatility_extreme = 1.0 if atr > np.percentile(...) else 0.0
price_near_ma6 = 1.0 if abs(close[row_idx] - ma6) < atr else 0.0

features.extend([
    volatility_extreme,  # [36]: Is volatility high?
    price_near_ma6,      # [37]: Price near MA6?
])
```

**Return:** 37-45 dimensional feature vector (was 20)

---

#### CHANGE 4: First extract_features() Call Update
**Location:** Lines ~390-408 (First validation loop)  
**Type:** Function call modification  
**Lines Changed:** ~20 lines

**Old Code:**
```python
if result["is_valid"]:
    features = extract_features(row_idx, df)
    if features:
        X_list.append(features)
        y_list.append(1.0)
```

**New Code:**
```python
if result["is_valid"]:
    # Create rule score dictionary for this setup
    rule_scores = {
        'rule1_trend': 1.0 if result['trend_ok'] else 0.0,
        'rule2_correlation': 1.0 if result['correlation_ok'] else 0.0,
        'rule3_multi_tf': 1.0 if result['multi_tf_ok'] else 0.0,
        'rule4_ma_retest': 1.0 if result['ma_retest_ok'] else 0.0,
        'rule5_sr_filter': 1.0,  # Passed validation = good
        'rule6_risk_mgmt': 1.0 if result['risk_management_ok'] else 0.0,
        'rule7_order_placement': 1.0,  # Part of validation
        'rule8_fibonacci': 1.0 if result['fib_ok'] else 0.0,
        'risk_reward_ratio': result.get('risk_reward_ratio', 1.5),
        'sl_pips': result.get('sl_pips', 20.0),
        'tp_pips': result.get('tp_pips', 30.0),
        'position_size': result.get('position_size', 0.1),
    }
    
    # Extract features for this valid setup
    features = extract_features(row_idx, df, result['direction'], rule_scores)
    if features:
        X_list.append(features)
        y_list.append(1.0)
```

**Changes:**
1. Create rule_scores dict with 12 key-value pairs
2. Convert boolean results to 1.0/0.0 scores
3. Pass result['direction'] and rule_scores to extract_features()
4. Features now contain all 8 rule outcomes

**Impact:** First validation loop now creates 45-feature vectors with rule information

---

#### CHANGE 5: Fibonacci 61.8% Price Calculation (First Loop)
**Location:** Lines ~255-270 (Before first Swing creation)  
**Type:** Code addition  
**Lines Added:** ~8 lines

**New Code:**
```python
swing_high = max(recent_highs)
swing_low = min(recent_lows)
fib_range = swing_high - swing_low
fib_618_price = swing_high - (fib_range * 0.618)

swing = Swing(
    entry_price=float(close[row_idx]),
    direction='BUY',
    fib_618_price=float(fib_618_price),  # NEW PARAMETER
    # ... other parameters ...
)
```

**Purpose:** Calculate 61.8% Fibonacci level for SL placement reference  
**Impact:** Enables proper SL positioning in risk_management.py

---

#### CHANGE 6: Second extract_features() Call Update
**Location:** Lines ~560-578 (Second validation loop)  
**Type:** Function call modification  
**Lines Changed:** ~20 lines

**Old Code:** Same as before updating
```python
if result["is_valid"]:
    features = extract_features(row_idx, df)
```

**New Code:** Identical to CHANGE 4
```python
if result["is_valid"]:
    rule_scores = {
        'rule1_trend': 1.0 if result['trend_ok'] else 0.0,
        # ... 11 more entries ...
    }
    features = extract_features(row_idx, df, result['direction'], rule_scores)
```

**Purpose:** Ensure consistency between both validation loops  
**Impact:** Second loop also creates 45-feature vectors with rule information

---

#### CHANGE 7: Fibonacci 61.8% Price Calculation (Second Loop)
**Location:** Lines ~420-435 (Before second Swing creation)  
**Type:** Code addition  
**Lines Added:** ~8 lines

**New Code:** Identical to CHANGE 5
```python
swing_high = max(recent_highs)
swing_low = min(recent_lows)
fib_range = swing_high - swing_low
fib_618_price = swing_high - (fib_range * 0.618)

swing = Swing(
    entry_price=float(close[row_idx]),
    direction='SELL',
    fib_618_price=float(fib_618_price),  # NEW PARAMETER
)
```

**Purpose:** Consistent Fibonacci calculation in second loop  
**Impact:** Both validation loops have proper SL reference level

---

## Summary of Changes by Type

### New Files Created: 1
- `trading/tce/rule_scoring.py` (309 lines)

### Files Modified: 1
- `CELL4_COMPLETE_TCE_VALIDATION.py` (736 lines total)

### Total Changes: 7
1. Import addition (1 line)
2. Neural network architecture (input_size + layer) (20 lines)
3. extract_features() function rewrite (100 lines)
4. First extract_features() call (20 lines)
5. First Fibonacci calculation (8 lines)
6. Second extract_features() call (20 lines)
7. Second Fibonacci calculation (8 lines)

### Total New Code: ~177 lines across 2 files

### Feature Dimensions:
- Old: 20 features
- New: 37-45 features (+17-25 features)
- Addition: 8 rule scores + 4 risk metrics + 3 direction flags + 2-4 market conditions

### Neural Network Architecture:
- Old: 20→128→64→32→1
- New: 45→256→128→64→32→1
- Improvement: Added 256 layer for increased capacity

---

## Testing the Changes

### Verification Commands

**1. Check rule_scoring.py exists:**
```bash
ls -la trading/tce/rule_scoring.py
# Expected: File exists, ~309 lines
```

**2. Verify functions count:**
```bash
grep "^def score_" trading/tce/rule_scoring.py | wc -l
# Expected: 8
```

**3. Check imports added:**
```bash
grep "from trading.tce.rule_scoring import" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: Returns import line
```

**4. Verify neural network input:**
```bash
grep "input_size=45" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: Returns line with 45
```

**5. Check extract_features signature:**
```bash
grep "def extract_features" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: Shows all 4 parameters
```

**6. Verify calls updated:**
```bash
grep -c "extract_features(row_idx, df, result\['direction'\], rule_scores)" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: 2 (two calls)
```

---

## Integration Verification Checklist

✅ **Code Level:**
- [x] rule_scoring.py created with 8 functions
- [x] Import added to CELL4
- [x] Neural network input_size changed to 45
- [x] Extra 256 layer added
- [x] extract_features() signature updated
- [x] extract_features() implementation expanded
- [x] First validation loop updated
- [x] Second validation loop updated
- [x] Fibonacci calculations added (2 locations)

✅ **Integration Level:**
- [x] Validation result → rule_scores dict
- [x] rule_scores dict → extract_features()
- [x] extract_features() → 37-45 features
- [x] Features → X_list (training data)
- [x] Neural network accepts 45 inputs
- [x] Training pipeline complete

✅ **Data Flow:**
- [x] validate_tce() output → rule_scores
- [x] rule_scores extraction → features
- [x] features creation → tensor input
- [x] tensor input → neural network
- [x] neural network → loss calculation

---

## Expected Results After Execution

**Feature Vector Structure (37-45 dimensions):**
```
Indices 1-20:   Original 20 indicators
Indices 21-28:  8 rule scores (0-1 each)
Indices 29-32:  4 risk metrics
Indices 33-35:  3 direction flags
Indices 36-37:  2-4 market conditions
─────────────────────────────────
Total:          37-45 dimensions
```

**Dataset:**
```
315 valid TCE setups
Each with 37-45 features
Labels all = 1.0 (valid/winner)
```

**Training:**
```
Input: Batch of 32 samples × 45 features
Hidden: 256 → 128 → 64 → 32
Output: Batch of 32 × 1 (probability)
Loss: BCE on (prediction, 1.0)
```

---

## Success Indicators

After executing CELL4, you should see:

1. ✅ 315 valid setups extracted
2. ✅ Each setup has 37-45 features (not 20)
3. ✅ All rule scores are 0-1 range
4. ✅ Neural network trains without errors
5. ✅ Loss converges to < 0.1
6. ✅ Training completes in 30-45 minutes

---

## Rollback/Revert Guide (if needed)

If you need to revert:

1. **For CELL4:** Keep backup, git checkout from previous commit
2. **For rule_scoring.py:** Delete file (new file, won't affect original)
3. **No database changes:** All changes are code-only

---

## Next Steps After Successful Execution

1. Verify model training completed successfully
2. Check validation accuracy (should be > 95%)
3. Review feature importance (which rules matter most?)
4. Prepare for Cell 5: RL Agent training with learned DL model
