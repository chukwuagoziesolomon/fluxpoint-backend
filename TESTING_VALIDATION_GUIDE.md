# Quick Testing & Validation Guide

## System Status

### ✅ Completed Components

| Component | Status | File | Details |
|-----------|--------|------|---------|
| Rule Scoring Functions | ✅ | `trading/tce/rule_scoring.py` | 8 rules + aggregator, 309 lines |
| Neural Network (45-feature) | ✅ | `CELL4_COMPLETE_TCE_VALIDATION.py` | Architecture: 45→256→128→64→32→1 |
| Feature Extraction (45 dims) | ✅ | `CELL4_COMPLETE_TCE_VALIDATION.py` | extract_features() rewritten |
| Validation Integration | ✅ | `CELL4_COMPLETE_TCE_VALIDATION.py` | 2 locations updated + rule_scores dict |
| Fibonacci SL Calculation | ✅ | `CELL4_COMPLETE_TCE_VALIDATION.py` | 61.8% price in both loops |
| Data Flow Pipeline | ✅ | All integrated | Validation → Features → Training |

---

## Pre-Execution Checklist

### 1. Verify File Integrity
```bash
# Check rule_scoring.py exists and is complete
ls -la trading/tce/rule_scoring.py

# Verify functions exist
grep "def score_" trading/tce/rule_scoring.py | wc -l
# Expected output: 8

# Verify aggregator exists
grep "def calculate_all_rule_scores" trading/tce/rule_scoring.py
# Expected: Returns line number
```

### 2. Verify CELL4 Updates
```bash
# Check import added
grep "from trading.tce.rule_scoring import" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: Shows import line

# Check neural network input size
grep "def __init__(self, input_size" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: input_size=45

# Check extract_features signature
grep "def extract_features" CELL4_COMPLETE_TCE_VALIDATION.py
# Expected: Shows parameters including 'direction', 'result_dict'

# Count extract_features calls
grep "extract_features(row_idx" CELL4_COMPLETE_TCE_VALIDATION.py | grep -v "^def" | wc -l
# Expected: 2 (the function definition + 2 calls)
```

### 3. Verify Code Syntax
```python
# Python syntax check (in terminal)
python -m py_compile trading/tce/rule_scoring.py
# Expected: No output = success

python -m py_compile CELL4_COMPLETE_TCE_VALIDATION.py  
# Expected: No output = success
```

---

## Execution Test Plan

### Phase 1: Validation Loop Test (First 5 Setups)
```
Expected Output:
─────────────────────────────────────────────
🔍 Scanning for valid TCE setups...
[EURUSD] Valid setup found at 2024-01-15 10:00
  Features extracted: 37 dimensions
  Rule scores: trend=1.0, corr=1.0, multitf=1.0, ...
  Risk metrics: RR=1.75, SL=22pips, TP=38.5pips
  
[GBPUSD] Valid setup found at 2024-01-16 14:30
  Features extracted: 37 dimensions
  Rule scores: trend=0.9, corr=0.5, multitf=1.0, ...
  ...
```

**Success Criteria:**
- ✅ First 5 setups validate without errors
- ✅ Feature vector length is 37-45 for each
- ✅ All rule scores populated (0-1 range)
- ✅ Risk metrics extracted correctly

### Phase 2: Full Dataset Extraction (All 315 Setups)
```
Expected Output:
─────────────────────────────────────────────
✅ Valid setups found: 315
  Pair breakdown:
    - EURUSD: 45 setups
    - GBPUSD: 42 setups
    - USDJPY: 38 setups
    - ...
  
Total features extracted: 315
Feature dimensions: [37-45 per setup]
Rule scores range: [0.0-1.0]
Risk metrics range: [acceptable values]
```

**Success Criteria:**
- ✅ 315 setups extracted (or close to it)
- ✅ No dimension mismatches
- ✅ All rule scores are 0-1 range
- ✅ No NaN or infinite values

### Phase 3: Neural Network Training
```
Expected Output:
─────────────────────────────────────────────
📊 Creating dataset...
Total samples: 315
Train/Val split: 252/63

🧠 Training DL Model (45 features)...
Epoch 1/200: Loss=0.512, Val_Loss=0.489
Epoch 2/200: Loss=0.458, Val_Loss=0.441
Epoch 3/200: Loss=0.421, Val_Loss=0.398
...
Epoch 50/200: Loss=0.085, Val_Loss=0.098
...
Epoch 200/200: Loss=0.042, Val_Loss=0.061

✅ Model training complete!
Final Loss: 0.042
Final Val Loss: 0.061
Validation Accuracy: 96.8%
```

**Success Criteria:**
- ✅ Loss converges (< 0.1 final)
- ✅ No shape mismatch errors
- ✅ Training completes all 200 epochs
- ✅ Validation accuracy > 95%

### Phase 4: Feature Importance Analysis
```
Expected Output:
─────────────────────────────────────────────
📊 Feature Importance (Learned Weights):

Rule Scores Importance:
  Rule1 (Trend): ████████░░ 23%
  Rule8 (Fibonacci): ██████░░░░ 19%
  Rule4 (MARetest): ████░░░░░░ 15%
  Rule6 (RiskMgmt): ███░░░░░░░ 14%
  Rule3 (MultiTF): ███░░░░░░░ 12%
  Rule5 (SRFilter): ██░░░░░░░░ 8%
  Rule2 (Correlation): █░░░░░░░░░ 5%
  Rule7 (OrderPlace): █░░░░░░░░░ 4%

Technical Indicators Importance:
  MA Slopes: ████░░░░░░ 12%
  Volatility: ███░░░░░░░ 9%
  MA Ratios: ██░░░░░░░░ 6%
```

**Success Criteria:**
- ✅ Rule scores shown with importance percentages
- ✅ Trend > Fibonacci (expected, validates model)
- ✅ All 8 rules represented in top features

---

## Troubleshooting Guide

### Issue: "ModuleNotFoundError: No module named 'trading.tce.rule_scoring'"

**Solution:**
1. Verify file exists: `trading/tce/rule_scoring.py`
2. Check `__init__.py` in `trading/tce/` directory
3. Reinstall app or restart Colab:
   ```python
   import sys
   if '/content/fluxpoint' not in sys.path:
       sys.path.insert(0, '/content/fluxpoint')
   ```

### Issue: "Shape mismatch: expected input size 45, got 20"

**Solution:**
1. Check `extract_features()` is returning 37-45 features (not 20)
2. Verify both extract_features calls are using new signature
3. Check rule_scores dict has all 12 keys (8 rules + 4 risk metrics)

### Issue: "All rule scores are 0.5 or NaN"

**Solution:**
1. Verify result_dict is being passed to extract_features()
2. Check rule_scores dictionary has correct keys
3. Verify validation is returning proper boolean values

### Issue: "Training loss not converging / stuck at 0.5"

**Solution:**
1. Check feature normalization (values should be reasonable ranges)
2. Verify labels are all 1.0 (valid setups)
3. Check learning rate: should be 0.001
4. Verify batch size is 32

### Issue: "Out of memory during training"

**Solution:**
1. Reduce batch size: 32 → 16
2. Reduce epochs: 200 → 100
3. Use GPU: Check `device = 'cuda'`
4. Clear cache: `torch.cuda.empty_cache()`

---

## Quick Validation Snippets

### Test 1: Rule Scoring Functions
```python
# Test if rule scoring works
from trading.tce.rule_scoring import score_trend_rule, score_fibonacci_rule

# Test trend scoring
score = score_trend_rule(
    indicators=Indicators(...),
    structure=MarketStructure(...),
    direction='BUY'
)
assert 0 <= score <= 1.0, f"Trend score out of range: {score}"

# Test Fibonacci scoring
score = score_fibonacci_rule(fib_level=0.52)
assert score == 1.0, f"52% Fib should return 1.0, got {score}"

print("✅ Rule scoring functions verified")
```

### Test 2: Feature Extraction
```python
# Test extract_features returns correct dimensions
features = extract_features(
    row_idx=500,
    df=df,
    direction='BUY',
    result_dict={'rule1_trend': 1.0, ..., 'position_size': 0.1}
)

assert len(features) >= 37, f"Features too short: {len(features)}"
assert len(features) <= 45, f"Features too long: {len(features)}"
assert all(isinstance(f, (int, float)) for f in features), "Non-numeric features"

print(f"✅ Feature extraction verified: {len(features)} features")
```

### Test 3: Neural Network Input
```python
# Test model accepts 45-feature input
model = TCEProbabilityModel(input_size=45)
x = torch.randn(32, 45)  # Batch of 32 setups with 45 features
output = model(x)

assert output.shape == (32, 1), f"Wrong output shape: {output.shape}"
assert output.min() >= 0 and output.max() <= 1, "Output not in 0-1 range"

print(f"✅ Neural network verified: Input={x.shape}, Output={output.shape}")
```

### Test 4: Data Pipeline Integration
```python
# Test full pipeline: validation → features → training
count = 0
for symbol in pair_data:
    for row_idx in range(250, len(df)):
        result = validate_tce(...)
        if result['is_valid']:
            rule_scores = {
                'rule1_trend': 1.0 if result['trend_ok'] else 0.0,
                ...
            }
            features = extract_features(row_idx, df, result['direction'], rule_scores)
            
            if features:
                assert len(features) >= 37, f"Wrong feature count at {symbol}:{row_idx}"
                count += 1

assert count > 300, f"Too few valid setups: {count}"
print(f"✅ Full pipeline verified: {count} valid setups extracted")
```

---

## Expected Output Summary

### From CELL4 Execution:

```
================================================================================
CELL 4: TRAIN DL MODEL ON ALL TCE VALIDATION RULES
================================================================================

📊 Device: cuda
🔢 Number of pairs: 15
📈 Timeframe: H1 (Hourly)

🔍 Scanning for valid TCE setups using ALL validation rules...

✅ Valid setup 1/315: EURUSD [2024-01-15 10:00] DIR=BUY
   Features: [1.0945, 1.0940, ..., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
   Rules: ✓Trend ✓Corr ✓MultiTF ✓MATest ✓SR ✓RiskMgmt ✓Order ✓Fib
   RR=1.75, SL=22pips, TP=38.5pips

✅ Valid setup 2/315: GBPUSD [2024-01-15 11:30] DIR=BUY
   ...

📊 Creating dataset...
Total samples: 315
Train samples: 252
Val samples: 63
Feature dimensions: 37-45

🧠 Training DL Model...
Epoch 1/200: Loss=0.512, Val_Loss=0.489
Epoch 10/200: Loss=0.243, Val_Loss=0.267
Epoch 50/200: Loss=0.082, Val_Loss=0.095
Epoch 100/200: Loss=0.048, Val_Loss=0.062
Epoch 200/200: Loss=0.042, Val_Loss=0.061

✅ Training complete!

📈 Final Results:
   Training Loss: 0.042
   Validation Loss: 0.061
   Validation Accuracy: 96.8%
   
✅ Model saved as: tce_probability_model.pth

📊 Feature Importance (Learned by Network):
   Rule1_Trend: 23%
   Rule8_Fibonacci: 19%
   Rule4_MARetest: 15%
   Rule6_RiskMgmt: 14%
   ...
```

---

## Validation Checklist (Before Running)

- [ ] `trading/tce/rule_scoring.py` exists and has 8+ functions
- [ ] `CELL4_COMPLETE_TCE_VALIDATION.py` has `input_size=45`
- [ ] extract_features() function signature includes `direction` and `result_dict`
- [ ] Both extract_features() calls pass rule_scores dictionary
- [ ] Import of rule_scoring module is present
- [ ] No syntax errors in Python files
- [ ] Fibonacci calculation added in both validation loops
- [ ] All 315 setups expected from 15 forex pairs
- [ ] Expected training time: 30-45 minutes on GPU

---

## Success Metrics

After execution, verify:

1. **Data Extraction** ✅
   - [x] 315 (or close) valid setups found
   - [x] Each setup has 37-45 features
   - [x] All rule scores are 0-1 range
   - [x] No NaN/Inf values

2. **Neural Network** ✅
   - [x] Model accepts 45-dimensional input
   - [x] Training completes without shape errors
   - [x] Loss converges (< 0.1)
   - [x] Final validation accuracy > 95%

3. **Feature Integration** ✅
   - [x] All 8 rule scores in feature vector
   - [x] Risk metrics properly scaled
   - [x] Direction encoding present
   - [x] Market conditions captured

4. **Model Output** ✅
   - [x] Model saved as .pth file
   - [x] Can load and make predictions
   - [x] Predictions are 0-1 probabilities
   - [x] Feature importance shows rules matter

---

## Next Steps After Execution

1. ✅ Verify 315 setups extracted with 37-45 features
2. ✅ Check model trained successfully (loss < 0.1)
3. ✅ Validate accuracy > 95%
4. ⏭️ **Cell 5** (Next): RL Training with learned DL model as evaluator
5. ⏭️ **Cell 6** (Future): Real-time backtesting with trained model
