# TCE Validation Framework - Complete Implementation

## 5 Core Validation Rules

### ✅ Rule 1: TREND CONFIRMATION WITH SWING STRUCTURE
**File**: `trading/tce/utils.py`, `trading/tce/structure.py`

**Logic**:
- **Uptrend**: MA6 > MA18 > MA50 + 2+ positive slopes + **HIGHER LOWS**
  - Higher lows = 2nd dip shallower than 1st (strengthening uptrend)
- **Downtrend**: MA50 > MA18 > MA6 + 2+ negative slopes + **LOWER HIGHS**
  - Lower highs = 2nd rally lower than 1st (strengthening downtrend)

**Implementation**: 
- `is_uptrend()` - Basic MA + slopes
- `is_uptrend_with_structure()` - Adds higher lows check
- `is_valid_uptrend()` - Main validation using structure

---

### ✅ Rule 2: CORRELATION PAIRS (WITH COEFFICIENTS)
**File**: `trading/tce/correlation.py` (NEW)

**Logic**:
- Calculate correlation coefficient between correlated pairs
- **Positive correlation** (e.g., +0.7): Both must move SAME direction
  - Example: EURUSD UP + EURGBP UP ✓ Valid
  - Example: EURUSD UP + EURGBP DOWN ✗ Invalid
- **Negative correlation** (e.g., -0.7): Must move OPPOSITE directions
  - Example: EURUSD UP + GBPUSD DOWN ✓ Valid (USD strong, GBP weak)
  - Example: EURUSD UP + GBPUSD UP ✗ Invalid

**Functions**:
- `calculate_correlation_coefficient(prices1, prices2)` - Pearson correlation
- `validate_correlation_directions(correlations, direction, threshold)` - Checks alignment

**Integration**: Updated `validate_tce()` to use new correlation logic

---

### ✅ Rule 3: MULTI-TIMEFRAME TREND CONFIRMATION
**File**: `trading/tce/validation.py`

**Logic**:
- When trading on timeframe X, ALL higher timeframes must confirm trend
- Examples:
  - Trading M15 → M30 & H1 must be in same trend
  - Trading H1 → H4 & D1 must be in same trend
  - Trading H4 → D1 & W1 must be in same trend

**Current Implementation**:
- `higher_timeframe_confirmed()` - Validates all HTF are in same direction
- Called in `validate_tce()`

---

### ✅ Rule 4: MA RETEST HIERARCHY (ACROSS TIMEFRAMES)
**File**: `trading/tce/validation.py`

**Logic**:
Each timeframe up tests a progressively SHORTER MA period, creating a cascade:

```
M15 is retesting 50EMA  → H1 must be retesting 18MA
M30 is retesting 20EMA  → H1 must be retesting 6MA
H1  is retesting 18MA   → H4 must be retesting 6MA
H4  is retesting 6MA    → D1 must be in trend
D1  just need trend confirmation
```

**Current Implementation**:
- `ma_hit_confirmed()` - Validates MA period matching across timeframes
- Ensures setups have proper confluence

---

### ✅ Rule 5: SUPPORT/RESISTANCE FILTER (NEW)
**File**: `trading/tce/sr_detection.py` (NEW)

**Logic**:
- Entry must NOT be AT a horizontal S/R level (price gets stuck)
- Entry BETWEEN S/R levels is VALID (clean bounce, overbought recovery)
- Detection based on ACTUAL BOUNCES, not potential levels

**Implementation**:
- `find_bounce_levels(candles)` - Detects V-shaped bounces (support) and ^-shaped bounces (resistance)
  - Returns levels with bounce counts
  - 2+ bounces = confirmed S/R
  - 1 bounce = potential S/R
- `is_at_support_resistance(entry_price, sr_levels, tolerance_pips)` - Checks if entry is within 5 pips of S/R
- `get_sr_analysis()` - Complete analysis with reason codes

**Integration**: Added to `validate_tce()` as step 6.5 (before risk management)

---

## Updated Files

### New Files Created:
1. **`trading/tce/correlation.py`** - Correlation coefficient validation
2. **`trading/tce/sr_detection.py`** - S/R bounce detection
3. **`test_all_five_rules.py`** - Comprehensive rule testing

### Modified Files:
1. **`trading/tce/validation.py`**
   - Added imports for correlation and sr_detection
   - Updated `correlation_confirmed()` to handle coefficients
   - Added S/R filter step (6.5)
   
2. **`trading/tce/utils.py`**
   - Added `is_uptrend_with_structure()` - Checks higher lows
   - Added `is_downtrend_with_structure()` - Checks lower highs

3. **`trading/tce/structure.py`**
   - Updated imports
   - Modified `is_valid_uptrend()` - Uses structure-aware validation
   - Modified `is_valid_downtrend()` - Uses structure-aware validation

---

## Function Signatures

### Rule 2 - Correlation
```python
def validate_correlation_directions(
    correlations: Dict[str, Tuple[float, str]],  # {pair: (coefficient, direction)}
    direction: str,  # 'UP' or 'DOWN'
    threshold: float = 0.5
) -> bool
```

### Rule 5 - S/R Filter
```python
def find_bounce_levels(
    candles: List[Candle],
    lookback: int = 50,
    tolerance_pips: float = 5
) -> List[Tuple[float, int]]  # [(level, bounce_count), ...]

def get_sr_analysis(
    candles: List[Candle],
    entry_price: float,
    direction: str,
    lookback: int = 50,
    tolerance_pips: float = 5
) -> dict  # {is_valid, reason, sr_levels, nearest_level, ...}
```

---

## Testing

Run tests:
```bash
python test_all_five_rules.py
```

Tests validate:
- ✅ Rule 1: Uptrend with higher lows
- ✅ Rule 2: Positive correlation (same direction)
- ✅ Rule 2: Negative correlation (opposite direction)
- ✅ Rule 5: Rejection at S/R
- ✅ Rule 5: Acceptance between S/R

---

## Next Steps

1. **Test in Colab Cell 4**: Run full validation on real MT5 data
2. **Verify Results**: Check that valid setup count changes with new rules
3. **Fine-tune Thresholds**:
   - Correlation threshold: Currently 0.5 (moderate)
   - S/R tolerance: Currently 5 pips
   - Bounce detection: Currently checks 50 candles lookback
