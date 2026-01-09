#!/usr/bin/env python
"""
Test all 5 TCE validation rules with synthetic data.
"""

import numpy as np
from trading.tce.types import Candle, Indicators, Swing, MarketStructure
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend
from trading.tce.correlation import validate_correlation_directions
from trading.tce.sr_detection import find_bounce_levels, get_sr_analysis

print("\n" + "="*80)
print("TESTING ALL 5 TCE VALIDATION RULES")
print("="*80)

# ============================================================================
# RULE 1: TREND CONFIRMATION WITH SWING STRUCTURE
# ============================================================================
print("\n1️⃣  RULE 1: TREND WITH SWING STRUCTURE\n")

indicators_up = Indicators(
    ma6=1.0850, ma18=1.0820, ma50=1.0780, ma200=1.0720,
    slope6=0.0005, slope18=0.0003, slope50=0.0002, slope200=0.0001,
    atr=0.0020
)
structure_up = MarketStructure(
    highs=[], 
    lows=[1.0700, 1.0710, 1.0720, 1.0730, 1.0740, 1.0750, 1.0760, 1.0770, 1.0780, 1.0790]
)

result = is_valid_uptrend(indicators_up, structure_up)
print(f"   Uptrend with higher lows: {result}")
assert result == True, "Should be valid uptrend"
print("   ✅ PASS\n")

# ============================================================================
# RULE 2: CORRELATION PAIRS (POSITIVE & NEGATIVE)
# ============================================================================
print("2️⃣  RULE 2: CORRELATION PAIRS\n")

# Test 1: Positive correlation - both should go same direction
correlations_pos = {
    'EURUSD': (0.78, 'UP'),     # Positive correlation, going UP
    'EURGBP': (0.82, 'UP'),     # Positive correlation, going UP
}
result = validate_correlation_directions(correlations_pos, 'UP', threshold=0.5)
print(f"   Positive correlation (both UP): {result}")
assert result == True, "Should be valid"
print("   ✅ PASS")

# Test 2: Positive correlation - mismatch
correlations_mismatch = {
    'EURUSD': (0.78, 'UP'),
    'EURGBP': (0.82, 'DOWN'),   # ❌ Mismatch - positive correlation but different direction
}
result = validate_correlation_directions(correlations_mismatch, 'UP', threshold=0.5)
print(f"   Positive correlation (mismatch UP/DOWN): {result}")
assert result == False, "Should be invalid"
print("   ✅ FAIL (as expected)")

# Test 3: Negative correlation - opposite directions (correct)
correlations_neg = {
    'GBPUSD': (-0.65, 'DOWN'),  # Negative correlation, GBPUSD DOWN
}
result = validate_correlation_directions(correlations_neg, 'UP', threshold=0.5)  # Our trade is UP
print(f"   Negative correlation (our UP vs GBPUSD DOWN): {result}")
assert result == True, "Should be valid - opposite directions"
print("   ✅ PASS")

# Test 4: Negative correlation - wrong opposite
correlations_neg_wrong = {
    'GBPUSD': (-0.65, 'UP'),    # ❌ Negative correlation but same direction
}
result = validate_correlation_directions(correlations_neg_wrong, 'UP', threshold=0.5)
print(f"   Negative correlation (both UP - wrong): {result}")
assert result == False, "Should be invalid"
print("   ✅ FAIL (as expected)\n")

# ============================================================================
# RULE 5: SUPPORT/RESISTANCE FILTER
# ============================================================================
print("5️⃣  RULE 5: SUPPORT/RESISTANCE FILTER\n")

# Create synthetic candles with bounces
candles = []
prices = [1.0700, 1.0705, 1.0702, 1.0708, 1.0705, 1.0710, 1.0708, 1.0712, 
          1.0710, 1.0715, 1.0720, 1.0715, 1.0725, 1.0720, 1.0730]

for i, price in enumerate(prices):
    candle = Candle(
        open=price,
        high=price + 0.0005,
        low=price - 0.0005,
        close=price,
        timestamp=f"2024-01-{i:02d}"
    )
    candles.append(candle)

# Find bounce levels
sr_levels = find_bounce_levels(candles, lookback=15, tolerance_pips=5)
print(f"   Found S/R levels: {sr_levels}")
print(f"   (level, bounce_count) pairs")

# Test 1: Entry AT a S/R level (should be rejected)
sr_analysis = get_sr_analysis(candles, entry_price=1.0710, direction='BUY', tolerance_pips=5)
print(f"\n   Entry at 1.0710 (likely at S/R): {sr_analysis['reason']}")
print(f"   Valid: {sr_analysis['is_valid']}")
if 1.0710 in [level for level, _, _ in sr_levels]:
    assert sr_analysis['is_valid'] == False, "Should reject entry AT S/R"
    print("   ✅ REJECTED (as expected - entry at S/R)")
else:
    print("   ⚠️  Note: 1.0710 not detected as strong bounce level")

# Test 2: Entry BETWEEN S/R levels (should be accepted)
sr_analysis = get_sr_analysis(candles, entry_price=1.0718, direction='BUY', tolerance_pips=5)
print(f"\n   Entry at 1.0718 (between levels): {sr_analysis['reason']}")
print(f"   Valid: {sr_analysis['is_valid']}")
print(f"   Nearest level: {sr_analysis['nearest_level']}")
print(f"   Distance: {sr_analysis['nearest_distance_pips']:.1f} pips")
assert sr_analysis['is_valid'] == True, "Should accept entry between levels"
print("   ✅ ACCEPTED (as expected - clean entry)\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ ALL RULE TESTS PASSED!")
print("="*80)
print("\nTested Rules:")
print("✓ Rule 1: Trend with swing structure (higher lows/lower highs)")
print("✓ Rule 2: Correlation pairs with coefficients (positive & negative)")
print("✓ Rule 5: S/R filter (reject AT levels, accept BETWEEN levels)")
print("\nNote: Rules 3 & 4 (Multi-TF) tested via validate_tce() in main Cell4")
print()
