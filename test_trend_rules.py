#!/usr/bin/env python
"""Test the updated trend rules with higher lows/lower highs validation."""

import numpy as np
from trading.tce.types import Indicators, MarketStructure
from trading.tce.structure import is_valid_uptrend, is_valid_downtrend

print("\n" + "="*80)
print("TESTING TREND RULES: MA ALIGNMENT + SWING STRUCTURE")
print("="*80)

# ============================================================================
# TEST 1: UPTREND with Higher Lows
# ============================================================================
print("\n1️⃣  TEST UPTREND - Should PASS (MA aligned + higher lows)")
indicators_uptrend = Indicators(
    ma6=1.0850,
    ma18=1.0820,
    ma50=1.0780,
    ma200=1.0720,
    slope6=0.0005,
    slope18=0.0003,
    slope50=0.0002,
    slope200=0.0001,
    atr=0.0020
)

# Higher lows: 1.0700, 1.0710, 1.0720, 1.0730, 1.0740, 1.0750, 1.0760, 1.0770, 1.0780, 1.0790
lows_uptrend = [1.0700, 1.0710, 1.0720, 1.0730, 1.0740, 1.0750, 1.0760, 1.0770, 1.0780, 1.0790]
structure_uptrend = MarketStructure(highs=[], lows=lows_uptrend)

result = is_valid_uptrend(indicators_uptrend, structure_uptrend)
print(f"   Result: {result}")
print(f"   MA Alignment: MA6(1.0850) > MA18(1.0820) > MA50(1.0780) ✓")
print(f"   Slopes: 2+ positive ✓")
print(f"   Lows: {lows_uptrend[-5:]} (Higher lows) ✓")
assert result == True, "UPTREND with higher lows should PASS"
print("   ✅ PASS\n")

# ============================================================================
# TEST 2: UPTREND with LOWER Lows (should FAIL)
# ============================================================================
print("2️⃣  TEST UPTREND INVALID - Should FAIL (MAs aligned but lower lows)")
lows_invalid = [1.0800, 1.0790, 1.0780, 1.0770, 1.0760, 1.0750, 1.0740, 1.0730, 1.0720, 1.0710]
structure_invalid = MarketStructure(highs=[], lows=lows_invalid)

result = is_valid_uptrend(indicators_uptrend, structure_invalid)
print(f"   Result: {result}")
print(f"   MA Alignment: ✓")
print(f"   Slopes: ✓")
print(f"   Lows: {lows_invalid[-5:]} (LOWER lows - trend weakening) ✗")
assert result == False, "UPTREND with lower lows should FAIL"
print("   ✅ FAIL (as expected)\n")

# ============================================================================
# TEST 3: DOWNTREND with Lower Highs
# ============================================================================
print("3️⃣  TEST DOWNTREND - Should PASS (MA aligned + lower highs)")
indicators_downtrend = Indicators(
    ma6=1.0720,
    ma18=1.0750,
    ma50=1.0800,
    ma200=1.0900,
    slope6=-0.0005,
    slope18=-0.0003,
    slope50=-0.0002,
    slope200=-0.0001,
    atr=0.0020
)

# Lower highs: 1.0900, 1.0890, 1.0880, 1.0870, 1.0860, 1.0850, 1.0840, 1.0830, 1.0820, 1.0810
highs_downtrend = [1.0900, 1.0890, 1.0880, 1.0870, 1.0860, 1.0850, 1.0840, 1.0830, 1.0820, 1.0810]
structure_downtrend = MarketStructure(highs=highs_downtrend, lows=[])

result = is_valid_downtrend(indicators_downtrend, structure_downtrend)
print(f"   Result: {result}")
print(f"   MA Alignment: MA50(1.0800) > MA18(1.0750) > MA6(1.0720) ✓")
print(f"   Slopes: 2+ negative ✓")
print(f"   Highs: {highs_downtrend[-5:]} (Lower highs) ✓")
assert result == True, "DOWNTREND with lower highs should PASS"
print("   ✅ PASS\n")

# ============================================================================
# TEST 4: DOWNTREND with HIGHER Highs (should FAIL)
# ============================================================================
print("4️⃣  TEST DOWNTREND INVALID - Should FAIL (MAs aligned but higher highs)")
highs_invalid = [1.0810, 1.0820, 1.0830, 1.0840, 1.0850, 1.0860, 1.0870, 1.0880, 1.0890, 1.0900]
structure_invalid = MarketStructure(highs=highs_invalid, lows=[])

result = is_valid_downtrend(indicators_downtrend, structure_invalid)
print(f"   Result: {result}")
print(f"   MA Alignment: ✓")
print(f"   Slopes: ✓")
print(f"   Highs: {highs_invalid[-5:]} (HIGHER highs - trend weakening) ✗")
assert result == False, "DOWNTREND with higher highs should FAIL"
print("   ✅ FAIL (as expected)\n")

# ============================================================================
# TEST 5: Mixed Lows (some higher, some lower - should pass ~60%)
# ============================================================================
print("5️⃣  TEST UPTREND with Mixed Lows - Should PASS (>60% higher)")
lows_mixed = [1.0700, 1.0715, 1.0710, 1.0725, 1.0720, 1.0745, 1.0750, 1.0770, 1.0780, 1.0790]
structure_mixed = MarketStructure(highs=[], lows=lows_mixed)

result = is_valid_uptrend(indicators_uptrend, structure_mixed)
print(f"   Result: {result}")
print(f"   Lows: {lows_mixed[-5:]}")
print(f"   Comparison: ", end="")
for i in range(1, len(lows_mixed)):
    if lows_mixed[i] > lows_mixed[i-1]:
        print("↑ ", end="")
    else:
        print("↓ ", end="")
print()
higher_count = sum(1 for i in range(1, len(lows_mixed)) if lows_mixed[i] > lows_mixed[i-1])
print(f"   Higher moves: {higher_count}/9 = {100*higher_count/9:.0f}% (need 60%)")
assert result == True, "Mixed lows with >60% higher should PASS"
print("   ✅ PASS\n")

print("="*80)
print("✅ ALL TREND RULE TESTS PASSED!")
print("="*80)
print("\nSummary:")
print("✓ Uptrend = MA6 > MA18 > MA50 + 2+ positive slopes + HIGHER LOWS")
print("✓ Downtrend = MA50 > MA18 > MA6 + 2+ negative slopes + LOWER HIGHS")
print("✓ This ensures the 2nd test is shallower than the 1st (trend strengthening)")
print()
