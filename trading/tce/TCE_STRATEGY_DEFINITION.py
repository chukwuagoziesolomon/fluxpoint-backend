"""
TCE (Trade Confluence Entry) STRATEGY DEFINITION

Complete trading strategy for both UPTREND and DOWNTREND
Based on MA bounce, retracement, and retest confluence.
"""

# ============================================================================
# UPTREND TCE PATTERN (BUY SETUP)
# ============================================================================

UPTREND_TCE = """
📈 UPTREND TCE PATTERN (BUY)

MAIN TREND CONFIRMATION:
├─ MA50 SLOPING UP (positive slope over last 20 bars)
└─ MA200 SLOPING UP (positive slope over last 20 bars)
   → This defines the PRIMARY trend direction

STRUCTURE:
├─ SWING UP: Price makes a strong upward move
│  └─ Creates swing high
│
├─ INITIAL BOUNCE: Price bounces OFF ONE MA level
│  ├─ Bounce happens from: MA50, MA18, or MA6
│  ├─ Price comes DOWN and touches the MA
│  ├─ Price bounces UP immediately (rejects lower prices)
│  └─ This MA = RETEST LEVEL for entry
│
├─ RETRACEMENT: Price pulls back from the bounce
│  ├─ Depth: 38.2% to 61.8% (Fibonacci)
│  ├─ Example:
│  │  • Swing high: 1.1050
│  │  • Swing low (initial): 1.1000
│  │  • Range: 50 pips
│  │  • 61.8% retracement: 1.1050 - (50 × 0.618) = 1.1019
│  │  • Price pulls back to ~1.1019 (within 38.2%-61.8% range)
│  └─ MA50 + MA200 STILL SLOPING UP during retracement (main trend intact)
│
├─ RETEST: Price bounces back and retests the SAME MA
│  ├─ Same MA as initial bounce (e.g., if bounced from MA50, retest MA50)
│  ├─ This is the SECOND TOUCH of that MA
│  ├─ The retest MA MUST be sloping UP
│  └─ Price approaches the MA from BELOW (like before)
│
└─ ENTRY: Candlestick confirmation at retest MA
   ├─ Confirmation patterns:
   │  • Pin bar (long lower wick, close at top)
   │  • Engulfing candle (bullish)
   │  • Reversal pattern
   └─ Entry point: Buy Stop 2-3 pips ABOVE confirmation candle HIGH

STOP LOSS:
├─ Placement: 2-5 pips below 61.8% Fibonacci level
└─ Example:
   • 61.8% level: 1.1019
   • SL: 1.1014 (5 pips below)

TAKE PROFIT:
├─ Based on Risk:Reward ratio
├─ Calculate from entry to previous swing high
└─ Example:
   • Entry: 1.1025
   • Risk: 11 pips (to SL at 1.1014)
   • RR 1:1.5 → TP = 1.1025 + (11 × 1.5) = 1.1041
   • RR 1:2.0 → TP = 1.1025 + (11 × 2.0) = 1.1047

VISUALIZATION:
                    SWING HIGH (1.1050)
                         ↓
              Price rallies UP strongly
                         ↓
              ┌──────────────────
              │  MA50 ↗ (sloping up)
   ┌─────────┤  MA200 ↗ (sloping up)
   │ BOUNCE  │
   │   ↓     └──────────────────
   └ MA50    
        ↓
   Price pulls back
   (38.2%-61.8%)
        ↓
   ┌─ RETEST - 
   │  Price touches MA50 again
   │  (SECOND TOUCH)
   └─ Entry: Buy Stop above candle
   
"""

# ============================================================================
# DOWNTREND TCE PATTERN (SELL SETUP)
# ============================================================================

DOWNTREND_TCE = """
📉 DOWNTREND TCE PATTERN (SELL)

MAIN TREND CONFIRMATION:
├─ MA50 SLOPING DOWN (negative slope over last 20 bars)
└─ MA200 SLOPING DOWN (negative slope over last 20 bars)
   → This defines the PRIMARY trend direction

STRUCTURE:
├─ SWING DOWN: Price makes a strong downward move
│  └─ Creates swing low
│
├─ INITIAL BOUNCE: Price bounces OFF ONE MA level
│  ├─ Bounce happens from: MA50, MA18, or MA6
│  ├─ Price comes UP and touches the MA
│  ├─ Price bounces DOWN immediately (rejects higher prices)
│  └─ This MA = RETEST LEVEL for entry
│
├─ RETRACEMENT: Price pulls back from the bounce
│  ├─ Depth: 38.2% to 61.8% (Fibonacci)
│  ├─ Example:
│  │  • Swing low: 1.0950
│  │  • Swing high (initial): 1.1000
│  │  • Range: 50 pips
│  │  • 38.2% retracement: 1.0950 + (50 × 0.382) = 1.0991
│  │  • Price pulls back to ~1.0991 (within 38.2%-61.8% range)
│  └─ MA50 + MA200 STILL SLOPING DOWN during retracement (main trend intact)
│
├─ RETEST: Price bounces back and retests the SAME MA
│  ├─ Same MA as initial bounce (e.g., if bounced from MA50, retest MA50)
│  ├─ This is the SECOND TOUCH of that MA
│  ├─ The retest MA MUST be sloping DOWN
│  └─ Price approaches the MA from ABOVE (like before)
│
└─ ENTRY: Candlestick confirmation at retest MA
   ├─ Confirmation patterns:
   │  • Pin bar (long upper wick, close at bottom)
   │  • Engulfing candle (bearish)
   │  • Reversal pattern
   └─ Entry point: Sell Stop 2-3 pips BELOW confirmation candle LOW

STOP LOSS:
├─ Placement: 2-5 pips above 61.8% Fibonacci level
└─ Example:
   • 61.8% level: 1.0991
   • SL: 1.0996 (5 pips above)

TAKE PROFIT:
├─ Based on Risk:Reward ratio
├─ Calculate from entry to previous swing low
└─ Example:
   • Entry: 1.0975
   • Risk: 11 pips (to SL at 1.0986)
   • RR 1:1.5 → TP = 1.0975 - (11 × 1.5) = 1.0958
   • RR 1:2.0 → TP = 1.0975 - (11 × 2.0) = 1.0955

VISUALIZATION:
   ┌─────────────────────────────────
   │  MA50 ↘ (sloping down)
   │  MA200 ↘ (sloping down)
   └─────────────────────────────────
        ↓
   Price rallies DOWN strongly
        ↓
           SWING LOW (1.0950)
        ↓
   ┌─ BOUNCE
   │  Price touches MA50
   └─ MA50 ↘
        ↓
   Price pulls back UP
   (38.2%-61.8%)
        ↓
   ┌─ RETEST - 
   │  Price touches MA50 again
   │  (SECOND TOUCH)
   └─ Entry: Sell Stop below candle

"""

# ============================================================================
# KEY RULES (BOTH DIRECTIONS)
# ============================================================================

KEY_RULES = """
🎯 TCE ENTRY RULES

1. MAIN TREND MUST BE CLEAR
   ├─ MA50 and MA200 BOTH sloping in same direction
   ├─ For BUY: both UP
   ├─ For SELL: both DOWN
   └─ This is NON-NEGOTIABLE

2. IDENTIFY THE RETEST MA (which one bounced from)
   ├─ Scan which MA was closest to the retracement low/high
   ├─ The MA price bounced from = retest level
   ├─ Can be MA50, MA18, or MA6
   ├─ NOT a horizontal support/resistance level
   └─ IS a moving average that price respects

3. RETRACEMENT MUST BE 38.2% - 61.8%
   ├─ Shallower than 38.2% = not enough retracement
   ├─ Deeper than 61.8% = too much retracement (trend broken)
   ├─ Fibonacci depth = (High - Low) / Total Range
   └─ Calculate from swing high to retracement low

4. RETEST IS SECOND TOUCH (NOT FIRST TOUCH)
   ├─ First touch = initial bounce (identifies MA level)
   ├─ Second touch = retest (where we enter)
   ├─ Must be the SAME MA level as initial bounce
   ├─ Cannot skip levels or use different MA
   └─ Entry happens at candlestick confirmation on second touch

5. RETEST MA MUST BE SLOPING UP (BUY) OR DOWN (SELL)
   ├─ For BUY: retest MA slope > 0
   ├─ For SELL: retest MA slope < 0
   ├─ MA cannot be flat or against the trend
   └─ This confirms MA is valid support/resistance

6. CANDLESTICK CONFIRMATION AT RETEST
   ├─ NOT just any touch - needs confirmation
   ├─ Pin bar, engulfing, or reversal pattern
   ├─ Shows price respecting the MA level
   └─ Entry: Buy Stop / Sell Stop 2-3 pips from candle

7. RISK MANAGEMENT
   ├─ SL: 2-5 pips BELOW 61.8% level (BUY)
   ├─ SL: 2-5 pips ABOVE 61.8% level (SELL)
   ├─ TP: From entry with RR ratio (1:1.5 or 1:2)
   └─ Position size: Risk 1% of account = (SL distance × lot size)

8. CONFLUENCE
   ├─ ALL rules must pass simultaneously
   ├─ Main trend: ✅ MA50 + MA200 sloping correctly
   ├─ Retest MA: ✅ Sloping in trend direction
   ├─ Retracement: ✅ 38.2%-61.8% depth
   ├─ Second touch: ✅ Confirmed at same MA
   ├─ Candlestick: ✅ Confirmation pattern present
   └─ Enter only when ALL 5 conditions met

"""

# ============================================================================
# ENTRY CHECKLIST
# ============================================================================

ENTRY_CHECKLIST = """
✅ TCE ENTRY CHECKLIST

BEFORE ENTERING:

MAIN TREND
□ MA50 slope > 0 (BUY) or < 0 (SELL)
□ MA200 slope > 0 (BUY) or < 0 (SELL)
□ Both aligned in same direction

RETEST MA IDENTIFICATION
□ Identified which MA price bounced from (MA50/MA18/MA6)
□ That MA is the retest level
□ Retest MA slope correct (up for BUY, down for SELL)

RETRACEMENT DEPTH
□ Calculated Fibonacci depth
□ Depth between 38.2% and 61.8%
□ Record: ___.__% (example: 52.3%)

SECOND TOUCH
□ Price returned to same MA level
□ This is SECOND touch (not first)
□ Within 3 pips of MA level

CANDLESTICK CONFIRMATION
□ Confirmation pattern present at retest
□ Pin bar, engulfing, or reversal
□ Price rejecting further movement

ENTRY EXECUTION
□ Place Buy/Sell Stop 2-3 pips from candle
□ SL placed 2-5 pips beyond 61.8% level
□ TP calculated from RR ratio (1:1.5 or 1:2)
□ Position size: risk = 1% of account

RISK MANAGEMENT
□ Risk pips: _____ (SL to entry distance)
□ Lot size: _____ (calculated for 1% risk)
□ RR ratio: 1:_____ (expected ratio)

Ready to enter: YES □  NO □

"""

print(UPTREND_TCE)
print("\n" + "="*80 + "\n")
print(DOWNTREND_TCE)
print("\n" + "="*80 + "\n")
print(KEY_RULES)
print("\n" + "="*80 + "\n")
print(ENTRY_CHECKLIST)
