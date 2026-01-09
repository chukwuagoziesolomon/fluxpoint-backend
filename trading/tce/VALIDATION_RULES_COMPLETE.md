# Complete TCE Validation Framework - 5 Rules

## Rule 1️⃣: TREND CONFIRMATION (with Swing Structure)
**Status**: ✅ IMPLEMENTED in utils.py & structure.py

**Logic**:
- **UPTREND**: MA6 > MA18 > MA50 + 2+ positive slopes + HIGHER LOWS
  - Higher lows = 2nd dip is shallower than 1st dip (trend strengthening)
- **DOWNTREND**: MA50 > MA18 > MA6 + 2+ negative slopes + LOWER HIGHS
  - Lower highs = 2nd rally is lower than 1st rally (trend strengthening)

---

## Rule 2️⃣: CORRELATION PAIRS (Related Pairs Must Align)
**Status**: ⏳ NEEDS CLARIFICATION & IMPLEMENTATION

**Logic**:
- Identify correlated pairs (e.g., EURUSD & GBPUSD, EURJPY & GBPJPY)
- Both pairs must be in the SAME trend direction
- If one is UP and another is DOWN → INVALID setup

**Questions**:
- How do we define which pairs are "related"?
- What correlation threshold? (0.6+, 0.7+, 0.8+?)
- Do we check correlation coefficient in real-time or use hardcoded pairs?

---

## Rule 3️⃣: MULTI-TIMEFRAME TREND CONFIRMATION
**Status**: ⏳ PARTIALLY IMPLEMENTED (needs enhancement)

**Logic**:
- When trading on timeframe X, ALL higher timeframes must confirm the trend

**Examples**:
- Trading **M15** → **M30 AND H1** must be in uptrend
- Trading **H1** → **H4 AND D1** must be in uptrend
- Trading **H4** → **D1 AND W1** must be in uptrend

**Current Code Issue**:
- `higher_timeframe_confirmed()` checks if HTF is in same direction
- But we need to ensure we're checking the RIGHT higher timeframes
- Need to validate that we receive candles from correct timeframes

---

## Rule 4️⃣: MA RETEST HIERARCHY (Cascade Across Timeframes)
**Status**: ⏳ PARTIALLY IMPLEMENTED (needs enhancement)

**Logic**: Each timeframe up tests progressively SHORTER MA period

**Examples**:
- **M15 is retesting 50EMA** → **H1 must be retesting 18MA**
- **H1 is retesting 18MA** → **H4 must be retesting 6MA**
- **H4 is retesting 6MA** → **D1 must be in uptrend** (just trend confirmation)

**Current Code**:
- `ma_hit_confirmed()` exists but logic needs refinement
- Need to map: M15→50EMA, M30→20EMA, H1→18MA, H4→6MA, D1→trend
- Each level down tests ONE period shorter

**MA Mapping**:
```
M15  → Tests 50EMA → H1 should test 18MA
M30  → Tests 20EMA → H1 should test 6MA
H1   → Tests 18MA  → H4 should test 6MA
H4   → Tests 6MA   → D1 should be in trend
D1   → Just trend confirmation
```

---

## Rule 5️⃣: SUPPORT/RESISTANCE FILTER (No Entries AT S/R)
**Status**: ❌ NEEDS IMPLEMENTATION

**Logic**:
- Check all timeframes for horizontal S/R levels
- If current setup is AT/NEAR a S/R level → REJECT (price might get stuck)
- If setup is BETWEEN S/R levels → ACCEPT (clean bounce, natural overbought recovery)

**Why**:
- Overbought bounce at S/R = price might get rejected/stuck
- Overbought bounce between S/R = clean move, likely to continue

**Implementation Needed**:
- Function: `is_at_support_resistance(candle, sr_levels, tolerance_pips=5)`
- Collect S/R from all timeframes
- Check if candle.low (for BUY) or candle.high (for SELL) is within tolerance of any S/R

---

## Rule 6️⃣: RISK MANAGEMENT & LOT SIZING
**Status**: ⏳ PARTIALLY IMPLEMENTED (needs position placement logic)

**Logic**:
- **BEFORE entering trade**, calculate position size
- Formula: `Lots = (Account Balance × Risk%) / (SL Distance in pips × Pip Value per Lot)`
- Example: Account=$10,000, Risk=1%, SL=30pips, PipValue=$10/lot
  - Lots = (10,000 × 0.01) / (30 × 10) = 100 / 300 = 0.33 lots

**Current Code**: ✅ Already implemented in risk_management.py

---

## Rule 7️⃣: ORDER PLACEMENT STRATEGY (Pending Orders)
**Status**: ❌ NEEDS IMPLEMENTATION

**Logic**:
- **Don't use market execution** (buy/sell immediately)
- Use **pending stop orders** to enter AFTER confirmation
- Place order **2-3 pips above/below** the confirmation candle

**BUY Entry (Uptrend)**:
```
Setup identified at MA
→ Confirmation candle closes above MA
→ Place BUY STOP at (ConfirmationCandle.High + 2-3 pips)
→ When price reaches BUY STOP, order executes
→ Captures momentum after confirmation
```

**SELL Entry (Downtrend)**:
```
Setup identified at MA
→ Confirmation candle closes below MA
→ Place SELL STOP at (ConfirmationCandle.Low - 2-3 pips)
→ When price reaches SELL STOP, order executes
→ Captures momentum after confirmation
```

**Why This Works**:
- Avoids whipsaws (entering too early)
- Captures confirmed momentum
- Gives buffer (2-3 pips) above/below candle

---

## Rule 8️⃣: FIBONACCI DEPTH VALIDATION (Clarification)
**Status**: ✅ IMPLEMENTED (needs understanding)

**Logic**:
- **Price rarely hits MA exactly** → it goes BELOW the MA (uptrend) or ABOVE the MA (downtrend)
- **Fibonacci measures**: How deep did price retrace?
  - Shallow dip = 38.2% (strong, quick bounce)
  - Medium dip = 50% (balanced)
  - Deep dip = 61.8% (weak, but valid limit)
  - **Beyond 61.8% = INVALID** (price crashed too hard, setup too weak)

**Example (Uptrend)**:
```
MA18 = 1.0820
Price dips to 1.0790
Depth = (1.0820 - 1.0790) / (1 * ATR) = how many ATRs below MA?

If depth = 0.382 ATRs → 38.2% Fib (strong dip, valid)
If depth = 0.618 ATRs → 61.8% Fib (maximum valid depth)
If depth = 0.786 ATRs → Beyond 61.8% (INVALID - too weak)
```

**Current Code**: ✅ Already implemented - rejects if beyond 61.8%

---

## Summary Table

| Rule | Name | Status | Priority |
|------|------|--------|----------|
| 1 | Trend + Swing | ✅ Done | Core |
| 2 | Correlation Pairs | ⏳ Uses coefficients | High |
| 3 | Multi-TF Trend | ⚠️ Partial | High |
| 4 | MA Retest Hierarchy | ⚠️ Partial | High |
| 5 | S/R Filter | ⏳ Uses bounce detection | High |
| 6 | Risk Management | ✅ Done | Core |
| 7 | Order Placement | ⏳ Needs implementation | High |
| 8 | Fibonacci Depth | ✅ Done | Core |

---

## COMPLETE TCE TRADING WORKFLOW

1. **Identify Setup** (Rules 1-5):
   - ✅ Trend confirmed (Rule 1)
   - ✅ Correlation aligned (Rule 2)
   - ✅ Multi-TF confirmed (Rule 3)
   - ✅ MA retest hierarchy confirmed (Rule 4)
   - ✅ Not at S/R level (Rule 5)

2. **Validate Entry Details**:
   - ✅ Fibonacci depth valid (Rule 8) - price went 38.2-61.8% below MA
   - ✅ Candlestick confirmation pattern formed
   - ✅ Calculate SL & TP (Rule 6)

3. **Place Order** (Rule 7):
   - Place BUY STOP 2-3 pips above confirmation candle high (uptrend)
   - Place SELL STOP 2-3 pips below confirmation candle low (downtrend)
   - This ensures entry AFTER momentum confirms

4. **Execute Trade**:
   - Risk management validated
   - Position size calculated
   - Stop loss at calculated level
   - Take profit at RR target
   - Trade executes on pending order fill
