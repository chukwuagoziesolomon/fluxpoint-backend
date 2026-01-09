# ============================================================================
# CELL 4 QUICK REFERENCE - WHAT GETS VALIDATED
# ============================================================================

## 📋 THE 7 VALIDATION RULES AT A GLANCE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1️⃣ TREND                         │ MA alignment + slopes + structure        │
│    MA6 > MA18 > MA50 > MA200 (uptrend)                                     │
│    All slopes positive (slope6, slope18, slope50 > 0)                       │
│    Higher highs AND higher lows in recent 50 candles                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2️⃣ FIBONACCI                     │ Price retraced 38.2%, 50%, or 61.8%      │
│    NOT deeper than 61.8% (no 78.6% or deeper)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2.5️⃣ SWING STRUCTURE             │ Curved pullback pattern (not sharp V)    │
│    Smooth retracement with multiple swing points                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3️⃣ AT MA LEVEL                  │ Price within 2% of: MA6, MA18, MA50, MA200│
│    Dynamic support/resistance (NOT horizontal S/R levels)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3.5️⃣ MA RETEST                  │ SECOND touch, not first bounce            │
│    Price touched MA → moved away → came back (retest)                      │
│    Lookback: 50-200 candles depending on timeframe                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4️⃣ CANDLESTICK PATTERN          │ At retest point, pattern must appear:    │
│    BUY: Pin Bar, Rejection, Engulfing, Morning Star                        │
│    SELL: Bearish equivalents (upper wick pin bar, etc)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5️⃣ HIGHER TIMEFRAME              │ If using HTF: must be in same direction  │
│    BUY: HTF in uptrend | SELL: HTF in downtrend                            │
│    ⚠️ SKIPPED for single-timeframe (H1-only) training                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6️⃣ CORRELATION                   │ Correlated pairs must move together       │
│    Correlation threshold: 0.6 (60% minimum)                                │
│    ⚠️ SKIPPED for single-pair training                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7️⃣ RISK MANAGEMENT               │ SL = 1.5×ATR, TP = dynamic RR ratio     │
│    Position size = Risk$ / (SL pips × $/pip)                               │
│    Account balance × Risk% / (SL distance × pip value)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 MOVING AVERAGES USED

**CORRECT MAs for Cell 4** (NOT MA5/MA20!):
```
MA6   = 6-period moving average
MA18  = 18-period moving average
MA50  = 50-period moving average
MA200 = 200-period moving average

UPTREND:   MA6 > MA18 > MA50 > MA200  ✅
DOWNTREND: MA200 > MA50 > MA18 > MA6  ✅
```

## 📊 CANDLESTICK PATTERNS DETECTED

**For BUY entries:**
```
PIN BAR:           Long lower wick + small body
  │
  ├─────
  │ ║ (body)
  │─┘ (long wick)

REJECTION:         Close at top after testing low
  ─── 
    │ (wick down)
  ──┘

ENGULFING:         Large green engulfs small red
  ┃    (green)
  ┗─┃  (red)

MORNING STAR:      3-candle V-pattern
  │
  ├─ (gap down)
  │─ (small)
  ├─ (gap up)
  └─ (bullish)
```

**For SELL entries:** Same patterns but inverted (upper wicks instead of lower)

## 💰 RISK MANAGEMENT CALCULATIONS

**Stop Loss:**
```
SL_distance = 1.5 × ATR (minimum 12 pips)
BUY:  SL_price = Entry - SL_distance
SELL: SL_price = Entry + SL_distance
```

**Take Profit (Dynamic):**
```
if SL_distance < 20 pips:
    TP_distance = 2 × SL_distance    (RR 1:2)
else:
    TP_distance = 1.5 × SL_distance  (RR 1:1.5)

BUY:  TP_price = Entry + TP_distance
SELL: TP_price = Entry - TP_distance
```

**Position Sizing:**
```
Risk_amount = Account_balance × Risk%
Lots = Risk_amount / (SL_pips × Pip_value_per_lot)

Example:
Account: $10,000
Risk: 1% = $100
SL: 20 pips
Pip value (EURUSD): $10 per pip per lot
Lots = $100 / (20 × $10) = 0.50 lots
```

## 🎯 WHAT CELL 4 DISPLAYS FOR EACH SETUP FOUND

```
Symbol:           EURUSD
Date:             2025-06-15 10:30:00
Entry Price:      1.10150
Direction:        BUY

Risk Management:
  • SL:           1.10350 (20 pips below entry)
  • TP:           1.10750 (40 pips above entry)
  • Risk/Reward:  1:2.0
  • Position:     0.50 lots
  • Risk $:       $100.00

Moving Averages:
  • MA6:          1.10100
  • MA18:         1.10080
  • MA50:         1.10050
  • MA200:        1.09900
  • ATR:          0.00100

Validation:
  1️⃣  Trend: ✅
  2️⃣  Fibonacci: ✅
  2.5️⃣ Swing: ✅
  3️⃣  MA Level: ✅
  3.5️⃣ MA Retest: ✅
  4️⃣  Candlestick: ✅
  5️⃣  Multi-TF: ✅
  6️⃣  Correlation: ✅
  7️⃣  Risk Mgmt: ✅
```

## ✅ SUCCESS CHECKLIST

When running Cell 4 in Colab:

- [ ] Cell runs without errors
- [ ] Scans 15 pairs for valid setups
- [ ] Shows count of valid setups found (should be >0, not 0!)
- [ ] Displays 3 sample setups with all details
- [ ] Shows Risk Management values for each setup
- [ ] All 7 validation rules showing ✅ for each setup
- [ ] Neural network trains on valid setups
- [ ] Shows training progress (epoch, loss)
- [ ] Final accuracy >70%
- [ ] Saves model to Google Drive

## ❌ TROUBLESHOOTING

**Problem: 0 valid setups found**

Likely causes:
- [ ] Wrong moving averages (are you using MA6/MA18 or MA5/MA20?)
- [ ] Trend structure broken (check MA alignment and slopes)
- [ ] Candlestick patterns not detected (check pattern detection logic)
- [ ] MA retest pattern not in historical data (need older price action)
- [ ] Fibonacci validation too strict

**Problem: Training accuracy too low (<50%)**

Likely causes:
- [ ] Invalid setups selected (check validation rules)
- [ ] Features not properly normalized
- [ ] Model architecture doesn't fit the data
- [ ] Training data size too small (<50 setups)

**Problem: Risk Management values seem wrong**

Check:
- [ ] ATR calculation (should be 14-period)
- [ ] SL should be 1.5×ATR (minimum 12 pips)
- [ ] TP should be 1:2 RR (or 1:1.5 for large SL)
- [ ] Position size should scale with risk

---

## 📚 REFERENCE FILES

- **CELL4_COMPLETE_TCE_VALIDATION.py** - Main Cell 4 code (paste into Colab)
- **CELL4_VALIDATION_CHECKLIST.md** - Detailed explanation of each rule
- **TCE_VALIDATION_RULES_COMPLETE.md** - Full rule documentation
- **DEBUG_VALIDATION_RULES.py** - Test script to verify rules work locally
