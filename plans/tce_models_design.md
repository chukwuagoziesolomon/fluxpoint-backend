# Adam Khoo TCE Strategy Data Models Design

## Overview
Based on the Adam Khoo TCE strategy rules, the following data models are designed to support market data (candles, indicators, swings), trades, and strategies. Models include support for multi-timeframe analysis, correlation checks, and outcome tracking.

## Key Entities Identified from Rules
- **Candles**: OHLCV data for price action, rejections, wicks.
- **Indicators**: Moving averages (MA6, MA18, MA50, MA200) with slopes.
- **Swings**: Swing highs/lows, retracements, Fibonacci levels.
- **Trades**: Entries, exits, outcomes, P&L.
- **Strategies**: TCE strategy with parameters.
- **Correlations**: Pair correlations for validation.

## Model Schemas

### 1. Candle Model
Stores OHLCV data for each symbol and timeframe.

```python
class Candle(models.Model):
    symbol = models.CharField(max_length=10)  # e.g., 'EURUSD'
    timeframe = models.CharField(max_length=10)  # e.g., 'M15', 'H1', 'D1'
    timestamp = models.DateTimeField()
    open_price = models.DecimalField(max_digits=10, decimal_places=5)
    high_price = models.DecimalField(max_digits=10, decimal_places=5)
    low_price = models.DecimalField(max_digits=10, decimal_places=5)
    close_price = models.DecimalField(max_digits=10, decimal_places=5)
    volume = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        unique_together = ('symbol', 'timeframe', 'timestamp')
        ordering = ['timestamp']
```

### 2. Indicator Model
Stores technical indicators linked to candles.

```python
class Indicator(models.Model):
    candle = models.ForeignKey(Candle, on_delete=models.CASCADE)
    type = models.CharField(max_length=20)  # e.g., 'MA6', 'MA18', 'MA50', 'MA200'
    value = models.DecimalField(max_digits=10, decimal_places=5)
    slope = models.DecimalField(max_digits=10, decimal_places=5)  # Positive for up, negative for down

    class Meta:
        unique_together = ('candle', 'type')
```

### 3. Swing Model
Stores swing points and retracements.

```python
class Swing(models.Model):
    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    type = models.CharField(max_length=10, choices=[('high', 'High'), ('low', 'Low')])
    price = models.DecimalField(max_digits=10, decimal_places=5)
    timestamp = models.DateTimeField()
    retracement_level = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)  # Fib level: 0.382, 0.5, 0.618
    is_valid = models.BooleanField(default=True)  # For Fib validation

    class Meta:
        ordering = ['timestamp']
```

### 4. Strategy Model
Defines the TCE strategy.

```python
class Strategy(models.Model):
    name = models.CharField(max_length=100, unique=True)  # 'TCE'
    description = models.TextField()
    parameters = models.JSONField()  # Store rules like MA periods, Fib levels, etc.
    is_active = models.BooleanField(default=True)
```

### 5. Correlation Model
Stores correlation data between pairs.

```python
class Correlation(models.Model):
    pair1 = models.CharField(max_length=10)
    pair2 = models.CharField(max_length=10)
    correlation_value = models.DecimalField(max_digits=5, decimal_places=2)  # -1 to 1
    timestamp = models.DateTimeField()
    timeframe = models.CharField(max_length=10)

    class Meta:
        unique_together = ('pair1', 'pair2', 'timestamp', 'timeframe')
```

### 6. Trade Model
Stores trade data with outcome tracking.

```python
class Trade(models.Model):
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    entry_candle = models.ForeignKey(Candle, on_delete=models.CASCADE, related_name='entry_trades')
    entry_price = models.DecimalField(max_digits=10, decimal_places=5)
    exit_candle = models.ForeignKey(Candle, on_delete=models.CASCADE, null=True, blank=True, related_name='exit_trades')
    exit_price = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    direction = models.CharField(max_length=4, choices=[('BUY', 'Buy'), ('SELL', 'Sell')])
    pnl = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    outcome = models.CharField(max_length=10, choices=[('WIN', 'Win'), ('LOSS', 'Loss'), ('OPEN', 'Open')], default='OPEN')
    correlation_checked = models.BooleanField(default=False)
    correlation_valid = models.BooleanField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['entry_candle__timestamp']
```

## Relationships
- **Indicators** are linked to **Candles** (1 Candle : Many Indicators).
- **Swings** are per symbol/timeframe, can reference **Candles** if needed (optional FK).
- **Trades** link to **Strategy**, **entry_candle**, and optionally **exit_candle**.
- **Correlations** are separate, used for checks in trade validation.
- Multi-timeframe: All models have `timeframe` field for filtering.
- Correlation checks: **Trade** has `correlation_checked` and `correlation_valid` fields; logic can query **Correlation** model.
- Outcome tracking: **Trade** has `pnl`, `outcome` fields.

## Mermaid Diagram
```mermaid
erDiagram
    Candle ||--o{ Indicator : has
    Candle ||--o{ Trade : "entry/exit"
    Strategy ||--o{ Trade : generates
    Swing
    Correlation
    Trade {
        strategy FK
        entry_candle FK
        exit_candle FK
        pnl
        outcome
        correlation_checked
    }
```

## Multi-Timeframe Support
- All models include `timeframe` field.
- For validation, queries can filter by timeframe (e.g., M15 entry requires H1 uptrend).

## Correlation Checks
- **Correlation** model stores historical correlations.
- **Trade** model flags if correlation was checked and valid.

## Outcome Tracking
- **Trade** model tracks entry/exit prices, calculates P&L, determines win/loss.

This design supports the TCE rules: trend via Indicators, swings via Swing model, Fib via retracement_level, S/R via Swings, rejections via Candle data, multi-TF via timeframe, correlations via Correlation model.