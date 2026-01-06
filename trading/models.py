from django.db import models


class Candle(models.Model):
    symbol = models.CharField(max_length=10)  # e.g., 'EURUSD'
    timeframe = models.CharField(max_length=10)  # e.g., 'M15', 'H1', 'D1'
    timestamp = models.DateTimeField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()

    class Meta:
        unique_together = ('symbol', 'timeframe', 'timestamp')
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.symbol} {self.timeframe} {self.timestamp}"


class Indicator(models.Model):
    candle = models.ForeignKey(Candle, on_delete=models.CASCADE)
    type = models.CharField(max_length=20)  # e.g., 'MA6', 'MA18', 'MA50', 'MA200'
    value = models.FloatField()
    slope = models.FloatField()  # Positive for up, negative for down

    class Meta:
        unique_together = ('candle', 'type')

    def __str__(self):
        return f"{self.candle} {self.type}: {self.value}"


class Swing(models.Model):
    SWING_TYPES = [
        ('high', 'High'),
        ('low', 'Low'),
    ]

    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    type = models.CharField(max_length=10, choices=SWING_TYPES)
    price = models.FloatField()
    timestamp = models.DateTimeField()
    retracement_level = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)  # Fib level: 0.382, 0.5, 0.618
    is_valid = models.BooleanField(default=True)  # For Fib validation

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.symbol} {self.timeframe} {self.type} at {self.price}"


class Strategy(models.Model):
    name = models.CharField(max_length=100, unique=True)  # 'TCE'
    description = models.TextField()
    parameters = models.JSONField()  # Store rules like MA periods, Fib levels, etc.
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class Correlation(models.Model):
    pair1 = models.CharField(max_length=10)
    pair2 = models.CharField(max_length=10)
    correlation_value = models.FloatField()  # -1 to 1
    timestamp = models.DateTimeField()
    timeframe = models.CharField(max_length=10)

    class Meta:
        unique_together = ('pair1', 'pair2', 'timestamp', 'timeframe')

    def __str__(self):
        return f"{self.pair1} vs {self.pair2}: {self.correlation_value}"


class Trade(models.Model):
    DIRECTIONS = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]

    OUTCOMES = [
        ('WIN', 'Win'),
        ('LOSS', 'Loss'),
        ('OPEN', 'Open'),
    ]

    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    entry_candle = models.ForeignKey(Candle, on_delete=models.CASCADE, related_name='entry_trades')
    entry_price = models.FloatField()
    exit_candle = models.ForeignKey(Candle, on_delete=models.CASCADE, null=True, blank=True, related_name='exit_trades')
    exit_price = models.FloatField(null=True, blank=True)
    direction = models.CharField(max_length=4, choices=DIRECTIONS)
    pnl = models.FloatField(null=True, blank=True)
    outcome = models.CharField(max_length=10, choices=OUTCOMES, default='OPEN')
    correlation_checked = models.BooleanField(default=False)
    correlation_valid = models.BooleanField(null=True, blank=True)
    risk_r = models.FloatField(null=True, blank=True)  # R-multiple
    stop_loss = models.FloatField(null=True, blank=True)
    take_profit = models.FloatField(null=True, blank=True)
    ml_probability = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['entry_candle__timestamp']

    def __str__(self):
        return f"{self.strategy.name} {self.direction} {self.symbol} {self.outcome}"
