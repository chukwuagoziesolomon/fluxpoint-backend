"""
No-Code Strategy Builder Models

Allows users to define trading strategies in natural language,
converts them to rule-based logic, trains ML models, and automates trading.
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import json


class UserStrategy(models.Model):
    """
    User-created trading strategy with natural language description.
    """
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('parsing', 'Parsing Description'),
        ('validated', 'Validated'),
        ('training', 'Training ML Model'),
        ('trained', 'Trained'),
        ('backtesting', 'Backtesting'),
        ('ready', 'Ready for Trading'),
        ('live', 'Live Trading'),
        ('paused', 'Paused'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='strategies')
    name = models.CharField(max_length=200)
    description = models.TextField(
        help_text="Describe your trading strategy in plain English"
    )
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Parsed rule-based representation
    parsed_rules = models.JSONField(null=True, blank=True)
    validation_errors = models.JSONField(null=True, blank=True)
    
    # Trading settings
    symbols = models.JSONField(default=list, help_text="List of symbols to trade")
    timeframes = models.JSONField(default=list, help_text="List of timeframes")
    
    # Risk management
    risk_percentage = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(10.0)],
        help_text="Risk per trade as % of account"
    )
    max_concurrent_trades = models.IntegerField(default=3)
    
    # Performance tracking
    win_rate = models.FloatField(null=True, blank=True)
    profit_factor = models.FloatField(null=True, blank=True)
    total_trades = models.IntegerField(default=0)
    is_active = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'User Strategy'
        verbose_name_plural = 'User Strategies'
    
    def __str__(self):
        return f"{self.user.username} - {self.name}"


class StrategyComponent(models.Model):
    """
    Individual components/rules of a strategy (entry, exit, filters, etc.)
    """
    COMPONENT_TYPES = [
        ('entry', 'Entry Condition'),
        ('exit', 'Exit Condition'),
        ('filter', 'Filter'),
        ('risk_management', 'Risk Management'),
        ('indicator', 'Indicator'),
        ('timeframe', 'Timeframe Condition'),
    ]
    
    strategy = models.ForeignKey(UserStrategy, on_delete=models.CASCADE, related_name='components')
    type = models.CharField(max_length=20, choices=COMPONENT_TYPES)
    description = models.TextField()
    
    # Parsed rule representation
    rule_type = models.CharField(max_length=100)  # e.g., 'ma_cross', 'rsi_oversold', 'price_above_ma'
    parameters = models.JSONField(default=dict)  # e.g., {'ma_period': 20, 'threshold': 70}
    
    # Logic operators
    operator = models.CharField(
        max_length=10,
        choices=[('AND', 'AND'), ('OR', 'OR')],
        default='AND'
    )
    priority = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['strategy', 'priority']
    
    def __str__(self):
        return f"{self.strategy.name} - {self.get_type_display()}"


class StrategyIndicator(models.Model):
    """
    Technical indicators used in a strategy.
    """
    strategy = models.ForeignKey(UserStrategy, on_delete=models.CASCADE, related_name='indicators')
    
    name = models.CharField(max_length=50)  # e.g., 'MA', 'RSI', 'MACD', 'ATR'
    parameters = models.JSONField(default=dict)  # e.g., {'period': 14, 'type': 'exponential'}
    
    # For display and reference
    display_name = models.CharField(max_length=100)
    variable_name = models.CharField(max_length=50)  # e.g., 'rsi_14', 'ma_20'
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('strategy', 'variable_name')
    
    def __str__(self):
        return f"{self.strategy.name} - {self.display_name}"


class StrategyMLModel(models.Model):
    """
    ML model trained for a user strategy.
    """
    strategy = models.OneToOneField(UserStrategy, on_delete=models.CASCADE, related_name='ml_model')
    
    # Model info
    model_type = models.CharField(
        max_length=50,
        choices=[
            ('dnn', 'Deep Neural Network'),
            ('lstm', 'LSTM'),
            ('rf', 'Random Forest'),
            ('xgboost', 'XGBoost'),
        ],
        default='dnn'
    )
    model_path = models.CharField(max_length=500)
    
    # Training metadata
    trained_at = models.DateTimeField(null=True, blank=True)
    training_samples = models.IntegerField(default=0)
    validation_accuracy = models.FloatField(null=True, blank=True)
    validation_precision = models.FloatField(null=True, blank=True)
    validation_recall = models.FloatField(null=True, blank=True)
    
    # Feature engineering
    feature_names = models.JSONField(default=list)
    feature_count = models.IntegerField(default=0)
    
    # Hyperparameters
    hyperparameters = models.JSONField(default=dict)
    
    # Performance
    ml_threshold = models.FloatField(
        default=0.65,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Minimum probability to take trade"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"ML Model for {self.strategy.name}"


class StrategyBacktest(models.Model):
    """
    Backtest results for a strategy.
    """
    strategy = models.ForeignKey(UserStrategy, on_delete=models.CASCADE, related_name='backtests')
    
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    
    # Backtest settings
    initial_capital = models.FloatField(default=10000.0)
    use_ml_filter = models.BooleanField(default=False)
    
    # Results
    total_trades = models.IntegerField(default=0)
    winning_trades = models.IntegerField(default=0)
    losing_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(null=True, blank=True)
    
    total_pnl = models.FloatField(default=0.0)
    profit_factor = models.FloatField(null=True, blank=True)
    sharpe_ratio = models.FloatField(null=True, blank=True)
    max_drawdown = models.FloatField(null=True, blank=True)
    
    avg_win = models.FloatField(null=True, blank=True)
    avg_loss = models.FloatField(null=True, blank=True)
    largest_win = models.FloatField(null=True, blank=True)
    largest_loss = models.FloatField(null=True, blank=True)
    
    # Detailed results
    trades_detail = models.JSONField(default=list)
    equity_curve = models.JSONField(default=list)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.strategy.name} - Backtest {self.created_at.date()}"


class StrategyTrade(models.Model):
    """
    Trades executed by user strategies.
    """
    DIRECTION_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('cancelled', 'Cancelled'),
    ]
    
    strategy = models.ForeignKey(UserStrategy, on_delete=models.CASCADE, related_name='trades')
    
    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    direction = models.CharField(max_length=4, choices=DIRECTION_CHOICES)
    
    # Entry
    entry_timestamp = models.DateTimeField()
    entry_price = models.FloatField()
    position_size = models.FloatField()  # Lots
    
    # Exit
    stop_loss = models.FloatField()
    take_profit = models.FloatField()
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    exit_price = models.FloatField(null=True, blank=True)
    
    # Result
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    pnl = models.FloatField(null=True, blank=True)
    r_multiple = models.FloatField(null=True, blank=True)
    
    # ML prediction (if used)
    ml_probability = models.FloatField(null=True, blank=True)
    
    # Execution details
    execution_details = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-entry_timestamp']
    
    def __str__(self):
        return f"{self.strategy.name} - {self.symbol} {self.direction} ({self.status})"


class ParsedCondition(models.Model):
    """
    Stores parsed trading conditions from natural language.
    Reusable library of trading logic patterns.
    """
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    
    # Pattern matching
    keywords = models.JSONField(default=list)  # Keywords that trigger this condition
    patterns = models.JSONField(default=list)  # Regex patterns for matching
    
    # Implementation
    condition_type = models.CharField(max_length=50)
    required_indicators = models.JSONField(default=list)
    logic_template = models.TextField()  # Python code template
    
    # Examples
    example_phrases = models.JSONField(default=list)
    
    # Usage stats
    usage_count = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name

