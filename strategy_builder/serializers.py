"""
API Serializers for No-Code Strategy Builder

Converts database models to/from JSON for API communication.
"""

from rest_framework import serializers
from .models import (
    UserStrategy,
    StrategyComponent,
    StrategyIndicator,
    StrategyMLModel,
    StrategyBacktest,
    StrategyTrade
)


class StrategyIndicatorSerializer(serializers.ModelSerializer):
    """Serializer for strategy indicators"""
    
    class Meta:
        model = StrategyIndicator
        fields = ['id', 'indicator_name', 'parameters', 'timeframe']


class StrategyComponentSerializer(serializers.ModelSerializer):
    """Serializer for strategy components (entry/exit rules)"""
    
    class Meta:
        model = StrategyComponent
        fields = ['id', 'type', 'description', 'rule_type', 'parameters', 'operator', 'priority']


class StrategyMLModelSerializer(serializers.ModelSerializer):
    """Serializer for ML model training status"""
    
    class Meta:
        model = StrategyMLModel
        fields = [
            'id', 'model_type', 'status', 'accuracy', 'precision', 'recall',
            'f1_score', 'training_started_at', 'training_completed_at',
            'hyperparameters', 'feature_config'
        ]


class StrategyBacktestSerializer(serializers.ModelSerializer):
    """Serializer for backtest results"""
    
    class Meta:
        model = StrategyBacktest
        fields = [
            'id', 'start_date', 'end_date', 'total_trades', 'winning_trades',
            'losing_trades', 'win_rate', 'profit_factor', 'total_return',
            'max_drawdown', 'sharpe_ratio', 'created_at'
        ]


class StrategyTradeSerializer(serializers.ModelSerializer):
    """Serializer for individual trades"""
    
    class Meta:
        model = StrategyTrade
        fields = [
            'id', 'symbol', 'entry_time', 'exit_time', 'direction',
            'entry_price', 'exit_price', 'stop_loss', 'take_profit',
            'position_size', 'profit_loss', 'profit_loss_pips',
            'ml_probability', 'rl_action', 'outcome'
        ]


class UserStrategyListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing strategies"""
    
    class Meta:
        model = UserStrategy
        fields = [
            'id', 'name', 'status', 'is_active', 'created_at',
            'win_rate', 'profit_factor', 'total_trades'
        ]


class UserStrategyDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer with nested components"""
    
    components = StrategyComponentSerializer(many=True, read_only=True)
    indicators = StrategyIndicatorSerializer(many=True, read_only=True, source='strategyindicator_set')
    ml_models = StrategyMLModelSerializer(many=True, read_only=True, source='strategymlmodel_set')
    backtests = StrategyBacktestSerializer(many=True, read_only=True, source='strategybacktest_set')
    
    class Meta:
        model = UserStrategy
        fields = [
            'id', 'name', 'description', 'status', 'is_active',
            'created_at', 'updated_at', 'parsed_rules', 'validation_errors',
            'symbols', 'timeframes', 'risk_percentage', 'max_concurrent_trades',
            'win_rate', 'profit_factor', 'total_trades',
            'components', 'indicators', 'ml_models', 'backtests', 'feature_config'
        ]


class UserStrategyCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new strategies"""
    
    class Meta:
        model = UserStrategy
        fields = ['name', 'description', 'symbols', 'timeframes', 'risk_percentage', 'max_concurrent_trades']
    
    def validate_description(self, value):
        """Validate description is not empty"""
        if not value or len(value.strip()) < 20:
            raise serializers.ValidationError(
                "Strategy description must be at least 20 characters long"
            )
        return value
    
    def validate_risk_percentage(self, value):
        """Validate risk percentage is reasonable"""
        if value < 0.1 or value > 10.0:
            raise serializers.ValidationError(
                "Risk percentage must be between 0.1% and 10%"
            )
        return value


class UserStrategyUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating strategies"""
    
    class Meta:
        model = UserStrategy
        fields = ['name', 'description', 'symbols', 'timeframes', 'risk_percentage', 'max_concurrent_trades', 'is_active']
        
    def validate_is_active(self, value):
        """Validate strategy can be activated"""
        if value and self.instance.status not in ['ready', 'live', 'paused']:
            raise serializers.ValidationError(
                f"Cannot activate strategy with status '{self.instance.status}'. Must complete training first."
            )
        return value
