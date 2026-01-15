"""
API Views for No-Code Strategy Builder

REST API endpoints for strategy management, training, backtesting, and monitoring.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404

from .models import UserStrategy, StrategyTrade, StrategyBacktest
from .serializers import (
    UserStrategyListSerializer,
    UserStrategyDetailSerializer,
    UserStrategyCreateSerializer,
    UserStrategyUpdateSerializer,
    StrategyTradeSerializer,
    StrategyBacktestSerializer
)
from .workflow import NoCodeStrategyBuilder


class UserStrategyViewSet(viewsets.ModelViewSet):
    """
    ViewSet for user strategy CRUD operations.
    
    Endpoints:
    - GET    /api/strategies/          List user's strategies
    - POST   /api/strategies/          Create new strategy
    - GET    /api/strategies/{id}/     Get strategy details
    - PUT    /api/strategies/{id}/     Update strategy
    - PATCH  /api/strategies/{id}/     Partial update
    - DELETE /api/strategies/{id}/     Delete strategy
    
    Custom Actions:
    - POST   /api/strategies/{id}/activate/    Activate for live trading
    - POST   /api/strategies/{id}/deactivate/  Deactivate trading
    - GET    /api/strategies/{id}/status/      Get training/backtest status
    - POST   /api/strategies/{id}/backtest/    Run backtest
    - GET    /api/strategies/{id}/trades/      Get trade history
    - GET    /api/strategies/{id}/performance/ Get performance metrics
    """
    
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return only strategies belonging to current user"""
        return UserStrategy.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action"""
        if self.action == 'list':
            return UserStrategyListSerializer
        elif self.action == 'create':
            return UserStrategyCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return UserStrategyUpdateSerializer
        return UserStrategyDetailSerializer
    
    def create(self, request):
        """
        Create new strategy from user description.
        
        POST /api/strategies/
        Body: {
            "name": "My RSI Strategy",
            "description": "Buy when RSI < 30, sell when RSI > 70...",
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframes": ["H1"],
            "risk_percentage": 1.0,
            "max_concurrent_trades": 3
        }
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Use workflow builder to create and parse strategy
        builder = NoCodeStrategyBuilder(user_id=request.user.id)
        
        result = builder.create_strategy(
            description=serializer.validated_data['description'],
            name=serializer.validated_data.get('name')
        )
        
        if not result.get('success', True):
            return Response(
                {'error': result.get('error', 'Strategy creation failed')},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get the created strategy
        strategy = UserStrategy.objects.get(id=result['strategy_id'])
        
        # Update additional fields
        strategy.symbols = serializer.validated_data.get('symbols', [])
        strategy.timeframes = serializer.validated_data.get('timeframes', [])
        strategy.risk_percentage = serializer.validated_data.get('risk_percentage', 1.0)
        strategy.max_concurrent_trades = serializer.validated_data.get('max_concurrent_trades', 3)
        strategy.save()
        
        # Return detailed strategy info
        detail_serializer = UserStrategyDetailSerializer(strategy)
        return Response(detail_serializer.data, status=status.HTTP_201_CREATED)
    
    def destroy(self, request, pk=None):
        """
        Delete strategy.
        
        DELETE /api/strategies/{id}/
        """
        strategy = self.get_object()
        
        # Check if strategy is active
        if strategy.is_active:
            return Response(
                {'error': 'Cannot delete active strategy. Deactivate first.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        strategy.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """
        Activate strategy for live trading.
        
        POST /api/strategies/{id}/activate/
        """
        strategy = self.get_object()
        
        # Check if strategy is ready
        if strategy.status not in ['ready', 'paused']:
            return Response(
                {
                    'error': f'Cannot activate strategy with status "{strategy.status}". '
                             'Complete training and backtesting first.'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if ML model exists and is trained
        ml_models = strategy.strategymlmodel_set.filter(status='completed')
        if not ml_models.exists():
            return Response(
                {'error': 'No trained ML model found. Complete training first.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        strategy.is_active = True
        strategy.status = 'live'
        strategy.save()
        
        return Response({
            'message': 'Strategy activated successfully',
            'strategy_id': strategy.id,
            'status': strategy.status
        })
    
    @action(detail=True, methods=['post'])
    def deactivate(self, request, pk=None):
        """
        Deactivate strategy (pause live trading).
        
        POST /api/strategies/{id}/deactivate/
        """
        strategy = self.get_object()
        
        strategy.is_active = False
        strategy.status = 'paused'
        strategy.save()
        
        return Response({
            'message': 'Strategy deactivated successfully',
            'strategy_id': strategy.id,
            'status': strategy.status
        })
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """
        Get strategy training and backtest status.
        
        GET /api/strategies/{id}/status/
        """
        strategy = self.get_object()
        builder = NoCodeStrategyBuilder(user_id=request.user.id)
        
        status_info = builder.get_strategy_status(strategy.id)
        return Response(status_info)
    
    @action(detail=True, methods=['post'])
    def backtest(self, request, pk=None):
        """
        Run backtest on strategy.
        
        POST /api/strategies/{id}/backtest/
        Body: {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_balance": 10000
        }
        """
        strategy = self.get_object()
        
        # Check if ML model is trained
        ml_models = strategy.strategymlmodel_set.filter(status='completed')
        if not ml_models.exists():
            return Response(
                {'error': 'ML model not trained yet. Wait for training to complete.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get parameters
        start_date = request.data.get('start_date')
        end_date = request.data.get('end_date')
        initial_balance = request.data.get('initial_balance', 10000.0)
        
        # TODO: Implement backtesting
        # For now, return placeholder
        return Response({
            'message': 'Backtesting queued',
            'strategy_id': strategy.id,
            'status': 'queued',
            'note': 'Backtesting implementation in progress'
        }, status=status.HTTP_202_ACCEPTED)
    
    @action(detail=True, methods=['get'])
    def trades(self, request, pk=None):
        """
        Get trade history for strategy.
        
        GET /api/strategies/{id}/trades/?limit=50&offset=0
        """
        strategy = self.get_object()
        
        # Get query parameters
        limit = int(request.query_params.get('limit', 50))
        offset = int(request.query_params.get('offset', 0))
        
        trades = StrategyTrade.objects.filter(strategy=strategy).order_by('-entry_time')[offset:offset+limit]
        serializer = StrategyTradeSerializer(trades, many=True)
        
        return Response({
            'count': StrategyTrade.objects.filter(strategy=strategy).count(),
            'results': serializer.data
        })
    
    @action(detail=True, methods=['get'])
    def performance(self, request, pk=None):
        """
        Get strategy performance metrics.
        
        GET /api/strategies/{id}/performance/
        """
        strategy = self.get_object()
        
        # Get all trades
        trades = StrategyTrade.objects.filter(strategy=strategy)
        total_trades = trades.count()
        
        if total_trades == 0:
            return Response({
                'strategy_id': strategy.id,
                'total_trades': 0,
                'message': 'No trades executed yet'
            })
        
        # Calculate metrics
        winning_trades = trades.filter(outcome='win').count()
        losing_trades = trades.filter(outcome='loss').count()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t.profit_loss for t in trades.filter(outcome='win'))
        total_loss = abs(sum(t.profit_loss for t in trades.filter(outcome='loss')))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        net_profit = total_profit - total_loss
        
        return Response({
            'strategy_id': strategy.id,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_profit': round(total_profit, 2),
            'total_loss': round(total_loss, 2),
            'net_profit': round(net_profit, 2),
            'average_win': round(total_profit / winning_trades, 2) if winning_trades > 0 else 0,
            'average_loss': round(total_loss / losing_trades, 2) if losing_trades > 0 else 0
        })


class StrategyBacktestViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing backtest results.
    
    Endpoints:
    - GET /api/backtests/           List backtests
    - GET /api/backtests/{id}/      Get backtest details
    """
    
    permission_classes = [IsAuthenticated]
    serializer_class = StrategyBacktestSerializer
    
    def get_queryset(self):
        """Return only backtests for user's strategies"""
        return StrategyBacktest.objects.filter(
            strategy__user=self.request.user
        ).order_by('-created_at')


class StrategyTradeViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing trade history.
    
    Endpoints:
    - GET /api/trades/              List trades
    - GET /api/trades/{id}/         Get trade details
    """
    
    permission_classes = [IsAuthenticated]
    serializer_class = StrategyTradeSerializer
    
    def get_queryset(self):
        """Return only trades for user's strategies"""
        return StrategyTrade.objects.filter(
            strategy__user=self.request.user
        ).order_by('-entry_time')
