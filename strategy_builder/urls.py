"""
URL Configuration for Strategy Builder API

Maps URL patterns to ViewSets.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserStrategyViewSet, StrategyBacktestViewSet, StrategyTradeViewSet

router = DefaultRouter()
router.register(r'strategies', UserStrategyViewSet, basename='strategy')
router.register(r'backtests', StrategyBacktestViewSet, basename='backtest')
router.register(r'trades', StrategyTradeViewSet, basename='trade')

app_name = 'strategy_builder'

urlpatterns = [
    path('', include(router.urls)),
]
