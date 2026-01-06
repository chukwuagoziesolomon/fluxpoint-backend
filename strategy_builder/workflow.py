"""
No-Code Strategy Builder Workflow

Complete flow for user-described strategies:
1. User describes strategy in plain English
2. AI (LLM) converts description → structured rules
3. Rules are validated for completeness
4. Features auto-generated from indicators
5. ML + RL training launched
6. User strategies isolated per account
"""

from typing import Dict, List, Optional
import json
import numpy as np
import pandas as pd
from datetime import datetime

from ..nlp.llm_parser import parse_with_llm
from ..nlp.parser import parse_strategy_description
from ..models import UserStrategy, StrategyComponent, StrategyIndicator, StrategyMLModel


class NoCodeStrategyBuilder:
    """
    Manages the complete no-code strategy workflow.
    """
    
    def __init__(self, user_id: int):
        """
        Args:
            user_id: User ID for strategy isolation
        """
        self.user_id = user_id
    
    def create_strategy(
        self,
        description: str,
        name: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict:
        """
        Create a new strategy from user description.
        
        Complete workflow:
        1. Parse description with LLM
        2. Validate rules
        3. Save to database
        4. Generate features
        5. Queue ML/RL training
        
        Args:
            description: User's strategy description in plain English
            name: Optional strategy name
            use_llm: Use LLM for parsing (True) or regex (False)
        
        Returns:
            Strategy creation result
        """
        print("="*60)
        print("NO-CODE STRATEGY BUILDER WORKFLOW")
        print("="*60)
        
        # Step 1: AI Parse Description
        print("\n[1/6] Parsing strategy with AI...")
        parsed = parse_strategy_description(description, use_llm=use_llm)
        
        if 'error' in parsed:
            return {
                'success': False,
                'error': 'Failed to parse strategy',
                'details': parsed['error']
            }
        
        print(f"✅ Parsed with {parsed.get('parsing_method', 'unknown')}")
        print(f"   - Indicators: {len(parsed.get('indicators', []))}")
        print(f"   - Entry conditions: {len(parsed.get('entry_conditions', []))}")
        print(f"   - Exit conditions: {len(parsed.get('exit_conditions', []))}")
        
        # Step 2: Validate Rules
        print("\n[2/6] Validating strategy rules...")
        validation_errors = parsed.get('validation_errors', [])
        
        if validation_errors:
            print(f"⚠️  Found {len(validation_errors)} validation issues:")
            for error in validation_errors:
                print(f"   - {error}")
        else:
            print("✅ Strategy is complete")
        
        # Step 3: Save to Database
        print("\n[3/6] Saving strategy to database...")
        strategy_name = name or parsed.get('strategy_name', 'Unnamed Strategy')
        
        strategy = UserStrategy.objects.create(
            user_id=self.user_id,
            name=strategy_name,
            description=description,
            parsed_rules=parsed,
            status='pending_validation' if validation_errors else 'ready_for_training',
            is_active=False
        )
        
        # Save strategy components
        self._save_components(strategy, parsed)
        self._save_indicators(strategy, parsed)
        
        print(f"✅ Strategy saved (ID: {strategy.id})")
        
        # Step 4: Generate Features
        print("\n[4/6] Generating ML features from indicators...")
        feature_config = self._generate_feature_config(parsed)
        strategy.feature_config = feature_config
        strategy.save()
        
        print(f"✅ Generated {len(feature_config.get('features', []))} features")
        
        # Step 5: Queue ML Training
        print("\n[5/6] Queueing ML training...")
        if not validation_errors:
            training_queued = self._queue_ml_training(strategy)
            if training_queued:
                print("✅ ML training queued")
            else:
                print("⚠️  ML training not queued (insufficient data)")
        else:
            print("⏭️  Skipping ML training (validation errors)")
        
        # Step 6: Queue RL Training
        print("\n[6/6] Queueing RL training...")
        if not validation_errors:
            rl_queued = self._queue_rl_training(strategy)
            if rl_queued:
                print("✅ RL training queued")
            else:
                print("⏭️  RL training will start after ML completes")
        else:
            print("⏭️  Skipping RL training (validation errors)")
        
        print("\n" + "="*60)
        print("STRATEGY CREATION COMPLETE")
        print("="*60)
        print(f"Strategy ID: {strategy.id}")
        print(f"Status: {strategy.status}")
        print(f"User Isolation: ✅ (User ID: {self.user_id})")
        print("="*60)
        
        return {
            'success': True,
            'strategy_id': strategy.id,
            'name': strategy_name,
            'status': strategy.status,
            'parsed_rules': parsed,
            'feature_count': len(feature_config.get('features', [])),
            'validation_errors': validation_errors,
            'isolated_user': self.user_id
        }
    
    def _save_components(self, strategy: UserStrategy, parsed: Dict):
        """Save strategy components (entry/exit conditions, filters)."""
        components = []
        
        # Entry conditions
        for condition in parsed.get('entry_conditions', []):
            StrategyComponent.objects.create(
                strategy=strategy,
                component_type='entry',
                condition_type=condition.get('type'),
                parameters=condition,
                logic_operator='AND'
            )
        
        # Exit conditions
        for condition in parsed.get('exit_conditions', []):
            StrategyComponent.objects.create(
                strategy=strategy,
                component_type='exit',
                condition_type=condition.get('type'),
                parameters=condition,
                logic_operator='OR'
            )
        
        # Filters
        for filter_cond in parsed.get('filters', []):
            StrategyComponent.objects.create(
                strategy=strategy,
                component_type='filter',
                condition_type=filter_cond.get('type'),
                parameters=filter_cond,
                logic_operator='AND'
            )
    
    def _save_indicators(self, strategy: UserStrategy, parsed: Dict):
        """Save strategy indicators."""
        for indicator in parsed.get('indicators', []):
            StrategyIndicator.objects.create(
                strategy=strategy,
                indicator_name=indicator.get('name'),
                parameters=indicator.get('parameters', {}),
                timeframe=indicator.get('timeframe', 'default')
            )
    
    def _generate_feature_config(self, parsed: Dict) -> Dict:
        """
        Generate ML feature configuration from parsed rules.
        
        Features are auto-generated based on indicators and conditions.
        """
        features = []
        
        # Extract indicator-based features
        for indicator in parsed.get('indicators', []):
            name = indicator.get('name')
            params = indicator.get('parameters', {})
            
            if name == 'MA':
                period = params.get('period', 20)
                features.extend([
                    f'distance_to_ma{period}',
                    f'ma{period}_slope',
                    f'price_above_ma{period}'
                ])
            
            elif name == 'RSI':
                period = params.get('period', 14)
                features.extend([
                    f'rsi{period}',
                    f'rsi{period}_slope',
                    f'rsi_oversold',
                    f'rsi_overbought'
                ])
            
            elif name == 'MACD':
                features.extend([
                    'macd_line',
                    'macd_signal',
                    'macd_histogram',
                    'macd_cross'
                ])
            
            elif name == 'BB':
                period = params.get('period', 20)
                features.extend([
                    f'bb{period}_upper',
                    f'bb{period}_lower',
                    f'bb{period}_position',
                    f'bb{period}_width'
                ])
            
            elif name == 'ATR':
                period = params.get('period', 14)
                features.extend([
                    f'atr{period}',
                    f'atr{period}_ratio'
                ])
        
        # Add market context features
        features.extend([
            'volatility',
            'trend_strength',
            'candle_body_size',
            'candle_wick_ratio',
            'volume_ratio'
        ])
        
        return {
            'features': features,
            'normalization': 'standard',
            'lookback_window': 50
        }
    
    def _queue_ml_training(self, strategy: UserStrategy) -> bool:
        """
        Queue ML training for the strategy.
        
        In production, this would add to a task queue (Celery).
        For now, we just mark it as queued.
        """
        # Check if enough data available
        # In production, check MT5 data availability
        
        # Create ML model record
        StrategyMLModel.objects.create(
            strategy=strategy,
            model_type='deep_learning',
            hyperparameters={
                'hidden_layers': [128, 64, 32],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            status='queued',
            training_started_at=None,
            training_completed_at=None
        )
        
        return True
    
    def _queue_rl_training(self, strategy: UserStrategy) -> bool:
        """
        Queue RL training for execution optimization.
        
        RL trains AFTER ML completes on valid setups.
        """
        # Create RL model record
        StrategyMLModel.objects.create(
            strategy=strategy,
            model_type='reinforcement_learning',
            hyperparameters={
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'gamma': 0.99,
                'timesteps': 100000
            },
            status='waiting_for_ml',
            training_started_at=None,
            training_completed_at=None
        )
        
        return True
    
    def list_user_strategies(self) -> List[Dict]:
        """
        List all strategies for this user (isolation).
        
        Returns:
            List of user's strategies
        """
        strategies = UserStrategy.objects.filter(user_id=self.user_id).order_by('-created_at')
        
        return [
            {
                'id': s.id,
                'name': s.name,
                'status': s.status,
                'is_active': s.is_active,
                'created_at': s.created_at.isoformat() if hasattr(s, 'created_at') else None,
                'performance': s.performance_metrics
            }
            for s in strategies
        ]
    
    def get_strategy_status(self, strategy_id: int) -> Dict:
        """
        Get status of a strategy.
        
        Args:
            strategy_id: Strategy ID
        
        Returns:
            Strategy status with training progress
        """
        try:
            strategy = UserStrategy.objects.get(id=strategy_id, user_id=self.user_id)
        except UserStrategy.DoesNotExist:
            return {'error': 'Strategy not found or access denied'}
        
        # Get ML training status
        ml_models = StrategyMLModel.objects.filter(strategy=strategy)
        
        return {
            'id': strategy.id,
            'name': strategy.name,
            'status': strategy.status,
            'is_active': strategy.is_active,
            'ml_training': [
                {
                    'type': m.model_type,
                    'status': m.status,
                    'accuracy': m.accuracy,
                    'started': m.training_started_at.isoformat() if m.training_started_at else None,
                    'completed': m.training_completed_at.isoformat() if m.training_completed_at else None
                }
                for m in ml_models
            ],
            'performance': strategy.performance_metrics
        }


def create_user_strategy(
    user_id: int,
    description: str,
    name: Optional[str] = None,
    use_llm: bool = True
) -> Dict:
    """
    Convenience function to create a user strategy.
    
    Args:
        user_id: User ID (for isolation)
        description: Strategy description in plain English
        name: Optional strategy name
        use_llm: Use LLM for parsing
    
    Returns:
        Strategy creation result
    """
    builder = NoCodeStrategyBuilder(user_id=user_id)
    return builder.create_strategy(description, name=name, use_llm=use_llm)
