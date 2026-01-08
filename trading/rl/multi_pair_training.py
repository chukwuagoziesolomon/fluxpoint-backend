"""
Multi-Pair RL Training

Trains a single RL agent on multiple currency pairs simultaneously.
The agent learns generalizable execution patterns across pairs.

Key Benefits:
- More training data = better generalization
- Agent learns pair-agnostic execution strategies
- Features are already normalized (ATR-based), so pair-agnostic
- Larger sample size = more robust learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .environment import create_execution_env
from .agent import create_execution_agent


class MultiPairRLTrainer:
    """
    Trains RL agent on multiple currency pairs simultaneously.
    """
    
    def __init__(
        self,
        model_name: str = "tce_execution_multipair",
        initial_balance: float = 10000,
        risk_percentage: float = 1.0,
        symbols: List[str] = None,
        require_candlestick_pattern: bool = True,
        enforce_risk_management: bool = True
    ):
        """
        Args:
            model_name: Name for the RL model
            initial_balance: Starting balance per symbol simulation
            risk_percentage: % of account to risk per trade
            symbols: List of trading pairs (e.g., ['EURUSD', 'GBPUSD', 'USDJPY'])
            require_candlestick_pattern: Enforce candlestick confirmation
            enforce_risk_management: Enforce risk management rules
        """
        self.model_name = model_name
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD']
        self.require_candlestick_pattern = require_candlestick_pattern
        self.enforce_risk_management = enforce_risk_management
        self.agent = None
        self.training_history = {
            'total_setups': 0,
            'setups_per_pair': {},
            'training_start': None,
            'training_end': None
        }
    
    def prepare_training_data(
        self,
        pair_data: Dict[str, Tuple[pd.DataFrame, List[Dict]]]
    ) -> Tuple:
        """
        Prepare multi-pair training data by combining all pairs.
        
        Args:
            pair_data: Dict mapping symbol -> (candles_df, valid_setups_list)
                Example:
                {
                    'EURUSD': (df_eur, setups_eur),
                    'GBPUSD': (df_gbp, setups_gbp),
                    'USDJPY': (df_jpy, setups_jpy)
                }
        
        Returns:
            (train_env, eval_env, statistics) tuple
        """
        print("\n" + "="*70)
        print("MULTI-PAIR RL TRAINING DATA PREPARATION")
        print("="*70)
        
        combined_candles = []
        combined_setups = []
        pair_stats = {}
        
        # Process each pair
        for symbol in self.symbols:
            if symbol not in pair_data:
                print(f"⚠️  Skipping {symbol}: Not in provided data")
                continue
            
            candles_df, setups = pair_data[symbol]
            
            if len(setups) == 0:
                print(f"⚠️  Skipping {symbol}: No valid setups found")
                continue
            
            print(f"\n📊 Processing {symbol}:")
            print(f"   Candles: {len(candles_df)} bars")
            print(f"   Valid setups: {len(setups)} trades")
            
            # Verify pair-agnostic features
            if len(setups) > 0 and 'features' in setups[0]:
                feature_dim = len(setups[0]['features'])
                print(f"   Feature dimension: {feature_dim}")
            
            # Add symbol to each setup for tracking
            for setup in setups:
                setup['symbol'] = symbol
            
            combined_setups.extend(setups)
            pair_stats[symbol] = {
                'candles': len(candles_df),
                'setups': len(setups),
                'date_range': f"{candles_df.index[0]} to {candles_df.index[-1]}" if len(candles_df) > 0 else "N/A"
            }
        
        if len(combined_setups) == 0:
            raise ValueError("No valid setups found across any pairs")
        
        print(f"\n{'='*70}")
        print(f"✅ Total Combined Setups: {len(combined_setups)}")
        print(f"{'='*70}")
        
        # Split data: 80% train, 20% eval
        # IMPORTANT: Split by setup index to avoid temporal leakage
        split_idx = int(len(combined_setups) * 0.8)
        
        train_setups = combined_setups[:split_idx]
        eval_setups = combined_setups[split_idx:]
        
        print(f"\nData Split:")
        print(f"  Training setups:   {len(train_setups)}")
        print(f"  Evaluation setups: {len(eval_setups)}")
        
        # Group setups by pair for environment creation
        train_setups_by_pair = {}
        eval_setups_by_pair = {}
        
        for setup in train_setups:
            symbol = setup.get('symbol', 'UNKNOWN')
            if symbol not in train_setups_by_pair:
                train_setups_by_pair[symbol] = []
            train_setups_by_pair[symbol].append(setup)
        
        for setup in eval_setups:
            symbol = setup.get('symbol', 'UNKNOWN')
            if symbol not in eval_setups_by_pair:
                eval_setups_by_pair[symbol] = []
            eval_setups_by_pair[symbol].append(setup)
        
        print(f"\nTraining setups per pair:")
        for symbol, setups in train_setups_by_pair.items():
            print(f"  {symbol}: {len(setups)} setups")
        
        print(f"\nEvaluation setups per pair:")
        for symbol, setups in eval_setups_by_pair.items():
            print(f"  {symbol}: {len(setups)} setups")
        
        # Create combined environments
        # Note: In a real system, you'd interleave pairs or use multi-env wrappers
        # For now, we'll concatenate candles and setups
        train_env = self._create_combined_env(
            pair_data=pair_data,
            setups=train_setups,
            is_training=True
        )
        
        eval_env = self._create_combined_env(
            pair_data=pair_data,
            setups=eval_setups,
            is_training=False
        )
        
        self.training_history['total_setups'] = len(combined_setups)
        self.training_history['setups_per_pair'] = pair_stats
        
        return train_env, eval_env, {
            'total_setups': len(combined_setups),
            'train_setups': len(train_setups),
            'eval_setups': len(eval_setups),
            'pair_stats': pair_stats,
            'setups_per_pair': train_setups_by_pair
        }
    
    def _create_combined_env(
        self,
        pair_data: Dict[str, Tuple[pd.DataFrame, List[Dict]]],
        setups: List[Dict],
        is_training: bool = True
    ):
        """
        Create environment with combined multi-pair data.
        
        Args:
            pair_data: Dict of symbol -> (candles, setups)
            setups: Combined setups for this split (train/eval)
            is_training: Whether this is training or evaluation
        
        Returns:
            Combined environment
        """
        # Combine candles from all pairs
        combined_candles = []
        
        for symbol in self.symbols:
            if symbol not in pair_data:
                continue
            
            candles_df, _ = pair_data[symbol]
            
            # Add symbol column for tracking
            candles_df_copy = candles_df.copy()
            candles_df_copy['symbol'] = symbol
            
            combined_candles.append(candles_df_copy)
        
        # Concatenate all candles (will be in symbol-based order)
        if combined_candles:
            combined_candles_df = pd.concat(combined_candles, ignore_index=True)
        else:
            raise ValueError("No candle data found")
        
        # Create environment with combined data
        env = create_execution_env(
            candles=combined_candles_df,
            valid_setups=setups,
            initial_balance=self.initial_balance,
            risk_percentage=self.risk_percentage,
            symbol="MULTIPAIR",  # Generic multi-pair identifier
            require_candlestick_pattern=self.require_candlestick_pattern,
            enforce_risk_management=self.enforce_risk_management
        )
        
        return env
    
    def train(
        self,
        train_env,
        eval_env=None,
        total_timesteps: int = 200000,
        eval_freq: int = 10000,
        save_freq: int = 20000
    ) -> Dict:
        """
        Train RL agent on multi-pair data.
        
        Args:
            train_env: Training environment (combined pairs)
            eval_env: Evaluation environment (combined pairs)
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            save_freq: Save frequency
        
        Returns:
            Training metrics
        """
        print("\n" + "="*70)
        print("MULTI-PAIR RL TRAINING")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Training on pairs: {', '.join(self.symbols)}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Risk per trade: {self.risk_percentage}%")
        print("="*70 + "\n")
        
        # Create agent
        self.agent = create_execution_agent(
            env=train_env,
            model_name=self.model_name
        )
        
        # Train
        self.training_history['training_start'] = datetime.now().isoformat()
        
        metrics = self.agent.train(
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            eval_env=eval_env,
            save_freq=save_freq
        )
        
        self.training_history['training_end'] = datetime.now().isoformat()
        
        # Evaluate
        if eval_env is not None:
            eval_metrics = self.agent.evaluate(eval_env, n_episodes=20)
            metrics['eval'] = eval_metrics
            
            print("\n" + "="*70)
            print("MULTI-PAIR EVALUATION RESULTS")
            print("="*70)
            print(f"Mean Reward:      {eval_metrics['mean_reward']:.2f}")
            print(f"Mean R-Multiple:  {eval_metrics['mean_r_multiple']:.2f}R")
            print(f"Mean Win Rate:    {eval_metrics['mean_win_rate']:.2%}")
            print("="*70)
            
            # Per-pair analysis (if tracking available)
            if hasattr(eval_env, 'get_pair_stats'):
                pair_stats = eval_env.get_pair_stats()
                print("\nPer-Pair Performance:")
                for pair, stats in pair_stats.items():
                    print(f"  {pair}: {stats}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model."""
        if self.agent is None:
            raise ValueError("No trained agent to save")
        
        self.agent.save(path)
        
        # Save training history
        history_path = Path(path).parent / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        print(f"Model saved to {path}")
        print(f"History saved to {history_path}")
    
    def load_model(self, path: Optional[str] = None):
        """Load trained model."""
        if self.agent is None:
            raise ValueError("Must initialize agent first")
        
        self.agent.load(path)
        print(f"Model loaded from {path}")


def train_rl_multipair(
    pair_data: Dict[str, Tuple[pd.DataFrame, List[Dict]]],
    symbols: List[str] = None,
    model_name: str = "tce_execution_multipair",
    initial_balance: float = 10000,
    risk_percentage: float = 1.0,
    total_timesteps: int = 200000
) -> Dict:
    """
    Convenience function to train RL agent on multiple pairs.
    
    Args:
        pair_data: Dict mapping symbol -> (candles_df, valid_setups)
        symbols: Specific symbols to train on (subset of pair_data.keys())
        model_name: Model name
        initial_balance: Starting balance
        risk_percentage: Risk per trade
        total_timesteps: Training steps
    
    Returns:
        Training metrics
    
    Example:
        >>> pair_data = {
        ...     'EURUSD': (df_eur, setups_eur),
        ...     'GBPUSD': (df_gbp, setups_gbp),
        ...     'USDJPY': (df_jpy, setups_jpy)
        ... }
        >>> metrics = train_rl_multipair(pair_data, total_timesteps=200000)
    """
    trainer = MultiPairRLTrainer(
        model_name=model_name,
        initial_balance=initial_balance,
        risk_percentage=risk_percentage,
        symbols=symbols or list(pair_data.keys())
    )
    
    # Prepare data
    train_env, eval_env, stats = trainer.prepare_training_data(pair_data)
    
    print(f"\n📈 Training Stats:")
    print(f"   Total setups: {stats['total_setups']}")
    print(f"   Train/Eval split: {stats['train_setups']}/{stats['eval_setups']}")
    
    # Train
    metrics = trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=total_timesteps
    )
    
    return metrics
