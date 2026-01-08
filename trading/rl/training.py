"""
RL Training Pipeline for Trade Execution Optimization

Trains RL agent to optimize execution of VALID trading setups.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json

from .environment import create_execution_env
from .agent import create_execution_agent


class RLExecutionTrainer:
    """
    Manages RL training for trade execution optimization.
    """
    
    def __init__(
        self,
        model_name: str = "tce_execution",
        initial_balance: float = 10000,
        risk_percentage: float = 1.0,
        symbol: str = "EURUSD",
        require_candlestick_pattern: bool = True,
        enforce_risk_management: bool = True
    ):
        """
        Args:
            model_name: Name for the RL model
            initial_balance: Starting balance for simulation
            risk_percentage: % of account to risk per trade
            symbol: Trading pair (e.g., 'EURUSD')
            require_candlestick_pattern: Enforce candlestick confirmation
            enforce_risk_management: Enforce risk management rules
        """
        self.model_name = model_name
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.symbol = symbol
        self.require_candlestick_pattern = require_candlestick_pattern
        self.enforce_risk_management = enforce_risk_management
        self.agent = None
    
    def prepare_training_data(
        self,
        candles: pd.DataFrame,
        valid_setups: List[Dict]
    ) -> tuple:
        """
        Prepare data for RL training.
        
        Args:
            candles: Historical OHLCV data
            valid_setups: List of validated trading setups with features
        
        Returns:
            (train_env, eval_env) tuple
        """
        # Split data: 80% train, 20% eval
        split_idx = int(len(candles) * 0.8)
        
        train_candles = candles.iloc[:split_idx].reset_index(drop=True)
        eval_candles = candles.iloc[split_idx:].reset_index(drop=True)
        
        # Split setups based on timestamp
        train_setups = [
            s for s in valid_setups 
            if s.get('timestamp_idx', 0) < split_idx
        ]
        eval_setups = [
            s for s in valid_setups 
            if s.get('timestamp_idx', 0) >= split_idx
        ]
        
        # Create environments with risk management enabled
        train_env = create_execution_env(
            candles=train_candles,
            valid_setups=train_setups,
            initial_balance=self.initial_balance,
            risk_percentage=self.risk_percentage,
            symbol=self.symbol,
            require_candlestick_pattern=self.require_candlestick_pattern,
            enforce_risk_management=self.enforce_risk_management
        )
        
        eval_env = create_execution_env(
            candles=eval_candles,
            valid_setups=eval_setups,
            initial_balance=self.initial_balance,
            risk_percentage=self.risk_percentage,
            symbol=self.symbol,
            require_candlestick_pattern=self.require_candlestick_pattern,
            enforce_risk_management=self.enforce_risk_management
        )
        
        print(f"Training data: {len(train_candles)} candles, {len(train_setups)} setups")
        print(f"Evaluation data: {len(eval_candles)} candles, {len(eval_setups)} setups")
        print(f"Risk management: enforce={self.enforce_risk_management}, candlestick={self.require_candlestick_pattern}")
        print(f"Risk per trade: {self.risk_percentage}% of account")
        
        return train_env, eval_env
    
    def train(
        self,
        train_env,
        eval_env=None,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        save_freq: int = 10000
    ) -> Dict:
        """
        Train RL agent.
        
        Args:
            train_env: Training environment
            eval_env: Evaluation environment
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            save_freq: Save frequency
        
        Returns:
            Training metrics
        """
        # Create agent
        self.agent = create_execution_agent(
            env=train_env,
            model_name=self.model_name
        )
        
        # Train
        metrics = self.agent.train(
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            eval_env=eval_env,
            save_freq=save_freq
        )
        
        # Evaluate
        if eval_env is not None:
            eval_metrics = self.agent.evaluate(eval_env, n_episodes=10)
            metrics['eval'] = eval_metrics
            
            print("\n" + "="*60)
            print("Evaluation Results:")
            print("="*60)
            print(f"Mean Reward: {eval_metrics['mean_reward']:.2f}")
            print(f"Mean R-Multiple: {eval_metrics['mean_r_multiple']:.2f}R")
            print(f"Mean Win Rate: {eval_metrics['mean_win_rate']:.2%}")
            print("="*60)
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model."""
        if self.agent is None:
            raise ValueError("No trained agent to save")
        
        self.agent.save(path)
    
    def load_model(self, path: Optional[str] = None):
        """Load trained model."""
        # Need to create agent first with dummy env
        # In practice, this would be loaded with the actual environment
        pass


def train_rl_execution(
    candles: pd.DataFrame,
    valid_setups: List[Dict],
    model_name: str = "tce_execution",
    initial_balance: float = 10000,
    total_timesteps: int = 100000
) -> Dict:
    """
    Convenience function to train RL execution agent.
    
    Args:
        candles: Historical OHLCV data
        valid_setups: List of validated setups from strategy
        model_name: Model name
        initial_balance: Starting balance
        total_timesteps: Training steps
    
    Returns:
        Training metrics
    """
    trainer = RLExecutionTrainer(
        model_name=model_name,
        initial_balance=initial_balance
    )
    
    # Prepare data
    train_env, eval_env = trainer.prepare_training_data(candles, valid_setups)
    
    # Train
    metrics = trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=total_timesteps
    )
    
    return metrics
