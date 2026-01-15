"""
RL Training Pipeline for User Strategies

Trains PPO agent to optimize execution (when to enter, position sizing, stop management).
Adapted from TCE RL approach but works with any user strategy.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
from typing import Dict, List, Optional
import os

from .models import UserStrategy
from .data_collection import DataCollectionPipeline
from .ml_training import MLTrainingPipeline


class StrategyTradingEnv(gym.Env):
    """
    OpenAI Gym environment for user strategy execution.
    
    State: Market features + ML probability + current position state
    Actions: 0=hold, 1=enter, 2=adjust SL, 3=take profit
    Reward: Profit from trades + risk-adjusted return
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        strategy_rules: Dict,
        ml_model: torch.nn.Module,
        ml_pipeline: MLTrainingPipeline,
        initial_balance: float = 10000.0,
        max_steps: int = 1000
    ):
        super(StrategyTradingEnv, self).__init__()
        
        self.df = df
        self.strategy_rules = strategy_rules
        self.ml_model = ml_model
        self.ml_pipeline = ml_pipeline
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        
        # Define action and observation space
        # Actions: 0=hold, 1=enter_trade, 2=adjust_sl, 3=take_profit
        self.action_space = spaces.Discrete(4)
        
        # Observation: market features (dynamic size based on strategy) + position state
        feature_size = self._get_feature_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_size + 5,), dtype=np.float32
        )
        
        self.reset()
    
    def _get_feature_size(self) -> int:
        """Get feature size from first valid row"""
        for i in range(50, len(self.df)):
            features = self._extract_features(i)
            if features is not None:
                return len(features)
        return 20  # Default
    
    def reset(self):
        """Reset environment to start"""
        self.current_step = 50  # Start after indicators are valid
        self.balance = self.initial_balance
        self.in_position = False
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trades_taken = 0
        self.winning_trades = 0
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return next state"""
        reward = 0.0
        done = False
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # Action 0: Hold
        if action == 0:
            reward = -0.001  # Small penalty for inaction
        
        # Action 1: Enter trade
        elif action == 1 and not self.in_position:
            # Check if ML model approves
            features = self._extract_features(self.current_step)
            if features is not None:
                ml_probability = self.ml_pipeline.predict_probability(self.ml_model, features)
                
                if ml_probability >= 0.65:  # ML filter
                    self.in_position = True
                    self.entry_price = current_price
                    self.position_size = self.balance * 0.01  # 1% risk
                    self.stop_loss = current_price * 0.998  # 0.2% SL
                    self.take_profit = current_price * 1.004  # 0.4% TP
                    self.trades_taken += 1
                    reward = 0.1  # Reward for taking action
                else:
                    reward = -0.05  # Penalty for trying bad setup
        
        # Action 2: Adjust stop loss (trail)
        elif action == 2 and self.in_position:
            new_sl = current_price * 0.999  # Trail to 0.1%
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
                reward = 0.05  # Reward for risk management
        
        # Action 3: Take profit early
        elif action == 3 and self.in_position:
            profit = (current_price - self.entry_price) / self.entry_price
            if profit > 0:
                pnl = self.position_size * profit
                self.balance += pnl
                self.winning_trades += 1
                reward = profit * 10  # Reward proportional to profit
                self.in_position = False
            else:
                reward = -0.1  # Penalty for closing losing trade
        
        # Check if SL/TP hit
        if self.in_position:
            if current_price <= self.stop_loss:
                pnl = self.position_size * (self.stop_loss - self.entry_price) / self.entry_price
                self.balance += pnl
                reward = pnl / self.position_size * 10  # Negative reward
                self.in_position = False
            
            elif current_price >= self.take_profit:
                pnl = self.position_size * (self.take_profit - self.entry_price) / self.entry_price
                self.balance += pnl
                self.winning_trades += 1
                reward = pnl / self.position_size * 10  # Positive reward
                self.in_position = False
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.df) - 10 or self.current_step >= self.max_steps:
            done = True
            # Final reward based on total return
            total_return = (self.balance - self.initial_balance) / self.initial_balance
            reward += total_return * 100
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Extract market features
        features = self._extract_features(self.current_step)
        if features is None:
            features = np.zeros(self._get_feature_size())
        
        # Add position state
        position_state = np.array([
            float(self.in_position),
            (self.entry_price / self.df.iloc[self.current_step]['close'] - 1) if self.in_position else 0,
            (self.stop_loss / self.df.iloc[self.current_step]['close'] - 1) if self.in_position else 0,
            (self.take_profit / self.df.iloc[self.current_step]['close'] - 1) if self.in_position else 0,
            self.balance / self.initial_balance - 1
        ], dtype=np.float32)
        
        return np.concatenate([features, position_state])
    
    def _extract_features(self, idx: int) -> Optional[np.ndarray]:
        """Extract features using data collector"""
        collector = DataCollectionPipeline()
        return collector._extract_features(self.df, idx, self.strategy_rules)


class TrainingCallback(BaseCallback):
    """Callback for monitoring RL training"""
    
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            self.episode_rewards.append(mean_reward)
            self.episode_lengths.append(mean_length)
            
            if self.verbose > 0:
                print(f"Rollout: Mean Reward={mean_reward:.2f}, Mean Length={mean_length:.0f}")


class RLTrainingPipeline:
    """
    RL training pipeline for user strategies.
    
    Pipeline:
    1. Load trained ML model
    2. Collect historical data
    3. Create gym environment
    4. Train PPO agent
    5. Save agent and update database
    """
    
    def __init__(self):
        self.data_collector = DataCollectionPipeline()
        self.ml_pipeline = MLTrainingPipeline()
    
    def train_rl_agent(
        self,
        strategy_id: int,
        start_date: datetime,
        end_date: datetime,
        total_timesteps: int = 100000,
        learning_rate: float = 0.0003
    ) -> Dict:
        """
        Train RL agent for strategy execution.
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date for training data
            end_date: End date for training data
            total_timesteps: Total training timesteps
            learning_rate: PPO learning rate
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*80}")
        print(f"RL TRAINING PIPELINE - STRATEGY {strategy_id}")
        print(f"{'='*80}\n")
        
        # Get strategy
        try:
            strategy = UserStrategy.objects.get(id=strategy_id)
        except UserStrategy.DoesNotExist:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Load ML model
        print("🧠 Loading ML model...")
        try:
            ml_model = self.ml_pipeline.load_model(strategy_id)
            print("✅ ML model loaded")
        except Exception as e:
            raise ValueError(f"Cannot train RL without ML model: {str(e)}")
        
        # Collect training data
        print("\n📊 Collecting training data...")
        all_envs = []
        
        for symbol in strategy.symbols[:2]:  # Limit to 2 symbols for speed
            for timeframe in strategy.timeframes[:1]:  # Use primary timeframe
                print(f"  - Fetching {symbol} {timeframe}...")
                
                df = self.data_collector.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate indicators
                df = self.data_collector.indicator_calc.calculate_all(
                    df, strategy.parsed_rules.get('indicators', [])
                )
                
                # Create environment
                env = StrategyTradingEnv(
                    df=df,
                    strategy_rules=strategy.parsed_rules,
                    ml_model=ml_model,
                    ml_pipeline=self.ml_pipeline,
                    initial_balance=10000.0
                )
                
                all_envs.append(env)
        
        print(f"✅ Created {len(all_envs)} training environments")
        
        # Train PPO agent
        print(f"\n🤖 Training PPO agent ({total_timesteps:,} timesteps)...")
        
        # Use first environment for training
        env = all_envs[0]
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        callback = TrainingCallback(verbose=1)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        # Save agent
        model_path = self._save_agent(model, strategy_id)
        
        print(f"\n{'='*80}")
        print(f"RL TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Environments: {len(all_envs)}")
        print(f"Model Saved: {model_path}")
        print(f"{'='*80}\n")
        
        return {
            'success': True,
            'strategy_id': strategy_id,
            'model_path': model_path,
            'total_timesteps': total_timesteps
        }
    
    def _save_agent(self, model: PPO, strategy_id: int) -> str:
        """Save trained RL agent to disk"""
        os.makedirs('models/user_strategies', exist_ok=True)
        model_path = f'models/user_strategies/strategy_{strategy_id}_rl.zip'
        
        model.save(model_path)
        print(f"✅ RL agent saved: {model_path}")
        
        return model_path
    
    def load_agent(self, strategy_id: int) -> PPO:
        """Load trained RL agent from disk"""
        model_path = f'models/user_strategies/strategy_{strategy_id}_rl.zip'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RL agent not found: {model_path}")
        
        model = PPO.load(model_path)
        return model
    
    def get_action(self, model: PPO, observation: np.ndarray) -> int:
        """
        Get action from trained agent.
        
        Args:
            model: Trained PPO model
            observation: Current state
            
        Returns:
            Action (0=hold, 1=enter, 2=adjust_sl, 3=take_profit)
        """
        action, _ = model.predict(observation, deterministic=True)
        return int(action)
