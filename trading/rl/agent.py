"""
Reinforcement Learning Agent for Trade Execution

Uses PPO (Proximal Policy Optimization) to learn optimal execution:
- When to enter (now vs wait for better price)
- Position sizing (full vs partial)
- When to exit (manual exit vs let SL/TP hit)
- Stop loss trailing (lock in profits)
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from typing import Dict, List, Optional
import os
from pathlib import Path


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_r_multiples = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log custom metrics if episode done
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            
            if 'balance' in info:
                self.logger.record('rollout/balance', info['balance'])
            
            if 'equity' in info:
                self.logger.record('rollout/equity', info['equity'])
        
        return True


class ExecutionAgent:
    """
    RL Agent for optimizing trade execution.
    
    Does NOT find strategies - optimizes execution of VALID setups.
    """
    
    def __init__(
        self,
        env,
        model_name: str = "tce_execution_ppo",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1
    ):
        """
        Args:
            env: TradeExecutionEnv instance
            model_name: Name for saving model
            learning_rate: Learning rate for PPO
            n_steps: Steps per rollout
            batch_size: Batch size for training
            n_epochs: Epochs per rollout
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping
            ent_coef: Entropy coefficient (exploration)
            verbose: Verbosity level
        """
        self.env = DummyVecEnv([lambda: Monitor(env)])
        self.model_name = model_name
        
        # Create PPO agent
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log="./logs/tensorboard/"
        )
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        eval_env=None,
        save_freq: int = 10000
    ) -> Dict:
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            eval_env: Environment for evaluation
            save_freq: Save frequency
        
        Returns:
            Training metrics
        """
        # Callbacks
        callbacks = [TensorboardCallback()]
        
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f'./models/rl/{self.model_name}_best/',
                log_path=f'./logs/rl/{self.model_name}/',
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train
        print(f"Training {self.model_name} for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=self.model_name
        )
        
        # Save final model
        self.save()
        
        return {
            'timesteps': total_timesteps,
            'model_name': self.model_name
        }
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action for given observation.
        
        Args:
            observation: Current state
            deterministic: Use deterministic policy (vs stochastic)
        
        Returns:
            (action, state) tuple
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, _states
    
    def save(self, path: Optional[str] = None):
        """Save model to disk."""
        if path is None:
            path = f"./models/rl/{self.model_name}"
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load model from disk."""
        if path is None:
            path = f"./models/rl/{self.model_name}"
        
        if os.path.exists(path + ".zip"):
            self.model = PPO.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, using untrained model")
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate agent performance.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
        
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_r_multiples = []
        win_rates = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            trades = []
            
            while not done:
                action, _states = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Get final metrics
            if hasattr(env, 'trade_history'):
                trades = env.trade_history
                r_multiples = [t['r_multiple'] for t in trades]
                wins = sum(1 for r in r_multiples if r > 0)
                win_rate = wins / len(r_multiples) if r_multiples else 0
                avg_r = np.mean(r_multiples) if r_multiples else 0
                
                episode_r_multiples.append(avg_r)
                win_rates.append(win_rate)
            
            episode_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_r_multiple': np.mean(episode_r_multiples) if episode_r_multiples else 0,
            'mean_win_rate': np.mean(win_rates) if win_rates else 0,
            'n_episodes': n_episodes
        }


def create_execution_agent(
    env,
    model_name: str = "tce_execution_ppo"
) -> ExecutionAgent:
    """
    Factory function to create execution agent.
    
    Args:
        env: TradeExecutionEnv instance
        model_name: Model name for saving
    
    Returns:
        Configured ExecutionAgent
    """
    return ExecutionAgent(env, model_name=model_name)
