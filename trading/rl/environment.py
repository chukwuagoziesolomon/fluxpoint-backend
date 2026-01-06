"""
Reinforcement Learning Environment for Trade Execution Optimization

RL DOES NOT find strategies - it optimizes execution of VALID setups:
- Entry timing (enter now vs wait)
- SL/TP placement (dynamic adjustment)
- Partial exits (take profits incrementally)
- Position sizing (adjust based on confidence)

State: TCE features + ML probability + trade context
Actions: Enter, Wait, Exit, Trail Stop
Reward: R-multiple (not raw profit)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd


class TradeExecutionEnv(gym.Env):
    """
    RL Environment for optimizing trade execution.
    
    The agent receives VALID trading setups from the strategy validator
    and decides HOW to execute them (timing, sizing, exit management).
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        candles: pd.DataFrame,
        valid_setups: List[Dict],
        initial_balance: float = 10000,
        max_position_size: float = 1.0,
        commission: float = 0.0001  # 1 pip spread
    ):
        """
        Args:
            candles: Historical OHLCV data
            valid_setups: List of validated trading setups from strategy
            initial_balance: Starting account balance
            max_position_size: Maximum position size (in lots)
            commission: Trading commission (as decimal)
        """
        super().__init__()
        
        self.candles = candles.reset_index(drop=True)
        self.valid_setups = valid_setups
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.commission = commission
        
        # State space: 20 TCE features + ML probability + 9 context features = 30 dimensions
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(30,), dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        # 0 = Enter full position
        # 1 = Enter half position
        # 2 = Wait (skip this setup)
        # 3 = Exit (if in trade)
        # 4 = Trail stop (tighten stop loss)
        self.action_space = spaces.Discrete(5)
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = None  # Dict with entry, size, sl, tp
        self.trade_history = []
        self.current_setup_idx = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.trade_history = []
        self.current_setup_idx = 0
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: 0=Enter full, 1=Enter half, 2=Wait, 3=Exit, 4=Trail stop
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        info = {}
        
        # Get current setup
        if self.current_setup_idx >= len(self.valid_setups):
            return self._get_observation(), 0.0, True, False, {'reason': 'no_more_setups'}
        
        setup = self.valid_setups[self.current_setup_idx]
        current_price = self.candles.iloc[self.current_step]['close']
        
        # Execute action
        if action == 0:  # Enter full position
            if self.position is None:
                reward = self._enter_trade(setup, current_price, size_multiplier=1.0)
                info['action'] = 'enter_full'
        
        elif action == 1:  # Enter half position
            if self.position is None:
                reward = self._enter_trade(setup, current_price, size_multiplier=0.5)
                info['action'] = 'enter_half'
        
        elif action == 2:  # Wait
            reward = -0.01  # Small penalty for waiting (opportunity cost)
            info['action'] = 'wait'
        
        elif action == 3:  # Exit
            if self.position is not None:
                reward = self._exit_trade(current_price, reason='manual_exit')
                info['action'] = 'exit'
        
        elif action == 4:  # Trail stop
            if self.position is not None:
                reward = self._trail_stop(current_price)
                info['action'] = 'trail_stop'
        
        # Update position if in trade
        if self.position is not None:
            reward += self._update_position(current_price)
        
        # Move to next candle
        self.current_step += 1
        
        # Check if episode done
        terminated = (
            self.current_step >= len(self.candles) - 1 or
            self.balance <= self.initial_balance * 0.5  # 50% drawdown = stop
        )
        
        truncated = False
        
        # Update info
        info['balance'] = self.balance
        info['equity'] = self._get_equity(current_price)
        info['position'] = self.position is not None
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        State = [TCE features (20) + ML probability (1) + Context (9)]
        """
        if self.current_setup_idx >= len(self.valid_setups):
            return np.zeros(30, dtype=np.float32)
        
        setup = self.valid_setups[self.current_setup_idx]
        
        # TCE features (20 dimensions)
        tce_features = np.array(setup.get('features', np.zeros(20)), dtype=np.float32)
        
        # ML probability (1 dimension)
        ml_prob = np.array([setup.get('ml_probability', 0.5)], dtype=np.float32)
        
        # Trade context (9 dimensions)
        context = np.array([
            1.0 if self.position is not None else 0.0,  # In trade?
            self.position['size'] / self.max_position_size if self.position else 0.0,  # Position size
            self.position['unrealized_r'] if self.position else 0.0,  # Current R-multiple
            self.balance / self.initial_balance - 1.0,  # Account % change
            len(self.trade_history) / 100.0,  # Trade count (normalized)
            self._win_rate(),  # Recent win rate
            self._avg_r_multiple(),  # Average R-multiple
            setup.get('risk_reward_ratio', 1.5) / 3.0,  # RR ratio (normalized)
            self.current_step / len(self.candles)  # Progress through data
        ], dtype=np.float32)
        
        return np.concatenate([tce_features, ml_prob, context])
    
    def _enter_trade(
        self, 
        setup: Dict, 
        entry_price: float, 
        size_multiplier: float = 1.0
    ) -> float:
        """
        Enter a trade based on setup.
        
        Args:
            setup: Validated setup with SL, TP, position size
            entry_price: Current market price
            size_multiplier: 1.0 = full position, 0.5 = half position
        
        Returns:
            Immediate reward (negative for entering - cost of entry)
        """
        # Extract setup parameters
        stop_loss = setup.get('stop_loss', entry_price * 0.98)
        take_profit = setup.get('take_profit', entry_price * 1.02)
        position_size = setup.get('position_size', 0.1) * size_multiplier
        
        # Calculate risk
        sl_distance = abs(entry_price - stop_loss)
        risk_amount = self.balance * 0.01 * size_multiplier  # 1% risk per full position
        
        # Create position
        self.position = {
            'entry_price': entry_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'entry_step': self.current_step,
            'unrealized_r': 0.0
        }
        
        # Immediate cost: commission
        commission_cost = position_size * entry_price * self.commission
        self.balance -= commission_cost
        
        # Negative reward for entering (encourages selectivity)
        return -0.05 * size_multiplier
    
    def _exit_trade(self, exit_price: float, reason: str = 'manual') -> float:
        """
        Exit current trade and calculate R-multiple reward.
        
        Args:
            exit_price: Exit price
            reason: Exit reason (tp_hit, sl_hit, manual_exit)
        
        Returns:
            R-multiple reward
        """
        if self.position is None:
            return 0.0
        
        entry_price = self.position['entry_price']
        stop_loss = self.position['stop_loss']
        risk_amount = self.position['risk_amount']
        size = self.position['size']
        
        # Calculate profit/loss
        pnl = (exit_price - entry_price) * size * 10000  # Simplified pip value
        
        # Apply commission
        commission_cost = size * exit_price * self.commission
        pnl -= commission_cost
        
        # Update balance
        self.balance += pnl
        
        # Calculate R-multiple
        sl_distance = abs(entry_price - stop_loss)
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
        
        # Record trade
        self.trade_history.append({
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'r_multiple': r_multiple,
            'reason': reason,
            'duration': self.current_step - self.position['entry_step']
        })
        
        # Clear position
        self.position = None
        
        # Reward = R-multiple (this is the key!)
        # Positive R = positive reward, negative R = negative reward
        return r_multiple
    
    def _trail_stop(self, current_price: float) -> float:
        """
        Trail stop loss to lock in profits.
        
        Returns:
            Small positive reward for risk management
        """
        if self.position is None:
            return 0.0
        
        entry_price = self.position['entry_price']
        stop_loss = self.position['stop_loss']
        
        # Calculate new stop loss (50% of current profit)
        if current_price > entry_price:  # Long position
            new_sl = entry_price + (current_price - entry_price) * 0.5
            if new_sl > stop_loss:
                self.position['stop_loss'] = new_sl
                return 0.02  # Small reward for good risk management
        
        return 0.0
    
    def _update_position(self, current_price: float) -> float:
        """
        Update position and check for SL/TP hits.
        
        Returns:
            Reward if position closed (R-multiple), else 0
        """
        if self.position is None:
            return 0.0
        
        entry_price = self.position['entry_price']
        stop_loss = self.position['stop_loss']
        take_profit = self.position['take_profit']
        risk_amount = self.position['risk_amount']
        
        # Update unrealized R-multiple
        unrealized_pnl = (current_price - entry_price) * self.position['size'] * 10000
        self.position['unrealized_r'] = unrealized_pnl / risk_amount if risk_amount > 0 else 0.0
        
        # Check stop loss hit
        if current_price <= stop_loss:
            return self._exit_trade(stop_loss, reason='sl_hit')
        
        # Check take profit hit
        if current_price >= take_profit:
            return self._exit_trade(take_profit, reason='tp_hit')
        
        return 0.0
    
    def _get_equity(self, current_price: float) -> float:
        """Calculate current account equity including open position."""
        equity = self.balance
        
        if self.position is not None:
            unrealized_pnl = (
                (current_price - self.position['entry_price']) * 
                self.position['size'] * 10000
            )
            equity += unrealized_pnl
        
        return equity
    
    def _win_rate(self) -> float:
        """Calculate recent win rate (last 20 trades)."""
        if len(self.trade_history) == 0:
            return 0.5
        
        recent_trades = self.trade_history[-20:]
        wins = sum(1 for trade in recent_trades if trade['r_multiple'] > 0)
        return wins / len(recent_trades)
    
    def _avg_r_multiple(self) -> float:
        """Calculate average R-multiple (last 20 trades)."""
        if len(self.trade_history) == 0:
            return 0.0
        
        recent_trades = self.trade_history[-20:]
        return np.mean([trade['r_multiple'] for trade in recent_trades])
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{len(self.candles)}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Equity: ${self._get_equity(self.candles.iloc[self.current_step]['close']):.2f}")
            print(f"Position: {self.position is not None}")
            print(f"Trades: {len(self.trade_history)}")
            print(f"Win Rate: {self._win_rate():.2%}")
            print(f"Avg R: {self._avg_r_multiple():.2f}R")
            print("-" * 50)


def create_execution_env(
    candles: pd.DataFrame,
    valid_setups: List[Dict],
    initial_balance: float = 10000
) -> TradeExecutionEnv:
    """
    Factory function to create trade execution environment.
    
    Args:
        candles: Historical OHLCV data
        valid_setups: List of validated setups from strategy
        initial_balance: Starting balance
    
    Returns:
        Configured TradeExecutionEnv
    """
    return TradeExecutionEnv(
        candles=candles,
        valid_setups=valid_setups,
        initial_balance=initial_balance
    )
