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

CONSTRAINTS:
- Must have candlestick pattern confirmation before entry
- Must respect risk management rules (SL, TP, position sizing)
- Position size calculated from account risk % and SL distance
- Risk/reward ratio validation enforced
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys
from pathlib import Path

# Add tce module to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'tce'))

from ..tce.utils import has_candlestick_confirmation
from ..tce.risk_management import (
    calculate_position_size,
    calculate_stop_loss,
    determine_risk_reward_ratio,
    get_pip_value_per_lot,
    validate_risk_management
)
from ..tce.types import Candle, Indicators, Swing, MarketStructure


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
        commission: float = 0.0001,  # 1 pip spread
        risk_percentage: float = 1.0,  # % of account to risk per trade
        symbol: str = "EURUSD",
        require_candlestick_pattern: bool = True,
        enforce_risk_management: bool = True
    ):
        """
        Args:
            candles: Historical OHLCV data
            valid_setups: List of validated trading setups from strategy
            initial_balance: Starting account balance
            max_position_size: Maximum position size (in lots)
            commission: Trading commission (as decimal)
            risk_percentage: % of account to risk per trade (e.g., 1.0 for 1%)
            symbol: Trading pair (e.g., 'EURUSD')
            require_candlestick_pattern: Enforce candlestick confirmation before entry
            enforce_risk_management: Enforce SL/TP/position sizing rules
        """
        super().__init__()
        
        self.candles = candles.reset_index(drop=True)
        self.valid_setups = valid_setups
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.commission = commission
        self.risk_percentage = risk_percentage
        self.symbol = symbol
        self.require_candlestick_pattern = require_candlestick_pattern
        self.enforce_risk_management = enforce_risk_management
        self.pip_value_per_lot = get_pip_value_per_lot(symbol)
        
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
        Enter a trade based on setup with risk management constraints.
        
        CONSTRAINTS:
        - Candlestick pattern confirmation required (pin bar, engulfing, etc.)
        - Position size calculated from account risk % and SL distance
        - Risk/reward ratio must be > 1.0
        - Stop loss enforced by risk management rules
        - Take profit enforced by risk management rules
        
        Args:
            setup: Validated setup with SL, TP, position size
            entry_price: Current market price
            size_multiplier: 1.0 = full position, 0.5 = half position
        
        Returns:
            Immediate reward or penalty
        """
        # ===== 1. CANDLESTICK PATTERN VALIDATION =====
        if self.require_candlestick_pattern:
            # Get recent candles from current position
            recent_candles = self.candles.iloc[max(0, self.current_step-3):self.current_step+1].values
            
            # Convert to Candle objects for pattern detection
            recent_candle_objs = [
                Candle(
                    open=c[1], high=c[2], low=c[3], close=c[4], volume=c[5] if len(c) > 5 else 0
                )
                for c in recent_candles
            ]
            
            direction = setup.get('direction', 'BUY')
            
            # Check if candlestick pattern is confirmed
            if not has_candlestick_confirmation(recent_candle_objs, direction):
                # Penalty for trying to enter without candlestick confirmation
                return -0.5  # Large penalty
        
        # ===== 2. RISK MANAGEMENT VALIDATION =====
        if self.enforce_risk_management:
            # Extract or calculate risk management parameters
            stop_loss = setup.get('stop_loss', None)
            take_profit = setup.get('take_profit', None)
            
            # If not provided, use risk management defaults
            if stop_loss is None or take_profit is None:
                direction = setup.get('direction', 'BUY')
                
                # Calculate SL using ATR or default
                atr = setup.get('atr', 0.001)
                sl_distance = max(1.5 * atr, 0.0012)  # Min 12 pips
                
                if direction == 'BUY':
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + (sl_distance * 1.5)  # 1:1.5 RR
                else:  # SELL
                    stop_loss = entry_price + sl_distance
                    take_profit = entry_price - (sl_distance * 1.5)  # 1:1.5 RR
            
            # Calculate SL distance in pips
            sl_pips = abs(entry_price - stop_loss) / 0.0001
            
            # Calculate position size based on risk management
            try:
                position_sizing = calculate_position_size(
                    account_balance=self.balance,
                    risk_percentage=self.risk_percentage * size_multiplier,
                    stop_loss_pips=max(sl_pips, 12),  # Min 12 pips
                    pip_value_per_lot=self.pip_value_per_lot,
                    symbol=self.symbol
                )
                position_size = position_sizing['lots']
                risk_amount = position_sizing['risk_amount']
            except:
                # Fallback to default calculation
                position_size = self.max_position_size * size_multiplier * 0.1
                risk_amount = self.balance * (self.risk_percentage / 100.0) * size_multiplier
            
            # Validate risk/reward ratio
            tp_distance = abs(entry_price - take_profit)
            sl_distance_pips = abs(entry_price - stop_loss) / 0.0001
            tp_distance_pips = tp_distance / 0.0001
            
            if sl_distance_pips > 0:
                rr_ratio = tp_distance_pips / sl_distance_pips
                if rr_ratio < 1.0:
                    # Reward: good risk management (TP > SL distance)
                    return -0.3  # Penalty for bad RR ratio
        else:
            # Use setup defaults
            stop_loss = setup.get('stop_loss', entry_price * 0.98)
            take_profit = setup.get('take_profit', entry_price * 1.02)
            position_size = setup.get('position_size', 0.1) * size_multiplier
            risk_amount = self.balance * 0.01 * size_multiplier
        
        # ===== 3. CREATE POSITION =====
        self.position = {
            'entry_price': entry_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'entry_step': self.current_step,
            'unrealized_r': 0.0,
            'direction': setup.get('direction', 'BUY')
        }
        
        # Immediate cost: commission
        commission_cost = position_size * entry_price * self.commission
        self.balance -= commission_cost
        
        # Small negative reward for entering (encourages selectivity)
        # Better reward if position size is reasonable and risk is managed
        reward = -0.05 * size_multiplier
        if self.enforce_risk_management and risk_amount > 0:
            # Bonus if risk is properly sized
            risk_ratio = risk_amount / self.initial_balance
            if risk_ratio <= (self.risk_percentage / 100.0):
                reward += 0.02  # Small bonus for proper risk sizing
        
        return reward
    
    def _exit_trade(self, exit_price: float, reason: str = 'manual') -> float:
        """
        Exit current trade and calculate R-multiple reward.
        
        R-multiple = profit/loss ÷ risk_amount
        - Positive R: profitable trade
        - Negative R: losing trade
        - 1R = risked amount
        
        Args:
            exit_price: Exit price
            reason: Exit reason (tp_hit, sl_hit, manual_exit)
        
        Returns:
            R-multiple reward (main reward signal for RL agent)
        """
        if self.position is None:
            return 0.0
        
        entry_price = self.position['entry_price']
        stop_loss = self.position['stop_loss']
        risk_amount = self.position['risk_amount']
        size = self.position['size']
        direction = self.position.get('direction', 'BUY')
        
        # Calculate profit/loss in pips (normalized)
        pip_value = 0.0001
        
        if direction == 'BUY':
            pips_gained = (exit_price - entry_price) / pip_value
        else:  # SELL
            pips_gained = (entry_price - exit_price) / pip_value
        
        # Convert pips to profit/loss
        pnl = pips_gained * self.pip_value_per_lot * size
        
        # Apply commission
        commission_cost = size * exit_price * self.commission
        pnl -= commission_cost
        
        # Update balance
        self.balance += pnl
        
        # Calculate R-multiple (the key reward metric!)
        # R-multiple = P&L ÷ Risk Amount
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
        
        # Record trade with detailed metrics
        self.trade_history.append({
            'entry': entry_price,
            'exit': exit_price,
            'direction': direction,
            'pnl': pnl,
            'r_multiple': r_multiple,
            'reason': reason,
            'duration': self.current_step - self.position['entry_step'],
            'size': size,
            'stop_loss': stop_loss,
            'position_size_lots': size
        })
        
        # Clear position
        self.position = None
        
        # Reward = R-multiple (this is the main training signal!)
        # - Positive R = positive reward
        # - Negative R = negative reward
        # - Larger R = larger reward
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
    initial_balance: float = 10000,
    risk_percentage: float = 1.0,
    symbol: str = "EURUSD",
    require_candlestick_pattern: bool = True,
    enforce_risk_management: bool = True
) -> TradeExecutionEnv:
    """
    Factory function to create trade execution environment.
    
    Args:
        candles: Historical OHLCV data
        valid_setups: List of validated setups from strategy
        initial_balance: Starting balance
        risk_percentage: % of account to risk per trade
        symbol: Trading pair (e.g., 'EURUSD')
        require_candlestick_pattern: Enforce candlestick confirmation before entry
        enforce_risk_management: Enforce SL/TP/position sizing rules
    
    Returns:
        Configured TradeExecutionEnv with risk management enabled
    """
    return TradeExecutionEnv(
        candles=candles,
        valid_setups=valid_setups,
        initial_balance=initial_balance,
        risk_percentage=risk_percentage,
        symbol=symbol,
        require_candlestick_pattern=require_candlestick_pattern,
        enforce_risk_management=enforce_risk_management
    )
