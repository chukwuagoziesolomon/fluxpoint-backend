"""
RL Module for Trade Execution Optimization

RL DOES NOT find strategies - it optimizes execution of VALID setups:
- Entry timing (enter now vs wait)
- SL/TP placement (dynamic adjustment)  
- Partial exits
- Position sizing

Components:
- environment.py: Gym environment with state, actions, rewards
- agent.py: PPO agent for learning optimal execution
- training.py: Training pipeline
- integration.py: Integration with TCE strategy
"""

from .environment import TradeExecutionEnv, create_execution_env
from .agent import ExecutionAgent, create_execution_agent
from .training import RLExecutionTrainer, train_rl_execution
from .integration import rl_optimizer, add_rl_execution_to_validation

__all__ = [
    'TradeExecutionEnv',
    'create_execution_env',
    'ExecutionAgent',
    'create_execution_agent',
    'RLExecutionTrainer',
    'train_rl_execution',
    'rl_optimizer',
    'add_rl_execution_to_validation'
]
