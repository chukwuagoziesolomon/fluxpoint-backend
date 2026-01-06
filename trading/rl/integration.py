"""
RL Integration with TCE Strategy

Integrates RL agent with TCE validation for optimized execution.
"""

import numpy as np
from typing import Dict, Optional
from .agent import ExecutionAgent


class RLExecutionOptimizer:
    """
    Singleton class to manage RL execution optimization.
    """
    
    _instance = None
    _agent = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str):
        """Load trained RL model."""
        try:
            # In production, this would load the actual trained model
            # For now, we'll mark as loaded
            self._model_loaded = True
            print(f"RL model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load RL model: {e}")
            self._model_loaded = False
    
    def optimize_execution(
        self,
        setup: Dict,
        observation: np.ndarray,
        use_rl: bool = True
    ) -> Dict:
        """
        Use RL agent to optimize execution of a valid setup.
        
        Args:
            setup: Validated trading setup
            observation: Current state (30-dim: TCE features + ML prob + context)
            use_rl: Whether to use RL (vs default execution)
        
        Returns:
            Optimized execution parameters
        """
        if not use_rl or not self._model_loaded:
            # Default execution (no RL optimization)
            return {
                'action': 'enter_full',
                'position_size_multiplier': 1.0,
                'stop_loss': setup.get('stop_loss'),
                'take_profit': setup.get('take_profit'),
                'rl_optimized': False
            }
        
        # Use RL agent to decide action
        try:
            if self._agent is None:
                return self._default_execution(setup)
            
            action, _ = self._agent.predict(observation, deterministic=True)
            
            # Map action to execution parameters
            if action == 0:  # Enter full
                return {
                    'action': 'enter_full',
                    'position_size_multiplier': 1.0,
                    'stop_loss': setup.get('stop_loss'),
                    'take_profit': setup.get('take_profit'),
                    'rl_optimized': True
                }
            elif action == 1:  # Enter half
                return {
                    'action': 'enter_half',
                    'position_size_multiplier': 0.5,
                    'stop_loss': setup.get('stop_loss'),
                    'take_profit': setup.get('take_profit'),
                    'rl_optimized': True
                }
            elif action == 2:  # Wait
                return {
                    'action': 'wait',
                    'position_size_multiplier': 0.0,
                    'rl_optimized': True
                }
            elif action == 3:  # Exit (if in trade)
                return {
                    'action': 'exit',
                    'rl_optimized': True
                }
            elif action == 4:  # Trail stop
                return {
                    'action': 'trail_stop',
                    'rl_optimized': True
                }
        
        except Exception as e:
            print(f"RL execution error: {e}")
            return self._default_execution(setup)
    
    def _default_execution(self, setup: Dict) -> Dict:
        """Default execution without RL."""
        return {
            'action': 'enter_full',
            'position_size_multiplier': 1.0,
            'stop_loss': setup.get('stop_loss'),
            'take_profit': setup.get('take_profit'),
            'rl_optimized': False
        }


# Global instance
rl_optimizer = RLExecutionOptimizer()


def add_rl_execution_to_validation(
    validation_result: Dict,
    observation: np.ndarray,
    use_rl: bool = True
) -> Dict:
    """
    Add RL execution optimization to validation result.
    
    Args:
        validation_result: Result from TCE validation
        observation: Current state observation
        use_rl: Whether to use RL
    
    Returns:
        Validation result with RL execution parameters
    """
    if not validation_result.get('is_valid', False):
        # Setup not valid, no execution
        return validation_result
    
    # Get RL execution optimization
    execution = rl_optimizer.optimize_execution(
        setup=validation_result,
        observation=observation,
        use_rl=use_rl
    )
    
    # Add to validation result
    validation_result['rl_execution'] = execution
    
    return validation_result
