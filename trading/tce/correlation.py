"""
Correlation analysis for TCE validation.
Handles positive and negative correlations.
"""

from typing import Dict, Tuple
import numpy as np


def calculate_correlation_coefficient(
    prices1: list,
    prices2: list,
    lookback: int = 50
) -> float:
    """
    Calculate Pearson correlation coefficient between two price series.
    
    Args:
        prices1: First price series (e.g., EURUSD closes)
        prices2: Second price series (e.g., GBPUSD closes)
        lookback: Number of candles to use
    
    Returns:
        Correlation coefficient: -1.0 to +1.0
        +1.0 = perfect positive (move together)
        -1.0 = perfect negative (move opposite)
        0.0 = no correlation
    """
    if len(prices1) < lookback or len(prices2) < lookback:
        return 0.0
    
    recent1 = np.array(prices1[-lookback:])
    recent2 = np.array(prices2[-lookback:])
    
    # Calculate returns (% change)
    returns1 = np.diff(recent1) / recent1[:-1]
    returns2 = np.diff(recent2) / recent2[:-1]
    
    # Pearson correlation
    correlation = np.corrcoef(returns1, returns2)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def validate_correlation_directions(
    correlations: Dict[str, Tuple[float, str]],
    direction: str,
    threshold: float = 0.5
) -> bool:
    """
    Validate that correlated pairs move in expected directions.
    
    Args:
        correlations: {
            'GBPUSD': (0.72, 'UP'),      # positive correlation, GBPUSD going UP
            'GBPJPY': (-0.65, 'DOWN'),   # negative correlation, GBPJPY going DOWN
            ...
        }
        direction: Current setup direction ('UP' or 'DOWN')
        threshold: Minimum correlation strength (0.5 = moderate)
    
    Returns:
        True if all correlated pairs align with their expected directions
    """
    for pair_name, (coefficient, pair_direction) in correlations.items():
        # Skip weak correlations
        if abs(coefficient) < threshold:
            continue
        
        # Check if correlation direction matches expected direction
        if coefficient > 0:  # Positive correlation
            # Pair should move in SAME direction as our setup
            if pair_direction != direction:
                return False
        else:  # Negative correlation
            # Pair should move in OPPOSITE direction to our setup
            if pair_direction == direction:
                return False
    
    return True


# Common forex pair correlations (reference - these would be calculated dynamically)
COMMON_CORRELATIONS = {
    'EURUSD': {
        'GBPUSD': -0.65,      # Negative (GBP stronger when EUR weak)
        'USDCHF': -0.71,      # Negative (CHF safer, opposite of EUR)
        'EURJPY': 0.78,       # Positive (both yen crosses)
        'EURGBP': 0.82,       # Positive (both EUR-based)
    },
    'GBPUSD': {
        'EURUSD': -0.65,      # Negative
        'AUDUSD': 0.68,       # Positive (risk-on currencies)
        'NZDUSD': 0.72,       # Positive (risk-on currencies)
    },
    'USDJPY': {
        'EURUSD': -0.78,      # Negative (JPY safe-haven)
        'GBPUSD': -0.72,      # Negative
        'AUDJPY': 0.85,       # Positive (both yen pairs)
    }
}
