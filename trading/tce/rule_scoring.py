"""
Rule Scoring Functions for Deep Learning Model Training

Each rule returns a score 0-1 representing how well the setup satisfies that rule.
The model learns which rules matter most for predicting winning trades.
"""

from typing import Dict, List
import numpy as np
from .types import Indicators, Candle, Swing, MarketStructure


def score_trend_rule(indicators: Indicators, structure: MarketStructure, direction: str) -> float:
    """
    Rule 1: Trend Confirmation (MA alignment + swing structure)
    
    Score components:
    - MA alignment: How well ordered are the MAs? (0-0.5)
    - Slopes: How many slopes point in correct direction? (0-0.5)
    - Swing structure: Higher lows (BUY) / Lower highs (SELL) (0-1.0)
    
    Returns: 0-1 score
    """
    score = 0.0
    
    # 1. MA Alignment Score (0-0.5)
    if direction == "BUY":
        # Ideal: MA6 > MA18 > MA50 > MA200
        ma_aligned = 0
        if indicators.ma6 > indicators.ma18:
            ma_aligned += 0.25
        if indicators.ma18 > indicators.ma50:
            ma_aligned += 0.25
        if indicators.ma50 > indicators.ma200:
            ma_aligned += 0.25
        if indicators.ma6 > indicators.ma50:
            ma_aligned += 0.25
        ma_score = min(ma_aligned / 1.0, 0.5)  # Cap at 0.5
    else:  # SELL
        # Ideal: MA6 < MA18 < MA50 < MA200
        ma_aligned = 0
        if indicators.ma6 < indicators.ma18:
            ma_aligned += 0.25
        if indicators.ma18 < indicators.ma50:
            ma_aligned += 0.25
        if indicators.ma50 < indicators.ma200:
            ma_aligned += 0.25
        if indicators.ma6 < indicators.ma50:
            ma_aligned += 0.25
        ma_score = min(ma_aligned / 1.0, 0.5)  # Cap at 0.5
    
    # 2. Slopes Score (0-0.5)
    slopes_correct = 0
    if direction == "BUY":
        if indicators.slope6 > 0:
            slopes_correct += 1
        if indicators.slope18 > 0:
            slopes_correct += 1
        if indicators.slope50 > 0:
            slopes_correct += 1
    else:  # SELL
        if indicators.slope6 < 0:
            slopes_correct += 1
        if indicators.slope18 < 0:
            slopes_correct += 1
        if indicators.slope50 < 0:
            slopes_correct += 1
    
    slopes_score = (slopes_correct / 3.0) * 0.5  # Max 0.5
    
    # 3. Swing Structure Score (0-1.0)
    swing_score = 0.0
    if len(structure.lows) >= 3 or len(structure.highs) >= 3:
        if direction == "BUY" and len(structure.lows) >= 3:
            # Check for higher lows (at least 60% of recent moves)
            recent = structure.lows[-10:] if len(structure.lows) >= 10 else structure.lows[-3:]
            higher_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
            swing_score = (higher_count / len(recent)) * 1.0
        elif direction == "SELL" and len(structure.highs) >= 3:
            # Check for lower highs (at least 60% of recent moves)
            recent = structure.highs[-10:] if len(structure.highs) >= 10 else structure.highs[-3:]
            lower_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
            swing_score = (lower_count / len(recent)) * 1.0
    
    # Total: 0-1
    score = min((ma_score + slopes_score + swing_score) / 2.0, 1.0)
    return score


def score_correlation_rule(correlations: Dict[str, float]) -> float:
    """
    Rule 2: Correlation Pairs Alignment
    
    Score: How aligned are related pairs with the trade direction?
    - Positive correlation: Both should go same direction
    - Negative correlation: Should go opposite directions
    
    Returns: 0-1 score (1.0 = all aligned, 0.0 = misaligned)
    """
    if not correlations:
        return 0.5  # No correlation data = neutral
    
    alignment_count = 0
    total = len(correlations)
    
    for value in correlations.values():
        # If correlation is strong (high absolute value) and aligned, it's good
        if abs(value) > 0.6:
            alignment_count += 1
    
    score = alignment_count / total if total > 0 else 0.5
    return min(score, 1.0)


def score_multi_tf_rule(higher_tf_results: Dict[str, bool]) -> float:
    """
    Rule 3: Multi-Timeframe Confirmation
    
    Score: What percentage of higher timeframes confirm the trend?
    
    Args:
        higher_tf_results: Dict like {'H1': True, 'H4': True, 'D1': False}
    
    Returns: 0-1 score
    """
    if not higher_tf_results:
        return 0.5  # No HTF data = neutral
    
    confirmed = sum(1 for v in higher_tf_results.values() if v)
    total = len(higher_tf_results)
    
    score = confirmed / total if total > 0 else 0.5
    return min(score, 1.0)


def score_ma_retest_rule(retest_depth: float) -> float:
    """
    Rule 4: MA Retest Hierarchy
    
    Score: How well does the retest match the expected MA hierarchy?
    
    Args:
        retest_depth: How many times the price has retested (1-3 typical)
    
    Returns: 0-1 score (2+ retests = 1.0)
    """
    # Score increases with retest depth (2 retests = 0.7, 3+ = 1.0)
    if retest_depth < 1:
        return 0.0
    elif retest_depth == 1:
        return 0.3
    elif retest_depth == 2:
        return 0.7
    else:  # 3+
        return 1.0


def score_sr_filter_rule(sr_distance_pips: float, atr: float) -> float:
    """
    Rule 5: Support/Resistance Filter
    
    Score: How far is the setup from the nearest S/R level?
    - Close to S/R (< 5 pips): 0.0 (bad - price might get stuck)
    - Moderate distance (5-20 pips): 0.5 (okay)
    - Far from S/R (> 20 pips): 1.0 (good - clean move)
    
    Returns: 0-1 score
    """
    if sr_distance_pips < 0:
        sr_distance_pips = abs(sr_distance_pips)
    
    # Normalize by ATR
    normalized_distance = sr_distance_pips / (atr * 100) if atr > 0 else sr_distance_pips
    
    if normalized_distance < 5:
        return 0.0  # Too close to S/R
    elif normalized_distance < 20:
        return 0.5 + (normalized_distance - 5) / 30  # Interpolate
    else:
        return 1.0  # Far from S/R


def score_risk_management_rule(risk_reward_ratio: float, sl_pips: float, position_size: float) -> float:
    """
    Rule 6: Risk Management Quality
    
    Score components:
    - Risk:Reward ratio (1:1.5 = good, 1:2 = excellent)
    - Stop loss distance (12-30 pips = good)
    - Position size (0.01-0.5 lots = reasonable for risk)
    
    Returns: 0-1 score
    """
    score = 0.0
    
    # 1. Risk:Reward component (0-0.4)
    if risk_reward_ratio >= 2.0:
        rr_score = 0.4
    elif risk_reward_ratio >= 1.5:
        rr_score = 0.3
    elif risk_reward_ratio >= 1.0:
        rr_score = 0.2
    else:
        rr_score = 0.0
    
    # 2. Stop loss distance component (0-0.3)
    if 12 <= sl_pips <= 30:
        sl_score = 0.3
    elif 30 < sl_pips <= 50:
        sl_score = 0.25
    elif sl_pips < 12:
        sl_score = 0.0
    else:  # > 50
        sl_score = 0.15
    
    # 3. Position size component (0-0.3)
    if 0.01 <= position_size <= 0.5:
        ps_score = 0.3
    elif position_size > 0.5:
        ps_score = 0.15
    else:
        ps_score = 0.0
    
    score = rr_score + sl_score + ps_score
    return min(score, 1.0)


def score_order_placement_rule(entry_offset_pips: float) -> float:
    """
    Rule 7: Order Placement Quality
    
    Score: How well is the entry placed above/below confirmation candle?
    - 2-3 pips offset: 1.0 (ideal)
    - 1-4 pips offset: 0.8 (good)
    - 4-10 pips offset: 0.5 (okay)
    - > 10 pips offset: 0.2 (poor)
    
    Returns: 0-1 score
    """
    entry_offset_pips = abs(entry_offset_pips)
    
    if 2 <= entry_offset_pips <= 3:
        return 1.0
    elif 1 <= entry_offset_pips <= 4:
        return 0.8
    elif 4 < entry_offset_pips <= 10:
        return 0.5
    else:
        return 0.2


def score_fibonacci_rule(fib_level: float, max_fib: float = 0.618) -> float:
    """
    Rule 8: Fibonacci Depth Validation
    
    Score: How valid is the Fibonacci depth?
    - 0.382 (38.2%): 1.0 (ideal shallow retracement)
    - 0.500 (50%):   0.9 (good medium retracement)
    - 0.618 (61.8%): 0.7 (acceptable deep retracement)
    - > 0.618: 0.0 (invalid - too deep)
    
    Returns: 0-1 score
    """
    if fib_level <= 0.382:
        return 1.0
    elif fib_level <= 0.500:
        return 0.9
    elif fib_level <= 0.618:
        return 0.7
    else:
        return 0.0


def calculate_all_rule_scores(
    indicators: Indicators,
    structure: MarketStructure,
    direction: str,
    correlations: Dict[str, float],
    higher_tf_results: Dict[str, bool],
    retest_depth: float,
    sr_distance_pips: float,
    risk_reward_ratio: float,
    sl_pips: float,
    position_size: float,
    entry_offset_pips: float,
    fib_level: float
) -> Dict[str, float]:
    """
    Calculate all 8 rule scores and return as a dictionary.
    
    Returns:
        Dict with keys: rule1_score, rule2_score, ..., rule8_score, overall_score
    """
    scores = {
        'rule1_trend': score_trend_rule(indicators, structure, direction),
        'rule2_correlation': score_correlation_rule(correlations),
        'rule3_multi_tf': score_multi_tf_rule(higher_tf_results),
        'rule4_ma_retest': score_ma_retest_rule(retest_depth),
        'rule5_sr_filter': score_sr_filter_rule(sr_distance_pips, indicators.atr),
        'rule6_risk_mgmt': score_risk_management_rule(risk_reward_ratio, sl_pips, position_size),
        'rule7_order_placement': score_order_placement_rule(entry_offset_pips),
        'rule8_fibonacci': score_fibonacci_rule(fib_level)
    }
    
    # Overall score = average of all rules
    scores['overall_score'] = np.mean(list(scores.values()))
    
    return scores
