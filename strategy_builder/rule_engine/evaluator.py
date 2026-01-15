"""
Rule Evaluator - Converts parsed rules into executable trading logic

This is the core engine that evaluates entry/exit conditions dynamically
based on user-defined rules from natural language.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .indicators import IndicatorCalculator


class RuleEvaluator:
    """
    Evaluates trading rules dynamically based on parsed strategy.
    
    Converts parsed rules like:
    {"type": "rsi_below", "threshold": 30}
    
    Into executable logic:
    if rsi < 30: return True
    """
    
    def __init__(self):
        self.indicator_calc = IndicatorCalculator()
    
    def evaluate_entry_conditions(
        self,
        df: pd.DataFrame,
        row_idx: int,
        entry_conditions: List[Dict],
        operator: str = 'AND'
    ) -> Tuple[bool, str]:
        """
        Evaluate if entry conditions are met at specific candle.
        
        Args:
            df: DataFrame with calculated indicators
            row_idx: Current candle index
            entry_conditions: List of entry condition rules
            operator: 'AND' or 'OR'
            
        Returns:
            (is_valid, reason)
        """
        if not entry_conditions:
            return False, "No entry conditions defined"
        
        results = []
        reasons = []
        
        for condition in entry_conditions:
            result, reason = self._evaluate_single_condition(df, row_idx, condition)
            results.append(result)
            reasons.append(reason)
        
        # Apply operator
        if operator == 'AND':
            is_valid = all(results)
            if not is_valid:
                failed = [r for r, res in zip(reasons, results) if not res]
                return False, f"Failed: {', '.join(failed)}"
        else:  # OR
            is_valid = any(results)
            if not is_valid:
                return False, f"All conditions failed: {', '.join(reasons)}"
        
        return True, "All entry conditions met"
    
    def evaluate_exit_conditions(
        self,
        df: pd.DataFrame,
        row_idx: int,
        exit_conditions: List[Dict],
        entry_price: float,
        entry_candle_idx: int
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Evaluate if exit conditions are met.
        
        Args:
            df: DataFrame with calculated indicators
            row_idx: Current candle index
            exit_conditions: List of exit condition rules
            entry_price: Entry price of the trade
            entry_candle_idx: Index where trade was entered
            
        Returns:
            (should_exit, reason, exit_type)
        """
        if not exit_conditions:
            return False, "No exit conditions defined", None
        
        for condition in exit_conditions:
            condition_type = condition.get('type', '')
            
            # Risk-Reward based exits
            if condition_type == 'take_profit':
                result, reason = self._evaluate_take_profit(
                    df, row_idx, entry_price, condition
                )
                if result:
                    return True, reason, 'take_profit'
            
            elif condition_type == 'stop_loss':
                result, reason = self._evaluate_stop_loss(
                    df, row_idx, entry_price, condition
                )
                if result:
                    return True, reason, 'stop_loss'
            
            # Indicator-based exits
            else:
                result, reason = self._evaluate_single_condition(df, row_idx, condition)
                if result:
                    return True, f"Exit signal: {reason}", 'signal'
        
        return False, "No exit conditions met", None
    
    def _evaluate_single_condition(
        self,
        df: pd.DataFrame,
        row_idx: int,
        condition: Dict
    ) -> Tuple[bool, str]:
        """
        Evaluate a single condition.
        
        Condition types:
        - cross_above: Price/indicator crosses above another
        - cross_below: Price/indicator crosses below another
        - rsi_oversold: RSI below threshold
        - rsi_overbought: RSI above threshold
        - macd_cross: MACD crosses signal line
        - price_above_ma: Price above moving average
        - price_below_ma: Price below moving average
        - bb_touch: Price touches Bollinger Band
        - trend_up: Uptrend confirmation
        - trend_down: Downtrend confirmation
        """
        condition_type = condition.get('type', '')
        variables = condition.get('variables', {})
        
        try:
            # Price cross conditions
            if condition_type == 'cross_above':
                return self._check_cross_above(df, row_idx, variables)
            
            elif condition_type == 'cross_below':
                return self._check_cross_below(df, row_idx, variables)
            
            # RSI conditions
            elif condition_type in ['rsi_oversold', 'rsi_below']:
                return self._check_rsi_below(df, row_idx, variables)
            
            elif condition_type in ['rsi_overbought', 'rsi_above']:
                return self._check_rsi_above(df, row_idx, variables)
            
            elif condition_type == 'rsi_between':
                return self._check_rsi_between(df, row_idx, variables)
            
            # MACD conditions
            elif condition_type == 'macd_cross':
                return self._check_macd_cross(df, row_idx, variables)
            
            elif condition_type == 'macd_histogram_positive':
                return self._check_macd_histogram(df, row_idx, positive=True)
            
            elif condition_type == 'macd_histogram_negative':
                return self._check_macd_histogram(df, row_idx, positive=False)
            
            # MA conditions
            elif condition_type == 'price_above_ma':
                return self._check_price_above_ma(df, row_idx, variables)
            
            elif condition_type == 'price_below_ma':
                return self._check_price_below_ma(df, row_idx, variables)
            
            # Bollinger Band conditions
            elif condition_type == 'bb_upper_touch':
                return self._check_bb_touch(df, row_idx, variables, 'upper')
            
            elif condition_type == 'bb_lower_touch':
                return self._check_bb_touch(df, row_idx, variables, 'lower')
            
            # Trend conditions
            elif condition_type in ['trend_up', 'uptrend']:
                return self._check_trend(df, row_idx, 'up')
            
            elif condition_type in ['trend_down', 'downtrend']:
                return self._check_trend(df, row_idx, 'down')
            
            # Stochastic conditions
            elif condition_type == 'stoch_oversold':
                return self._check_stoch_oversold(df, row_idx, variables)
            
            elif condition_type == 'stoch_overbought':
                return self._check_stoch_overbought(df, row_idx, variables)
            
            else:
                return False, f"Unknown condition type: {condition_type}"
        
        except Exception as e:
            return False, f"Error evaluating condition: {str(e)}"
    
    def _check_cross_above(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if indicator1 crosses above indicator2"""
        ind1 = variables.get('indicator1', 'close')
        ind2 = variables.get('indicator2', variables.get('indicator', 'sma20'))
        
        if row_idx < 1:
            return False, "Not enough history"
        
        current = df.iloc[row_idx][ind1]
        previous = df.iloc[row_idx - 1][ind1]
        
        target_current = df.iloc[row_idx][ind2]
        target_previous = df.iloc[row_idx - 1][ind2]
        
        crossed = previous <= target_previous and current > target_current
        
        if crossed:
            return True, f"{ind1} crossed above {ind2}"
        return False, f"{ind1} below {ind2}"
    
    def _check_cross_below(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if indicator1 crosses below indicator2"""
        ind1 = variables.get('indicator1', 'close')
        ind2 = variables.get('indicator2', variables.get('indicator', 'sma20'))
        
        if row_idx < 1:
            return False, "Not enough history"
        
        current = df.iloc[row_idx][ind1]
        previous = df.iloc[row_idx - 1][ind1]
        
        target_current = df.iloc[row_idx][ind2]
        target_previous = df.iloc[row_idx - 1][ind2]
        
        crossed = previous >= target_previous and current < target_current
        
        if crossed:
            return True, f"{ind1} crossed below {ind2}"
        return False, f"{ind1} above {ind2}"
    
    def _check_rsi_below(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if RSI is below threshold"""
        threshold = variables.get('threshold', 30)
        period = variables.get('period', 14)
        col_name = f'rsi{period}'
        
        if col_name not in df.columns:
            return False, f"RSI{period} not calculated"
        
        rsi = df.iloc[row_idx][col_name]
        
        if pd.isna(rsi):
            return False, "RSI not available"
        
        if rsi < threshold:
            return True, f"RSI({rsi:.1f}) < {threshold}"
        return False, f"RSI({rsi:.1f}) >= {threshold}"
    
    def _check_rsi_above(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if RSI is above threshold"""
        threshold = variables.get('threshold', 70)
        period = variables.get('period', 14)
        col_name = f'rsi{period}'
        
        if col_name not in df.columns:
            return False, f"RSI{period} not calculated"
        
        rsi = df.iloc[row_idx][col_name]
        
        if pd.isna(rsi):
            return False, "RSI not available"
        
        if rsi > threshold:
            return True, f"RSI({rsi:.1f}) > {threshold}"
        return False, f"RSI({rsi:.1f}) <= {threshold}"
    
    def _check_rsi_between(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if RSI is between min and max"""
        min_val = variables.get('min', 30)
        max_val = variables.get('max', 70)
        period = variables.get('period', 14)
        col_name = f'rsi{period}'
        
        if col_name not in df.columns:
            return False, f"RSI{period} not calculated"
        
        rsi = df.iloc[row_idx][col_name]
        
        if pd.isna(rsi):
            return False, "RSI not available"
        
        if min_val <= rsi <= max_val:
            return True, f"RSI({rsi:.1f}) in range [{min_val}, {max_val}]"
        return False, f"RSI({rsi:.1f}) outside range [{min_val}, {max_val}]"
    
    def _check_macd_cross(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if MACD line crosses signal line"""
        direction = variables.get('direction', 'bullish')
        
        if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
            return False, "MACD not calculated"
        
        if row_idx < 1:
            return False, "Not enough history"
        
        macd_current = df.iloc[row_idx]['macd_line']
        macd_previous = df.iloc[row_idx - 1]['macd_line']
        signal_current = df.iloc[row_idx]['macd_signal']
        signal_previous = df.iloc[row_idx - 1]['macd_signal']
        
        if direction == 'bullish':
            crossed = macd_previous <= signal_previous and macd_current > signal_current
            if crossed:
                return True, "MACD bullish cross"
        else:
            crossed = macd_previous >= signal_previous and macd_current < signal_current
            if crossed:
                return True, "MACD bearish cross"
        
        return False, "No MACD cross"
    
    def _check_macd_histogram(self, df: pd.DataFrame, row_idx: int, positive: bool) -> Tuple[bool, str]:
        """Check if MACD histogram is positive/negative"""
        if 'macd_histogram' not in df.columns:
            return False, "MACD not calculated"
        
        hist = df.iloc[row_idx]['macd_histogram']
        
        if pd.isna(hist):
            return False, "MACD histogram not available"
        
        if positive and hist > 0:
            return True, f"MACD histogram positive ({hist:.4f})"
        elif not positive and hist < 0:
            return True, f"MACD histogram negative ({hist:.4f})"
        
        return False, f"MACD histogram {'negative' if positive else 'positive'}"
    
    def _check_price_above_ma(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if price is above moving average"""
        period = variables.get('period', 20)
        ma_type = variables.get('ma_type', 'sma')
        col_name = f'{ma_type}{period}'
        
        if col_name not in df.columns:
            return False, f"{ma_type.upper()}{period} not calculated"
        
        price = df.iloc[row_idx]['close']
        ma = df.iloc[row_idx][col_name]
        
        if pd.isna(ma):
            return False, f"{ma_type.upper()}{period} not available"
        
        if price > ma:
            return True, f"Price({price:.5f}) > {ma_type.upper()}{period}({ma:.5f})"
        return False, f"Price({price:.5f}) <= {ma_type.upper()}{period}({ma:.5f})"
    
    def _check_price_below_ma(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if price is below moving average"""
        period = variables.get('period', 20)
        ma_type = variables.get('ma_type', 'sma')
        col_name = f'{ma_type}{period}'
        
        if col_name not in df.columns:
            return False, f"{ma_type.upper()}{period} not calculated"
        
        price = df.iloc[row_idx]['close']
        ma = df.iloc[row_idx][col_name]
        
        if pd.isna(ma):
            return False, f"{ma_type.upper()}{period} not available"
        
        if price < ma:
            return True, f"Price({price:.5f}) < {ma_type.upper()}{period}({ma:.5f})"
        return False, f"Price({price:.5f}) >= {ma_type.upper()}{period}({ma:.5f})"
    
    def _check_bb_touch(self, df: pd.DataFrame, row_idx: int, variables: Dict, band: str) -> Tuple[bool, str]:
        """Check if price touches Bollinger Band"""
        period = variables.get('period', 20)
        tolerance = variables.get('tolerance', 0.001)  # 0.1% tolerance
        
        col_name = f'bb{period}_{band}'
        
        if col_name not in df.columns:
            return False, f"Bollinger Band not calculated"
        
        price = df.iloc[row_idx]['close']
        bb_value = df.iloc[row_idx][col_name]
        
        if pd.isna(bb_value):
            return False, "Bollinger Band not available"
        
        distance = abs(price - bb_value) / bb_value
        
        if distance <= tolerance:
            return True, f"Price touching {band} BB"
        return False, f"Price not at {band} BB"
    
    def _check_trend(self, df: pd.DataFrame, row_idx: int, direction: str) -> Tuple[bool, str]:
        """Check trend direction using MA alignment"""
        # Use 50 and 200 MA for trend
        if 'sma50' not in df.columns or 'sma200' not in df.columns:
            return False, "MAs not calculated for trend check"
        
        price = df.iloc[row_idx]['close']
        sma50 = df.iloc[row_idx]['sma50']
        sma200 = df.iloc[row_idx]['sma200']
        
        if pd.isna(sma50) or pd.isna(sma200):
            return False, "MAs not available"
        
        if direction == 'up':
            if price > sma50 > sma200:
                return True, "Uptrend confirmed"
        else:
            if price < sma50 < sma200:
                return True, "Downtrend confirmed"
        
        return False, f"{direction}trend not confirmed"
    
    def _check_stoch_oversold(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if Stochastic is oversold"""
        threshold = variables.get('threshold', 20)
        
        if 'stoch_k' not in df.columns:
            return False, "Stochastic not calculated"
        
        stoch_k = df.iloc[row_idx]['stoch_k']
        
        if pd.isna(stoch_k):
            return False, "Stochastic not available"
        
        if stoch_k < threshold:
            return True, f"Stochastic oversold ({stoch_k:.1f})"
        return False, f"Stochastic not oversold ({stoch_k:.1f})"
    
    def _check_stoch_overbought(self, df: pd.DataFrame, row_idx: int, variables: Dict) -> Tuple[bool, str]:
        """Check if Stochastic is overbought"""
        threshold = variables.get('threshold', 80)
        
        if 'stoch_k' not in df.columns:
            return False, "Stochastic not calculated"
        
        stoch_k = df.iloc[row_idx]['stoch_k']
        
        if pd.isna(stoch_k):
            return False, "Stochastic not available"
        
        if stoch_k > threshold:
            return True, f"Stochastic overbought ({stoch_k:.1f})"
        return False, f"Stochastic not overbought ({stoch_k:.1f})"
    
    def _evaluate_take_profit(
        self,
        df: pd.DataFrame,
        row_idx: int,
        entry_price: float,
        condition: Dict
    ) -> Tuple[bool, str]:
        """Evaluate take profit condition"""
        current_price = df.iloc[row_idx]['close']
        
        tp_type = condition.get('variables', {}).get('type', 'fixed_pips')
        
        if tp_type == 'fixed_pips':
            pips = condition.get('variables', {}).get('pips', 20)
            pip_value = 0.0001  # For most forex pairs
            tp_price = entry_price + (pips * pip_value)
            
            if current_price >= tp_price:
                return True, f"Take profit hit ({pips} pips)"
        
        elif tp_type == 'risk_reward':
            rr_ratio = condition.get('variables', {}).get('ratio', 2.0)
            # Need stop loss to calculate
            # For now, simplified
            profit = current_price - entry_price
            if profit > 0:  # Simplified check
                return True, f"Risk-reward target hit"
        
        return False, "Take profit not hit"
    
    def _evaluate_stop_loss(
        self,
        df: pd.DataFrame,
        row_idx: int,
        entry_price: float,
        condition: Dict
    ) -> Tuple[bool, str]:
        """Evaluate stop loss condition"""
        current_price = df.iloc[row_idx]['close']
        
        sl_type = condition.get('variables', {}).get('type', 'fixed_pips')
        
        if sl_type == 'fixed_pips':
            pips = condition.get('variables', {}).get('pips', 20)
            pip_value = 0.0001
            sl_price = entry_price - (pips * pip_value)
            
            if current_price <= sl_price:
                return True, f"Stop loss hit ({pips} pips)"
        
        elif sl_type == 'atr':
            multiplier = condition.get('variables', {}).get('multiplier', 1.5)
            if 'atr14' in df.columns:
                atr = df.iloc[row_idx]['atr14']
                sl_price = entry_price - (atr * multiplier)
                
                if current_price <= sl_price:
                    return True, f"ATR stop loss hit ({multiplier}x ATR)"
        
        return False, "Stop loss not hit"
