"""
Generic Indicator Calculator Library

Calculates technical indicators for user-defined strategies.
Supports all common indicators that users might describe.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


class IndicatorCalculator:
    """
    Generic indicator calculator that supports all common technical indicators.
    """
    
    def __init__(self):
        self.supported_indicators = {
            'MA', 'EMA', 'SMA', 'RSI', 'MACD', 'BB', 'ATR', 
            'STOCH', 'ADX', 'CCI', 'MOM', 'ROC', 'OBV'
        }
    
    def calculate_all(self, df: pd.DataFrame, indicator_configs: List[Dict]) -> pd.DataFrame:
        """
        Calculate all indicators specified in user strategy.
        
        Args:
            df: DataFrame with OHLCV data
            indicator_configs: List of indicator configs from parsed rules
            
        Returns:
            DataFrame with all indicators calculated
        """
        df = df.copy()
        
        for config in indicator_configs:
            name = config.get('name', '').upper()
            params = config.get('parameters', {})
            timeframe = config.get('timeframe', 'default')
            
            if name in ['MA', 'SMA']:
                df = self._calculate_sma(df, params, timeframe)
            elif name == 'EMA':
                df = self._calculate_ema(df, params, timeframe)
            elif name == 'RSI':
                df = self._calculate_rsi(df, params, timeframe)
            elif name == 'MACD':
                df = self._calculate_macd(df, params, timeframe)
            elif name == 'BB':
                df = self._calculate_bollinger_bands(df, params, timeframe)
            elif name == 'ATR':
                df = self._calculate_atr(df, params, timeframe)
            elif name == 'STOCH':
                df = self._calculate_stochastic(df, params, timeframe)
            elif name == 'ADX':
                df = self._calculate_adx(df, params, timeframe)
            elif name == 'CCI':
                df = self._calculate_cci(df, params, timeframe)
            else:
                print(f"⚠️  Unknown indicator: {name}")
        
        return df
    
    def _calculate_sma(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Simple Moving Average"""
        period = params.get('period', 20)
        column_name = f'sma{period}' if timeframe == 'default' else f'sma{period}_{timeframe}'
        
        df[column_name] = df['close'].rolling(window=period).mean()
        df[f'{column_name}_slope'] = df[column_name].diff()
        
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Exponential Moving Average"""
        period = params.get('period', 20)
        column_name = f'ema{period}' if timeframe == 'default' else f'ema{period}_{timeframe}'
        
        df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'{column_name}_slope'] = df[column_name].diff()
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Relative Strength Index"""
        period = params.get('period', 14)
        column_name = f'rsi{period}' if timeframe == 'default' else f'rsi{period}_{timeframe}'
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        df[column_name] = rsi
        df[f'{column_name}_slope'] = df[column_name].diff()
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """MACD Indicator"""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)
        
        prefix = 'macd' if timeframe == 'default' else f'macd_{timeframe}'
        
        # Calculate MACD line
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        df[f'{prefix}_line'] = macd_line
        df[f'{prefix}_signal'] = signal_line
        df[f'{prefix}_histogram'] = histogram
        df[f'{prefix}_cross'] = (macd_line > signal_line).astype(int)
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Bollinger Bands"""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        prefix = f'bb{period}' if timeframe == 'default' else f'bb{period}_{timeframe}'
        
        # Calculate middle band (SMA)
        middle = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        df[f'{prefix}_upper'] = upper
        df[f'{prefix}_middle'] = middle
        df[f'{prefix}_lower'] = lower
        df[f'{prefix}_width'] = (upper - lower) / middle  # Normalized width
        df[f'{prefix}_position'] = (df['close'] - lower) / (upper - lower)  # 0-1 position
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Average True Range"""
        period = params.get('period', 14)
        column_name = f'atr{period}' if timeframe == 'default' else f'atr{period}_{timeframe}'
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        df[column_name] = atr
        df[f'{column_name}_ratio'] = atr / df['close']  # Volatility as % of price
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Stochastic Oscillator"""
        k_period = params.get('k_period', 14)
        d_period = params.get('d_period', 3)
        
        prefix = 'stoch' if timeframe == 'default' else f'stoch_{timeframe}'
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        df[f'{prefix}_k'] = k
        df[f'{prefix}_d'] = d
        df[f'{prefix}_cross'] = (k > d).astype(int)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Average Directional Index"""
        period = params.get('period', 14)
        column_name = f'adx{period}' if timeframe == 'default' else f'adx{period}_{timeframe}'
        
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate TR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        df[column_name] = adx
        df[f'{column_name}_plus_di'] = plus_di
        df[f'{column_name}_minus_di'] = minus_di
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, params: Dict, timeframe: str) -> pd.DataFrame:
        """Commodity Channel Index"""
        period = params.get('period', 20)
        column_name = f'cci{period}' if timeframe == 'default' else f'cci{period}_{timeframe}'
        
        # Calculate Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate SMA of Typical Price
        sma_tp = tp.rolling(window=period).mean()
        
        # Calculate Mean Deviation
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        # Calculate CCI
        cci = (tp - sma_tp) / (0.015 * mad)
        
        df[column_name] = cci
        
        return df
    
    def get_indicator_value(self, df: pd.DataFrame, row_idx: int, indicator_name: str) -> float:
        """
        Get indicator value at specific row.
        
        Args:
            df: DataFrame with calculated indicators
            row_idx: Row index
            indicator_name: Name of indicator column
            
        Returns:
            Indicator value
        """
        if indicator_name not in df.columns:
            raise ValueError(f"Indicator '{indicator_name}' not found in dataframe")
        
        return df.iloc[row_idx][indicator_name]
