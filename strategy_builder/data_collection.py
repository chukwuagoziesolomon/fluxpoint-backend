"""
Data Collection Pipeline for User Strategies

Fetches MT5 data, calculates indicators, scans for setups, and labels outcomes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5

from .rule_engine.indicators import IndicatorCalculator
from .rule_engine.evaluator import RuleEvaluator


class DataCollectionPipeline:
    """
    Collects and labels training data for user strategies.
    
    Pipeline:
    1. Fetch historical data from MT5
    2. Calculate user's indicators
    3. Scan for valid entry setups
    4. Track outcomes (TP hit first = 1, SL hit first = 0)
    5. Extract features for ML training
    """
    
    def __init__(self):
        self.indicator_calc = IndicatorCalculator()
        self.evaluator = RuleEvaluator()
    
    def fetch_mt5_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_aggressive_fetching: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data from MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'H1', 'M15', 'D1')
            start_date: Start date
            end_date: End date
            use_aggressive_fetching: If True, fetch maximum available data regardless of date range
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map timeframe string to MT5 constant and maximum bars
        timeframe_config = {
            'M1': {'mt5_code': mt5.TIMEFRAME_M1, 'max_bars': 20000},    # ~2 weeks
            'M5': {'mt5_code': mt5.TIMEFRAME_M5, 'max_bars': 50000},    # ~6 months
            'M15': {'mt5_code': mt5.TIMEFRAME_M15, 'max_bars': 80000},  # ~2 years
            'M30': {'mt5_code': mt5.TIMEFRAME_M30, 'max_bars': 100000}, # ~4 years
            'H1': {'mt5_code': mt5.TIMEFRAME_H1, 'max_bars': 50000},    # ~5.7 years
            'H4': {'mt5_code': mt5.TIMEFRAME_H4, 'max_bars': 50000},    # ~22 years
            'D1': {'mt5_code': mt5.TIMEFRAME_D1, 'max_bars': 15000},    # ~41 years
            'W1': {'mt5_code': mt5.TIMEFRAME_W1, 'max_bars': 3000},     # ~57 years
            'MN1': {'mt5_code': mt5.TIMEFRAME_MN1, 'max_bars': 1000}    # ~83 years
        }
        
        config = timeframe_config.get(timeframe, {'mt5_code': mt5.TIMEFRAME_H1, 'max_bars': 30000})
        mt5_timeframe = config['mt5_code']
        max_bars = config['max_bars']
        
        # Initialize MT5
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        
        try:
            # Try to enable symbol in Market Watch
            mt5.symbol_select(symbol, True)
            
            # AGGRESSIVE FETCHING: Get maximum available historical data
            if use_aggressive_fetching:
                print(f"  📥 Fetching maximum available data ({max_bars:,} bars)...")
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, max_bars)
            else:
                # Standard date range fetching
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                raise ValueError(f"No data returned for {symbol} {timeframe}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to lowercase
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Filter to requested date range if using aggressive fetching
            if use_aggressive_fetching:
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                print(f"✅ Fetched {len(rates):,} total bars, filtered to {len(df):,} bars in date range")
            else:
                print(f"✅ Fetched {len(df):,} candles for {symbol} {timeframe}")
            
            return df
            
        finally:
            mt5.shutdown()
    
    def collect_training_data(
        self,
        strategy_id: int,
        parsed_rules: Dict,
        symbols: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        stop_loss_pips: float = 20,
        take_profit_pips: float = 40
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Collect and label training data for a strategy.
        
        Args:
            strategy_id: Strategy ID
            parsed_rules: Parsed strategy rules
            symbols: List of symbols to scan
            timeframes: List of timeframes to scan
            start_date: Start date for data collection
            end_date: End date for data collection
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            
        Returns:
            (X_train, y_train, setup_details)
            X_train: Feature matrix
            y_train: Labels (1=win, 0=loss)
            setup_details: List of setup metadata
        """
        X_train_list = []
        y_train_list = []
        setup_details = []
        
        print(f"\n{'='*80}")
        print(f"DATA COLLECTION FOR STRATEGY {strategy_id}")
        print(f"{'='*80}")
        print(f"Symbols: {symbols}")
        print(f"Timeframes: {timeframes}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Stop Loss: {stop_loss_pips} pips")
        print(f"Take Profit: {take_profit_pips} pips\n")
        
        for symbol in symbols:
            for timeframe in timeframes:
                print(f"📊 Processing {symbol} {timeframe}...")
                
                try:
                    # Fetch data
                    df = self.fetch_mt5_data(symbol, timeframe, start_date, end_date)
                    
                    # Calculate indicators
                    df = self.indicator_calc.calculate_all(df, parsed_rules.get('indicators', []))
                    
                    # Scan for valid setups
                    setups = self._scan_for_setups(
                        df=df,
                        entry_conditions=parsed_rules.get('entry_conditions', []),
                        symbol=symbol,
                        timeframe=timeframe,
                        stop_loss_pips=stop_loss_pips,
                        take_profit_pips=take_profit_pips
                    )
                    
                    print(f"  ✅ Found {len(setups)} valid setups")
                    
                    # Extract features and labels
                    for setup in setups:
                        features = self._extract_features(df, setup['index'], parsed_rules)
                        if features is not None:
                            X_train_list.append(features)
                            y_train_list.append(setup['label'])
                            setup_details.append(setup)
                    
                except Exception as e:
                    print(f"  ❌ Error processing {symbol} {timeframe}: {str(e)}")
                    continue
        
        if len(X_train_list) == 0:
            raise ValueError("No valid training data collected")
        
        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.float32)
        
        print(f"\n{'='*80}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total setups: {len(X_train)}")
        print(f"Features per setup: {X_train.shape[1]}")
        print(f"Win rate: {(y_train.sum() / len(y_train) * 100):.1f}%")
        print(f"Wins: {int(y_train.sum())}")
        print(f"Losses: {int(len(y_train) - y_train.sum())}\n")
        
        return X_train, y_train, setup_details
    
    def _scan_for_setups(
        self,
        df: pd.DataFrame,
        entry_conditions: List[Dict],
        symbol: str,
        timeframe: str,
        stop_loss_pips: float,
        take_profit_pips: float
    ) -> List[Dict]:
        """
        Scan dataframe for valid entry setups and label outcomes.
        """
        setups = []
        pip_value = 0.0001  # For most forex pairs
        
        # Need at least 50 candles for indicators and 100 forward for outcomes
        min_idx = 50
        max_idx = len(df) - 100
        
        for i in range(min_idx, max_idx):
            # Check if entry conditions are met
            is_valid, reason = self.evaluator.evaluate_entry_conditions(
                df=df,
                row_idx=i,
                entry_conditions=entry_conditions,
                operator='AND'
            )
            
            if not is_valid:
                continue
            
            # Valid setup found - determine outcome
            entry_price = df.iloc[i]['close']
            entry_time = df.iloc[i]['timestamp']
            
            # Calculate SL and TP
            sl_price = entry_price - (stop_loss_pips * pip_value)
            tp_price = entry_price + (take_profit_pips * pip_value)
            
            # Look forward to see which hits first
            label = self._label_outcome(df, i + 1, entry_price, sl_price, tp_price)
            
            if label is not None:  # Valid outcome
                setups.append({
                    'index': i,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'label': label,
                    'reason': reason
                })
        
        return setups
    
    def _label_outcome(
        self,
        df: pd.DataFrame,
        start_idx: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        max_candles: int = 100
    ) -> Optional[int]:
        """
        Label outcome by checking which hits first: TP or SL.
        
        Returns:
            1 if TP hit first (win)
            0 if SL hit first (loss)
            None if neither hit within max_candles
        """
        for i in range(start_idx, min(start_idx + max_candles, len(df))):
            candle = df.iloc[i]
            
            # Check if SL hit
            if candle['low'] <= sl_price:
                return 0  # Loss
            
            # Check if TP hit
            if candle['high'] >= tp_price:
                return 1  # Win
        
        return None  # Inconclusive
    
    def _extract_features(
        self,
        df: pd.DataFrame,
        idx: int,
        parsed_rules: Dict
    ) -> Optional[np.ndarray]:
        """
        Extract features from a setup for ML training.
        
        Features are auto-generated from user's indicators.
        """
        features = []
        
        try:
            # Extract indicator values
            for indicator in parsed_rules.get('indicators', []):
                name = indicator.get('name', '').upper()
                params = indicator.get('parameters', {})
                
                if name in ['MA', 'SMA']:
                    period = params.get('period', 20)
                    col_name = f'sma{period}'
                    if col_name in df.columns:
                        features.append(df.iloc[idx][col_name])
                        features.append(df.iloc[idx][f'{col_name}_slope'])
                
                elif name == 'EMA':
                    period = params.get('period', 20)
                    col_name = f'ema{period}'
                    if col_name in df.columns:
                        features.append(df.iloc[idx][col_name])
                        features.append(df.iloc[idx][f'{col_name}_slope'])
                
                elif name == 'RSI':
                    period = params.get('period', 14)
                    col_name = f'rsi{period}'
                    if col_name in df.columns:
                        features.append(df.iloc[idx][col_name])
                        features.append(df.iloc[idx][f'{col_name}_slope'])
                
                elif name == 'MACD':
                    if 'macd_line' in df.columns:
                        features.append(df.iloc[idx]['macd_line'])
                        features.append(df.iloc[idx]['macd_signal'])
                        features.append(df.iloc[idx]['macd_histogram'])
                
                elif name == 'ATR':
                    period = params.get('period', 14)
                    col_name = f'atr{period}'
                    if col_name in df.columns:
                        features.append(df.iloc[idx][col_name])
                        features.append(df.iloc[idx][f'{col_name}_ratio'])
            
            # Add market context features
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Volatility (20-candle std dev)
            if idx >= 20:
                volatility = np.std(close[idx-20:idx])
                features.append(volatility)
            
            # Candle characteristics
            candle_range = (high[idx] - low[idx]) / close[idx]
            candle_body = abs(df.iloc[idx]['close'] - df.iloc[idx]['open']) / close[idx]
            features.append(candle_range)
            features.append(candle_body)
            
            # Ensure all features are valid
            if any(pd.isna(f) for f in features):
                return None
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"  ⚠️  Error extracting features at idx {idx}: {str(e)}")
            return None
