"""
Backtesting Engine for User Strategies

Simulates strategy execution on historical data with ML/RL filtering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import torch

from .models import UserStrategy, StrategyBacktest, StrategyTrade
from .data_collection import DataCollectionPipeline
from .ml_training import MLTrainingPipeline
from .rule_engine.evaluator import RuleEvaluator


class BacktestEngine:
    """
    Backtest engine for user strategies.
    
    Simulates:
    1. Real-time signal generation
    2. ML probability filtering (P >= 0.65)
    3. Position management (entry, exit, SL, TP)
    4. Performance metrics calculation
    """
    
    def __init__(self):
        self.data_collector = DataCollectionPipeline()
        self.ml_pipeline = MLTrainingPipeline()
        self.evaluator = RuleEvaluator()
    
    def run_backtest(
        self,
        strategy_id: int,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        risk_percentage: float = 1.0,
        use_ml_filter: bool = True,
        ml_threshold: float = 0.65,
        use_rl: bool = False
    ) -> Dict:
        """
        Run backtest on strategy.
        
        Args:
            strategy_id: Strategy ID
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Starting balance
            risk_percentage: Risk per trade (% of balance)
            use_ml_filter: Whether to use ML probability filter
            ml_threshold: Minimum probability to take trade
            use_rl: Whether to use RL agent for execution
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING STRATEGY {strategy_id}")
        print(f"{'='*80}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Risk Per Trade: {risk_percentage}%")
        print(f"ML Filter: {'Enabled' if use_ml_filter else 'Disabled'} (threshold: {ml_threshold})")
        print(f"RL Agent: {'Enabled' if use_rl else 'Disabled'}\n")
        
        # Get strategy
        try:
            strategy = UserStrategy.objects.get(id=strategy_id)
        except UserStrategy.DoesNotExist:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Load ML model if using filter
        ml_model = None
        if use_ml_filter:
            try:
                ml_model = self.ml_pipeline.load_model(strategy_id)
                print("✅ ML model loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load ML model: {str(e)}")
                use_ml_filter = False
        
        # Initialize tracking
        balance = initial_balance
        trades: List[Dict] = []
        equity_curve = [balance]
        
        # Process each symbol/timeframe
        for symbol in strategy.symbols:
            for timeframe in strategy.timeframes:
                print(f"\n📊 Processing {symbol} {timeframe}...")
                
                try:
                    # Fetch data
                    df = self.data_collector.fetch_mt5_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Calculate indicators
                    df = self.data_collector.indicator_calc.calculate_all(
                        df, strategy.parsed_rules.get('indicators', [])
                    )
                    
                    # Simulate trading
                    symbol_trades = self._simulate_trading(
                        df=df,
                        strategy=strategy,
                        symbol=symbol,
                        timeframe=timeframe,
                        balance=balance,
                        risk_percentage=risk_percentage,
                        ml_model=ml_model,
                        ml_threshold=ml_threshold,
                        use_rl=use_rl
                    )
                    
                    trades.extend(symbol_trades)
                    print(f"  ✅ Executed {len(symbol_trades)} trades")
                    
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    continue
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(trades, initial_balance)
        
        # Save backtest results
        backtest = StrategyBacktest.objects.create(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_return=metrics['total_return'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0)
        )
        
        # Save individual trades
        for trade_data in trades:
            StrategyTrade.objects.create(
                strategy=strategy,
                backtest=backtest,
                symbol=trade_data['symbol'],
                entry_time=trade_data['entry_time'],
                exit_time=trade_data['exit_time'],
                direction=trade_data['direction'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                stop_loss=trade_data['stop_loss'],
                take_profit=trade_data['take_profit'],
                position_size=trade_data['position_size'],
                profit_loss=trade_data['profit_loss'],
                profit_loss_pips=trade_data['profit_loss_pips'],
                ml_probability=trade_data.get('ml_probability'),
                outcome=trade_data['outcome']
            )
        
        print(f"\n{'='*80}")
        print(f"BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"{'='*80}\n")
        
        return {
            'success': True,
            'backtest_id': backtest.id,
            'metrics': metrics,
            'trades': trades
        }
    
    def _simulate_trading(
        self,
        df: pd.DataFrame,
        strategy: UserStrategy,
        symbol: str,
        timeframe: str,
        balance: float,
        risk_percentage: float,
        ml_model: Optional[torch.nn.Module],
        ml_threshold: float,
        use_rl: bool
    ) -> List[Dict]:
        """Simulate trading on historical data"""
        trades = []
        pip_value = 0.0001
        
        # Get parsed rules
        entry_conditions = strategy.parsed_rules.get('entry_conditions', [])
        exit_conditions = strategy.parsed_rules.get('exit_conditions', [])
        
        # Default SL/TP from strategy or use defaults
        stop_loss_pips = strategy.parsed_rules.get('stop_loss_pips', 20)
        take_profit_pips = strategy.parsed_rules.get('take_profit_pips', 40)
        
        # Scan for entries
        for i in range(50, len(df) - 50):  # Leave room for indicators and outcomes
            # Check entry conditions
            is_valid, reason = self.evaluator.evaluate_entry_conditions(
                df=df,
                row_idx=i,
                entry_conditions=entry_conditions,
                operator='AND'
            )
            
            if not is_valid:
                continue
            
            # Valid signal - apply ML filter if enabled
            if ml_model is not None:
                features = self._extract_features(df, i, strategy.parsed_rules)
                if features is None:
                    continue
                
                probability = self.ml_pipeline.predict_probability(ml_model, features)
                
                if probability < ml_threshold:
                    continue  # Filtered out
            else:
                probability = None
            
            # Execute trade
            entry_price = df.iloc[i]['close']
            entry_time = df.iloc[i]['timestamp']
            
            # Calculate SL/TP
            sl_price = entry_price - (stop_loss_pips * pip_value)
            tp_price = entry_price + (take_profit_pips * pip_value)
            
            # Calculate position size based on risk
            risk_amount = balance * (risk_percentage / 100)
            position_size = risk_amount / (stop_loss_pips * pip_value * 100000)  # Standard lot
            
            # Simulate outcome
            outcome_data = self._simulate_outcome(
                df=df,
                start_idx=i + 1,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                position_size=position_size
            )
            
            if outcome_data is not None:
                trades.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'entry_time': entry_time,
                    'exit_time': outcome_data['exit_time'],
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': outcome_data['exit_price'],
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'position_size': position_size,
                    'profit_loss': outcome_data['profit_loss'],
                    'profit_loss_pips': outcome_data['profit_loss_pips'],
                    'ml_probability': probability,
                    'outcome': outcome_data['outcome']
                })
        
        return trades
    
    def _simulate_outcome(
        self,
        df: pd.DataFrame,
        start_idx: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        position_size: float,
        max_candles: int = 100
    ) -> Optional[Dict]:
        """Simulate trade outcome"""
        pip_value = 0.0001
        
        for i in range(start_idx, min(start_idx + max_candles, len(df))):
            candle = df.iloc[i]
            
            # Check if SL hit
            if candle['low'] <= sl_price:
                pips = (sl_price - entry_price) / pip_value
                profit_loss = pips * pip_value * 100000 * position_size
                
                return {
                    'exit_time': candle['timestamp'],
                    'exit_price': sl_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pips': pips,
                    'outcome': 'loss'
                }
            
            # Check if TP hit
            if candle['high'] >= tp_price:
                pips = (tp_price - entry_price) / pip_value
                profit_loss = pips * pip_value * 100000 * position_size
                
                return {
                    'exit_time': candle['timestamp'],
                    'exit_price': tp_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pips': pips,
                    'outcome': 'win'
                }
        
        return None  # No clear outcome
    
    def _extract_features(self, df: pd.DataFrame, idx: int, parsed_rules: Dict) -> Optional[np.ndarray]:
        """Extract features for ML prediction"""
        # This should match feature extraction in data_collection.py
        return self.data_collector._extract_features(df, idx, parsed_rules)
    
    def _calculate_metrics(self, trades: List[Dict], initial_balance: float) -> Dict:
        """Calculate backtest performance metrics"""
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'final_balance': initial_balance
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['outcome'] == 'win')
        losing_trades = sum(1 for t in trades if t['outcome'] == 'loss')
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t['profit_loss'] for t in trades if t['outcome'] == 'win')
        total_loss = abs(sum(t['profit_loss'] for t in trades if t['outcome'] == 'loss'))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        net_profit = total_profit - total_loss
        final_balance = initial_balance + net_profit
        total_return = (net_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        # Calculate max drawdown
        balance_curve = [initial_balance]
        running_balance = initial_balance
        
        for trade in trades:
            running_balance += trade['profit_loss']
            balance_curve.append(running_balance)
        
        peak = balance_curve[0]
        max_dd = 0.0
        
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            
            dd = ((peak - balance) / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_profit': round(total_profit, 2),
            'total_loss': round(total_loss, 2),
            'net_profit': round(net_profit, 2),
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_dd, 2),
            'final_balance': round(final_balance, 2),
            'average_win': round(total_profit / winning_trades, 2) if winning_trades > 0 else 0,
            'average_loss': round(total_loss / losing_trades, 2) if losing_trades > 0 else 0
        }
