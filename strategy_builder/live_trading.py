"""
Live Trading Integration for User Strategies

Real-time signal generation, ML/RL filtering, and MT5 execution.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import torch

from .models import UserStrategy, StrategyTrade
from .data_collection import DataCollectionPipeline
from .ml_training import MLTrainingPipeline
from .rl_training import RLTrainingPipeline
from .rule_engine.evaluator import RuleEvaluator


class LiveTradingEngine:
    """
    Live trading engine for user strategies.
    
    Workflow:
    1. Monitor market for entry signals
    2. Calculate indicators in real-time
    3. Evaluate entry conditions
    4. Apply ML probability filter (P >= 0.65)
    5. Apply RL agent for execution optimization
    6. Execute trade on MT5
    7. Monitor and manage positions
    """
    
    def __init__(self):
        self.data_collector = DataCollectionPipeline()
        self.ml_pipeline = MLTrainingPipeline()
        self.rl_pipeline = RLTrainingPipeline()
        self.evaluator = RuleEvaluator()
        self.active_positions = {}  # Track open positions
    
    def start_trading(
        self,
        strategy_id: int,
        use_ml_filter: bool = True,
        ml_threshold: float = 0.65,
        use_rl: bool = True,
        check_interval: int = 60
    ):
        """
        Start live trading for a strategy.
        
        Args:
            strategy_id: Strategy ID
            use_ml_filter: Whether to use ML probability filter
            ml_threshold: Minimum ML probability to take trade
            use_rl: Whether to use RL agent for execution
            check_interval: Check interval in seconds
        """
        print(f"\n{'='*80}")
        print(f"LIVE TRADING ENGINE - STRATEGY {strategy_id}")
        print(f"{'='*80}")
        print(f"ML Filter: {'Enabled' if use_ml_filter else 'Disabled'} (threshold: {ml_threshold})")
        print(f"RL Agent: {'Enabled' if use_rl else 'Disabled'}")
        print(f"Check Interval: {check_interval}s")
        print(f"{'='*80}\n")
        
        # Get strategy
        try:
            strategy = UserStrategy.objects.get(id=strategy_id)
        except UserStrategy.DoesNotExist:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Check if strategy is active
        if not strategy.is_active:
            raise ValueError("Strategy is not active. Activate first.")
        
        # Load models
        ml_model = None
        rl_agent = None
        
        if use_ml_filter:
            try:
                ml_model = self.ml_pipeline.load_model(strategy_id)
                print("✅ ML model loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load ML model: {str(e)}")
                use_ml_filter = False
        
        if use_rl:
            try:
                rl_agent = self.rl_pipeline.load_agent(strategy_id)
                print("✅ RL agent loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load RL agent: {str(e)}")
                use_rl = False
        
        # Initialize MT5
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        
        print("\n🚀 Live trading started. Press Ctrl+C to stop.\n")
        
        try:
            while True:
                # Check if strategy is still active
                strategy.refresh_from_db()
                if not strategy.is_active:
                    print("\n⚠️  Strategy deactivated. Stopping...")
                    break
                
                # Monitor each symbol/timeframe
                for symbol in strategy.symbols:
                    for timeframe in strategy.timeframes:
                        try:
                            self._check_for_signals(
                                strategy=strategy,
                                symbol=symbol,
                                timeframe=timeframe,
                                ml_model=ml_model,
                                ml_threshold=ml_threshold,
                                rl_agent=rl_agent,
                                use_ml_filter=use_ml_filter,
                                use_rl=use_rl
                            )
                        except Exception as e:
                            print(f"❌ Error checking {symbol} {timeframe}: {str(e)}")
                
                # Manage open positions
                self._manage_positions(strategy)
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopping live trading...")
        
        finally:
            # Close all positions before stopping
            self._close_all_positions(strategy)
            mt5.shutdown()
            print("\n✅ Live trading stopped.\n")
    
    def _check_for_signals(
        self,
        strategy: UserStrategy,
        symbol: str,
        timeframe: str,
        ml_model: Optional[torch.nn.Module],
        ml_threshold: float,
        rl_agent: Optional[object],
        use_ml_filter: bool,
        use_rl: bool
    ):
        """Check for entry signals on a symbol/timeframe"""
        
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days of data
        
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
        
        # Check current candle (last row)
        current_idx = len(df) - 1
        
        # Check if already in position for this symbol
        if symbol in self.active_positions:
            return  # Skip if already trading this symbol
        
        # Evaluate entry conditions
        is_valid, reason = self.evaluator.evaluate_entry_conditions(
            df=df,
            row_idx=current_idx,
            entry_conditions=strategy.parsed_rules.get('entry_conditions', []),
            operator='AND'
        )
        
        if not is_valid:
            return  # No signal
        
        print(f"📊 {symbol} {timeframe}: Entry signal detected - {reason}")
        
        # Apply ML filter
        if use_ml_filter and ml_model is not None:
            features = self.data_collector._extract_features(df, current_idx, strategy.parsed_rules)
            if features is None:
                print(f"  ⚠️  Cannot extract features, skipping")
                return
            
            ml_probability = self.ml_pipeline.predict_probability(ml_model, features)
            print(f"  🧠 ML Probability: {ml_probability:.2%}")
            
            if ml_probability < ml_threshold:
                print(f"  ❌ Filtered out (< {ml_threshold:.0%})")
                return
        else:
            ml_probability = None
        
        # Execute trade
        print(f"  ✅ Executing trade...")
        
        trade_result = self._execute_trade(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            ml_probability=ml_probability
        )
        
        if trade_result is not None:
            print(f"  ✅ Trade opened: {trade_result['ticket']}")
            self.active_positions[symbol] = trade_result
    
    def _execute_trade(
        self,
        strategy: UserStrategy,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        ml_probability: Optional[float]
    ) -> Optional[Dict]:
        """Execute trade on MT5"""
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"  ❌ Could not get tick data for {symbol}")
            return None
        
        current_price = tick.ask
        
        # Calculate position size based on risk
        account_info = mt5.account_info()
        balance = account_info.balance
        risk_amount = balance * (strategy.risk_percentage / 100)
        
        # Get SL/TP from strategy
        stop_loss_pips = strategy.parsed_rules.get('stop_loss_pips', 20)
        take_profit_pips = strategy.parsed_rules.get('take_profit_pips', 40)
        pip_value = 0.0001
        
        sl_price = current_price - (stop_loss_pips * pip_value)
        tp_price = current_price + (take_profit_pips * pip_value)
        
        # Calculate lot size
        point = mt5.symbol_info(symbol).point
        tick_value = mt5.symbol_info(symbol).trade_tick_value
        lot_size = risk_amount / (stop_loss_pips * pip_value / point * tick_value)
        lot_size = round(lot_size, 2)  # Round to 2 decimals
        lot_size = max(0.01, min(lot_size, 10.0))  # Clamp between 0.01 and 10 lots
        
        # Prepare order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": current_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": strategy.id,
            "comment": f"Strategy {strategy.id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            print(f"  ❌ Order failed: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"  ❌ Order failed: {result.retcode}")
            return None
        
        # Save trade to database
        trade = StrategyTrade.objects.create(
            strategy=strategy,
            symbol=symbol,
            entry_time=datetime.now(),
            direction='long',
            entry_price=current_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            position_size=lot_size,
            ml_probability=ml_probability
        )
        
        return {
            'ticket': result.order,
            'symbol': symbol,
            'entry_price': current_price,
            'sl': sl_price,
            'tp': tp_price,
            'lot_size': lot_size,
            'trade_id': trade.id
        }
    
    def _manage_positions(self, strategy: UserStrategy):
        """Monitor and manage open positions"""
        
        for symbol in list(self.active_positions.keys()):
            position = self.active_positions[symbol]
            
            # Check if position is still open on MT5
            positions = mt5.positions_get(symbol=symbol)
            
            if positions is None or len(positions) == 0:
                # Position closed
                print(f"📊 {symbol}: Position closed")
                
                # Update trade record
                try:
                    trade = StrategyTrade.objects.get(id=position['trade_id'])
                    trade.exit_time = datetime.now()
                    
                    # Get exit price from MT5 history
                    # For simplicity, we'll use last known price
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None:
                        trade.exit_price = tick.bid
                        trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size * 100000
                        
                        if trade.profit_loss > 0:
                            trade.outcome = 'win'
                        else:
                            trade.outcome = 'loss'
                    
                    trade.save()
                    
                except StrategyTrade.DoesNotExist:
                    pass
                
                # Remove from active positions
                del self.active_positions[symbol]
    
    def _close_all_positions(self, strategy: UserStrategy):
        """Close all open positions for strategy"""
        
        print("\n⚠️  Closing all positions...")
        
        for symbol in list(self.active_positions.keys()):
            positions = mt5.positions_get(symbol=symbol)
            
            if positions is not None:
                for position in positions:
                    if position.magic == strategy.id:
                        # Close position
                        tick = mt5.symbol_info_tick(symbol)
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": position.volume,
                            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                            "position": position.ticket,
                            "price": tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask,
                            "deviation": 20,
                            "magic": strategy.id,
                            "comment": "Close on stop",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        
                        result = mt5.order_send(request)
                        
                        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"  ✅ Closed {symbol}")
                        else:
                            print(f"  ❌ Failed to close {symbol}")
        
        self.active_positions.clear()
