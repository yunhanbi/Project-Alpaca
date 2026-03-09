"""
Integrated Backtesting Framework - Complete strategy backtesting with performance analysis
Part 3: Strategy Backtesting (Main Integration Component)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import all components
from trading_strategy import BABAAlgoTradingStrategy
from gateway import MarketDataGateway
from orderbook import OrderBook, Order, OrderSide, OrderType
from order_manager import OrderManager
from order_gateway import OrderGateway
from matching_engine import MatchingEngine, ExecutionOutcome
from performance_analytics import PerformanceAnalytics
from visualization import TradingVisualizer


class BacktestingFramework:
    """Complete integrated backtesting framework for BABA strategy"""
    
    def __init__(self, initial_capital: float = 100000, 
                 csv_file_path: str = 'market_data_baba.csv'):
        
        self.initial_capital = initial_capital
        self.csv_file_path = csv_file_path
        
        # Initialize all components
        self.strategy = BABAAlgoTradingStrategy(portfolio_value=initial_capital)
        self.data_gateway = MarketDataGateway(csv_file_path)
        self.order_book = OrderBook('BABA')
        self.order_manager = OrderManager(
            initial_capital=initial_capital,
            risk_limits={
                'max_orders_per_minute': 10,
                'max_position': 1000,
                'max_order_size': initial_capital * 0.1,
                'max_price_deviation': 0.05
            }
        )
        self.order_gateway = OrderGateway('backtest_orders.log')
        self.matching_engine = MatchingEngine()
        self.performance_analytics = PerformanceAnalytics(initial_capital)
        self.visualizer = TradingVisualizer()
        
        # Backtesting state
        self.trades_log = []
        self.signals_history = {}
        self.market_data = pd.DataFrame()
        self.backtest_results = {}
        
    def prepare_data(self, news_text: str = None) -> bool:
        """Prepare market data and train strategy"""
        
        print("📊 Preparing market data for backtesting...")
        
        try:
            # Load market data
            self.data_gateway.load_data()
            self.market_data = self.data_gateway.data.copy()
            
            if self.market_data.empty:
                print("❌ No market data loaded")
                return False
            
            print(f"✅ Loaded {len(self.market_data)} market data points")
            
            # Train strategy ML model
            print("🤖 Training strategy ML model...")
            self.strategy.train_ml_model(self.market_data)
            
            # Generate signals for the entire dataset
            print("📈 Generating trading signals...")
            self.signals_history = self.strategy.generate_signals(
                self.market_data, news_text
            )
            
            print("✅ Data preparation completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return False
    
    def run_backtest(self, 
                    order_size_base: float = 100,
                    simulate_latency: bool = True,
                    verbose: bool = True) -> Dict:
        """Run comprehensive strategy backtest"""
        
        print("\n🚀 Starting BABA Strategy Backtest...")
        print("=" * 50)
        
        if self.market_data.empty:
            print("❌ Market data not prepared. Call prepare_data() first.")
            return {}
        
        # Backtest state variables
        current_position = 0
        position_entry_time = None
        position_entry_price = 0.0
        trades_executed = 0
        
        # Progress tracking
        total_ticks = len(self.market_data)
        progress_interval = max(1, total_ticks // 20)  # Show progress 20 times
        
        print(f"📊 Processing {total_ticks} market ticks...")
        
        for i, (timestamp, market_row) in enumerate(self.market_data.iterrows()):
            
            # Progress indicator
            if verbose and i % progress_interval == 0:
                progress = (i / total_ticks) * 100
                print(f"  Progress: {progress:.1f}% ({i}/{total_ticks})")
            
            # Get current market data
            current_price = market_row['Close']
            market_tick = {
                'timestamp': timestamp,
                'open': market_row['Open'],
                'high': market_row['High'],
                'low': market_row['Low'],
                'close': current_price,
                'volume': market_row['Volume']
            }
            
            # Check for trading signals
            if timestamp in self.signals_history['combined'].index:
                signal = self.signals_history['combined'][timestamp]
                signal_strength = self.signals_history['strength'][timestamp]
                
                # Entry logic
                if signal != 0 and current_position == 0:
                    trades_executed += 1
                    entry_result = self._execute_entry_order(
                        timestamp, signal, signal_strength, 
                        current_price, order_size_base, market_tick
                    )
                    
                    if entry_result['executed']:
                        current_position = signal
                        position_entry_time = timestamp
                        position_entry_price = entry_result['execution_price']
                        
                        # Log entry trade
                        self.trades_log.append({
                            'timestamp': timestamp,
                            'action': 'ENTRY',
                            'signal': 'BUY' if signal > 0 else 'SELL',
                            'price': entry_result['execution_price'],
                            'quantity': entry_result['quantity'],
                            'position_size': entry_result['position_size'],
                            'strength': signal_strength,
                            'order_id': entry_result['order_id']
                        })
                
                # Exit logic
                elif current_position != 0:
                    should_exit, exit_reason = self._check_exit_conditions(
                        current_position, position_entry_time, position_entry_price,
                        current_price, timestamp, signal
                    )
                    
                    if should_exit:
                        exit_result = self._execute_exit_order(
                            timestamp, current_position, current_price, 
                            order_size_base, market_tick, exit_reason
                        )
                        
                        if exit_result['executed']:
                            # Calculate P&L
                            if current_position > 0:  # Long position
                                pnl_pct = (exit_result['execution_price'] - position_entry_price) / position_entry_price
                            else:  # Short position
                                pnl_pct = (position_entry_price - exit_result['execution_price']) / position_entry_price
                            
                            hold_time_minutes = (timestamp - position_entry_time).total_seconds() / 60
                            
                            # Log exit trade
                            self.trades_log.append({
                                'timestamp': timestamp,
                                'action': 'EXIT',
                                'signal': exit_reason,
                                'price': exit_result['execution_price'],
                                'quantity': exit_result['quantity'],
                                'pnl_pct': pnl_pct,
                                'hold_time_min': hold_time_minutes,
                                'order_id': exit_result['order_id']
                            })
                            
                            # Reset position
                            current_position = 0
                            position_entry_time = None
                            position_entry_price = 0.0
        
        print("✅ Backtest execution completed!")
        print(f"📊 Total trades executed: {trades_executed}")
        print(f"📊 Total orders processed: {len(self.trades_log)}")
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades_log)
        
        # Calculate performance metrics
        print("\n📈 Calculating performance metrics...")
        performance_metrics = self.performance_analytics.calculate_portfolio_performance(
            trades_df, self.market_data
        )
        
        # Store results
        self.backtest_results = {
            'trades_df': trades_df,
            'performance_metrics': performance_metrics,
            'signals_history': self.signals_history,
            'market_data': self.market_data,
            'order_manager_summary': self.order_manager.get_portfolio_summary(),
            'execution_stats': self.matching_engine.get_execution_statistics(),
            'order_log_summary': self.order_gateway.get_log_summary()
        }
        
        return self.backtest_results
    
    def _execute_entry_order(self, timestamp: datetime, signal: int, signal_strength: int,
                           current_price: float, order_size_base: float, 
                           market_tick: Dict) -> Dict:
        """Execute entry order with full validation and simulation"""
        
        # Calculate position size
        position_size = self.strategy.calculate_position_size(
            signal_strength, 0.02, 0.015  # Mock volatility values
        )
        order_quantity = min(order_size_base, position_size / current_price)
        
        # Create order
        order_id = f"ORDER_{uuid.uuid4().hex[:8]}_{timestamp.strftime('%H%M%S')}"
        order = Order(
            order_id=order_id,
            side=OrderSide.BUY if signal > 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=order_quantity,
            price=current_price,
            timestamp=timestamp
        )
        
        # Validate order
        validation = self.order_manager.validate_order(order, current_price)
        self.order_manager.record_order(order, validation)
        
        if not validation['valid']:
            self.order_gateway.log_order_rejected(order, '; '.join(validation['errors']))
            return {'executed': False, 'reason': 'validation_failed'}
        
        # Log order sent
        self.order_gateway.log_order_sent(order)
        
        # Simulate execution
        execution_result = self.matching_engine.simulate_execution(order, market_tick)
        
        if execution_result['outcome'] == ExecutionOutcome.FILLED:
            # Add to order book
            trades = self.order_book.add_order(order)
            
            # Log successful execution
            if trades:
                for trade in trades:
                    self.order_gateway.log_order_filled(trade)
                    self.order_manager.update_position(trade)
            
            return {
                'executed': True,
                'order_id': order_id,
                'execution_price': execution_result['execution_price'],
                'quantity': execution_result['filled_quantity'],
                'position_size': position_size
            }
        
        elif execution_result['outcome'] == ExecutionOutcome.PARTIALLY_FILLED:
            self.order_gateway.log_order_partially_filled(
                order_id, execution_result['filled_quantity'],
                execution_result['remaining_quantity'],
                execution_result['execution_price']
            )
            
            return {
                'executed': True,
                'order_id': order_id,
                'execution_price': execution_result['execution_price'],
                'quantity': execution_result['filled_quantity'],
                'position_size': position_size
            }
        
        else:  # REJECTED
            self.order_gateway.log_order_rejected(order, execution_result['rejection_reason'])
            return {'executed': False, 'reason': 'execution_rejected'}
    
    def _execute_exit_order(self, timestamp: datetime, position: int, current_price: float,
                          order_size_base: float, market_tick: Dict, exit_reason: str) -> Dict:
        """Execute exit order with full validation and simulation"""
        
        # Create exit order (opposite side of position)
        order_id = f"EXIT_{uuid.uuid4().hex[:8]}_{timestamp.strftime('%H%M%S')}"
        order = Order(
            order_id=order_id,
            side=OrderSide.SELL if position > 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=order_size_base,  # Simplified - should match position size
            price=current_price,
            timestamp=timestamp
        )
        
        # Log order sent
        self.order_gateway.log_order_sent(order)
        
        # Simulate execution (exit orders typically have higher fill rate)
        execution_result = self.matching_engine.simulate_execution(order, market_tick)
        
        if execution_result['outcome'] in [ExecutionOutcome.FILLED, ExecutionOutcome.PARTIALLY_FILLED]:
            # Log execution
            if execution_result['filled_quantity'] > 0:
                mock_trade = {
                    'timestamp': timestamp,
                    'buy_order_id': order_id if position < 0 else 'MARKET',
                    'sell_order_id': order_id if position > 0 else 'MARKET',
                    'quantity': execution_result['filled_quantity'],
                    'price': execution_result['execution_price'],
                    'symbol': 'BABA'
                }
                self.order_gateway.log_order_filled(mock_trade)
                
                return {
                    'executed': True,
                    'order_id': order_id,
                    'execution_price': execution_result['execution_price'],
                    'quantity': execution_result['filled_quantity']
                }
        
        # If not executed, log rejection
        self.order_gateway.log_order_rejected(order, execution_result.get('rejection_reason', 'Unknown'))
        return {'executed': False, 'reason': 'execution_failed'}
    
    def _check_exit_conditions(self, position: int, entry_time: datetime, entry_price: float,
                             current_price: float, current_time: datetime, signal: int) -> Tuple[bool, str]:
        """Check if position should be exited"""
        
        # Calculate P&L
        if position > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Profit target (4%)
        if pnl_pct >= 0.04:
            return True, "PROFIT_TARGET"
        
        # Stop loss (2.5%)
        if pnl_pct <= -0.025:
            return True, "STOP_LOSS"
        
        # Signal reversal
        if (position > 0 and signal < 0) or (position < 0 and signal > 0):
            return True, "SIGNAL_REVERSAL"
        
        # Time-based exit (5 hours = 300 minutes)
        hold_time_minutes = (current_time - entry_time).total_seconds() / 60
        if hold_time_minutes > 300:
            return True, "TIME_EXIT"
        
        return False, ""
    
    def generate_performance_report(self, save_plots: bool = False) -> str:
        """Generate comprehensive performance report"""
        
        if not self.backtest_results:
            return "No backtest results available. Run backtest first."
        
        print("\n📋 Generating Comprehensive Performance Report...")
        print("=" * 60)
        
        # Text report
        performance_metrics = self.backtest_results['performance_metrics']
        text_report = self.performance_analytics.generate_performance_report(performance_metrics)
        
        # Visual report
        self.visualizer.create_comprehensive_report(
            performance_metrics=performance_metrics,
            trades_df=self.backtest_results['trades_df'],
            signals_data=self.backtest_results['signals_history'],
            market_data=self.backtest_results['market_data'],
            save_plots=save_plots
        )
        
        return text_report
    
    def run_parameter_sensitivity_analysis(self, 
                                         parameter_ranges: Dict,
                                         sample_size: int = 10) -> Dict:
        """Run parameter sensitivity analysis"""
        
        print("\n🔍 Running Parameter Sensitivity Analysis...")
        print("=" * 50)
        
        results = {}
        
        for param_name, param_range in parameter_ranges.items():
            print(f"Testing parameter: {param_name}")
            param_results = []
            
            for param_value in np.linspace(param_range[0], param_range[1], sample_size):
                # Modify strategy parameter
                if hasattr(self.strategy, param_name):
                    setattr(self.strategy, param_name, param_value)
                
                # Run mini backtest
                mini_results = self.run_backtest(verbose=False)
                
                if mini_results:
                    performance = mini_results['performance_metrics']
                    param_results.append({
                        'parameter_value': param_value,
                        'total_return': performance['return_metrics']['total_return'],
                        'sharpe_ratio': performance['risk_metrics']['sharpe_ratio'],
                        'max_drawdown': performance['risk_metrics']['max_drawdown'],
                        'win_rate': performance['basic_metrics']['win_rate']
                    })
            
            results[param_name] = pd.DataFrame(param_results)
        
        print("✅ Parameter sensitivity analysis completed!")
        return results
    
    def compare_strategy_variants(self, variant_configs: List[Dict]) -> Dict:
        """Compare different strategy configurations"""
        
        print("\n⚖️  Comparing Strategy Variants...")
        print("=" * 50)
        
        comparison_results = {}
        
        for i, config in enumerate(variant_configs):
            variant_name = config.get('name', f'Variant_{i+1}')
            print(f"Testing {variant_name}...")
            
            # Apply configuration
            for param, value in config.items():
                if param != 'name' and hasattr(self.strategy, param):
                    setattr(self.strategy, param, value)
            
            # Run backtest
            results = self.run_backtest(verbose=False)
            
            if results:
                comparison_results[variant_name] = results['performance_metrics']
        
        print("✅ Strategy variant comparison completed!")
        return comparison_results
    
    def save_backtest_results(self, filename: str = None) -> str:
        """Save backtest results to file"""
        
        if not self.backtest_results:
            return "No results to save"
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baba_backtest_results_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Save trades
                self.backtest_results['trades_df'].to_excel(
                    writer, sheet_name='Trades', index=False
                )
                
                # Save performance metrics (flattened)
                metrics_df = pd.json_normalize(self.backtest_results['performance_metrics'])
                metrics_df.to_excel(
                    writer, sheet_name='Performance_Metrics', index=False
                )
                
                # Save market data sample
                self.backtest_results['market_data'].head(1000).to_excel(
                    writer, sheet_name='Market_Data_Sample'
                )
            
            return f"Results saved to: {filename}"
            
        except Exception as e:
            return f"Error saving results: {e}"


# Example usage and testing functions
def run_complete_backtest_example():
    """Run a complete backtesting example"""
    
    print("🎯 BABA ALGORITHMIC TRADING STRATEGY - COMPLETE BACKTEST")
    print("=" * 70)
    
    # Initialize framework
    backtester = BacktestingFramework(
        initial_capital=100000,
        csv_file_path='market_data_baba.csv'
    )
    
    # Prepare data
    news_text = "Alibaba reports strong quarterly earnings with cloud growth momentum"
    if not backtester.prepare_data(news_text):
        print("❌ Failed to prepare data")
        return
    
    # Run backtest
    results = backtester.run_backtest(
        order_size_base=100,
        simulate_latency=True,
        verbose=True
    )
    
    if not results:
        print("❌ Backtest failed")
        return
    
    # Generate comprehensive report
    text_report = backtester.generate_performance_report(save_plots=True)
    print("\n" + text_report)
    
    # Save results
    save_message = backtester.save_backtest_results()
    print(f"\n💾 {save_message}")
    
    print("\n🎉 Complete backtest analysis finished!")
    return backtester


if __name__ == "__main__":
    # Run example backtest
    backtester = run_complete_backtest_example()