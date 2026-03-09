"""
Main Integration File - Demonstrates how to use all trading system components together
Combines Part 1 (Trading Strategy) with Part 2 (Backtester Framework)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import uuid

# Import all custom modules
from trading_strategy import BABAAlgoTradingStrategy
from gateway import MarketDataGateway
from orderbook import OrderBook, Order, OrderSide, OrderType
from order_manager import OrderManager
from order_gateway import OrderGateway
from matching_engine import MatchingEngine


def create_trading_system(csv_file_path: str, initial_capital: float = 100000):
    """Create and initialize complete trading system"""
    
    print("🚀 Initializing BABA Algorithmic Trading System...")
    
    # Initialize all components
    components = {
        # Part 1: Trading Strategy
        'strategy': BABAAlgoTradingStrategy(portfolio_value=initial_capital),
        
        # Part 2: Backtester Framework
        'data_gateway': MarketDataGateway(csv_file_path),
        'order_book': OrderBook('BABA'),
        'order_manager': OrderManager(
            initial_capital=initial_capital,
            risk_limits={
                'max_orders_per_minute': 10,
                'max_position': 1000,
                'max_order_size': initial_capital * 0.1,  # 10% of capital per order
                'max_price_deviation': 0.05  # 5% price deviation limit
            }
        ),
        'order_gateway': OrderGateway('trading_system_orders.log'),
        'matching_engine': MatchingEngine(
            fill_probability=0.85,
            partial_fill_probability=0.10,
            rejection_probability=0.05
        )
    }
    
    print("✅ All components initialized successfully!")
    return components


def run_backtest(components: dict):
    """Run complete backtest simulation"""
    
    print("\n📊 Starting Backtest Simulation...")
    
    # Load and prepare data
    gateway = components['data_gateway']
    strategy = components['strategy']
    order_manager = components['order_manager']
    order_gateway = components['order_gateway']
    matching_engine = components['matching_engine']
    order_book = components['order_book']
    
    # Load historical data
    gateway.load_data()
    data = gateway.data
    
    # Train strategy ML model
    print("🤖 Training ML model...")
    strategy.train_ml_model(data)
    
    # Generate trading signals
    print("📈 Generating trading signals...")
    sample_news = "Alibaba reports strong quarterly earnings with cloud growth"
    signals_data = strategy.generate_signals(data, sample_news)
    
    # Execute backtest
    print("⚡ Executing backtest...")
    trades = []
    position = 0
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing tick {i}/{len(data)}...")
            
        # Get current market data
        market_tick = {
            'timestamp': timestamp,
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        
        # Check for trading signals
        if timestamp in signals_data['combined'].index:
            signal = signals_data['combined'][timestamp]
            strength = signals_data['strength'][timestamp]
            
            if signal != 0 and position == 0:  # Entry signal
                # Create order
                order_id = f"ORDER_{uuid.uuid4().hex[:8]}"
                order_quantity = 100  # Fixed quantity for demo
                order_price = row['Close']
                
                order = Order(
                    order_id=order_id,
                    side=OrderSide.BUY if signal > 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=order_quantity,
                    price=order_price,
                    timestamp=timestamp
                )
                
                # Validate order
                validation = order_manager.validate_order(order, row['Close'])
                order_manager.record_order(order, validation)
                
                if validation['valid']:
                    # Log order
                    order_gateway.log_order_sent(order)
                    
                    # Simulate execution
                    execution_result = matching_engine.simulate_execution(order, market_tick)
                    
                    if execution_result['outcome'].value == 'FILLED':
                        # Add to order book and execute
                        trade_results = order_book.add_order(order)
                        
                        # Log execution
                        if trade_results:
                            for trade in trade_results:
                                order_gateway.log_order_filled(trade)
                                order_manager.update_position(trade)
                                
                        position = signal
                        
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'ENTRY',
                            'price': execution_result['execution_price'],
                            'quantity': execution_result['filled_quantity'],
                            'signal_strength': strength
                        })
                        
                    elif execution_result['outcome'].value == 'REJECTED':
                        order_gateway.log_order_rejected(order, execution_result['rejection_reason'])
                        
                else:
                    # Order rejected by risk management
                    order_gateway.log_order_rejected(order, '; '.join(validation['errors']))
    
    print("✅ Backtest completed!")
    return trades


def generate_performance_report(components: dict, trades: list):
    """Generate comprehensive performance report"""
    
    print("\n📋 Generating Performance Report...")
    
    # Get component statistics
    order_manager = components['order_manager']
    order_gateway = components['order_gateway']
    matching_engine = components['matching_engine']
    order_book = components['order_book']
    
    # Portfolio summary
    portfolio_summary = order_manager.get_portfolio_summary()
    risk_metrics = order_manager.get_risk_metrics()
    execution_stats = matching_engine.get_execution_statistics()
    log_summary = order_gateway.get_log_summary()
    
    # Trading performance
    if trades:
        total_trades = len(trades)
        entry_trades = [t for t in trades if t['action'] == 'ENTRY']
        avg_signal_strength = np.mean([t['signal_strength'] for t in entry_trades])
    else:
        total_trades = 0
        avg_signal_strength = 0
    
    report = {
        'backtest_summary': {
            'total_trades_executed': total_trades,
            'average_signal_strength': avg_signal_strength,
        },
        'portfolio_performance': portfolio_summary,
        'risk_management': risk_metrics,
        'execution_quality': execution_stats,
        'order_logging': log_summary,
        'order_book_status': {
            'best_bid': order_book.get_best_bid(),
            'best_ask': order_book.get_best_ask(),
            'spread': order_book.get_spread(),
            'total_trades_matched': len(order_book.trade_history)
        }
    }
    
    # Print report
    print("\n" + "="*60)
    print("           BABA ALGORITHMIC TRADING SYSTEM REPORT")
    print("="*60)
    
    for section, data in report.items():
        print(f"\n📊 {section.upper().replace('_', ' ')}:")
        for key, value in data.items():
            if isinstance(value, float):
                if 'rate' in key or 'utilization' in key or 'pct' in key:
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    return report


def main():
    """Main execution function"""
    
    # Configuration
    CSV_FILE = 'market_data_baba.csv'  # Make sure this file exists
    INITIAL_CAPITAL = 100000
    
    try:
        # Create trading system
        system_components = create_trading_system(CSV_FILE, INITIAL_CAPITAL)
        
        # Run backtest
        trade_results = run_backtest(system_components)
        
        # Generate report
        performance_report = generate_performance_report(system_components, trade_results)
        
        print(f"\n🎯 Backtest completed successfully!")
        print(f"📁 Order logs saved to: trading_system_orders.log")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()