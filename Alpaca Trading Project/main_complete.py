"""
Updated Main Integration File - Now includes Part 3 Backtesting Framework
Combines Part 1 (Trading Strategy), Part 2 (Backtester Framework), and Part 3 (Performance Analysis)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import warnings
warnings.filterwarnings('ignore')

# Import all custom modules
from trading_strategy import BABAAlgoTradingStrategy
from gateway import MarketDataGateway
from orderbook import OrderBook, Order, OrderSide, OrderType
from order_manager import OrderManager
from order_gateway import OrderGateway
from matching_engine import MatchingEngine
from performance_analytics import PerformanceAnalytics
from visualization import TradingVisualizer
from backtesting_framework import BacktestingFramework


def run_complete_baba_trading_system():
    """Run the complete BABA algorithmic trading system with comprehensive analysis"""
    
    print("🎯 BABA ALGORITHMIC TRADING SYSTEM - COMPLETE EXECUTION")
    print("=" * 70)
    
    # Configuration
    INITIAL_CAPITAL = 100000
    CSV_FILE = 'market_data_baba.csv'
    NEWS_TEXT = "Alibaba reports strong quarterly earnings with cloud growth momentum and improved profitability"
    
    print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"📊 Data Source: {CSV_FILE}")
    print(f"📰 News Sentiment: {NEWS_TEXT[:50]}...")
    
    try:
        # Initialize the comprehensive backtesting framework
        print("\n🚀 Initializing Backtesting Framework...")
        backtester = BacktestingFramework(
            initial_capital=INITIAL_CAPITAL,
            csv_file_path=CSV_FILE
        )
        
        # Step 1: Prepare data
        print("\n📊 Step 1: Preparing Market Data...")
        if not backtester.prepare_data(news_text=NEWS_TEXT):
            print("❌ Failed to prepare data")
            return
        
        data_stats = {
            'data_points': len(backtester.market_data),
            'date_range': f"{backtester.market_data.index.min()} to {backtester.market_data.index.max()}",
            'price_range': f"${backtester.market_data['Close'].min():.2f} - ${backtester.market_data['Close'].max():.2f}"
        }
        
        for key, value in data_stats.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        # Step 2: Execute backtest
        print("\n⚡ Step 2: Executing Strategy Backtest...")
        results = backtester.run_backtest(
            order_size_base=100,
            simulate_latency=True,
            verbose=False  # Reduced verbosity for cleaner output
        )
        
        if not results:
            print("❌ Backtest failed")
            return
        
        # Step 3: Analyze performance
        print("\n📈 Step 3: Performance Analysis...")
        performance_report = backtester.generate_performance_report(save_plots=False)
        
        # Show key metrics
        performance = results['performance_metrics']
        trades_df = results['trades_df']
        
        print(f"\n🏆 KEY PERFORMANCE METRICS:")
        if performance and 'return_metrics' in performance:
            total_return = performance['return_metrics']['total_return']
            total_pnl = performance['return_metrics']['total_pnl_dollar']
            print(f"  • Total Return: {total_return:.2%}")
            print(f"  • Total P&L: ${total_pnl:,.2f}")
        
        if performance and 'basic_metrics' in performance:
            win_rate = performance['basic_metrics']['win_rate']
            total_trades = performance['basic_metrics']['total_trades']
            profit_factor = performance['basic_metrics']['profit_factor']
            print(f"  • Win Rate: {win_rate:.2%}")
            print(f"  • Total Trades: {total_trades}")
            print(f"  • Profit Factor: {profit_factor:.2f}")
        
        if performance and 'risk_metrics' in performance:
            sharpe_ratio = performance['risk_metrics']['sharpe_ratio']
            max_drawdown = performance['risk_metrics']['max_drawdown']
            print(f"  • Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  • Max Drawdown: {max_drawdown:.2%}")
        
        # Step 4: Save results
        print("\n💾 Step 4: Saving Results...")
        save_message = backtester.save_backtest_results()
        print(f"  • {save_message}")
        print(f"  • Order logs: backtest_orders.log")
        
        # Performance grade
        grade = "A+"
        if performance and 'risk_metrics' in performance:
            sharpe_ratio = performance['risk_metrics']['sharpe_ratio']
            if sharpe_ratio < 1.0:
                grade = "B"
            if sharpe_ratio < 0.5:
                grade = "C"
            if total_return < 0:
                grade = "D"
        
        print(f"\n🎓 PERFORMANCE GRADE: {grade}")
        status = '✅ PROFITABLE' if total_return > 0 else '❌ UNPROFITABLE'
        print(f"🎯 STRATEGY STATUS: {status}")
        
        print(f"\n🎊 SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return backtester
        
    except Exception as e:
        print(f"❌ Error during system execution: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simple_component_test():
    """Run a simple test of individual components"""
    
    print("🧪 COMPONENT TESTING MODE")
    print("=" * 40)
    
    components_to_test = [
        ('Trading Strategy', BABAAlgoTradingStrategy),
        ('Market Data Gateway', MarketDataGateway),
        ('Order Book', OrderBook),
        ('Order Manager', OrderManager),
        ('Order Gateway', OrderGateway),
        ('Matching Engine', MatchingEngine),
        ('Performance Analytics', PerformanceAnalytics),
        ('Trading Visualizer', TradingVisualizer),
        ('Backtesting Framework', BacktestingFramework)
    ]
    
    print("Testing component initialization...")
    
    for name, component_class in components_to_test:
        try:
            if name == 'Trading Strategy':
                obj = component_class(portfolio_value=50000)
            elif name == 'Market Data Gateway':
                obj = component_class('market_data_baba.csv')
            elif name == 'Order Book':
                obj = component_class('BABA')
            elif name == 'Order Manager':
                obj = component_class(50000, {'max_orders_per_minute': 10})
            elif name == 'Order Gateway':
                obj = component_class('test_orders.log')
            elif name == 'Performance Analytics':
                obj = component_class(50000)
            elif name == 'Backtesting Framework':
                obj = component_class(50000, 'market_data_baba.csv')
            else:
                obj = component_class()
            
            print(f"  ✅ {name}: {type(obj).__name__}")
            
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    print("\n✅ Component testing completed!")


if __name__ == "__main__":
    print("Choose execution mode:")
    print("1. Complete BABA Trading System (recommended)")
    print("2. Component Testing Only")
    
    try:
        # Default to complete system
        choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
        
        if choice == "1":
            backtester = run_complete_baba_trading_system()
        elif choice == "2":
            run_simple_component_test()
        else:
            print("Invalid choice. Running complete system...")
            backtester = run_complete_baba_trading_system()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        # Fallback to component testing
        print("\n🔄 Falling back to component testing...")
        run_simple_component_test()