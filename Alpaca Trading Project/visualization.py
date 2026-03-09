"""
Visualization Module - Create comprehensive trading performance visualizations
Part 3: Strategy Backtesting (Reporting Component)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TradingVisualizer:
    """Create comprehensive trading performance visualizations"""
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_equity_curve(self, equity_curve: pd.Series, 
                         benchmark_data: pd.DataFrame = None,
                         title: str = "BABA Strategy Equity Curve") -> None:
        """Plot equity curve with optional benchmark comparison"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot strategy equity curve
        ax.plot(equity_curve.index, equity_curve.values, 
               linewidth=2, label='Strategy', color='blue')
        
        # Plot benchmark if provided
        if benchmark_data is not None:
            benchmark_curve = self._calculate_benchmark_equity(equity_curve, benchmark_data)
            ax.plot(benchmark_curve.index, benchmark_curve.values,
                   linewidth=2, label='Buy & Hold Benchmark', 
                   color='orange', alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value annotations
        start_value = equity_curve.iloc[0]
        end_value = equity_curve.iloc[-1]
        total_return = (end_value - start_value) / start_value
        
        ax.annotate(f'Total Return: {total_return:.2%}',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=11, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_analysis(self, equity_curve: pd.Series,
                             title: str = "Drawdown Analysis") -> None:
        """Plot drawdown analysis"""
        
        if equity_curve.empty:
            print("No equity curve data available for drawdown analysis")
            return
        
        # Calculate drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       height_ratios=[2, 1])
        
        # Plot equity curve with peaks
        ax1.plot(equity_curve.index, equity_curve.values, 
                linewidth=2, label='Equity', color='blue')
        ax1.plot(peak.index, peak.values, 
                linewidth=1, label='Peak', color='red', alpha=0.7)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown.values, 
                linewidth=1, color='red')
        ax2.set_ylabel('Drawdown %', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add max drawdown annotation
        max_dd = abs(drawdown.min())
        max_dd_date = drawdown.idxmin()
        
        ax2.annotate(f'Max Drawdown: {max_dd:.2%}',
                    xy=(max_dd_date, drawdown.min()),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=10)
        
        # Format dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame,
                               title: str = "Trade Distribution Analysis") -> None:
        """Plot trade return distribution and statistics"""
        
        if trades_df.empty:
            print("No trade data available for distribution analysis")
            return
        
        # Calculate returns for completed trades
        completed_trades = trades_df[trades_df['action'] == 'EXIT']['pnl_pct'].dropna()
        
        if completed_trades.empty:
            print("No completed trades available for analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Histogram of returns
        ax1.hist(completed_trades, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(completed_trades.mean(), color='red', linestyle='--', 
                   label=f'Mean: {completed_trades.mean():.2%}')
        ax1.set_title('Distribution of Trade Returns', fontweight='bold')
        ax1.set_xlabel('Return %')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2.boxplot(completed_trades, vert=True)
        ax2.set_title('Trade Returns Box Plot', fontweight='bold')
        ax2.set_ylabel('Return %')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative returns
        cumulative_returns = (1 + completed_trades).cumprod()
        ax3.plot(range(len(cumulative_returns)), cumulative_returns, 
                linewidth=2, color='green')
        ax3.set_title('Cumulative Trade Returns', fontweight='bold')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative Return Multiple')
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling statistics
        window = min(10, len(completed_trades) // 2) if len(completed_trades) > 10 else len(completed_trades)
        if window > 1:
            rolling_mean = completed_trades.rolling(window=window).mean()
            rolling_std = completed_trades.rolling(window=window).std()
            
            ax4.plot(range(len(rolling_mean)), rolling_mean, 
                    label=f'Rolling Mean (window={window})', linewidth=2)
            ax4.fill_between(range(len(rolling_mean)), 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.3, label='±1 Std Dev')
            ax4.set_title('Rolling Statistics', fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Return %')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, performance_metrics: Dict,
                                title: str = "Performance Metrics Dashboard") -> None:
        """Plot comprehensive performance metrics dashboard"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Win Rate and Trade Statistics
        basic_metrics = performance_metrics['basic_metrics']
        
        win_rate = basic_metrics['win_rate']
        lose_rate = 1 - win_rate
        
        ax1.pie([win_rate, lose_rate], labels=['Winning Trades', 'Losing Trades'],
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'],
               startangle=90)
        ax1.set_title(f'Win Rate: {win_rate:.1%}\n(Total Trades: {basic_metrics["total_trades"]})',
                     fontweight='bold')
        
        # 2. Risk-Return Scatter
        risk_metrics = performance_metrics['risk_metrics']
        return_metrics = performance_metrics['return_metrics']
        
        scatter_data = {
            'Strategy': (risk_metrics['volatility'], return_metrics['total_return']),
        }
        
        for name, (vol, ret) in scatter_data.items():
            ax2.scatter(vol, ret, s=100, label=name, alpha=0.7)
        
        ax2.set_xlabel('Volatility (Risk)')
        ax2.set_ylabel('Total Return')
        ax2.set_title('Risk-Return Profile', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Key Metrics Bar Chart
        metrics_names = ['Sharpe Ratio', 'Calmar Ratio', 'Profit Factor']
        metrics_values = [
            risk_metrics['sharpe_ratio'],
            risk_metrics['calmar_ratio'],
            basic_metrics['profit_factor']
        ]
        
        colors = ['skyblue' if v > 0 else 'lightcoral' for v in metrics_values]
        bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax3.set_title('Key Performance Ratios', fontweight='bold')
        ax3.set_ylabel('Ratio Value')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 4. Monthly/Period Performance (if enough data)
        equity_curve = performance_metrics['trading_metrics'].get('equity_curve', pd.Series())
        
        if not equity_curve.empty and len(equity_curve) > 1:
            # Calculate period returns
            equity_df = pd.DataFrame({'equity': equity_curve})
            equity_df['period'] = equity_df.index.to_period('D')  # Daily periods
            
            period_returns = equity_df.groupby('period')['equity'].last().pct_change().dropna()
            
            if len(period_returns) > 0:
                period_returns.plot(kind='bar', ax=ax4, color='lightblue', alpha=0.7)
                ax4.set_title('Daily Returns Distribution', fontweight='bold')
                ax4.set_xlabel('Trading Day')
                ax4.set_ylabel('Daily Return %')
                ax4.grid(True, axis='y', alpha=0.3)
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor period analysis', 
                        ha='center', va='center', transform=ax4.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                ax4.set_title('Period Analysis', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No equity curve data\navailable', 
                    ha='center', va='center', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax4.set_title('Equity Analysis', fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_signal_analysis(self, signals_data: Dict, market_data: pd.DataFrame,
                           title: str = "Signal Analysis") -> None:
        """Plot trading signals analysis"""
        
        if not signals_data or market_data.empty:
            print("Insufficient data for signal analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Align data
        common_index = market_data.index.intersection(signals_data['combined'].index)
        
        if len(common_index) == 0:
            print("No common data points between signals and market data")
            return
        
        price_data = market_data.loc[common_index, 'Close']
        combined_signals = signals_data['combined'].loc[common_index]
        
        # 1. Price with signals
        ax1.plot(price_data.index, price_data.values, linewidth=1, color='black', alpha=0.7)
        
        # Mark buy signals
        buy_signals = combined_signals == 1
        if buy_signals.any():
            buy_points = price_data[buy_signals]
            ax1.scatter(buy_points.index, buy_points.values, 
                       color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
        
        # Mark sell signals
        sell_signals = combined_signals == -1
        if sell_signals.any():
            sell_points = price_data[sell_signals]
            ax1.scatter(sell_points.index, sell_points.values, 
                       color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
        
        ax1.set_title('Price with Trading Signals', fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Signal strength over time
        if 'strength' in signals_data:
            strength_data = signals_data['strength'].loc[common_index]
            ax2.plot(strength_data.index, strength_data.values, 
                    linewidth=2, color='purple')
            ax2.fill_between(strength_data.index, 0, strength_data.values, 
                           alpha=0.3, color='purple')
            ax2.set_title('Signal Strength Over Time', fontweight='bold')
            ax2.set_ylabel('Signal Strength')
            ax2.grid(True, alpha=0.3)
        
        # 3. Individual signal components
        signal_names = ['momentum', 'ml', 'sentiment']
        colors = ['blue', 'orange', 'green']
        
        for i, (signal_name, color) in enumerate(zip(signal_names, colors)):
            if signal_name in signals_data:
                signal_data = signals_data[signal_name].loc[common_index]
                ax3.plot(signal_data.index, signal_data.values + i * 0.1, 
                        label=signal_name.capitalize(), color=color, alpha=0.7)
        
        ax3.set_title('Individual Signal Components', fontweight='bold')
        ax3.set_ylabel('Signal Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Signal statistics
        signal_stats = {}
        for signal_name in signal_names:
            if signal_name in signals_data:
                signal_data = signals_data[signal_name].loc[common_index]
                signal_stats[signal_name] = {
                    'buy_signals': (signal_data == 1).sum(),
                    'sell_signals': (signal_data == -1).sum(),
                    'neutral': (signal_data == 0).sum()
                }
        
        if signal_stats:
            # Create stacked bar chart
            signal_types = ['buy_signals', 'sell_signals', 'neutral']
            signal_labels = ['Buy', 'Sell', 'Neutral']
            colors = ['green', 'red', 'gray']
            
            bottom = np.zeros(len(signal_stats))
            
            for signal_type, label, color in zip(signal_types, signal_labels, colors):
                values = [signal_stats[name][signal_type] for name in signal_stats.keys()]
                ax4.bar(signal_stats.keys(), values, bottom=bottom, 
                       label=label, color=color, alpha=0.7)
                bottom += values
            
            ax4.set_title('Signal Distribution by Component', fontweight='bold')
            ax4.set_ylabel('Number of Signals')
            ax4.legend()
            ax4.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def _calculate_benchmark_equity(self, equity_curve: pd.Series, 
                                   benchmark_data: pd.DataFrame) -> pd.Series:
        """Calculate benchmark equity curve for comparison"""
        
        if equity_curve.empty or benchmark_data.empty:
            return pd.Series()
        
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        initial_value = equity_curve.iloc[0]
        
        # Filter benchmark data to the same period
        benchmark_period = benchmark_data.loc[start_date:end_date]
        
        if len(benchmark_period) < 2:
            return pd.Series()
        
        # Calculate benchmark returns
        benchmark_returns = benchmark_period['Close'].pct_change().fillna(0)
        benchmark_equity = initial_value * (1 + benchmark_returns).cumprod()
        
        return benchmark_equity
    
    def create_comprehensive_report(self, performance_metrics: Dict,
                                   trades_df: pd.DataFrame,
                                   signals_data: Dict,
                                   market_data: pd.DataFrame,
                                   save_plots: bool = False,
                                   plot_prefix: str = "baba_strategy") -> None:
        """Create comprehensive visual performance report"""
        
        print("🎨 Generating Comprehensive Visual Performance Report...")
        print("=" * 60)
        
        # 1. Equity Curve
        equity_curve = performance_metrics['trading_metrics'].get('equity_curve', pd.Series())
        if not equity_curve.empty:
            self.plot_equity_curve(equity_curve, market_data, 
                                 "BABA Strategy - Equity Curve vs Benchmark")
            if save_plots:
                plt.savefig(f"{plot_prefix}_equity_curve.png", dpi=300, bbox_inches='tight')
        
        # 2. Drawdown Analysis
        if not equity_curve.empty:
            self.plot_drawdown_analysis(equity_curve, "BABA Strategy - Drawdown Analysis")
            if save_plots:
                plt.savefig(f"{plot_prefix}_drawdown.png", dpi=300, bbox_inches='tight')
        
        # 3. Trade Distribution
        if not trades_df.empty:
            self.plot_trade_distribution(trades_df, "BABA Strategy - Trade Analysis")
            if save_plots:
                plt.savefig(f"{plot_prefix}_trades.png", dpi=300, bbox_inches='tight')
        
        # 4. Performance Metrics
        self.plot_performance_metrics(performance_metrics, "BABA Strategy - Performance Dashboard")
        if save_plots:
            plt.savefig(f"{plot_prefix}_metrics.png", dpi=300, bbox_inches='tight')
        
        # 5. Signal Analysis
        if signals_data and not market_data.empty:
            self.plot_signal_analysis(signals_data, market_data, 
                                    "BABA Strategy - Signal Analysis")
            if save_plots:
                plt.savefig(f"{plot_prefix}_signals.png", dpi=300, bbox_inches='tight')
        
        print("✅ Visual report generation completed!")
        if save_plots:
            print(f"📁 Plots saved with prefix: {plot_prefix}")