"""
Performance Analytics - Calculate comprehensive trading performance metrics
Part 3: Strategy Backtesting (Performance Tracking Component)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceAnalytics:
    """Calculate comprehensive trading performance metrics"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades = []
        self.portfolio_values = []
        self.drawdowns = []
        
    def add_trade(self, trade: Dict):
        """Add a trade to performance tracking"""
        self.trades.append(trade)
        
    def calculate_returns(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trade-by-trade returns"""
        if trades_df.empty:
            return pd.DataFrame()
            
        # Separate entry and exit trades
        entries = trades_df[trades_df['action'] == 'ENTRY'].copy()
        exits = trades_df[trades_df['action'] == 'EXIT'].copy()
        
        # Calculate returns for completed trades
        completed_trades = []
        for _, exit_trade in exits.iterrows():
            # Find corresponding entry (simplified - assumes trades are paired)
            entry_trades = entries[entries.index < exit_trade.name]
            if not entry_trades.empty:
                entry_trade = entry_trades.iloc[-1]  # Get most recent entry
                
                trade_return = exit_trade.get('pnl_pct', 0.0)
                trade_pnl = trade_return * entry_trade.get('position_size', 0.0)
                
                completed_trades.append({
                    'entry_time': entry_trade['timestamp'],
                    'exit_time': exit_trade['timestamp'],
                    'entry_price': entry_trade['price'],
                    'exit_price': exit_trade['price'],
                    'position_size': entry_trade.get('position_size', 0.0),
                    'return_pct': trade_return,
                    'pnl_dollar': trade_pnl,
                    'hold_time_hours': exit_trade.get('hold_time_min', 0) / 60,
                    'exit_reason': exit_trade.get('signal', 'UNKNOWN')
                })
        
        return pd.DataFrame(completed_trades)
    
    def calculate_portfolio_performance(self, trades_df: pd.DataFrame, 
                                      market_data: pd.DataFrame = None) -> Dict:
        """Calculate comprehensive portfolio performance metrics"""
        
        if trades_df.empty:
            return self._empty_performance_metrics()
        
        # Calculate trade-level metrics
        returns_df = self.calculate_returns(trades_df)
        
        if returns_df.empty:
            return self._empty_performance_metrics()
        
        # Basic performance metrics
        total_trades = len(returns_df)
        winning_trades = len(returns_df[returns_df['return_pct'] > 0])
        losing_trades = len(returns_df[returns_df['return_pct'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = returns_df['pnl_dollar'].sum()
        total_return = total_pnl / self.initial_capital
        
        avg_win = returns_df[returns_df['return_pct'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_loss = returns_df[returns_df['return_pct'] < 0]['return_pct'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        returns_series = returns_df['return_pct']
        volatility = returns_series.std() if len(returns_series) > 1 else 0
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if volatility > 0:
            mean_return = returns_series.mean()
            sharpe_ratio = (mean_return * 252) / (volatility * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Drawdown calculation
        equity_curve = self._calculate_equity_curve(returns_df)
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_curve)
        
        # Additional metrics
        profit_factor = (returns_df[returns_df['return_pct'] > 0]['return_pct'].sum() /
                        abs(returns_df[returns_df['return_pct'] < 0]['return_pct'].sum())) if losing_trades > 0 else float('inf')
        
        avg_hold_time = returns_df['hold_time_hours'].mean()
        
        # Benchmark comparison (if market data provided)
        benchmark_return = 0
        beta = 0
        alpha = 0
        
        if market_data is not None:
            benchmark_return = self._calculate_benchmark_return(returns_df, market_data)
            beta = self._calculate_beta(returns_df, market_data)
            alpha = total_return - (benchmark_return * beta)
        
        return {
            'basic_metrics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor
            },
            'return_metrics': {
                'total_return': total_return,
                'total_pnl_dollar': total_pnl,
                'average_trade_return': returns_series.mean(),
                'best_trade': returns_series.max(),
                'worst_trade': returns_series.min()
            },
            'risk_metrics': {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration_days': max_drawdown_duration,
                'calmar_ratio': total_return / max_drawdown if max_drawdown > 0 else float('inf')
            },
            'trading_metrics': {
                'average_hold_time_hours': avg_hold_time,
                'trades_per_day': self._calculate_trades_per_day(returns_df),
                'equity_curve': equity_curve
            },
            'benchmark_metrics': {
                'benchmark_return': benchmark_return,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': alpha / volatility if volatility > 0 else 0
            }
        }
    
    def _empty_performance_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'basic_metrics': {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0},
            'return_metrics': {'total_return': 0, 'total_pnl_dollar': 0},
            'risk_metrics': {'sharpe_ratio': 0, 'max_drawdown': 0},
            'trading_metrics': {'average_hold_time_hours': 0},
            'benchmark_metrics': {'alpha': 0, 'beta': 0}
        }
    
    def _calculate_equity_curve(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate cumulative equity curve"""
        if returns_df.empty:
            return pd.Series()
        
        # Sort by exit time
        returns_df = returns_df.sort_values('exit_time')
        
        # Calculate cumulative returns
        cumulative_pnl = returns_df['pnl_dollar'].cumsum()
        equity_curve = self.initial_capital + cumulative_pnl
        
        # Create time series
        equity_series = pd.Series(equity_curve.values, index=returns_df['exit_time'])
        return equity_series
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if equity_curve.empty:
            return 0.0, 0
        
        # Calculate running maximum
        peak = equity_curve.cummax()
        
        # Calculate drawdown as percentage from peak
        drawdown = (equity_curve - peak) / peak
        
        # Maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        # Calculate drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_lengths = []
        current_length = 0
        
        for in_drawdown in drawdown_periods:
            if in_drawdown:
                current_length += 1
            else:
                if current_length > 0:
                    drawdown_lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            drawdown_lengths.append(current_length)
        
        max_drawdown_duration = max(drawdown_lengths) if drawdown_lengths else 0
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_benchmark_return(self, returns_df: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate buy-and-hold benchmark return"""
        if returns_df.empty or market_data.empty:
            return 0.0
        
        start_date = returns_df['entry_time'].min()
        end_date = returns_df['exit_time'].max()
        
        # Filter market data to trading period
        period_data = market_data.loc[start_date:end_date]
        
        if len(period_data) < 2:
            return 0.0
        
        start_price = period_data['Close'].iloc[0]
        end_price = period_data['Close'].iloc[-1]
        
        benchmark_return = (end_price - start_price) / start_price
        return benchmark_return
    
    def _calculate_beta(self, returns_df: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate beta against market benchmark"""
        if returns_df.empty or market_data.empty:
            return 0.0
        
        # This is a simplified beta calculation
        # In practice, you'd align returns periods and calculate correlation
        strategy_volatility = returns_df['return_pct'].std()
        market_returns = market_data['Close'].pct_change().dropna()
        market_volatility = market_returns.std()
        
        if market_volatility > 0:
            # Simplified beta estimation
            return strategy_volatility / market_volatility
        return 0.0
    
    def _calculate_trades_per_day(self, returns_df: pd.DataFrame) -> float:
        """Calculate average trades per day"""
        if returns_df.empty:
            return 0.0
        
        start_date = returns_df['entry_time'].min()
        end_date = returns_df['exit_time'].max()
        
        total_days = (end_date - start_date).days + 1
        total_trades = len(returns_df)
        
        return total_trades / total_days if total_days > 0 else 0.0
    
    def generate_performance_report(self, performance_metrics: Dict) -> str:
        """Generate formatted performance report"""
        
        report = []
        report.append("=" * 60)
        report.append("           BABA STRATEGY PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Basic metrics
        basic = performance_metrics['basic_metrics']
        report.append(f"\n📊 BASIC TRADING METRICS:")
        report.append(f"  Total Trades: {basic['total_trades']}")
        report.append(f"  Winning Trades: {basic['winning_trades']}")
        report.append(f"  Losing Trades: {basic['losing_trades']}")
        report.append(f"  Win Rate: {basic['win_rate']:.2%}")
        report.append(f"  Average Win: {basic['average_win']:.2%}")
        report.append(f"  Average Loss: {basic['average_loss']:.2%}")
        report.append(f"  Profit Factor: {basic['profit_factor']:.2f}")
        
        # Return metrics
        returns = performance_metrics['return_metrics']
        report.append(f"\n💰 RETURN METRICS:")
        report.append(f"  Total Return: {returns['total_return']:.2%}")
        report.append(f"  Total P&L: ${returns['total_pnl_dollar']:,.2f}")
        report.append(f"  Average Trade Return: {returns['average_trade_return']:.2%}")
        report.append(f"  Best Trade: {returns['best_trade']:.2%}")
        report.append(f"  Worst Trade: {returns['worst_trade']:.2%}")
        
        # Risk metrics
        risk = performance_metrics['risk_metrics']
        report.append(f"\n⚠️  RISK METRICS:")
        report.append(f"  Volatility: {risk['volatility']:.2%}")
        report.append(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        report.append(f"  Max Drawdown: {risk['max_drawdown']:.2%}")
        report.append(f"  Max DD Duration: {risk['max_drawdown_duration_days']} periods")
        report.append(f"  Calmar Ratio: {risk['calmar_ratio']:.2f}")
        
        # Trading metrics
        trading = performance_metrics['trading_metrics']
        report.append(f"\n🕒 TRADING METRICS:")
        report.append(f"  Avg Hold Time: {trading['average_hold_time_hours']:.1f} hours")
        report.append(f"  Trades Per Day: {trading['trades_per_day']:.1f}")
        
        # Benchmark metrics
        benchmark = performance_metrics['benchmark_metrics']
        report.append(f"\n📈 BENCHMARK COMPARISON:")
        report.append(f"  Benchmark Return: {benchmark['benchmark_return']:.2%}")
        report.append(f"  Alpha: {benchmark['alpha']:.2%}")
        report.append(f"  Beta: {benchmark['beta']:.2f}")
        report.append(f"  Information Ratio: {benchmark['information_ratio']:.2f}")
        
        return "\n".join(report)