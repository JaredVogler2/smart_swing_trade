# monitoring/performance.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and analyzes trading performance metrics
    """

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Trade history
        self.trades = []
        self.positions = []
        self.daily_pnl = []

        # Performance metrics
        self.metrics = {
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'avg_holding_days': 0
        }

        # Equity curve
        self.equity_curve = [{'date': datetime.now(), 'equity': initial_capital}]
        self.peak_equity = initial_capital

    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trades.append(trade)

        # Update metrics
        self._update_metrics(trade)

        # Update equity
        self.current_capital += trade.get('pnl', 0)
        self.equity_curve.append({
            'date': trade.get('exit_time', datetime.now()),
            'equity': self.current_capital
        })

        # Update peak equity
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital

    def _update_metrics(self, trade: Dict):
        """Update performance metrics with new trade"""
        pnl = trade.get('pnl', 0)

        self.metrics['total_trades'] += 1

        if pnl > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['largest_win'] = max(self.metrics['largest_win'], pnl)
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['largest_loss'] = min(self.metrics['largest_loss'], pnl)

        # Update return
        self.metrics['total_return'] = (self.current_capital - self.initial_capital) / self.initial_capital

        # Update win rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']

        # Update average win/loss
        self._update_averages()

        # Update ratios
        self._update_ratios()

    def _update_averages(self):
        """Update average win/loss metrics"""
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]

        self.metrics['avg_win'] = np.mean(wins) if wins else 0
        self.metrics['avg_loss'] = np.mean(losses) if losses else 0

        # Average holding period
        holding_periods = []
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                days = (trade['exit_time'] - trade['entry_time']).days
                holding_periods.append(days)

        self.metrics['avg_holding_days'] = np.mean(holding_periods) if holding_periods else 0

    def _update_ratios(self):
        """Update performance ratios"""
        # Profit factor
        total_wins = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))

        if total_losses > 0:
            self.metrics['profit_factor'] = total_wins / total_losses
        else:
            self.metrics['profit_factor'] = float('inf') if total_wins > 0 else 0

        # Sharpe ratio (simplified daily)
        if len(self.equity_curve) > 2:
            returns = []
            for i in range(1, len(self.equity_curve)):
                daily_return = (self.equity_curve[i]['equity'] - self.equity_curve[i - 1]['equity']) / \
                               self.equity_curve[i - 1]['equity']
                returns.append(daily_return)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)

                if std_return > 0:
                    # Annualized Sharpe (assuming 252 trading days)
                    self.metrics['sharpe_ratio'] = (avg_return * 252) / (std_return * np.sqrt(252))
                else:
                    self.metrics['sharpe_ratio'] = 0

                # Sortino ratio (downside deviation)
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        self.metrics['sortino_ratio'] = (avg_return * 252) / (downside_std * np.sqrt(252))
                    else:
                        self.metrics['sortino_ratio'] = 0
                else:
                    self.metrics['sortino_ratio'] = self.metrics['sharpe_ratio']

        # Maximum drawdown
        self._calculate_drawdown()

        # Calmar ratio
        if self.metrics['max_drawdown'] > 0:
            annualized_return = self.metrics['total_return'] * (252 / max(len(self.equity_curve) - 1, 1))
            self.metrics['calmar_ratio'] = annualized_return / self.metrics['max_drawdown']

    def _calculate_drawdown(self):
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return

        peak = self.equity_curve[0]['equity']
        max_dd = 0

        for point in self.equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

        self.metrics['max_drawdown'] = max_dd

    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics.copy()

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_curve)
        df.set_index('date', inplace=True)

        # Add returns column
        df['returns'] = df['equity'].pct_change()

        # Add drawdown column
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']

        return df

    def get_trade_analysis(self) -> Dict:
        """Analyze trades by various dimensions"""
        if not self.trades:
            return {}

        df = pd.DataFrame(self.trades)

        analysis = {}

        # By symbol
        if 'symbol' in df.columns:
            symbol_performance = df.groupby('symbol').agg({
                'pnl': ['count', 'sum', 'mean'],
                'return_pct': 'mean'
            }).round(2)
            analysis['by_symbol'] = symbol_performance.to_dict()

        # By setup type
        if 'setup_type' in df.columns:
            setup_performance = df.groupby('setup_type').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            analysis['by_setup'] = setup_performance.to_dict()

        # By day of week
        if 'entry_time' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['entry_time']).dt.day_name()
            dow_performance = df.groupby('day_of_week').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            analysis['by_day_of_week'] = dow_performance.to_dict()

        # By holding period
        if 'holding_days' in df.columns:
            df['holding_bucket'] = pd.cut(df['holding_days'],
                                          bins=[0, 1, 3, 5, 10, 100],
                                          labels=['1d', '2-3d', '4-5d', '6-10d', '>10d'])
            holding_performance = df.groupby('holding_bucket').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            analysis['by_holding_period'] = holding_performance.to_dict()

        return analysis

    def calculate_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns"""
        equity_df = self.get_equity_curve()

        if equity_df.empty:
            return pd.Series()

        # Resample to monthly
        monthly = equity_df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()

        return monthly_returns

    def calculate_risk_metrics(self) -> Dict:
        """Calculate detailed risk metrics"""
        if not self.trades:
            return {}

        returns = [t.get('return_pct', 0) for t in self.trades]

        risk_metrics = {
            'value_at_risk_95': np.percentile(returns, 5) if returns else 0,
            'conditional_var_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if returns else 0,
            'skewness': pd.Series(returns).skew() if len(returns) > 2 else 0,
            'kurtosis': pd.Series(returns).kurtosis() if len(returns) > 3 else 0,
            'max_consecutive_losses': self._max_consecutive_losses(),
            'max_consecutive_wins': self._max_consecutive_wins(),
            'recovery_factor': self.metrics['total_return'] / self.metrics['max_drawdown'] if self.metrics[
                                                                                                  'max_drawdown'] > 0 else 0,
            'expectancy': self._calculate_expectancy()
        }

        return risk_metrics

    def _max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        if not self.trades:
            return 0

        max_losses = 0
        current_losses = 0

        for trade in self.trades:
            if trade.get('pnl', 0) < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def _max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins"""
        if not self.trades:
            return 0

        max_wins = 0
        current_wins = 0

        for trade in self.trades:
            if trade.get('pnl', 0) > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0

        return max_wins

    def _calculate_expectancy(self) -> float:
        """Calculate trading expectancy"""
        if not self.trades:
            return 0

        win_rate = self.metrics['win_rate']
        avg_win = self.metrics['avg_win']
        avg_loss = abs(self.metrics['avg_loss'])

        if avg_loss == 0:
            return avg_win * win_rate

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return expectancy

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = ["=== Trading Performance Report ==="]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Account Summary
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"Current Capital: ${self.current_capital:,.2f}")
        report.append(f"Total Return: {self.metrics['total_return']:.2%}")
        report.append(f"Peak Equity: ${self.peak_equity:,.2f}\n")

        # Trade Statistics
        report.append("Trade Statistics:")
        report.append(f"  Total Trades: {self.metrics['total_trades']}")
        report.append(f"  Winning Trades: {self.metrics['winning_trades']}")
        report.append(f"  Losing Trades: {self.metrics['losing_trades']}")
        report.append(f"  Win Rate: {self.metrics['win_rate']:.1%}")
        report.append(f"  Avg Holding Period: {self.metrics['avg_holding_days']:.1f} days\n")

        # Profit/Loss
        report.append("Profit/Loss Analysis:")
        report.append(f"  Average Win: ${self.metrics['avg_win']:.2f}")
        report.append(f"  Average Loss: ${self.metrics['avg_loss']:.2f}")
        report.append(f"  Largest Win: ${self.metrics['largest_win']:.2f}")
        report.append(f"  Largest Loss: ${self.metrics['largest_loss']:.2f}")
        report.append(f"  Profit Factor: {self.metrics['profit_factor']:.2f}\n")

        # Risk Metrics
        report.append("Risk Metrics:")
        report.append(f"  Maximum Drawdown: {self.metrics['max_drawdown']:.1%}")
        report.append(f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"  Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        report.append(f"  Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")

        # Additional Risk Metrics
        risk_metrics = self.calculate_risk_metrics()
        report.append(f"  Value at Risk (95%): {risk_metrics['value_at_risk_95']:.2%}")
        report.append(f"  Max Consecutive Losses: {risk_metrics['max_consecutive_losses']}")
        report.append(f"  Expectancy: ${risk_metrics['expectancy']:.2f}")

        return "\n".join(report)

    def save_performance_data(self, filepath: str):
        """Save performance data to file"""
        data = {
            'metrics': self.metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_performance_data(self, filepath: str):
        """Load performance data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics = data['metrics']
        self.trades = data['trades']
        self.equity_curve = data['equity_curve']
        self.initial_capital = data['initial_capital']
        self.current_capital = data['current_capital']

        # Convert date strings back to datetime
        for point in self.equity_curve:
            if isinstance(point['date'], str):
                point['date'] = datetime.fromisoformat(point['date'])

        for trade in self.trades:
            for key in ['entry_time', 'exit_time']:
                if key in trade and isinstance(trade[key], str):
                    trade[key] = datetime.fromisoformat(trade[key])