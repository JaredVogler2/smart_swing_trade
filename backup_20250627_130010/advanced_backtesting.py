# advanced_backtesting.py
"""
Advanced Backtesting System with Walk-Forward Analysis and GPU Acceleration
Prevents data leakage and provides realistic performance metrics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import torch
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import json
from dataclasses import dataclass
from collections import defaultdict

# Import custom modules
from models.enhanced_features import EnhancedFeatureEngineer
from models.ensemble_gpu_windows import GPUEnsembleModel
from risk.risk_manager import RiskManager
from config.watchlist import WATCHLIST

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    position_size_pct: float = 0.05  # 5% per position
    max_positions: int = 10
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.001  # 0.1% slippage
    min_holding_period: int = 1  # Minimum 1 day holding (no day trading)

        
    # Sophisticated quant-style settings
    purge_days: int = 3          # Remove days around train/test boundary
    embargo_days: int = 2        # Wait days before trading after training
    
    # Multi-timeframe settings (for future enhancement)
    use_multi_timeframe: bool = False
    timeframes: dict = None
    # Walk-forward analysis parameters
    train_period_days: int = 504  # 2 years of trading days  # 1 year training
    test_period_days: int = 63  # 3 months testing
    retrain_frequency: int = 21  # Retrain every month

    # Feature engineering parameters
    min_data_points: int = 252  # Minimum data points for training
    sequence_length: int = 20  # LSTM sequence length

    # Risk management
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    max_correlation: float = 0.7  # Max correlation between positions

    # Performance thresholds
    min_sharpe_ratio: float = 1.0
    min_win_rate: float = 0.5


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    side: str = 'long'  # Only long positions allowed
    stop_loss: float = 0
    take_profit: float = 0
    commission_paid: float = 0
    slippage_cost: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    holding_period: int = 0
    exit_reason: str = ''
    ml_confidence: float = 0
    ml_probability: float = 0
    features_snapshot: Dict = None


class GPUBacktester:
    """Advanced backtesting system with GPU acceleration"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.feature_engineer = EnhancedFeatureEngineer(use_gpu=torch.cuda.is_available())
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.daily_returns = []
        self.model_performance = defaultdict(list)

        # Setup GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"GPU Backtesting enabled: {torch.cuda.get_device_name()}")

    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch historical data with proper handling"""
        print(f"DEBUG: fetch_historical_data called with start_date={start_date}, end_date={end_date}")
        data = {}

        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False,
                                 auto_adjust=True)
                print(f"DEBUG: Downloaded {len(df)} rows for {symbol}")

                # If MultiIndex, flatten it
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if len(df) >= self.config.min_data_points:
                    # Add additional price data
                    df['returns'] = df['Close'].pct_change()
                    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                    df['dollar_volume'] = df['Close'] * df['Volume']

                    # Rename columns to lowercase
                    df.columns = df.columns.str.lower()

                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} days of data for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} days")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        return data

    def walk_forward_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Perform walk-forward backtesting to prevent data leakage
        """
        logger.info("Starting walk-forward backtest...")

        # Fetch all historical data
        all_data = self.fetch_historical_data(
            symbols,
            start_date,
            end_date
        )
        print(f"\nDEBUG: Fetched data for {len(all_data)} symbols")
        for symbol in list(all_data.keys())[:3]:
            df = all_data[symbol]
            print(f"  {symbol}: {len(df)} rows, dates {df.index[0]} to {df.index[-1]}")

        if not all_data:
            logger.error("No data available for backtesting")
            return {}

        # Initialize portfolio
        portfolio = {
            'cash': self.config.initial_capital,
            'positions': {},
            'equity': self.config.initial_capital,
            'trades': []
        }

        # Get date range
        # Start walk-forward after we have enough training data
        min_start_date = pd.to_datetime(start_date) + timedelta(days=self.config.train_period_days * 2)  # Need 2x training period for sufficient data
        all_dates = pd.date_range(start=min_start_date, end=end_date, freq='B')  # Business days only

        # Walk-forward loop
        current_model = None
        last_train_date = None

        print(f"\nDEBUG: Walk-forward date range: {all_dates[0]} to {all_dates[-1]}")
        print(f"Total dates to process: {len(all_dates)}")
        for current_date in all_dates:
            # Check if we need to retrain
            if (last_train_date is None or
                    (current_date - last_train_date).days >= self.config.retrain_frequency):

                # Train new model
                train_end = current_date - timedelta(days=1)
                train_start = train_end - timedelta(days=self.config.train_period_days)
                print(f"DEBUG: current_date={current_date}, train_start={train_start}, train_end={train_end}")

                logger.info(f"Training model: {train_start.date()} to {train_end.date()}")

                print(f"DEBUG: Calling _train_model with train_start={train_start}, train_end={train_end}")
                current_model = self._train_model(all_data, train_start, train_end)
                last_train_date = current_date

                # Validate model performance
                val_start = train_end + timedelta(days=1)
                val_end = min(val_start + timedelta(days=21), current_date)

                if val_end > val_start:
                    val_metrics = self._validate_model(current_model, all_data, val_start, val_end)
                    self.model_performance[current_date].append(val_metrics)

            # Generate predictions for current date
            predictions = self._generate_predictions(current_model, all_data, current_date)

            # Update existing positions
            self._update_positions(portfolio, all_data, current_date)

            # Check for new trading opportunities
            self._check_trading_signals(portfolio, predictions, all_data, current_date)

            # Record daily metrics
            self._record_daily_metrics(portfolio, current_date)

        # Close all remaining positions
        self._close_all_positions(portfolio, all_data, all_dates[-1])

        # Calculate performance metrics
        results = self._calculate_performance_metrics()

        return results

    def _train_model(self, all_data: Dict[str, pd.DataFrame],
                     train_start: pd.Timestamp, train_end: pd.Timestamp) -> GPUEnsembleModel:
        """Train model on specific date range"""

        # Prepare training data
        train_data = {}

        for symbol, df in all_data.items():
            # Filter to training period
            print(f"Before filter: {symbol} has {len(df)} rows")
            print(f"df date range: {df.index[0]} to {df.index[-1]}")
            print(f"Filter range: {train_start} to {train_end}")
            mask = (df.index >= train_start) & (df.index <= train_end)
            train_df = df[mask].copy()
            print(f"After filter: {len(train_df)} rows pass the date filter")

            if len(train_df) >= self.config.sequence_length + 50:
                train_data[symbol] = train_df

        # Create and train model
        model = GPUEnsembleModel()
        model.train(train_data)

        return model

    def _validate_model(self, model: GPUEnsembleModel, all_data: Dict[str, pd.DataFrame],
                        val_start: pd.Timestamp, val_end: pd.Timestamp) -> Dict:
        """Validate model performance on out-of-sample data"""

        predictions = []
        actuals = []

        for symbol, df in all_data.items():
            # Filter to validation period
            mask = (df.index >= val_start) & (df.index <= val_end)
            val_df = df[mask].copy()

            if len(val_df) > 0:
                # Get predictions
                pred = model.predict(symbol, df[df.index <= val_end])

                if 'error' not in pred:
                    predictions.append(pred['probability'])

                    # Calculate actual return
                    future_return = val_df['close'].iloc[-1] / val_df['close'].iloc[0] - 1
                    actuals.append(1 if future_return > 0.02 else 0)

        # Calculate validation metrics
        if predictions and actuals:
            from sklearn.metrics import accuracy_score, precision_score, recall_score

            pred_binary = [1 if p > 0.5 else 0 for p in predictions]

            metrics = {
                'accuracy': accuracy_score(actuals, pred_binary),
                'precision': precision_score(actuals, pred_binary, zero_division=0),
                'recall': recall_score(actuals, pred_binary, zero_division=0),
                'n_samples': len(predictions)
            }
        else:
            metrics = {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'n_samples': 0
            }

        return metrics

    def _generate_predictions(self, model: GPUEnsembleModel, all_data: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> Dict:
        """Generate predictions for current date"""

        predictions = {}

        if model is None or not model.is_trained:
            return predictions

        for symbol, df in all_data.items():
            # Get data up to current date (no future data leakage!)
            historical_data = df[df.index <= current_date].copy()

            if len(historical_data) >= self.config.sequence_length + 50:
                try:
                    pred = model.predict(symbol, historical_data)

                    if 'error' not in pred:
                        predictions[symbol] = pred

                except Exception as e:
                    logger.error(f"Prediction error for {symbol}: {e}")

        return predictions

    def _update_positions(self, portfolio: Dict, all_data: Dict[str, pd.DataFrame],
                          current_date: pd.Timestamp):
        """Update existing positions with current prices"""

        positions_to_close = []

        for symbol, position in portfolio['positions'].items():
            if symbol in all_data:
                current_data = all_data[symbol]

                # Get current price
                if current_date in current_data.index:
                    current_price = current_data.loc[current_date, 'close']

                    # Check stop loss
                    if current_price <= position['stop_loss']:
                        positions_to_close.append((symbol, current_price, 'stop_loss'))

                    # Check take profit
                    elif current_price >= position['take_profit']:
                        positions_to_close.append((symbol, current_price, 'take_profit'))

                    # Check minimum holding period
                    elif (current_date - position['entry_date']).days >= self.config.min_holding_period:
                        # Optional: Add other exit conditions
                        pass

        # Close positions
        for symbol, exit_price, exit_reason in positions_to_close:
            self._close_position(portfolio, symbol, exit_price, current_date, exit_reason)

    def _check_trading_signals(self, portfolio: Dict, predictions: Dict,
                               all_data: Dict[str, pd.DataFrame], current_date: pd.Timestamp):
        """Check for new trading opportunities"""

        # Check portfolio constraints
        current_positions = len(portfolio['positions'])
        if current_positions >= self.config.max_positions:
            return

        # Sort predictions by confidence
        sorted_predictions = sorted(
            [(s, p) for s, p in predictions.items() if p['prediction'] == 1],
            key=lambda x: x[1]['confidence'],
            reverse=True
        )

        for symbol, pred in sorted_predictions:
            # Check if already in position
            if symbol in portfolio['positions']:
                continue

            # Check correlation with existing positions
            if not self._check_correlation(symbol, portfolio['positions'], all_data, current_date):
                continue

            # Get current price
            if symbol in all_data and current_date in all_data[symbol].index:
                current_price = all_data[symbol].loc[current_date, 'close']

                # Calculate position size
                position_value = portfolio['equity'] * self.config.position_size_pct
                quantity = int(position_value / current_price)

                if quantity > 0:
                    # Apply slippage
                    entry_price = current_price * (1 + self.config.slippage)

                    # Calculate costs
                    commission = position_value * self.config.commission
                    total_cost = (entry_price * quantity) + commission

                    # Check if we have enough cash
                    if total_cost <= portfolio['cash']:
                        # Open position
                        trade = Trade(
                            symbol=symbol,
                            entry_date=current_date,
                            entry_price=entry_price,
                            quantity=quantity,
                            stop_loss=entry_price * (1 - self.config.stop_loss_pct),
                            take_profit=entry_price * (1 + self.config.take_profit_pct),
                            commission_paid=commission,
                            slippage_cost=quantity * self.config.slippage * current_price,
                            ml_confidence=pred['confidence'],
                            ml_probability=pred['probability'],
                            features_snapshot=pred.get('model_probabilities', {})
                        )

                        # Update portfolio
                        portfolio['positions'][symbol] = {
                            'trade': trade,
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'stop_loss': trade.stop_loss,
                            'take_profit': trade.take_profit
                        }

                        portfolio['cash'] -= total_cost

                        logger.info(f"Opened position: {symbol} @ ${entry_price:.2f} x{quantity}")

                        # Stop if max positions reached
                        if len(portfolio['positions']) >= self.config.max_positions:
                            break

    def _check_correlation(self, symbol: str, existing_positions: Dict,
                           all_data: Dict[str, pd.DataFrame], current_date: pd.Timestamp) -> bool:
        """Check correlation with existing positions"""

        if not existing_positions:
            return True

        # Get returns for symbol
        if symbol not in all_data:
            return False

        symbol_data = all_data[symbol][all_data[symbol].index <= current_date]
        if len(symbol_data) < 50:
            return False

        symbol_returns = symbol_data['returns'].iloc[-50:]

        # Check correlation with each existing position
        for pos_symbol in existing_positions:
            if pos_symbol in all_data:
                pos_data = all_data[pos_symbol][all_data[pos_symbol].index <= current_date]
                if len(pos_data) >= 50:
                    pos_returns = pos_data['returns'].iloc[-50:]

                    # Calculate correlation
                    correlation = symbol_returns.corr(pos_returns)

                    if abs(correlation) > self.config.max_correlation:
                        return False

        return True

    def _close_position(self, portfolio: Dict, symbol: str, exit_price: float,
                        exit_date: pd.Timestamp, exit_reason: str):
        """Close a position and record the trade"""

        if symbol not in portfolio['positions']:
            return

        position = portfolio['positions'][symbol]
        trade = position['trade']

        # Apply slippage on exit
        exit_price = exit_price * (1 - self.config.slippage)

        # Calculate commission
        exit_commission = exit_price * trade.quantity * self.config.commission

        # Update trade
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.holding_period = (exit_date - trade.entry_date).days

        # Calculate P&L
        gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        total_commission = trade.commission_paid + exit_commission
        total_slippage = trade.slippage_cost + (trade.quantity * self.config.slippage * exit_price)

        trade.pnl = gross_pnl - total_commission - total_slippage
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)

        # Update portfolio
        portfolio['cash'] += (exit_price * trade.quantity) - exit_commission
        del portfolio['positions'][symbol]

        # Record trade
        self.trades.append(trade)

        logger.info(f"Closed position: {symbol} @ ${exit_price:.2f}, "
                    f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct * 100:.2f}%), "
                    f"Reason: {exit_reason}")

    def _close_all_positions(self, portfolio: Dict, all_data: Dict[str, pd.DataFrame],
                             final_date: pd.Timestamp):
        """Close all remaining positions at end of backtest"""

        positions_to_close = list(portfolio['positions'].keys())

        for symbol in positions_to_close:
            if symbol in all_data and final_date in all_data[symbol].index:
                exit_price = all_data[symbol].loc[final_date, 'close']
                self._close_position(portfolio, symbol, exit_price, final_date, 'end_of_backtest')

    def _record_daily_metrics(self, portfolio: Dict, current_date: pd.Timestamp):
        """Record daily portfolio metrics"""

        # Calculate portfolio value
        positions_value = 0

        for symbol, position in portfolio['positions'].items():
            # This would need current price - simplified here
            positions_value += position['entry_price'] * position['quantity']

        total_equity = portfolio['cash'] + positions_value
        portfolio['equity'] = total_equity

        # Record equity curve
        self.equity_curve.append({
            'date': current_date,
            'equity': total_equity,
            'cash': portfolio['cash'],
            'positions_value': positions_value,
            'n_positions': len(portfolio['positions'])
        })

        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (total_equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        if not self.trades:
            return {
                'error': 'No trades executed',
                'n_trades': 0
            }

        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'holding_period': t.holding_period,
                'exit_reason': t.exit_reason,
                'ml_confidence': t.ml_confidence
            }
            for t in self.trades if t.exit_date is not None
        ])

        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else np.inf

        # Return metrics
        total_return = (equity_df['equity'].iloc[-1] - self.config.initial_capital) / self.config.initial_capital

        if self.daily_returns:
            daily_returns = np.array(self.daily_returns)
            annual_return = np.mean(daily_returns) * 252
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

            # Maximum drawdown
            equity_curve = equity_df['equity'].values
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            annual_return = 0
            annual_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0

        # Trade analysis
        avg_holding_period = trades_df['holding_period'].mean()

        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

        # ML performance
        avg_confidence = trades_df['ml_confidence'].mean()
        high_confidence_trades = trades_df[trades_df['ml_confidence'] > 0.8]
        high_confidence_win_rate = len(high_confidence_trades[high_confidence_trades['pnl'] > 0]) / len(
            high_confidence_trades) if len(high_confidence_trades) > 0 else 0

        # Monthly returns
        if not equity_df.empty:
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()

            monthly_stats = {
                'avg_monthly_return': monthly_returns.mean(),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': len(monthly_returns[monthly_returns > 0]),
                'negative_months': len(monthly_returns[monthly_returns < 0])
            }
        else:
            monthly_stats = {}

        # Compile all metrics
        metrics = {
            # Trade Statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,

            # P&L Metrics
            'total_pnl': total_pnl,
            'average_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),

            # Return Metrics
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,

            # Trade Analysis
            'avg_holding_period': avg_holding_period,
            'exit_reasons': exit_reasons,

            # ML Performance
            'avg_ml_confidence': avg_confidence,
            'high_confidence_win_rate': high_confidence_win_rate,

            # Monthly Stats
            'monthly_stats': monthly_stats,

            # Final Portfolio
            'final_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else self.config.initial_capital,
            'initial_capital': self.config.initial_capital
        }

        return metrics

    def generate_report(self, results: Dict, output_path: str = 'backtest_report.html'):
        """Generate comprehensive HTML report"""

        html_content = f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-label {{ font-weight: bold; color: #666; }}
                .metric-value {{ font-size: 24px; color: #333; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Summary Statistics</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {('positive' if results['total_return'] > 0 else 'negative')}">
                        {results['total_return'] * 100:.2f}%
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{results['sharpe_ratio']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{results['max_drawdown'] * 100:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{results['win_rate'] * 100:.1f}%</div>
                </div>
            </div>

            <h2>Trade Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{results['total_trades']}</td>
                </tr>
                <tr>
                    <td>Winning Trades</td>
                    <td>{results['winning_trades']}</td>
                </tr>
                <tr>
                    <td>Losing Trades</td>
                    <td>{results['losing_trades']}</td>
                </tr>
                <tr>
                    <td>Average Win</td>
                    <td class="positive">${results['average_win']:.2f}</td>
                </tr>
                <tr>
                    <td>Average Loss</td>
                    <td class="negative">${results['average_loss']:.2f}</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{results['profit_factor']:.2f}</td>
                </tr>
                <tr>
                    <td>Average Holding Period</td>
                    <td>{results['avg_holding_period']:.1f} days</td>
                </tr>
            </table>

            <h2>Risk Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Annual Return</td>
                    <td>{results['annual_return'] * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Annual Volatility</td>
                    <td>{results['annual_volatility'] * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{results['calmar_ratio']:.2f}</td>
                </tr>
                <tr>
                    <td>ML Avg Confidence</td>
                    <td>{results['avg_ml_confidence'] * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>High Confidence Win Rate</td>
                    <td>{results['high_confidence_win_rate'] * 100:.1f}%</td>
                </tr>
            </table>

            <h2>Monthly Performance</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Monthly Return</td>
                    <td>{results['monthly_stats'].get('avg_monthly_return', 0) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Best Month</td>
                    <td class="positive">{results['monthly_stats'].get('best_month', 0) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Worst Month</td>
                    <td class="negative">{results['monthly_stats'].get('worst_month', 0) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Positive Months</td>
                    <td>{results['monthly_stats'].get('positive_months', 0)}</td>
                </tr>
                <tr>
                    <td>Negative Months</td>
                    <td>{results['monthly_stats'].get('negative_months', 0)}</td>
                </tr>
            </table>

            <h2>Exit Reason Analysis</h2>
            <table>
                <tr>
                    <th>Exit Reason</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {"".join([f'<tr><td>{reason}</td><td>{count}</td><td>{count / results["total_trades"] * 100:.1f}%</td></tr>'
                          for reason, count in results['exit_reasons'].items()])}
            </table>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Report generated: {output_path}")

    def plot_results(self, save_path: str = 'backtest_results.png'):
        """Generate visualization plots"""

        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results', fontsize=16)

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)

        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(equity_df.index, equity_df['equity'], 'b-', linewidth=2)
        ax1.fill_between(equity_df.index, self.config.initial_capital, equity_df['equity'],
                         where=equity_df['equity'] >= self.config.initial_capital,
                         facecolor='green', alpha=0.3)
        ax1.fill_between(equity_df.index, self.config.initial_capital, equity_df['equity'],
                         where=equity_df['equity'] < self.config.initial_capital,
                         facecolor='red', alpha=0.3)
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = axes[0, 1]
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.5)
        ax2.set_title('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Monthly Returns
        ax3 = axes[1, 0]
        monthly_returns = equity_df['equity'].resample('M').last().pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
        ax3.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.7)
        ax3.set_title('Monthly Returns %')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)

        # 4. Trade P&L Distribution
        ax4 = axes[1, 1]
        if self.trades:
            pnls = [t.pnl for t in self.trades if t.exit_date is not None]
            ax4.hist(pnls, bins=30, color='blue', alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax4.set_title('Trade P&L Distribution')
            ax4.set_xlabel('P&L ($)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Results plotted: {save_path}")


def run_comprehensive_backtest():
    """Run a comprehensive backtest with the provided configuration"""

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.05,
        max_positions=10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        train_period_days=252,
        test_period_days=63,
        retrain_frequency=21
    )

    # Initialize backtester
    backtester = GPUBacktester(config)

    # Define test period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')  # 2 years

    # Run backtest
    results = backtester.walk_forward_backtest(
        symbols=WATCHLIST[:20],  # Top 20 symbols
        start_date=start_date,
        end_date=end_date
    )

    # Generate reports
    if results and 'error' not in results:
        backtester.generate_report(results, 'backtest_report.html')
        backtester.plot_results('backtest_results.png')

        # Print summary
        print("\n" + "=" * 50)
        print("BACKTEST SUMMARY")
        print("=" * 50)
        print(f"Total Return: {results['total_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print("=" * 50)

        # Save detailed results
        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print("Backtest failed:", results.get('error', 'Unknown error'))


if __name__ == "__main__":
    run_comprehensive_backtest()