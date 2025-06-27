# enhanced_advanced_backtesting.py
"""
Sophisticated Hedge Fund Style Backtesting System
- Full watchlist support (188 symbols)
- Realistic execution modeling
- Advanced risk management
- Proper data handling without leakage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

# Import custom modules
from models.ensemble_gpu_windows import GPUEnsembleModel
from models.enhanced_features import EnhancedFeatureEngineer
from config.watchlist import WATCHLIST, SECTOR_MAPPING
from risk.risk_manager import RiskManager

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Enhanced configuration for sophisticated backtesting"""
    # Capital and position sizing
    initial_capital: float = 100000
    position_size_pct: float = 0.02  # 2% per position (more positions, smaller size)
    max_positions: int = 20  # More positions for diversification
    max_sector_exposure: float = 0.30  # Max 30% in one sector

    # Risk parameters
    stop_loss_pct: float = 0.02  # Tighter stop loss
    trailing_stop_pct: float = 0.015  # Trailing stop
    take_profit_pct: float = 0.05  # 5% take profit
    max_drawdown_pct: float = 0.15  # 15% max drawdown

    # Transaction costs (realistic)
    commission_per_share: float = 0.005  # $0.005 per share
    min_commission: float = 1.0  # $1 minimum
    slippage_bps: float = 5  # 5 basis points slippage
    market_impact_factor: float = 0.1  # 10% of daily volume impacts price

    # Timing constraints
    min_holding_period: int = 1  # No day trading
    max_holding_period: int = 20  # Maximum 20 days

    # ML parameters - UPDATED FOR LONGER HISTORY
    train_period_days: int = 756  # 3 years training (increased from 2)
    validation_period_days: int = 252  # 1 year validation (increased from 6 months)
    test_period_days: int = 63  # 3 months testing
    retrain_frequency: int = 21  # Retrain monthly
    min_data_points: int = 504  # 2 years minimum (increased from 1)

    # Walk-forward parameters
    purge_days: int = 5  # Remove 5 days around train/test boundary
    embargo_days: int = 2  # Wait 2 days before trading after training

    # Performance thresholds
    min_sharpe_ratio: float = 1.5
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.5

    # Execution parameters
    use_limit_orders: bool = True
    limit_order_offset_bps: float = 10  # 10 bps better than market
    order_timeout_minutes: int = 30

    # Advanced features
    use_market_regime: bool = True
    use_correlation_filter: bool = True
    max_correlation: float = 0.7
    use_volume_filter: bool = True
    min_dollar_volume: float = 1000000  # $1M daily volume

    # Multi-timeframe analysis
    use_multi_timeframe: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1d', '1h', '15m'])

    # Market condition analysis
    analyze_market_cycles: bool = True
    min_cycles_required: int = 2  # Need at least 2 full market cycles


@dataclass
class Trade:
    """Enhanced trade record with detailed metrics"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    quantity: int = 0

    # Order details
    order_type: str = 'market'
    limit_price: Optional[float] = None

    # Risk management
    initial_stop: float = 0
    trailing_stop: float = 0
    take_profit: float = 0

    # Costs
    commission_paid: float = 0
    slippage_cost: float = 0
    market_impact_cost: float = 0

    # Performance
    pnl: float = 0
    pnl_pct: float = 0
    max_profit: float = 0
    max_loss: float = 0
    holding_period: int = 0

    # Exit details
    exit_reason: str = ''

    # ML details
    ml_confidence: float = 0
    ml_probability: float = 0
    predicted_return: float = 0

    # Market context
    entry_market_regime: str = ''
    entry_volatility: float = 0
    entry_volume_ratio: float = 0

    # Risk metrics
    position_size_pct: float = 0
    portfolio_heat: float = 0
    sector: str = ''

    # Feature snapshot
    features_snapshot: Dict = field(default_factory=dict)


class DataManager:
    """Efficient data management for backtesting"""

    def __init__(self, cache_dir: str = 'cache/backtest_data'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data_cache = {}

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   force_refresh: bool = False) -> pd.DataFrame:
        """Fetch data with caching and error handling"""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")

        # Check memory cache first
        if cache_key in self.data_cache and not force_refresh:
            return self.data_cache[cache_key]

        # Check disk cache
        if os.path.exists(cache_path) and not force_refresh:
            df = pd.read_parquet(cache_path)
            self.data_cache[cache_key] = df
            return df

        # Fetch from yfinance
        try:
            df = yf.download(symbol, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty:
                # Add essential calculations
                df['returns'] = df['Close'].pct_change()
                df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['dollar_volume'] = df['Close'] * df['Volume']
                df['volatility'] = df['returns'].rolling(20).std()
                df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

                # Rename columns
                df.columns = df.columns.str.lower()

                # Save to cache
                df.to_parquet(cache_path)
                self.data_cache[cache_key] = df

                return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all_data(self, symbols: List[str], start_date: str, end_date: str,
                       n_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols in parallel"""
        all_data = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_symbol),
                               total=len(symbols), desc="Fetching data"):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty and len(data) >= 252:  # At least 1 year
                        all_data[symbol] = data
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        logger.info(f"Successfully fetched data for {len(all_data)}/{len(symbols)} symbols")
        return all_data


class ExecutionSimulator:
    """Realistic execution simulation with market microstructure"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def simulate_entry(self, symbol: str, signal_price: float, quantity: int,
                       daily_volume: float, spread: float = 0.01) -> Tuple[float, float, float]:
        """Simulate realistic entry execution"""
        # Base slippage
        slippage_pct = self.config.slippage_bps / 10000

        # Market impact based on order size vs daily volume
        order_value = signal_price * quantity
        daily_dollar_volume = daily_volume * signal_price
        volume_fraction = order_value / daily_dollar_volume if daily_dollar_volume > 0 else 0.01

        # Non-linear market impact
        market_impact_pct = self.config.market_impact_factor * np.sqrt(volume_fraction)

        # Total price impact
        total_impact = slippage_pct + market_impact_pct + (spread / signal_price / 2)

        # Execution price
        execution_price = signal_price * (1 + total_impact)

        # Calculate costs
        slippage_cost = quantity * signal_price * slippage_pct
        market_impact_cost = quantity * signal_price * market_impact_pct

        # Commission
        commission = max(quantity * self.config.commission_per_share,
                         self.config.min_commission)

        return execution_price, slippage_cost + market_impact_cost, commission

    def simulate_exit(self, symbol: str, signal_price: float, quantity: int,
                      daily_volume: float, spread: float = 0.01) -> Tuple[float, float, float]:
        """Simulate realistic exit execution"""
        # Similar to entry but in reverse direction
        slippage_pct = self.config.slippage_bps / 10000

        order_value = signal_price * quantity
        daily_dollar_volume = daily_volume * signal_price
        volume_fraction = order_value / daily_dollar_volume if daily_dollar_volume > 0 else 0.01

        market_impact_pct = self.config.market_impact_factor * np.sqrt(volume_fraction)
        total_impact = slippage_pct + market_impact_pct + (spread / signal_price / 2)

        execution_price = signal_price * (1 - total_impact)

        slippage_cost = quantity * signal_price * slippage_pct
        market_impact_cost = quantity * signal_price * market_impact_pct
        commission = max(quantity * self.config.commission_per_share,
                         self.config.min_commission)

        return execution_price, slippage_cost + market_impact_cost, commission


class PortfolioManager:
    """Sophisticated portfolio management with risk controls"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Trade] = {}
        self.cash = config.initial_capital
        self.equity_curve = []
        self.trades = []
        self.sector_exposure = defaultdict(float)

    def can_open_position(self, symbol: str, position_value: float) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return False, "max_positions"

        # Check if already have position
        if symbol in self.positions:
            return False, "already_in_position"

        # Check cash available
        if position_value > self.cash * 0.95:  # Keep 5% cash buffer
            return False, "insufficient_cash"

        # Check sector exposure
        sector = SECTOR_MAPPING.get(symbol, 'Unknown')
        current_sector_exposure = self.sector_exposure[sector]
        total_equity = self.get_total_equity()

        if (current_sector_exposure + position_value) / total_equity > self.config.max_sector_exposure:
            return False, "sector_concentration"

        return True, "ok"

    def open_position(self, trade: Trade, execution_price: float,
                      costs: float, commission: float):
        """Open a new position"""
        total_cost = (execution_price * trade.quantity) + costs + commission

        self.cash -= total_cost
        self.positions[trade.symbol] = trade

        # Update sector exposure
        sector = SECTOR_MAPPING.get(trade.symbol, 'Unknown')
        self.sector_exposure[sector] += execution_price * trade.quantity

        # Update trade record
        trade.entry_price = execution_price
        trade.commission_paid = commission
        trade.slippage_cost = costs

    def close_position(self, symbol: str, execution_price: float,
                       exit_date: pd.Timestamp, exit_reason: str,
                       costs: float, commission: float):
        """Close an existing position"""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        proceeds = (execution_price * trade.quantity) - costs - commission

        self.cash += proceeds

        # Update trade record
        trade.exit_price = execution_price
        trade.exit_date = exit_date
        trade.exit_reason = exit_reason
        trade.holding_period = (exit_date - trade.entry_date).days
        trade.commission_paid += commission
        trade.slippage_cost += costs

        # Calculate PnL
        gross_pnl = (execution_price - trade.entry_price) * trade.quantity
        net_pnl = gross_pnl - trade.commission_paid - trade.slippage_cost
        trade.pnl = net_pnl
        trade.pnl_pct = net_pnl / (trade.entry_price * trade.quantity)

        # Update sector exposure
        sector = SECTOR_MAPPING.get(symbol, 'Unknown')
        self.sector_exposure[sector] -= execution_price * trade.quantity

        # Record completed trade
        self.trades.append(trade)
        del self.positions[symbol]

    def get_total_equity(self) -> float:
        """Calculate total portfolio equity"""
        positions_value = sum(
            pos.quantity * pos.entry_price for pos in self.positions.values()
        )
        return self.cash + positions_value

    def update_trailing_stops(self, current_prices: Dict[str, float]):
        """Update trailing stops for all positions"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]

                # Update max profit
                profit = (current_price - position.entry_price) * position.quantity
                position.max_profit = max(position.max_profit, profit)

                # Update trailing stop
                new_stop = current_price * (1 - self.config.trailing_stop_pct)
                if new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop


class HedgeFundBacktester:
    """Sophisticated hedge fund style backtesting engine"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.data_manager = DataManager()
        self.execution_sim = ExecutionSimulator(self.config)
        self.feature_engineer = EnhancedFeatureEngineer(use_gpu=torch.cuda.is_available())

        # Setup GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.models = {}
        self.model_performance = defaultdict(list)
        self.regime_states = []

    def run_backtest(self, start_date: str, end_date: str,
                     symbols: List[str] = None) -> Dict:
        """Run sophisticated walk-forward backtest"""
        symbols = symbols or WATCHLIST
        logger.info(f"Starting backtest with {len(symbols)} symbols")

        # Extend date range for training data
        extended_start = (pd.to_datetime(start_date) - timedelta(days=800)).strftime('%Y-%m-%d')

        # Fetch all data
        logger.info("Fetching historical data...")
        all_data = self.data_manager.fetch_all_data(symbols, extended_start, end_date)

        if len(all_data) < 10:
            logger.error("Insufficient data for backtesting")
            return {}

        # Initialize portfolio
        portfolio = PortfolioManager(self.config)

        # Setup walk-forward dates
        test_start = pd.to_datetime(start_date)
        test_end = pd.to_datetime(end_date)

        # Initialize first training window
        current_date = test_start
        last_train_date = None
        current_model = None

        # Performance tracking
        daily_returns = []
        equity_curve = []

        # Main backtest loop
        trading_days = pd.bdate_range(start=test_start, end=test_end)

        for date in tqdm(trading_days, desc="Backtesting"):
            # Check if we need to retrain
            days_since_training = (date - last_train_date).days if last_train_date else float('inf')

            if current_model is None or days_since_training >= self.config.retrain_frequency:
                logger.info(f"Training model for {date.date()}")
                current_model = self._train_model_for_date(all_data, date)
                last_train_date = date

                # Apply embargo period
                continue

            # Skip if within embargo period
            if (date - last_train_date).days < self.config.embargo_days:
                continue

            # Get current market data
            market_snapshot = self._get_market_snapshot(all_data, date)
            if not market_snapshot:
                continue

            # Update portfolio with current prices
            current_prices = {s: data['close'] for s, data in market_snapshot.items()}
            portfolio.update_trailing_stops(current_prices)

            # Check exits
            self._check_exits(portfolio, market_snapshot, date)

            # Generate new signals
            if len(portfolio.positions) < self.config.max_positions:
                signals = self._generate_signals(
                    current_model, market_snapshot,
                    portfolio.positions.keys(), date
                )

                # Execute signals
                self._execute_signals(portfolio, signals, market_snapshot, date)

            # Record daily metrics
            total_equity = portfolio.get_total_equity()
            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': portfolio.cash,
                'n_positions': len(portfolio.positions),
                'positions': list(portfolio.positions.keys())
            })

            if len(equity_curve) > 1:
                daily_return = (total_equity - equity_curve[-2]['equity']) / equity_curve[-2]['equity']
                daily_returns.append(daily_return)

        # Close all remaining positions
        self._close_all_positions(portfolio, all_data, test_end)

        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            portfolio, equity_curve, daily_returns
        )

        # Generate visualizations
        self._generate_reports(results, portfolio, equity_curve)

        return results

    def _train_model_for_date(self, all_data: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> GPUEnsembleModel:
        """Train model for specific date with proper data handling"""
        # Calculate training window
        train_end = current_date - timedelta(days=self.config.purge_days)
        train_start = train_end - timedelta(days=self.config.train_period_days)
        val_end = train_end
        val_start = val_end - timedelta(days=self.config.validation_period_days)

        # Prepare training data
        train_data = {}
        val_data = {}

        for symbol, df in all_data.items():
            # Training data
            train_mask = (df.index >= train_start) & (df.index < val_start)
            train_df = df[train_mask].copy()

            # Validation data
            val_mask = (df.index >= val_start) & (df.index < val_end)
            val_df = df[val_mask].copy()

            if len(train_df) >= self.config.min_data_points and len(val_df) >= 50:
                train_data[symbol] = train_df
                val_data[symbol] = val_df

        # Create and train model
        model = GPUEnsembleModel(max_gpu_memory_mb=8192)

        # Train with both training and validation data
        logger.info(f"Training on {len(train_data)} symbols")
        model.train(train_data, validation_data=val_data)

        return model

    def _get_market_snapshot(self, all_data: Dict[str, pd.DataFrame],
                             date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Get market data snapshot for specific date"""
        snapshot = {}

        for symbol, df in all_data.items():
            # Get data up to current date
            historical = df[df.index <= date].copy()

            if len(historical) >= 50 and date in df.index:
                # Include some future data for exit checking (realistic)
                snapshot[symbol] = historical

        return snapshot

    def _generate_signals(self, model: GPUEnsembleModel,
                          market_snapshot: Dict[str, pd.DataFrame],
                          current_positions: Set[str],
                          date: pd.Timestamp) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        # Filter symbols
        available_symbols = [
            s for s in market_snapshot.keys()
            if s not in current_positions
        ]

        # Check volume and liquidity filters
        liquid_symbols = []
        for symbol in available_symbols:
            df = market_snapshot[symbol]
            if not df.empty:
                avg_dollar_volume = df['dollar_volume'].rolling(20).mean().iloc[-1]
                if avg_dollar_volume >= self.config.min_dollar_volume:
                    liquid_symbols.append(symbol)

        # Generate predictions
        for symbol in liquid_symbols:
            try:
                prediction = model.predict(symbol, market_snapshot[symbol])

                if prediction['prediction'] == 1 and prediction['confidence'] >= 0.65:
                    signals.append({
                        'symbol': symbol,
                        'date': date,
                        'ml_confidence': prediction['confidence'],
                        'ml_probability': prediction['probability'],
                        'predicted_return': prediction.get('expected_return', 0.05),
                        'current_price': market_snapshot[symbol]['close'].iloc[-1],
                        'volatility': market_snapshot[symbol]['volatility'].iloc[-1],
                        'volume_ratio': market_snapshot[symbol]['volume_ratio'].iloc[-1],
                        'model_predictions': prediction.get('model_predictions', {})
                    })

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        # Sort by confidence and return top signals
        signals.sort(key=lambda x: x['ml_confidence'], reverse=True)

        # Apply correlation filter
        if self.config.use_correlation_filter:
            signals = self._filter_correlated_signals(signals, market_snapshot)

        return signals[:5]  # Take top 5 signals

    def _filter_correlated_signals(self, signals: List[Dict],
                                   market_snapshot: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Filter out highly correlated signals"""
        filtered_signals = []

        for signal in signals:
            # Check correlation with already selected signals
            is_correlated = False

            for selected in filtered_signals:
                corr = self._calculate_correlation(
                    signal['symbol'], selected['symbol'], market_snapshot
                )

                if abs(corr) > self.config.max_correlation:
                    is_correlated = True
                    break

            if not is_correlated:
                filtered_signals.append(signal)

        return filtered_signals

    def _calculate_correlation(self, symbol1: str, symbol2: str,
                               market_snapshot: Dict[str, pd.DataFrame]) -> float:
        """Calculate correlation between two symbols"""
        if symbol1 not in market_snapshot or symbol2 not in market_snapshot:
            return 0.0

        returns1 = market_snapshot[symbol1]['returns'].iloc[-50:]
        returns2 = market_snapshot[symbol2]['returns'].iloc[-50:]

        if len(returns1) == len(returns2) and len(returns1) >= 30:
            return returns1.corr(returns2)

        return 0.0

    def _execute_signals(self, portfolio: PortfolioManager,
                         signals: List[Dict],
                         market_snapshot: Dict[str, pd.DataFrame],
                         date: pd.Timestamp):
        """Execute trading signals with realistic simulation"""
        for signal in signals:
            symbol = signal['symbol']

            # Calculate position size
            total_equity = portfolio.get_total_equity()
            position_value = total_equity * self.config.position_size_pct

            # Risk-based position sizing
            if signal['volatility'] > 0:
                # Reduce size for high volatility
                volatility_adj = min(0.02 / signal['volatility'], 1.0)
                position_value *= volatility_adj

            # Check if we can open position
            can_open, reason = portfolio.can_open_position(symbol, position_value)
            if not can_open:
                logger.debug(f"Cannot open {symbol}: {reason}")
                continue

            # Calculate shares
            current_price = signal['current_price']
            shares = int(position_value / current_price)

            if shares == 0:
                continue

            # Get market data for execution
            df = market_snapshot[symbol]
            daily_volume = df['volume'].iloc[-1]

            # Simulate execution
            exec_price, costs, commission = self.execution_sim.simulate_entry(
                symbol, current_price, shares, daily_volume
            )

            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_date=date,
                entry_price=exec_price,
                quantity=shares,
                initial_stop=exec_price * (1 - self.config.stop_loss_pct),
                trailing_stop=exec_price * (1 - self.config.trailing_stop_pct),
                take_profit=exec_price * (1 + self.config.take_profit_pct),
                ml_confidence=signal['ml_confidence'],
                ml_probability=signal['ml_probability'],
                predicted_return=signal['predicted_return'],
                entry_volatility=signal['volatility'],
                entry_volume_ratio=signal['volume_ratio'],
                position_size_pct=position_value / total_equity,
                sector=SECTOR_MAPPING.get(symbol, 'Unknown')
            )

            # Open position
            portfolio.open_position(trade, exec_price, costs, commission)

            logger.info(f"Opened {symbol} @ ${exec_price:.2f} x{shares}")

    def _check_exits(self, portfolio: PortfolioManager,
                     market_snapshot: Dict[str, pd.DataFrame],
                     date: pd.Timestamp):
        """Check and execute exit conditions"""
        positions_to_close = []

        for symbol, position in portfolio.positions.items():
            if symbol not in market_snapshot:
                continue

            df = market_snapshot[symbol]
            current_price = df['close'].iloc[-1]
            daily_volume = df['volume'].iloc[-1]

            # Check exit conditions
            exit_reason = None

            # Stop loss
            if current_price <= position.initial_stop:
                exit_reason = "stop_loss"

            # Trailing stop
            elif current_price <= position.trailing_stop:
                exit_reason = "trailing_stop"

            # Take profit
            elif current_price >= position.take_profit:
                exit_reason = "take_profit"

            # Time-based exit
            elif position.holding_period >= self.config.max_holding_period:
                exit_reason = "max_holding_period"

            # ML signal reversal (optional)
            # Could add logic to check if ML model now predicts negative

            if exit_reason:
                positions_to_close.append((symbol, current_price, daily_volume, exit_reason))

        # Execute exits
        for symbol, price, volume, reason in positions_to_close:
            exec_price, costs, commission = self.execution_sim.simulate_exit(
                symbol, price, portfolio.positions[symbol].quantity, volume
            )

            portfolio.close_position(symbol, exec_price, date, reason, costs, commission)

            logger.info(f"Closed {symbol} @ ${exec_price:.2f} - {reason}")

    def _close_all_positions(self, portfolio: PortfolioManager,
                             all_data: Dict[str, pd.DataFrame],
                             end_date: pd.Timestamp):
        """Close all remaining positions at end of backtest"""
        for symbol in list(portfolio.positions.keys()):
            if symbol in all_data:
                df = all_data[symbol]
                last_data = df[df.index <= end_date]

                if not last_data.empty:
                    last_price = last_data['close'].iloc[-1]
                    last_volume = last_data['volume'].iloc[-1]

                    exec_price, costs, commission = self.execution_sim.simulate_exit(
                        symbol, last_price,
                        portfolio.positions[symbol].quantity, last_volume
                    )

                    portfolio.close_position(
                        symbol, exec_price, end_date,
                        "end_of_backtest", costs, commission
                    )

    def _calculate_performance_metrics(self, portfolio: PortfolioManager,
                                       equity_curve: List[Dict],
                                       daily_returns: List[float]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not portfolio.trades:
            return {'error': 'No trades executed'}

        # Convert to arrays
        returns = np.array(daily_returns)

        # Basic metrics
        total_return = (equity_curve[-1]['equity'] - self.config.initial_capital) / self.config.initial_capital

        # Risk metrics
        if len(returns) > 0:
            annual_return = np.mean(returns) * 252
            annual_vol = np.std(returns) * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = annual_return / downside_vol if downside_vol > 0 else 0

            # Maximum drawdown
            equity_values = [e['equity'] for e in equity_curve]
            running_max = pd.Series(equity_values).expanding().max()
            drawdown = (pd.Series(equity_values) - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calmar ratio
            calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            annual_return = annual_vol = sharpe = sortino = max_drawdown = calmar = 0

        # Trade analysis
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'holding_period': t.holding_period,
            'exit_reason': t.exit_reason,
            'ml_confidence': t.ml_confidence,
            'sector': t.sector
        } for t in portfolio.trades])

        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(
            losing_trades) > 0 else float('inf')

        # Exit reason analysis
        exit_analysis = trades_df['exit_reason'].value_counts().to_dict()

        # Sector analysis
        sector_performance = trades_df.groupby('sector')['pnl_pct'].agg(['mean', 'count']).to_dict('index')

        # ML performance analysis
        confidence_bins = pd.cut(trades_df['ml_confidence'], bins=[0, 0.7, 0.8, 0.9, 1.0])
        ml_performance = trades_df.groupby(confidence_bins)['pnl_pct'].agg(['mean', 'count'])

        # Monthly returns
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()

        # Compile results
        results = {
            # Overall Performance
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,

            # Trade Statistics
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': trades_df['holding_period'].mean(),

            # Risk Analysis
            'total_pnl': trades_df['pnl'].sum(),
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'avg_trade_pnl': trades_df['pnl'].mean(),

            # Exit Analysis
            'exit_reasons': exit_analysis,

            # Sector Performance
            'sector_performance': sector_performance,

            # ML Performance
            'avg_ml_confidence': trades_df['ml_confidence'].mean(),

            # Monthly Analysis
            'monthly_returns': {
                'mean': monthly_returns.mean(),
                'std': monthly_returns.std(),
                'best': monthly_returns.max(),
                'worst': monthly_returns.min(),
                'positive_months': len(monthly_returns[monthly_returns > 0]),
                'negative_months': len(monthly_returns[monthly_returns < 0])
            },

            # Final State
            'final_equity': equity_curve[-1]['equity'],
            'initial_capital': self.config.initial_capital,

            # Additional Details
            'trades': portfolio.trades,
            'equity_curve': equity_curve
        }

        return results

    def _generate_reports(self, results: Dict, portfolio: PortfolioManager,
                          equity_curve: List[Dict]):
        """Generate comprehensive reports and visualizations"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # 1. Equity Curve
        ax1 = plt.subplot(2, 3, 1)
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        ax1.plot(equity_df['date'], equity_df['equity'], 'b-', linewidth=2)
        ax1.fill_between(equity_df['date'], self.config.initial_capital, equity_df['equity'],
                         where=equity_df['equity'] >= self.config.initial_capital,
                         facecolor='green', alpha=0.3)
        ax1.fill_between(equity_df['date'], self.config.initial_capital, equity_df['equity'],
                         where=equity_df['equity'] < self.config.initial_capital,
                         facecolor='red', alpha=0.3)
        ax1.set_title('Portfolio Equity Curve', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = plt.subplot(2, 3, 2)
        equity_values = equity_df['equity'].values
        running_max = pd.Series(equity_values).expanding().max()
        drawdown = (pd.Series(equity_values) - running_max) / running_max * 100
        ax2.fill_between(equity_df['date'], 0, drawdown.values, color='red', alpha=0.5)
        ax2.set_title('Drawdown %', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Monthly Returns Heatmap
        ax3 = plt.subplot(2, 3, 3)
        monthly_returns = equity_df.set_index('date')['equity'].resample('M').last().pct_change()
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year,
                                                 monthly_returns.index.month]).mean().unstack()
        sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    ax=ax3, cbar_kws={'label': 'Return %'})
        ax3.set_title('Monthly Returns Heatmap', fontsize=14)

        # 4. Trade Distribution
        ax4 = plt.subplot(2, 3, 4)
        trade_pnls = [t.pnl for t in portfolio.trades]
        ax4.hist(trade_pnls, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Trade P&L Distribution', fontsize=14)
        ax4.set_xlabel('P&L ($)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

        # 5. Exit Reason Analysis
        ax5 = plt.subplot(2, 3, 5)
        exit_reasons = pd.Series(results['exit_reasons'])
        exit_reasons.plot(kind='bar', ax=ax5, color='skyblue')
        ax5.set_title('Exit Reason Analysis', fontsize=14)
        ax5.set_xlabel('Exit Reason')
        ax5.set_ylabel('Count')
        ax5.grid(True, alpha=0.3)

        # 6. Sector Performance
        ax6 = plt.subplot(2, 3, 6)
        sector_perf = pd.DataFrame(results['sector_performance']).T
        if not sector_perf.empty and 'mean' in sector_perf.columns:
            sector_perf['mean'].sort_values().plot(kind='barh', ax=ax6, color='green')
            ax6.set_title('Sector Performance', fontsize=14)
            ax6.set_xlabel('Average Return %')
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_report.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Generate detailed HTML report
        self._generate_html_report(results)

        # Save results to JSON
        results_to_save = {k: v for k, v in results.items()
                           if k not in ['trades', 'equity_curve']}
        with open('backtest_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        logger.info("Reports generated successfully")

    def _generate_html_report(self, results: Dict):
        """Generate detailed HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hedge Fund Style Backtest Report</title>
            <style>
                body {{ font-family: 'Arial', sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .container {{ background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
                th {{ background-color: #34495e; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px 20px; background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; font-size: 14px; }}
                .metric-value {{ font-size: 24px; color: #2c3e50; margin-top: 5px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .warning {{ background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hedge Fund Style Backtest Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="container">
                <h2>Executive Summary</h2>
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
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{results['sortino_ratio']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{results['max_drawdown'] * 100:.2f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{results['win_rate'] * 100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{results['profit_factor']:.2f}</div>
                    </div>
                </div>

                {self._generate_performance_warnings(results)}
            </div>

            <div class="container">
                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Benchmark</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>Annual Return</td>
                        <td>{results['annual_return'] * 100:.2f}%</td>
                        <td>15%</td>
                        <td>{'✓' if results['annual_return'] > 0.15 else '✗'}</td>
                    </tr>
                    <tr>
                        <td>Annual Volatility</td>
                        <td>{results['annual_volatility'] * 100:.2f}%</td>
                        <td>&lt; 20%</td>
                        <td>{'✓' if results['annual_volatility'] < 0.20 else '✗'}</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>{results['sharpe_ratio']:.2f}</td>
                        <td>&gt; 1.5</td>
                        <td>{'✓' if results['sharpe_ratio'] > 1.5 else '✗'}</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td>{results['calmar_ratio']:.2f}</td>
                        <td>&gt; 1.0</td>
                        <td>{'✓' if results['calmar_ratio'] > 1.0 else '✗'}</td>
                    </tr>
                </table>
            </div>

            <div class="container">
                <h2>Trade Analysis</h2>
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
                        <td>Average Win</td>
                        <td class="positive">${results['avg_win']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative">${results['avg_loss']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Best Trade</td>
                        <td class="positive">${results['best_trade']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Worst Trade</td>
                        <td class="negative">${results['worst_trade']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Holding Period</td>
                        <td>{results['avg_holding_period']:.1f} days</td>
                    </tr>
                </table>
            </div>

            <div class="container">
                <h2>Exit Reason Analysis</h2>
                <table>
                    <tr>
                        <th>Exit Reason</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {self._generate_exit_reason_rows(results)}
                </table>
            </div>

            <div class="footer">
                <p>This report is for informational purposes only and does not constitute investment advice.</p>
            </div>
        </body>
        </html>
        """

        with open('backtest_report.html', 'w') as f:
            f.write(html)

    def _generate_performance_warnings(self, results: Dict) -> str:
        """Generate performance warnings"""
        warnings = []

        if results['sharpe_ratio'] < self.config.min_sharpe_ratio:
            warnings.append(
                f"Sharpe ratio ({results['sharpe_ratio']:.2f}) below target ({self.config.min_sharpe_ratio})")

        if results['win_rate'] < self.config.min_win_rate:
            warnings.append(
                f"Win rate ({results['win_rate'] * 100:.1f}%) below target ({self.config.min_win_rate * 100}%)")

        if abs(results['max_drawdown']) > self.config.max_drawdown_pct:
            warnings.append(
                f"Maximum drawdown ({abs(results['max_drawdown']) * 100:.1f}%) exceeds limit ({self.config.max_drawdown_pct * 100}%)")

        if warnings:
            warning_html = '<div class="warning"><strong>⚠️ Warnings:</strong><ul>'
            for warning in warnings:
                warning_html += f'<li>{warning}</li>'
            warning_html += '</ul></div>'
            return warning_html

        return ''

    def _generate_exit_reason_rows(self, results: Dict) -> str:
        """Generate exit reason table rows"""
        rows = []
        total_exits = sum(results['exit_reasons'].values())

        for reason, count in results['exit_reasons'].items():
            percentage = (count / total_exits * 100) if total_exits > 0 else 0
            rows.append(f"""
                <tr>
                    <td>{reason.replace('_', ' ').title()}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """)

        return ''.join(rows)


def run_sophisticated_backtest():
    """Run the sophisticated hedge fund style backtest"""
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,  # 2% per position
        max_positions=20,
        stop_loss_pct=0.02,
        trailing_stop_pct=0.015,
        take_profit_pct=0.05,
        train_period_days=504,  # 2 years
        validation_period_days=126,  # 6 months
        test_period_days=63,  # 3 months
        retrain_frequency=21,  # Monthly
        use_multi_timeframe=True,
        use_market_regime=True,
        use_correlation_filter=True
    )

    # Initialize backtester
    backtester = HedgeFundBacktester(config)

    # Define test period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year backtest

    # Run backtest on full watchlist
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        symbols=WATCHLIST  # Use full 188 symbol watchlist
    )

    # Print summary
    if results and 'error' not in results:
        print("\n" + "=" * 60)
        print("SOPHISTICATED HEDGE FUND BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Total Return: {results['total_return'] * 100:.2f}%")
        print(f"Annual Return: {results['annual_return'] * 100:.2f}%")
        print(f"Annual Volatility: {results['annual_volatility'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Avg Holding Period: {results['avg_holding_period']:.1f} days")
        print("=" * 60)
        print("\nReports saved:")
        print("- backtest_report.html")
        print("- backtest_report.png")
        print("- backtest_results.json")
    else:
        print(f"Backtest failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # Set up directories
    import os

    os.makedirs('cache/backtest_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Run the backtest
    run_sophisticated_backtest()