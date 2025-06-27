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
import os
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
from execution_simulator import ExecutionSimulator, ExecutionConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Enhanced configuration for sophisticated backtesting"""
    # Capital and position sizing
    initial_capital: float = 100000
    position_size_pct: float = 0.02  # 2% per position
    max_positions: int = 20
    max_sector_exposure: float = 0.30
    
    # Risk parameters
    stop_loss_pct: float = 0.02
    trailing_stop_pct: float = 0.015
    take_profit_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    
    # Transaction costs
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 5
    market_impact_factor: float = 0.1
    
    # Timing constraints
    min_holding_period: int = 1
    max_holding_period: int = 20
    
    # ML parameters - MORE REASONABLE
    train_period_days: int = 252  # 1 year training (reduced from 756)
    validation_period_days: int = 63  # 3 months validation (reduced from 252)
    test_period_days: int = 21  # 1 month testing
    retrain_frequency: int = 21  # Monthly
    min_data_points: int = 100  # Minimum 100 days (reduced from 504)
    sequence_length: int = 20  # LSTM sequence
    
    # Walk-forward parameters
    purge_days: int = 5
    embargo_days: int = 2
    
    # Performance thresholds
    min_sharpe_ratio: float = 1.0  # Reduced from 1.5
    min_win_rate: float = 0.40  # Reduced from 0.45
    min_profit_factor: float = 1.2  # Reduced from 1.5
    
    # Execution parameters
    use_limit_orders: bool = True
    limit_order_offset_bps: float = 10
    order_timeout_minutes: int = 30
    
    # Advanced features
    use_market_regime: bool = True
    use_correlation_filter: bool = True
    max_correlation: float = 0.7
    use_volume_filter: bool = True
    min_dollar_volume: float = 1000000
    
    # Multi-timeframe analysis
    use_multi_timeframe: bool = False  # Disable for now
    timeframes: List[str] = field(default_factory=lambda: ['1d'])
    
    # Market condition analysis
    analyze_market_cycles: bool = True
    min_cycles_required: int = 1  # Reduced from 2
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
            try:
                df = pd.read_parquet(cache_path)
                self.data_cache[cache_key] = df
                return df
            except:
                pass  # If cache is corrupted, re-fetch
        
        # Fetch from yfinance
        try:
            # Download single symbol only
            df = yf.download(
                tickers=symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                prepost=False,
                threads=False
            )
            
            # Handle empty data
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # If we somehow got MultiIndex columns, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns for {symbol}: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Check minimum data length
            if len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}: only {len(df)} rows")
                return pd.DataFrame()
            
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Add calculated fields using .loc to avoid warnings
            df.loc[:, 'returns'] = df['close'].pct_change()
            df.loc[:, 'log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df.loc[:, 'dollar_volume'] = df['close'] * df['volume']
            df.loc[:, 'volatility'] = df['returns'].rolling(20, min_periods=1).std()
            
            # Volume ratio with safety check
            vol_mean = df['volume'].rolling(20, min_periods=1).mean()
            df.loc[:, 'volume_ratio'] = df['volume'] / vol_mean.where(vol_mean > 0, 1.0)
            
            # Fill NaN values
            df['returns'] = df['returns'].fillna(0)
            df['log_returns'] = df['log_returns'].fillna(0)
            df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            # Save to cache
            try:
                df.to_parquet(cache_path)
            except:
                pass  # Don't fail if cache write fails
                
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_all_data(self, symbols: List[str], start_date: str, end_date: str,
                      n_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols in parallel"""
        all_data = {}
        failed_symbols = []
        
        logger.info(f"Fetching data from {start_date} to {end_date} for {len(symbols)} symbols")
        
        # Sequential fetch with progress bar for better debugging
        for symbol in tqdm(symbols, desc="Fetching data"):
            try:
                data = self.fetch_data(symbol, start_date, end_date)
                if not data.empty and len(data) >= 100:  # Minimum 100 days
                    all_data[symbol] = data
                    logger.debug(f"{symbol}: {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_symbols.append(symbol)
        
        logger.info(f"Successfully fetched data for {len(all_data)}/{len(symbols)} symbols")
        if failed_symbols:
            logger.info(f"Failed symbols: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")
        
        # Show date range of fetched data
        if all_data:
            min_date = min(df.index[0] for df in all_data.values())
            max_date = max(df.index[-1] for df in all_data.values())
            logger.info(f"Data date range: {min_date.date()} to {max_date.date()}")
        
        return all_data





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

        # Initialize ExecutionSimulator with config
        exec_config = ExecutionConfig()
        exec_config.commission_per_share = self.config.commission_per_share
        exec_config.min_commission = self.config.min_commission
        exec_config.base_slippage_bps = self.config.slippage_bps
        self.execution_sim = ExecutionSimulator(exec_config)

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
        
        # Calculate buffer period for training data
        buffer_days = self.config.train_period_days + self.config.validation_period_days + self.config.purge_days + 30
        
        # Extend date range for training data
        backtest_start = pd.to_datetime(start_date)
        extended_start = (backtest_start - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        
        logger.info(f"Backtest period: {start_date} to {end_date}")
        logger.info(f"Fetching data from {extended_start} (with {buffer_days} day buffer)")
        
        # Fetch all data including buffer period
        logger.info("Fetching historical data...")
        all_data = self.data_manager.fetch_all_data(symbols, extended_start, end_date)
        
        if len(all_data) < 10:
            logger.error("Insufficient data for backtesting")
            return {'error': 'Insufficient data'}
        
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
        
        logger.info(f"Training window: {train_start.date()} to {train_end.date()}")
        logger.info(f"Validation window: {val_start.date()} to {val_end.date()}")
        
        # Prepare training data
        train_data = {}
        val_data = {}
        
        for symbol, df in all_data.items():
            # Check if we have data for the required period
            if df.index[0] > train_start or df.index[-1] < val_end:
                logger.debug(f"Skipping {symbol}: insufficient historical data")
                continue
                
            # Training data
            train_mask = (df.index >= train_start) & (df.index < val_start)
            train_df = df[train_mask].copy()
            
            # Validation data  
            val_mask = (df.index >= val_start) & (df.index < val_end)
            val_df = df[val_mask].copy()
            
            # Check minimum lengths
            if len(train_df) >= 50 and len(val_df) >= 20:
                train_data[symbol] = train_df
                val_data[symbol] = val_df
        
        logger.info(f"Training data: {len(train_data)} symbols, Validation data: {len(val_data)} symbols")
        
        if len(train_data) < 5:  # Need at least 5 symbols
            logger.warning("Insufficient symbols for training, using dummy model")
            model = GPUEnsembleModel()
            model.is_trained = False
            return model
        
        # Create and train model
        model = GPUEnsembleModel()
        
        # Train with available data
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
                snapshot[symbol] = historical
        
        return snapshot
    
    def _generate_signals(self, model: GPUEnsembleModel, 
                         market_snapshot: Dict[str, pd.DataFrame],
                         current_positions: Set[str], 
                         date: pd.Timestamp) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        if not model or not model.is_trained:
            return signals
        
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
        
        return signals[:5]  # Take top 5 signals

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
            volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.02

            # ... exit condition checks ...

            if exit_reason:
                positions_to_close.append((symbol, current_price, daily_volume, volatility, exit_reason))

        # Execute exits
        for symbol, price, volume, volatility, reason in positions_to_close:
            exec_price, costs, commission = self.execution_sim.simulate_exit(
                symbol, price, portfolio.positions[symbol].quantity, volume, volatility
            )

            portfolio.close_position(symbol, exec_price, date, reason, costs, commission)
    
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
            'sector': t.sector,
            'entry_date': t.entry_date
        } for t in portfolio.trades])
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
        
        # Exit reason analysis
        exit_analysis = trades_df['exit_reason'].value_counts().to_dict()
        
        # Sector analysis
        sector_performance = trades_df.groupby('sector')['pnl_pct'].agg(['mean', 'count']).to_dict('index')
        
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
        # Save results to JSON
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['trades', 'equity_curve']}
        with open('backtest_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info("Backtest complete. Results saved to backtest_results.json")

