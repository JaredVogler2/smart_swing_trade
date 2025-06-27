# enhanced_main.py
"""
Enhanced Hedge Fund Style Trading System
Main orchestrator with sophisticated features
"""

import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
import time
import threading
import signal
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import concurrent.futures

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configurations
from config.settings import Config
from config.watchlist import WATCHLIST, SECTOR_MAPPING

# Import core modules
from data.database import Database
from data.market_data import MarketDataManager
from data.cache_manager import CacheManager

# Import enhanced modules
from models.ensemble_gpu_windows import GPUEnsembleModel
from models.regime_detector import MarketRegimeDetector
from enhanced_advanced_backtesting import HedgeFundBacktester, BacktestConfig, DataManager

# Import analysis modules
from analysis.news_sentiment import NewsSentimentAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.signals import SignalGenerator

# Import risk management
from risk.position_sizer import PositionSizer
from risk.risk_manager import RiskManager
from risk.stop_loss import StopLossManager

# Import execution modules
from execution.order_manager import OrderManager
from execution.broker_interface import BrokerInterface

# Import monitoring
from monitoring.performance import PerformanceTracker
from monitoring.dashboard import TradingDashboard


# Setup sophisticated logging
def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # File handler for all logs
    all_handler = logging.FileHandler(f'{log_dir}/trading_system.log')
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(all_handler)

    # File handler for errors only
    error_handler = logging.FileHandler(f'{log_dir}/errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # Trading specific logger
    trade_logger = logging.getLogger('trading')
    trade_handler = logging.FileHandler(f'{log_dir}/trades.log')
    trade_handler.setFormatter(detailed_formatter)
    trade_logger.addHandler(trade_handler)

    return logging.getLogger(__name__)


# Setup logging
logger = setup_logging()


@dataclass
class SystemState:
    """System state tracking"""
    is_running: bool = False
    trading_enabled: bool = True
    last_model_update: Optional[datetime] = None
    total_trades_today: int = 0
    daily_pnl: float = 0.0
    positions_count: int = 0
    pending_orders_count: int = 0
    last_scan_time: Optional[datetime] = None
    system_health: str = "healthy"
    alerts: List[Dict] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class EnhancedTradingSystem:
    """Sophisticated hedge fund style trading system"""

    def __init__(self, mode: str = 'paper', config_file: Optional[str] = None):
        """
        Initialize the enhanced trading system

        Args:
            mode: 'paper', 'live', or 'backtest'
            config_file: Optional custom configuration file
        """
        self.mode = mode
        self.state = SystemState()

        logger.info(f"Initializing Enhanced Trading System in {mode} mode")

        # Load configuration
        self._load_configuration(config_file)

        # Initialize components
        self._initialize_components()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Performance metrics
        self.daily_trades = []
        self.session_start_time = datetime.now()

    def _load_configuration(self, config_file: Optional[str]):
        """Load system configuration"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                # Update Config with custom settings
                for key, value in custom_config.items():
                    setattr(Config, key, value)

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate system configuration"""
        required_keys = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL']

        for key in required_keys:
            if not hasattr(Config, key) or not getattr(Config, key):
                raise ValueError(f"Missing required configuration: {key}")

        logger.info("Configuration validated successfully")

    def _initialize_components(self):
        """Initialize all system components with error handling"""
        try:
            # Core components
            self.db = Database()
            logger.info("Database initialized")

            self.broker = BrokerInterface()
            self._verify_broker_connection()

            self.market_data = MarketDataManager()
            self.cache = CacheManager()
            self.data_manager = DataManager()
            logger.info("Market data systems initialized")

            # ML models
            self.ml_model = GPUEnsembleModel(max_gpu_memory_mb=8192)
            self.regime_detector = MarketRegimeDetector()
            self._load_or_train_models()

            # Analysis modules
            self.news_analyzer = NewsSentimentAnalyzer()
            self.technical_analyzer = TechnicalAnalyzer()
            self.signal_generator = SignalGenerator(self.ml_model, self.news_analyzer)
            logger.info("Analysis modules initialized")

            # Risk management
            account_value = self._get_account_value()
            self.position_sizer = PositionSizer(account_value)
            self.risk_manager = RiskManager(account_value)
            self.stop_loss_manager = StopLossManager()
            logger.info(f"Risk management initialized with account value: ${account_value:,.2f}")

            # Order management
            self.order_manager = OrderManager(self.broker.api)
            logger.info("Order manager initialized")

            # Performance tracking
            self.performance_tracker = PerformanceTracker(account_value)
            logger.info("Performance tracker initialized")

            # Initialize thread pool for parallel operations
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

            self.state.system_health = "healthy"

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.state.system_health = "error"
            raise

    def _verify_broker_connection(self):
        """Verify broker connection"""
        try:
            account = self.broker.get_account_info()
            if account:
                logger.info("Broker connection verified")
                return True
        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            if self.mode == 'live':
                raise
        return False

    def _get_account_value(self) -> float:
        """Get current account value"""
        try:
            account_info = self.broker.get_account_info()
            return float(account_info.get('portfolio_value', Config.ACCOUNT_SIZE))
        except:
            return Config.ACCOUNT_SIZE

    def _load_or_train_models(self):
        """Load existing models or train new ones"""
        model_path = 'models/saved'

        if os.path.exists(model_path) and os.listdir(model_path):
            try:
                logger.info("Loading existing models...")
                self.ml_model.load_models(model_path)
                self.state.last_model_update = datetime.now()
                logger.info("Models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                self._train_models()
        else:
            logger.info("No saved models found")
            self._train_models()

    def _train_models(self):
        """Train ML models on historical data"""
        logger.info("Starting model training...")

        try:
            # Use FULL watchlist for training
            training_symbols = WATCHLIST  # ALL 188 symbols

            # Fetch training data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years

            logger.info(f"Fetching training data for {len(training_symbols)} symbols")
            training_data = self.data_manager.fetch_all_data(
                training_symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if len(training_data) < 20:
                logger.error("Insufficient training data")
                return False

            # Split into train/validation
            # Use more symbols for better model generalization
            validation_split = int(len(training_data) * 0.2)  # 20% validation
            all_symbols = list(training_data.keys())
            np.random.shuffle(all_symbols)  # Shuffle for randomness

            validation_symbols = all_symbols[:validation_split]
            train_symbols = all_symbols[validation_split:]

            train_data = {s: training_data[s] for s in train_symbols}
            val_data = {s: training_data[s] for s in validation_symbols}

            # Train model
            logger.info(f"Training on {len(train_data)} symbols...")
            self.ml_model.train(train_data, validation_data=val_data)

            # Save model
            self.ml_model.save_models('models/saved')
            self.state.last_model_update = datetime.now()

            logger.info("Model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return False

    def run(self):
        """Main execution loop"""
        self.state.is_running = True
        logger.info("=" * 60)
        logger.info("ENHANCED TRADING SYSTEM STARTED")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Time: {datetime.now()}")
        logger.info("=" * 60)

        # Main trading loop
        cycle_count = 0
        last_scan_time = datetime.now()

        while self.state.is_running:
            try:
                loop_start = time.time()

                # Check market status
                market_status = self.market_data.get_market_status()

                if market_status['is_open'] and self.state.trading_enabled:
                    # Regular trading cycle
                    self._execute_trading_cycle()

                    # Periodic tasks
                    if cycle_count % 12 == 0:  # Every minute
                        self._update_system_metrics()
                        self._check_system_health()

                    if cycle_count % 60 == 0:  # Every 5 minutes
                        self._rebalance_portfolio()
                        self._update_risk_limits()

                    if cycle_count % 720 == 0:  # Every hour
                        self._generate_reports()
                        self._cleanup_old_data()

                    # Model retraining check (daily)
                    if self._should_retrain_models():
                        self._retrain_models_async()

                else:
                    # Market closed - perform maintenance
                    if cycle_count % 300 == 0:  # Every 25 minutes
                        self._after_hours_maintenance()

                # Sleep to maintain 5-second intervals
                elapsed = time.time() - loop_start
                sleep_time = max(0, 5 - elapsed)
                time.sleep(sleep_time)

                cycle_count += 1

            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self._handle_system_error(e)
                time.sleep(30)  # Wait before retrying

        self._shutdown()

    def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            # Update portfolio state
            self._update_portfolio_state()

            # Risk checks
            if not self._perform_risk_checks():
                return

            # Update existing positions
            positions_updated = self._update_positions()

            # Scan for new opportunities
            current_time = datetime.now()
            if (current_time - self.state.last_scan_time).seconds >= 30:  # Scan every 30 seconds
                new_signals = self._scan_for_opportunities()

                if new_signals:
                    self._execute_signals(new_signals)

                self.state.last_scan_time = current_time

            # Update performance metrics
            self._update_performance_metrics()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.state.alerts.append({
                'severity': 'high',
                'message': f"Trading cycle error: {str(e)}",
                'timestamp': datetime.now()
            })

    def _update_portfolio_state(self):
        """Update current portfolio state"""
        try:
            # Get positions
            positions = self.broker.get_positions()
            self.state.positions_count = len(positions)

            # Get orders
            orders = self.broker.get_open_orders()
            self.state.pending_orders_count = len(orders)

            # Update account metrics
            account_info = self.broker.get_account_info()

            # Calculate daily P&L
            current_equity = float(account_info.get('equity', 0))
            last_equity = float(account_info.get('last_equity', current_equity))
            self.state.daily_pnl = current_equity - last_equity

            # Log portfolio state
            logger.debug(f"Portfolio state - Positions: {self.state.positions_count}, "
                         f"Orders: {self.state.pending_orders_count}, "
                         f"Daily P&L: ${self.state.daily_pnl:,.2f}")

        except Exception as e:
            logger.error(f"Failed to update portfolio state: {e}")

    def _perform_risk_checks(self) -> bool:
        """Perform comprehensive risk checks"""
        try:
            # Check daily loss limit
            if self.state.daily_pnl < -Config.MAX_DAILY_LOSS:
                logger.warning(f"Daily loss limit reached: ${self.state.daily_pnl:,.2f}")
                self.state.trading_enabled = False
                return False

            # Check position limits
            if self.state.positions_count >= Config.MAX_POSITIONS:
                logger.debug("Maximum positions reached")
                return False

            # Check system risk metrics
            should_stop, reason = self.risk_manager.should_stop_trading()
            if should_stop:
                logger.warning(f"Risk manager stop: {reason}")
                self.state.trading_enabled = False
                return False

            return True

        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return False

    def _update_positions(self) -> int:
        """Update existing positions with sophisticated logic"""
        updated_count = 0

        try:
            positions = self.broker.get_positions()

            if not positions:
                return 0

            # Get current market data
            symbols = [p['symbol'] for p in positions]
            current_prices = self.market_data.get_current_prices_batch(symbols)

            # Parallel position updates
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []

                for position in positions:
                    future = executor.submit(
                        self._update_single_position,
                        position,
                        current_prices.get(position['symbol'])
                    )
                    futures.append((position['symbol'], future))

                for symbol, future in futures:
                    try:
                        if future.result():
                            updated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to update position {symbol}: {e}")

            logger.info(f"Updated {updated_count} positions")

        except Exception as e:
            logger.error(f"Position update failed: {e}")

        return updated_count

    def _update_single_position(self, position: Dict, current_price: Optional[float]) -> bool:
        """Update a single position"""
        if not current_price:
            return False

        try:
            symbol = position['symbol']

            # Get recent market data
            bars = self.market_data.get_bars(symbol, '1Day', limit=20)
            if bars.empty:
                return False

            # Check for exit signals
            exit_signal = self._check_exit_conditions(position, current_price, bars)

            if exit_signal:
                logger.info(f"Exit signal for {symbol}: {exit_signal['reason']}")
                self.order_manager.create_exit_order(position, exit_signal)
                return True

            # Update stop loss
            updated_stop = self.stop_loss_manager.update_position_stop(
                position, current_price
            )

            if updated_stop and updated_stop.get('stop_moved'):
                self.order_manager.update_stop_loss(position, updated_stop['stop_loss_price'])
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating position {position.get('symbol')}: {e}")
            return False

    def _check_exit_conditions(self, position: Dict, current_price: float,
                               bars: pd.DataFrame) -> Optional[Dict]:
        """Check sophisticated exit conditions"""
        symbol = position['symbol']
        entry_price = position['avg_entry_price']

        # Calculate metrics
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        holding_days = (datetime.now() - position['entry_time']).days

        # ML-based exit signal
        if self.ml_model.is_trained:
            try:
                prediction = self.ml_model.predict(symbol, bars)
                if prediction['prediction'] == 0 and prediction['confidence'] > 0.7:
                    return {
                        'reason': 'ml_reversal',
                        'confidence': prediction['confidence'],
                        'urgency': 'high'
                    }
            except:
                pass

        # Time-based exit (swing trading)
        if holding_days >= 20:  # Max 20 days
            return {
                'reason': 'max_holding_period',
                'urgency': 'medium'
            }

        # Profit target
        if pnl_pct >= 5.0:  # 5% profit target
            return {
                'reason': 'profit_target',
                'urgency': 'low'
            }

        # Technical indicators
        technical_signal = self.technical_analyzer.check_exit_signals(bars)
        if technical_signal and technical_signal['strength'] > 0.8:
            return {
                'reason': 'technical',
                'signal': technical_signal,
                'urgency': 'medium'
            }

        # Volatility-based exit
        recent_volatility = bars['close'].pct_change().rolling(10).std().iloc[-1]
        if recent_volatility > 0.05:  # 5% daily volatility
            return {
                'reason': 'high_volatility',
                'volatility': recent_volatility,
                'urgency': 'high'
            }

        return None

    def _scan_for_opportunities(self) -> List[Dict]:
        """Scan market for trading opportunities"""
        logger.info("Scanning for opportunities...")

        try:
            # Get current positions to exclude
            positions = self.broker.get_positions()
            current_symbols = {p['symbol'] for p in positions}

            # Determine scan universe
            available_symbols = [s for s in WATCHLIST if s not in current_symbols]

            # Check position limits
            max_new_positions = Config.MAX_POSITIONS - len(positions)
            if max_new_positions <= 0:
                return []

            # Get market regime
            spy_data = self.market_data.get_bars('SPY', '1Day', limit=100)
            regime_info = self.regime_detector.detect_regime(spy_data)
            market_regime = regime_info['regime']

            # Pre-screen symbols
            scan_symbols = self._prescreen_symbols(available_symbols, market_regime)

            # Generate signals in parallel for ALL available symbols
            all_signals = []

            # Process in batches for efficiency but cover ALL symbols
            batch_size = 20
            for i in range(0, len(scan_symbols), batch_size):
                batch = scan_symbols[i:i + batch_size]

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_symbol = {
                        executor.submit(self._analyze_symbol, symbol, market_regime): symbol
                        for symbol in batch
                    }

                    for future in concurrent.futures.as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            signal = future.result()
                            if signal:
                                all_signals.append(signal)
                        except Exception as e:
                            logger.error(f"Error analyzing {symbol}: {e}")

            # Rank and filter signals
            ranked_signals = self._rank_signals(all_signals, max_new_positions)

            logger.info(f"Found {len(ranked_signals)} high-quality signals")

            return ranked_signals

        except Exception as e:
            logger.error(f"Opportunity scan failed: {e}")
            return []

    def _prescreen_symbols(self, symbols: List[str], market_regime: str) -> List[str]:
        """Pre-screen symbols for efficiency"""
        screened = []

        try:
            # Get snapshot data
            snapshot = self.market_data.get_snapshot(symbols)

            for symbol, data in snapshot.items():
                if self._passes_prescreen(data, market_regime):
                    screened.append(symbol)

            # Sort by volume and momentum
            screened.sort(key=lambda s: (
                snapshot[s]['volume'],
                snapshot[s]['change_pct']
            ), reverse=True)

        except Exception as e:
            logger.error(f"Pre-screening failed: {e}")
            return symbols  # Return ALL symbols on error

        return screened

    def _passes_prescreen(self, snapshot_data: Dict, market_regime: str) -> bool:
        """Check if symbol passes pre-screening criteria"""
        # Volume filter
        if snapshot_data.get('volume', 0) < 1000000:
            return False

        # Price filter
        price = snapshot_data.get('price', 0)
        if price < 10 or price > 1000:
            return False

        # Regime-specific filters
        change_pct = snapshot_data.get('change_pct', 0)

        if market_regime == 'bull' and change_pct < -2:
            return False
        elif market_regime == 'bear' and change_pct > 2:
            return False

        return True

    def _analyze_symbol(self, symbol: str, market_regime: str) -> Optional[Dict]:
        """Analyze a single symbol for trading opportunity"""
        try:
            # Get market data
            bars = self.market_data.get_bars(symbol, '1Day', limit=100)
            if bars.empty or len(bars) < 50:
                return None

            # ML prediction
            ml_signal = None
            if self.ml_model.is_trained:
                prediction = self.ml_model.predict(symbol, bars)
                if prediction['prediction'] == 1 and prediction['confidence'] >= 0.65:
                    ml_signal = prediction

            # Technical analysis
            technical_signal = self.technical_analyzer.analyze(bars)

            # News sentiment
            news_sentiment = self.news_analyzer.get_symbol_sentiment(symbol)

            # Combine signals
            if ml_signal or (technical_signal and technical_signal['strength'] > 0.7):
                signal = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'market_regime': market_regime,
                    'ml_signal': ml_signal,
                    'technical_signal': technical_signal,
                    'news_sentiment': news_sentiment,
                    'current_price': bars['close'].iloc[-1],
                    'volatility': bars['close'].pct_change().rolling(20).std().iloc[-1],
                    'volume_ratio': bars['volume'].iloc[-1] / bars['volume'].rolling(20).mean().iloc[-1]
                }

                # Calculate composite score
                signal['score'] = self._calculate_signal_score(signal)

                if signal['score'] >= 0.6:
                    return signal

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

        return None

    def _calculate_signal_score(self, signal: Dict) -> float:
        """Calculate composite signal score"""
        score = 0.0
        weights = {
            'ml': 0.4,
            'technical': 0.3,
            'sentiment': 0.2,
            'market': 0.1
        }

        # ML component
        if signal.get('ml_signal'):
            score += weights['ml'] * signal['ml_signal']['confidence']

        # Technical component
        if signal.get('technical_signal'):
            score += weights['technical'] * signal['technical_signal']['strength']

        # Sentiment component
        if signal.get('news_sentiment'):
            sentiment_score = (signal['news_sentiment']['score'] + 1) / 2  # Normalize to 0-1
            score += weights['sentiment'] * sentiment_score

        # Market regime component
        if signal['market_regime'] == 'bull':
            score += weights['market'] * 1.0
        elif signal['market_regime'] == 'neutral':
            score += weights['market'] * 0.5

        return min(score, 1.0)

    def _rank_signals(self, signals: List[Dict], max_signals: int) -> List[Dict]:
        """Rank and filter signals"""
        if not signals:
            return []

        # Sort by score
        signals.sort(key=lambda x: x['score'], reverse=True)

        # Diversification filter
        selected = []
        sectors_included = set()

        for signal in signals:
            if len(selected) >= max_signals:
                break

            symbol = signal['symbol']
            sector = SECTOR_MAPPING.get(symbol, 'Unknown')

            # Limit per sector
            sector_count = sum(1 for s in selected if SECTOR_MAPPING.get(s['symbol'], 'Unknown') == sector)
            if sector_count >= 2:  # Max 2 per sector
                continue

            # Correlation check
            if not self._is_correlated_with_selected(signal, selected):
                selected.append(signal)
                sectors_included.add(sector)

        return selected

    def _is_correlated_with_selected(self, signal: Dict, selected: List[Dict]) -> bool:
        """Check if signal is highly correlated with already selected signals"""
        if not selected:
            return False

        # Simplified correlation check
        # In production, would calculate actual correlation matrix
        symbol = signal['symbol']

        for sel_signal in selected:
            if self._are_symbols_correlated(symbol, sel_signal['symbol']):
                return True

        return False

    def _are_symbols_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are highly correlated"""
        # Known correlation pairs
        known_pairs = [
            ('GOOGL', 'GOOG'), ('BRK.A', 'BRK.B'),
            ('META', 'GOOGL'), ('V', 'MA'),
            ('HD', 'LOW'), ('UPS', 'FDX')
        ]

        for pair in known_pairs:
            if (symbol1, symbol2) in [pair, pair[::-1]]:
                return True

        # Same sector check for some sectors
        sector1 = SECTOR_MAPPING.get(symbol1, 'Unknown')
        sector2 = SECTOR_MAPPING.get(symbol2, 'Unknown')

        if sector1 == sector2 and sector1 in ['Energy', 'Utilities']:
            return True

        return False

    def _execute_signals(self, signals: List[Dict]):
        """Execute trading signals with sophisticated order management"""
        if not signals:
            return

        try:
            # Get current account state
            account_info = self.broker.get_account_info()
            buying_power = float(account_info['buying_power'])
            total_equity = float(account_info['portfolio_value'])

            executed_count = 0

            for signal in signals:
                # Final validation
                validation = self._validate_signal(signal, buying_power, total_equity)
                if not validation['valid']:
                    logger.warning(f"Signal validation failed for {signal['symbol']}: {validation['reason']}")
                    continue

                # Risk approval
                risk_check = self.risk_manager.check_trade_approval(
                    signal, self.broker.get_positions()
                )
                if not risk_check['approved']:
                    logger.warning(f"Risk check failed for {signal['symbol']}: {risk_check['reasons']}")
                    continue

                # Calculate position size
                position_size = self.position_sizer.calculate_position_size(
                    signal,
                    self.broker.get_positions(),
                    market_regime=signal.get('market_regime')
                )

                if position_size['shares'] == 0:
                    continue

                # Execute order
                try:
                    order_result = self._execute_order(signal, position_size)
                    if order_result['success']:
                        executed_count += 1
                        buying_power -= position_size['position_value']

                        # Record trade
                        self._record_trade(signal, order_result)

                        # Update state
                        self.state.total_trades_today += 1

                except Exception as e:
                    logger.error(f"Order execution failed for {signal['symbol']}: {e}")

            logger.info(f"Executed {executed_count}/{len(signals)} signals")

        except Exception as e:
            logger.error(f"Signal execution failed: {e}")

    def _validate_signal(self, signal: Dict, buying_power: float, total_equity: float) -> Dict:
        """Validate signal before execution"""
        symbol = signal['symbol']

        # Check if tradable
        asset_info = self.market_data.get_asset_info(symbol)
        if not asset_info.get('tradable', False):
            return {'valid': False, 'reason': 'not_tradable'}

        # Check buying power
        position_value = total_equity * 0.02  # 2% position size
        if position_value > buying_power:
            return {'valid': False, 'reason': 'insufficient_buying_power'}

        # Check signal age
        signal_age = (datetime.now() - signal['timestamp']).seconds
        if signal_age > 300:  # 5 minutes
            return {'valid': False, 'reason': 'stale_signal'}

        return {'valid': True}

    def _execute_order(self, signal: Dict, position_size: Dict) -> Dict:
        """Execute order with advanced order types"""
        symbol = signal['symbol']

        try:
            # Get current quote
            quote = self.market_data.get_quote(symbol)

            # Determine order type based on market conditions
            if signal.get('urgency') == 'high' or quote['spread_pct'] < 0.05:
                # Market order for urgent signals or tight spreads
                order = self.broker.api.submit_order(
                    symbol=symbol,
                    qty=position_size['shares'],
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
            else:
                # Limit order for better execution
                limit_price = quote['bid'] + (quote['spread'] * 0.3)  # 30% into spread
                order = self.broker.api.submit_order(
                    symbol=symbol,
                    qty=position_size['shares'],
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=round(limit_price, 2)
                )

            logger.info(f"Order submitted: {symbol} x{position_size['shares']} @ "
                        f"{'market' if order.order_type == 'market' else f'limit ${order.limit_price}'}")

            # Set stop loss
            stop_price = signal['current_price'] * 0.98  # 2% stop loss
            stop_order = self.broker.api.submit_order(
                symbol=symbol,
                qty=position_size['shares'],
                side='sell',
                type='stop',
                time_in_force='gtc',
                stop_price=round(stop_price, 2)
            )

            return {
                'success': True,
                'order_id': order.id,
                'stop_order_id': stop_order.id,
                'order_type': order.order_type,
                'limit_price': getattr(order, 'limit_price', None)
            }

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return {'success': False, 'error': str(e)}

    def _record_trade(self, signal: Dict, order_result: Dict):
        """Record trade details for analysis"""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'signal_score': signal['score'],
            'ml_confidence': signal.get('ml_signal', {}).get('confidence'),
            'order_id': order_result['order_id'],
            'order_type': order_result['order_type'],
            'market_regime': signal['market_regime'],
            'features': signal
        }

        self.daily_trades.append(trade_record)
        self.db.save_trade(trade_record)

    def _update_performance_metrics(self):
        """Update real-time performance metrics"""
        try:
            # Get recent filled orders
            recent_orders = self.order_manager.get_filled_orders(lookback_hours=24)

            for order in recent_orders:
                if order['side'] == 'sell':
                    # Find matching buy order
                    # This is simplified - in production, maintain proper trade matching
                    self.performance_tracker.record_trade({
                        'symbol': order['symbol'],
                        'exit_price': order['filled_price'],
                        'exit_time': order['filled_at'],
                        'shares': order['filled_qty']
                    })

            # Update risk metrics
            self.risk_manager.update_daily_metrics()

        except Exception as e:
            logger.error(f"Performance update failed: {e}")

    def _update_system_metrics(self):
        """Update system health metrics"""
        try:
            # API usage
            api_usage = self.market_data.get_api_usage()

            # System resources
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            # Trading metrics
            positions = self.broker.get_positions()
            total_unrealized_pnl = sum(float(p.get('unrealized_pl', 0)) for p in positions)

            logger.info(f"System metrics - CPU: {cpu_percent}%, Memory: {memory_percent}%, "
                        f"API calls: {api_usage['calls_last_minute']}/{api_usage['calls_limit']}, "
                        f"Unrealized P&L: ${total_unrealized_pnl:,.2f}")

            # Check system health
            if cpu_percent > 90 or memory_percent > 90:
                self.state.system_health = "degraded"
                logger.warning("System resources high")
            elif api_usage['usage_pct'] > 80:
                self.state.system_health = "api_limited"
                logger.warning("Approaching API rate limit")
            else:
                self.state.system_health = "healthy"

        except Exception as e:
            logger.error(f"System metrics update failed: {e}")

    def _check_system_health(self):
        """Perform system health checks"""
        alerts = []

        # Check broker connection
        if not self._verify_broker_connection():
            alerts.append({
                'severity': 'critical',
                'message': 'Broker connection lost',
                'timestamp': datetime.now()
            })

        # Check model staleness
        if self.state.last_model_update:
            model_age = (datetime.now() - self.state.last_model_update).days
            if model_age > 30:
                alerts.append({
                    'severity': 'medium',
                    'message': f'ML model is {model_age} days old',
                    'timestamp': datetime.now()
                })

        # Check daily limits
        if self.state.total_trades_today > 100:
            alerts.append({
                'severity': 'high',
                'message': 'High trading volume today',
                'timestamp': datetime.now()
            })

        self.state.alerts = alerts

        # Log critical alerts
        for alert in alerts:
            if alert['severity'] == 'critical':
                logger.error(f"CRITICAL ALERT: {alert['message']}")

    def _rebalance_portfolio(self):
        """Rebalance portfolio based on targets"""
        try:
            positions = self.broker.get_positions()
            if not positions:
                return

            account_value = self._get_account_value()

            # Calculate current allocations
            position_values = {}
            total_positions_value = 0

            for position in positions:
                value = float(position['market_value'])
                position_values[position['symbol']] = value
                total_positions_value += value

            # Check for overweight positions
            for symbol, value in position_values.items():
                allocation = value / account_value

                if allocation > 0.05:  # Over 5% allocation
                    # Calculate shares to sell
                    target_value = account_value * 0.04  # Reduce to 4%
                    excess_value = value - target_value

                    current_price = self.market_data.get_current_price(symbol)
                    if current_price:
                        shares_to_sell = int(excess_value / current_price)

                        if shares_to_sell > 0:
                            logger.info(f"Rebalancing {symbol}: selling {shares_to_sell} shares")
                            self.order_manager.create_rebalance_order(
                                symbol, shares_to_sell, 'sell'
                            )

        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}")

    def _update_risk_limits(self):
        """Dynamically update risk limits based on performance"""
        try:
            # Get recent performance
            recent_trades = self.performance_tracker.get_recent_trades(days=30)

            if len(recent_trades) > 20:
                # Calculate metrics
                win_rate = len([t for t in recent_trades if t['pnl'] > 0]) / len(recent_trades)
                avg_win = np.mean([t['pnl'] for t in recent_trades if t['pnl'] > 0])
                avg_loss = abs(np.mean([t['pnl'] for t in recent_trades if t['pnl'] < 0]))

                # Adjust risk based on performance
                if win_rate > 0.6 and avg_win > avg_loss * 1.5:
                    # Increase risk slightly
                    self.risk_manager.adjust_risk_multiplier(1.1)
                    logger.info("Increasing risk limits due to good performance")
                elif win_rate < 0.4 or avg_loss > avg_win * 1.5:
                    # Decrease risk
                    self.risk_manager.adjust_risk_multiplier(0.9)
                    logger.info("Decreasing risk limits due to poor performance")

        except Exception as e:
            logger.error(f"Risk limit update failed: {e}")

    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        if not self.state.last_model_update:
            return True

        # Retrain weekly
        days_since_update = (datetime.now() - self.state.last_model_update).days
        return days_since_update >= 7

    def _retrain_models_async(self):
        """Retrain models asynchronously"""

        def retrain():
            logger.info("Starting async model retraining...")
            try:
                self._train_models()
                logger.info("Async model retraining completed")
            except Exception as e:
                logger.error(f"Async model retraining failed: {e}")

        # Run in separate thread
        thread = threading.Thread(target=retrain)
        thread.daemon = True
        thread.start()

    def _generate_reports(self):
        """Generate performance and risk reports"""
        try:
            # Performance report
            perf_report = self.performance_tracker.generate_performance_report()

            # Risk report
            risk_report = self.risk_manager.generate_risk_report()

            # Save reports
            report_dir = 'reports'
            os.makedirs(report_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            with open(f'{report_dir}/performance_{timestamp}.json', 'w') as f:
                json.dump(perf_report, f, indent=2, default=str)

            with open(f'{report_dir}/risk_{timestamp}.json', 'w') as f:
                json.dump(risk_report, f, indent=2, default=str)

            logger.info(f"Reports generated: {timestamp}")

            # Log summary
            logger.info(f"Daily P&L: ${self.state.daily_pnl:,.2f}, "
                        f"Trades: {self.state.total_trades_today}, "
                        f"Positions: {self.state.positions_count}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    def _cleanup_old_data(self):
        """Clean up old data to save space"""
        try:
            # Clean old cache files
            self.cache.cleanup_old_files(days=7)

            # Clean old logs
            log_dir = 'logs'
            cutoff = datetime.now() - timedelta(days=30)

            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.getmtime(filepath) < cutoff.timestamp():
                    os.remove(filepath)
                    logger.debug(f"Removed old log: {filename}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _after_hours_maintenance(self):
        """Perform after-hours maintenance tasks"""
        try:
            logger.info("Running after-hours maintenance...")

            # Backup database
            self.db.backup_database()

            # Clear market data cache
            self.market_data.clear_cache()

            # Analyze today's trades
            self._analyze_daily_performance()

            # Prepare for next session
            self.state.total_trades_today = 0
            self.daily_trades = []

            logger.info("After-hours maintenance completed")

        except Exception as e:
            logger.error(f"After-hours maintenance failed: {e}")

    def _analyze_daily_performance(self):
        """Analyze daily trading performance"""
        if not self.daily_trades:
            return

        try:
            # Calculate metrics
            total_trades = len(self.daily_trades)

            # ML model performance
            ml_trades = [t for t in self.daily_trades if t.get('ml_confidence')]
            if ml_trades:
                avg_ml_confidence = np.mean([t['ml_confidence'] for t in ml_trades])
                logger.info(f"ML trades: {len(ml_trades)}, Avg confidence: {avg_ml_confidence:.2f}")

            # Signal distribution
            signal_scores = [t['signal_score'] for t in self.daily_trades]
            logger.info(f"Signal scores - Mean: {np.mean(signal_scores):.2f}, "
                        f"Std: {np.std(signal_scores):.2f}")

            # Save daily analysis
            analysis = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades': total_trades,
                'ml_trades': len(ml_trades),
                'avg_signal_score': np.mean(signal_scores),
                'daily_pnl': self.state.daily_pnl,
                'trades': self.daily_trades
            }

            self.db.save_daily_analysis(analysis)

        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")

    def _handle_system_error(self, error: Exception):
        """Handle system errors gracefully"""
        logger.error(f"System error: {error}", exc_info=True)

        # Determine severity
        if isinstance(error, ConnectionError):
            # Try to reconnect
            logger.info("Attempting to reconnect...")
            self._initialize_components()
        else:
            # Log and continue
            self.state.alerts.append({
                'severity': 'high',
                'message': f'System error: {str(error)}',
                'timestamp': datetime.now()
            })

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.state.is_running = False

    def _shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("Starting graceful shutdown...")

        try:
            # Cancel all pending orders
            self.order_manager.cancel_all_orders()

            # Save state
            self._save_state()

            # Generate final reports
            self._generate_reports()

            # Cleanup
            self.executor.shutdown(wait=True)

            # Close connections
            self.db.close()

            logger.info("Shutdown completed successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _save_state(self):
        """Save system state"""
        try:
            state_data = {
                'timestamp': datetime.now(),
                'session_duration': (datetime.now() - self.session_start_time).total_seconds(),
                'total_trades': self.state.total_trades_today,
                'daily_pnl': self.state.daily_pnl,
                'positions_count': self.state.positions_count,
                'system_health': self.state.system_health,
                'last_model_update': self.state.last_model_update,
                'alerts': self.state.alerts
            }

            with open('data/system_state.json', 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            # Save performance data
            self.performance_tracker.save_performance_data('data/performance.json')

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def run_backtest(self, start_date: str, end_date: str):
        """Run sophisticated backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Configure backtest
        config = BacktestConfig(
            initial_capital=100000,
            position_size_pct=0.02,
            max_positions=20,
            stop_loss_pct=0.02,
            use_correlation_filter=True,
            use_market_regime=True
        )

        # Run backtest on FULL watchlist
        backtester = HedgeFundBacktester(config)
        results = backtester.run_backtest(
            start_date=start_date,
            end_date=end_date,
            symbols=WATCHLIST  # Use ALL 188 symbols
        )

        return results

    def run_dashboard(self):
        """Launch the Streamlit dashboard"""
        logger.info("Launching dashboard...")

        # Import and run dashboard in subprocess
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_streamlit_dashboard.py"])


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Hedge Fund Trading System')
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest', 'train', 'dashboard'],
        default='paper',
        help='System mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Custom configuration file'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    # Create necessary directories
    directories = ['logs', 'data', 'cache', 'models/saved', 'reports', 'data/backups']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Initialize system
    system = EnhancedTradingSystem(mode=args.mode, config_file=args.config)

    # Run based on mode
    if args.mode == 'train':
        # Train models only
        system._train_models()

    elif args.mode == 'backtest':
        # Run backtest
        if not args.start_date or not args.end_date:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
            # Use 5 years of history by default - more representative of different market conditions
            args.start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

        results = system.run_backtest(args.start_date, args.end_date)

        if results and 'error' not in results:
            print("\n" + "=" * 60)
            print("BACKTEST RESULTS")
            print("=" * 60)
            print(f"Total Return: {results['total_return'] * 100:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
            print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
            print("=" * 60)

    elif args.mode == 'dashboard':
        # Run dashboard
        system.run_dashboard()

    else:
        # Run trading system
        system.run()


if __name__ == "__main__":
    main()