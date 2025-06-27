# main.py

import sys
import os
import logging
import argparse
from datetime import datetime
import time
import threading
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configurations
from config.settings import Config
from config.watchlist import WATCHLIST

# Import core modules
from data.database import Database
from data.market_data import MarketDataManager
from data.cache_manager import CacheManager

# Import models
from models.ensemble import EnsembleModel
from models.regime_detector import MarketRegimeDetector

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


# Temporary dashboard placeholder for GPU testing
class TradingDashboard:
    """Temporary placeholder for dashboard"""

    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        print("Dashboard is temporarily disabled. Testing GPU monitoring...")
        print("To test GPU monitoring, run: python main.py --mode train")

#from monitoring.dashboard import TradingDashboard as TradingDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check and log GPU availability for TensorFlow and PyTorch"""

    # Check TensorFlow GPU
    print("\n" + "=" * 50)
    print("GPU AVAILABILITY CHECK")
    print("=" * 50)

    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")

        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPUs available: {len(gpus)}")

        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")

            # Check if TensorFlow is built with CUDA
            print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

            # Test GPU with simple operation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"TensorFlow GPU test successful: {c.device}")
        else:
            print("No TensorFlow GPUs found!")

    except Exception as e:
        print(f"TensorFlow GPU check failed: {e}")

    # Check PyTorch GPU
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"PyTorch GPU count: {torch.cuda.device_count()}")
            print(f"PyTorch GPU name: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

            # Test PyTorch GPU
            x = torch.rand(5, 3).cuda()
            print(f"PyTorch GPU test successful: {x.device}")
        else:
            print("No PyTorch CUDA support!")

    except Exception as e:
        print(f"PyTorch GPU check failed: {e}")

    print("=" * 50 + "\n")


class SmartSwingTrader:
    """Main trading system orchestrator"""

    def __init__(self, mode='paper'):
        """
        Initialize the trading system

        Args:
            mode: 'paper' for paper trading, 'live' for real trading
        """
        self.mode = mode
        self.is_running = False

        logger.info(f"Initializing Smart Swing Trader in {mode} mode...")

        # Initialize components
        self._initialize_components()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_components(self):
        """Initialize all system components"""
        # Database
        self.db = Database()
        logger.info("Database initialized")

        # Broker interface
        self.broker = BrokerInterface()
        logger.info("Broker connection established")

        # Market data
        self.market_data = MarketDataManager()
        self.cache = CacheManager()
        logger.info("Market data manager initialized")

        # ML models
        self.ml_model = EnsembleModel(use_gpu=Config.USE_GPU)
        self.regime_detector = MarketRegimeDetector()
        logger.info("ML models initialized")

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
        logger.info("Risk management initialized")

        # Order management
        self.order_manager = OrderManager(self.broker.api)
        logger.info("Order manager initialized")

        # Performance tracking
        self.performance_tracker = PerformanceTracker(account_value)
        logger.info("Performance tracker initialized")

        # Dashboard
        self.dashboard = TradingDashboard(
            self.broker, self.db, self.performance_tracker
        )
        logger.info("Dashboard initialized")

    def _get_account_value(self):
        """Get current account value"""
        account_info = self.broker.get_account_info()
        return float(account_info.get('portfolio_value', Config.ACCOUNT_SIZE))

    def train_models(self):
        """Train ML models on historical data"""
        logger.info("Starting model training...")

        # Use ALL symbols in watchlist for training
        training_symbols = WATCHLIST  # Use all 198 symbols
        training_data = {}
        failed_symbols = []

        # Import yfinance for better historical data access
        import yfinance as yf

        logger.info(f"Attempting to fetch data for {len(training_symbols)} symbols...")

        for i, symbol in enumerate(training_symbols, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(training_symbols)} symbols processed...")

            try:
                # Get 2 years of data using yfinance
                ticker = yf.Ticker(symbol)
                bars = ticker.history(period="2y")

                if not bars.empty:
                    # Standardize column names to lowercase
                    bars.columns = [col.lower() for col in bars.columns]

                    # Ensure we have the required OHLCV columns
                    if all(col in bars.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                        bars = bars[['open', 'high', 'low', 'close', 'volume']]

                        # Check if we have enough data (at least 200 days)
                        if len(bars) > 200:
                            training_data[symbol] = bars
                        else:
                            logger.debug(f"Insufficient data for {symbol}: only {len(bars)} days")
                            failed_symbols.append((symbol, f"Only {len(bars)} days"))
                    else:
                        logger.debug(f"Missing required columns for {symbol}")
                        failed_symbols.append((symbol, "Missing columns"))
                else:
                    logger.debug(f"No data returned for {symbol}")
                    failed_symbols.append((symbol, "No data"))

            except Exception as e:
                logger.debug(f"Error fetching data for {symbol}: {e}")
                # Try using Alpaca as fallback
                try:
                    bars = self.market_data.get_bars(
                        symbol, '1Day', limit=1000
                    )
                    if not bars.empty and len(bars) > 200:
                        training_data[symbol] = bars
                    else:
                        failed_symbols.append((symbol, str(e)))
                except Exception as e2:
                    logger.debug(f"Alpaca fallback also failed for {symbol}: {e2}")
                    failed_symbols.append((symbol, f"Both sources failed: {str(e2)[:50]}"))

        # Log summary
        logger.info(f"\nData fetching complete:")
        logger.info(f"- Total symbols: {len(training_symbols)}")
        logger.info(f"- Successfully fetched: {len(training_data)}")
        logger.info(f"- Failed: {len(failed_symbols)}")
        logger.info(f"- Success rate: {len(training_data) / len(training_symbols) * 100:.1f}%")

        if failed_symbols:
            logger.info(f"\nFailed symbols: {[s[0] for s in failed_symbols[:10]]}..." if len(
                failed_symbols) > 10 else f"Failed symbols: {[s[0] for s in failed_symbols]}")

        # Require at least 95% of symbols to have valid data
        min_symbols_required = int(len(training_symbols) * 0.95)

        if len(training_data) < min_symbols_required:
            logger.error(
                f"Insufficient training data: got {len(training_data)} symbols, need at least {min_symbols_required} (95%)")
            logger.info("Consider lowering the threshold or checking data sources")
            return False

        # Train the model
        logger.info(f"\nTraining model on {len(training_data)} symbols...")
        self.ml_model.train(training_data)

        # Save the trained model
        self.ml_model.save_models('models/saved')
        logger.info("Model training completed and saved")

        return True

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("Starting trading cycle...")

        try:
            # 1. Check market status
            market_status = self.broker.get_market_status()
            if not market_status['is_open']:
                logger.info("Market is closed")
                return

            # 2. Update risk metrics
            self.risk_manager.reset_daily_metrics()

            # 3. Check if we should stop trading
            should_stop, reason = self.risk_manager.should_stop_trading()
            if should_stop:
                logger.warning(f"Trading halted: {reason}")
                return

            # 4. Update existing positions
            self._update_positions()

            # 5. Generate new signals
            signals = self._scan_for_opportunities()

            # 6. Execute signals
            self._execute_signals(signals)

            # 7. Update performance metrics
            self._update_performance()

            logger.info("Trading cycle completed")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

    def _update_positions(self):
        """Update existing positions"""
        logger.info("Updating positions...")

        positions = self.broker.get_positions()

        if not positions:
            logger.info("No open positions")
            return

        # Get current prices
        symbols = [p['symbol'] for p in positions]
        current_prices = self.market_data.get_current_prices_batch(symbols)

        # Check for exit signals
        market_data = {}
        for symbol in symbols:
            bars = self.market_data.get_bars(symbol, '1Day', limit=100)
            if not bars.empty:
                market_data[symbol] = bars

        exit_signals = self.signal_generator.get_exit_signals(
            positions, market_data
        )

        # Update stops
        positions_with_prices = []
        for pos in positions:
            if pos['symbol'] in current_prices:
                pos['current_price'] = current_prices[pos['symbol']]
                positions_with_prices.append(pos)

        updated_positions = self.stop_loss_manager.update_position_stops_batch(
            positions_with_prices, current_prices
        )

        # Execute exits
        for signal in exit_signals:
            logger.info(f"Exit signal for {signal['symbol']}: {signal['reasons']}")
            self.order_manager.create_exit_order(signal['position'], signal)

        # Update stop orders
        for pos in updated_positions:
            if pos.get('stop_moved'):
                self.order_manager.create_stop_loss_order(
                    pos, pos['stop_loss_price']
                )

    def _scan_for_opportunities(self):
        """Scan for new trading opportunities"""
        logger.info("Scanning for opportunities...")

        # Get current positions
        positions = self.broker.get_positions()
        current_symbols = [p['symbol'] for p in positions]

        # Filter watchlist
        available_symbols = [s for s in WATCHLIST if s not in current_symbols]

        # Limit scan based on available positions
        max_new_positions = Config.MAX_POSITIONS - len(positions)
        if max_new_positions <= 0:
            logger.info("Maximum positions reached")
            return []

        # Get market regime
        spy_data = self.market_data.get_bars('SPY', '1Day', limit=100)
        regime_info = self.regime_detector.detect_regime(spy_data)
        market_regime = regime_info['regime']

        # Batch fetch market data
        scan_symbols = available_symbols[:50]  # Limit for efficiency
        market_data = {}

        for symbol in scan_symbols:
            bars = self.market_data.get_bars(symbol, '1Day', limit=100)
            if not bars.empty:
                market_data[symbol] = bars

        # Generate signals
        signals = self.signal_generator.generate_signals(
            scan_symbols, market_data, market_regime
        )

        # Filter by confidence
        high_conf_signals = [s for s in signals
                             if s['confidence'] >= Config.MIN_CONFIDENCE]

        logger.info(f"Found {len(high_conf_signals)} high confidence signals")

        return high_conf_signals[:max_new_positions]

    def _execute_signals(self, signals):
        """Execute trading signals"""
        if not signals:
            return

        account_info = self.broker.get_account_info()
        buying_power = float(account_info['buying_power'])

        for signal in signals:
            # Validate signal
            validation = self.signal_generator.validate_signal(
                signal,
                self.broker.get_positions(),
                self._get_account_value()
            )

            if not validation['valid']:
                logger.warning(f"Signal validation failed for {signal['symbol']}: "
                               f"{validation['warnings']}")
                continue

            # Check risk approval
            risk_check = self.risk_manager.check_trade_approval(
                signal,
                self.broker.get_positions()
            )

            if not risk_check['approved']:
                logger.warning(f"Risk check failed for {signal['symbol']}: "
                               f"{risk_check['reasons']}")
                continue

            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                signal,
                self.broker.get_positions(),
                market_regime=signal.get('market_regime')
            )

            if position_size['shares'] == 0:
                logger.warning(f"Position size is zero for {signal['symbol']}")
                continue

            # Check buying power
            if position_size['position_value'] > buying_power:
                logger.warning(f"Insufficient buying power for {signal['symbol']}")
                continue

            # Create order
            logger.info(f"Executing signal for {signal['symbol']}: "
                        f"{position_size['shares']} shares")

            result = self.order_manager.create_entry_order(signal, position_size)

            if result['success']:
                # Update buying power
                buying_power -= position_size['position_value']

                # Save signal to database
                self.db.save_signal(signal)
            else:
                logger.error(f"Failed to execute order for {signal['symbol']}: "
                             f"{result.get('error')}")

    def _update_performance(self):
        """Update performance metrics"""
        # Get recent filled orders
        filled_orders = self.order_manager.get_filled_orders(lookback_hours=24)

        for order in filled_orders:
            # Check if this is a closing trade
            if order['side'] == 'sell':
                # Find the opening trade
                # This is simplified - in production, track trade pairs properly
                trade = {
                    'symbol': order['symbol'],
                    'entry_price': order.get('entry_price', order['filled_price']),
                    'exit_price': order['filled_price'],
                    'shares': order['filled_qty'],
                    'entry_time': order.get('entry_time', order['created_at']),
                    'exit_time': order['filled_at'],
                    'pnl': (order['filled_price'] - order.get('entry_price', order['filled_price'])) * order[
                        'filled_qty']
                }

                self.performance_tracker.record_trade(trade)
                self.risk_manager.record_trade_result(trade)

    def run(self):
        """Main run loop"""
        self.is_running = True
        logger.info("Smart Swing Trader started")

        # Initial setup
        if not os.path.exists('models/saved'):
            logger.info("No saved models found, training new models...")
            if not self.train_models():
                logger.error("Model training failed")
                return
        else:
            logger.info("Loading saved models...")
            self.ml_model.load_models('models/saved')

        # Main loop
        cycle_count = 0

        while self.is_running:
            try:
                # Run trading cycle
                self.run_trading_cycle()

                # Update order status
                self.order_manager.update_order_status()

                # Cancel stale orders
                if cycle_count % 12 == 0:  # Every hour
                    self.order_manager.cancel_stale_orders()

                # Save state periodically
                if cycle_count % 60 == 0:  # Every 5 minutes
                    self._save_state()

                # Generate reports
                if cycle_count % 720 == 0:  # Every hour
                    self._generate_reports()

                cycle_count += 1

                # Sleep for 5 seconds
                time.sleep(5)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(30)  # Wait before retrying

        self._shutdown()

    def _save_state(self):
        """Save system state"""
        try:
            # Save performance data
            self.performance_tracker.save_performance_data('data/performance.json')

            # Backup database
            self.db.backup_database()

            logger.info("System state saved")

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _generate_reports(self):
        """Generate performance reports"""
        try:
            # Performance report
            perf_report = self.performance_tracker.generate_performance_report()
            logger.info("\n" + perf_report)

            # Risk report
            risk_report = self.risk_manager.generate_risk_report()
            logger.info("\n" + risk_report)

            # Signal report
            signal_report = self.signal_generator.generate_signal_report()
            logger.info("\n" + signal_report)

        except Exception as e:
            logger.error(f"Error generating reports: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.is_running = False

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Smart Swing Trader...")

        # Cancel all open orders
        self.order_manager.shutdown()

        # Save final state
        self._save_state()

        # Generate final reports
        self._generate_reports()

        logger.info("Shutdown complete")

    def run_dashboard(self):
        """Run the web dashboard"""
        logger.info("Starting dashboard...")
        self.dashboard.run()


def main():
    """Main entry point"""
    # Run GPU check first
    check_gpu_availability()

    parser = argparse.ArgumentParser(description='Smart Swing Trading System')
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest', 'train', 'dashboard'],
        default='paper',
        help='Trading mode'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to trade (overrides watchlist)'
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/backups', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)

    # Initialize trader
    trader = SmartSwingTrader(mode=args.mode)

    # Override watchlist if specified
    if args.symbols:
        global WATCHLIST
        WATCHLIST = args.symbols

    # Run based on mode
    if args.mode == 'train':
        trader.train_models()
    elif args.mode == 'dashboard':
        trader.run_dashboard()
    elif args.mode == 'backtest':
        logger.info("Backtest mode not implemented yet")
    else:
        trader.run()


if __name__ == "__main__":
    main()