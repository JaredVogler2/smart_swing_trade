# main_enhanced.py

import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
import time
import threading
import signal
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional, Union

# GPU monitoring
import pynvml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configurations
from config.settings import Config
from config.watchlist import WATCHLIST

# Import core modules
from data.database import Database
from data.market_data import MarketDataManager
from data.cache_manager import CacheManager

# Import enhanced models
from models.ensemble_gpu_windows import GPUEnsembleModel
from models.regime_detector import MarketRegimeDetector
from models.features import FeatureEngineer

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

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU usage and performance"""

    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_metrics(self):
        """Get current GPU metrics"""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts

            return {
                'utilization': util.gpu,
                'memory_used': mem_info.used / 1024 ** 3,  # GB
                'memory_total': mem_info.total / 1024 ** 3,  # GB
                'temperature': temp,
                'power': power,
                'memory_percent': (mem_info.used / mem_info.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return None


class EnhancedSmartSwingTrader:
    """Enhanced trading system with GPU acceleration and advanced features"""

    def __init__(self, mode='paper'):
        """
        Initialize the enhanced trading system

        Args:
            mode: 'paper' for paper trading, 'live' for real trading
        """
        self.mode = mode
        self.is_running = False
        self.gpu_monitor = GPUMonitor()

        logger.info(f"Initializing Enhanced Smart Swing Trader in {mode} mode...")

        # Performance metrics
        self.cycle_times = []
        self.prediction_times = []

        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.signal_queue = queue.PriorityQueue()

        # Initialize components
        self._initialize_components()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_components(self):
        """Initialize all system components with enhancements"""
        # Database with connection pooling
        self.db = Database()
        logger.info("Database initialized")

        # Enhanced broker interface with WebSocket support
        self.broker = BrokerInterface()
        self._setup_broker_websocket()
        logger.info("Broker connection established")

        # Market data with parallel fetching
        self.market_data = MarketDataManager()
        self.cache = CacheManager()
        logger.info("Market data manager initialized")

        # GPU-accelerated ML models
        gpu_metrics = self.gpu_monitor.get_metrics()
        available_memory = gpu_metrics['memory_total'] - gpu_metrics['memory_used'] if gpu_metrics else 8

        self.ml_model = GPUEnsembleModel(
            max_gpu_memory_mb=int(available_memory * 1024 * 0.8)  # Use 80% of available
        )
        self.regime_detector = MarketRegimeDetector()
        logger.info(f"ML models initialized with {available_memory:.1f}GB available GPU memory")

        # Enhanced analysis modules
        self.news_analyzer = NewsSentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.signal_generator = SignalGenerator(self.ml_model, self.news_analyzer)
        logger.info("Analysis modules initialized")

        # Dynamic risk management
        account_value = self._get_account_value()
        self.position_sizer = PositionSizer(account_value)
        self.risk_manager = RiskManager(account_value)
        self.stop_loss_manager = StopLossManager()
        logger.info("Risk management initialized")

        # Enhanced order management with smart routing
        self.order_manager = OrderManager(self.broker.api)
        logger.info("Order manager initialized")

        # Real-time performance tracking
        self.performance_tracker = PerformanceTracker(account_value)
        logger.info("Performance tracker initialized")

        # Enhanced dashboard with WebSocket updates
        self.dashboard = TradingDashboard(
            self.broker, self.db, self.performance_tracker
        )
        self._setup_dashboard_websocket()
        logger.info("Dashboard initialized")

    def _get_account_value(self):
        """Get current account value"""
        try:
            account_info = self.broker.get_account_info()
            return float(account_info.get('portfolio_value', Config.ACCOUNT_SIZE))
        except Exception as e:
            logger.warning(f"Could not get account value: {e}")
            return Config.ACCOUNT_SIZE

    def _setup_broker_websocket(self):
        """Setup WebSocket connection for real-time data"""
        # This would connect to Alpaca's WebSocket for real-time updates
        pass

    def _setup_dashboard_websocket(self):
        """Setup WebSocket for dashboard updates"""
        # This would setup WebSocket server for real-time dashboard updates
        pass

    # Replace the train_models_async method in main_enhanced.py with this fixed version

    async def train_models_async(self):
        """Train ML models asynchronously with GPU optimization"""
        logger.info("Starting async model training...")

        # Use larger symbol set for serious quant trading
        training_symbols = WATCHLIST[:200] if len(WATCHLIST) >= 200 else WATCHLIST
        logger.info(f"Training on {len(training_symbols)} symbols")

        # Fetch training data in parallel
        symbols_chunks = [training_symbols[i:i + 10] for i in range(0, len(training_symbols), 10)]

        async def fetch_symbol_data(symbols, executor):
            """Fetch data for a chunk of symbols"""
            data = {}
            for symbol in symbols:
                try:
                    # Use the passed executor
                    bars = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda s=symbol: self.market_data.get_bars(s, '1Day', 1000)
                    )
                    if bars is not None and not bars.empty and len(bars) > 500:
                        data[symbol] = bars
                        logger.info(f"[OK] {symbol}: {len(bars)} days of data")
                    else:
                        logger.warning(f"[SKIP] {symbol}: Insufficient data")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            return data

        # Create tasks with the executor
        tasks = [fetch_symbol_data(chunk, self.executor) for chunk in symbols_chunks]
        results = await asyncio.gather(*tasks)

        # Combine results
        training_data = {}
        for result in results:
            training_data.update(result)

        logger.info(f"Fetched data for {len(training_data)} symbols")

        if len(training_data) < 30:
            logger.error("Insufficient training data")
            return False

        # Train model with GPU acceleration
        logger.info("Starting GPU training with advanced features...")

        # The model will create all the advanced features including:
        # - Golden/Death crosses
        # - Support/Resistance breakouts
        # - Volume-price divergences
        # - Market microstructure features
        # - 200+ technical indicators
        # - Feature interactions

        await asyncio.get_event_loop().run_in_executor(
            None,
            self.ml_model.train,
            training_data,
            None,
            True  # Use multi-GPU if available
        )

        # Save model
        self.ml_model.save_models('models/saved')
        logger.info("Model training completed and saved")

        # Log feature importance
        if hasattr(self.ml_model, 'feature_importance') and self.ml_model.feature_importance is not None:
            logger.info("\nTop 20 Most Important Features:")
            for feat, importance in self.ml_model.feature_importance.head(20).items():
                logger.info(f"  {feat}: {importance:.4f}")

        return True

    async def run_trading_cycle_async(self):
        """Run trading cycle with async operations"""
        cycle_start = time.time()
        logger.info("Starting async trading cycle...")

        try:
            # Check market status
            market_status = await self._check_market_status_async()
            if not market_status['is_open']:
                logger.info("Market is closed")
                return

            # Update GPU metrics
            gpu_metrics = self.gpu_monitor.get_metrics()
            if gpu_metrics:
                logger.info(f"GPU Status - Util: {gpu_metrics['utilization']}%, "
                            f"Memory: {gpu_metrics['memory_used']:.1f}GB/{gpu_metrics['memory_total']:.1f}GB, "
                            f"Temp: {gpu_metrics['temperature']}°C")

            # Run tasks in parallel
            tasks = [
                self._update_positions_async(),
                self._scan_for_opportunities_async(),
                self._update_news_sentiment_async(),
                self._check_market_regime_async()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            position_updates = results[0] if not isinstance(results[0], Exception) else None
            signals = results[1] if not isinstance(results[1], Exception) else []
            news_sentiment = results[2] if not isinstance(results[2], Exception) else None
            market_regime = results[3] if not isinstance(results[3], Exception) else None

            # Execute signals with priority queue
            if signals:
                await self._execute_signals_async(signals, market_regime)

            # Update performance
            self._update_performance()

            # Update dashboard
            await self._update_dashboard_async()

            # Log cycle time
            cycle_time = time.time() - cycle_start
            self.cycle_times.append(cycle_time)
            logger.info(f"Trading cycle completed in {cycle_time:.2f}s")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

    async def _scan_for_opportunities_async(self):
        """Scan for opportunities using parallel processing"""
        logger.info("Scanning for opportunities with GPU acceleration...")

        # Get current positions
        positions = self.broker.get_positions()
        current_symbols = [p['symbol'] for p in positions]

        # Filter available symbols
        available_symbols = [s for s in WATCHLIST if s not in current_symbols]
        max_new_positions = Config.MAX_POSITIONS - len(positions)

        if max_new_positions <= 0:
            logger.info("Maximum positions reached")
            return []

        # Batch process symbols in parallel on GPU
        scan_symbols = available_symbols[:100]  # Increased limit
        symbol_chunks = [scan_symbols[i:i + 20] for i in range(0, len(scan_symbols), 20)]

        async def process_symbol_chunk(symbols):
            """Process a chunk of symbols on GPU"""
            chunk_signals = []

            # Fetch data in parallel
            data_tasks = []
            for symbol in symbols:
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.market_data.get_bars,
                    symbol, '1Day', 200
                )
                data_tasks.append(task)

            bars_list = await asyncio.gather(*data_tasks, return_exceptions=True)

            # Process valid data
            valid_data = {}
            for symbol, bars in zip(symbols, bars_list):
                if not isinstance(bars, Exception) and not bars.empty:
                    valid_data[symbol] = bars

            if valid_data:
                # Run predictions on GPU in batch
                predictions = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._batch_predict,
                    valid_data
                )

                # Generate signals from predictions
                for symbol, prediction in predictions.items():
                    if prediction['confidence'] >= Config.MIN_CONFIDENCE:
                        signal = {
                            'symbol': symbol,
                            'type': 'BUY' if prediction['prediction'] == 1 else 'HOLD',
                            'confidence': prediction['confidence'],
                            'probability': prediction['probability'],
                            'expected_return': prediction['expected_return'],
                            'risk_score': prediction['risk_score'],
                            'market_context': prediction['market_context'],
                            'timestamp': datetime.now()
                        }
                        chunk_signals.append(signal)

            return chunk_signals

        # Process all chunks in parallel
        chunk_tasks = [process_symbol_chunk(chunk) for chunk in symbol_chunks]
        chunk_results = await asyncio.gather(*chunk_tasks)

        # Combine and sort signals
        all_signals = []
        for chunk_signals in chunk_results:
            all_signals.extend(chunk_signals)

        # Sort by confidence and expected return
        all_signals.sort(key=lambda x: (x['confidence'], x['expected_return']), reverse=True)

        logger.info(f"Found {len(all_signals)} signals above confidence threshold")

        return all_signals[:max_new_positions]

    def _batch_predict(self, symbol_data: Dict) -> Dict:
        """Batch predict multiple symbols on GPU"""
        predictions = {}

        for symbol, data in symbol_data.items():
            try:
                pred_start = time.time()
                prediction = self.ml_model.predict(symbol, data)
                pred_time = time.time() - pred_start
                self.prediction_times.append(pred_time)
                predictions[symbol] = prediction
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")

        return predictions

    async def _execute_signals_async(self, signals: List[Dict], market_regime: Dict):
        """Execute signals with advanced order routing"""
        if not signals:
            return

        account_info = self.broker.get_account_info()
        buying_power = float(account_info['buying_power'])

        # Priority queue for signal execution
        for signal in signals:
            # Calculate priority based on multiple factors
            priority = self._calculate_signal_priority(signal, market_regime)
            self.signal_queue.put((-priority, signal))  # Negative for max heap

        # Execute signals in priority order
        executed = 0
        while not self.signal_queue.empty() and executed < Config.MAX_POSITIONS:
            _, signal = self.signal_queue.get()

            # Enhanced validation
            validation = await self._validate_signal_async(signal)
            if not validation['valid']:
                logger.warning(f"Signal validation failed for {signal['symbol']}: {validation['reasons']}")
                continue

            # Dynamic position sizing based on confidence and market regime
            position_size = self._calculate_dynamic_position_size(
                signal, market_regime, buying_power
            )

            if position_size['shares'] == 0:
                continue

            # Smart order routing
            order_params = self._determine_smart_order_params(signal, position_size)

            # Execute order
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.order_manager.create_entry_order,
                signal, position_size
            )

            if result['success']:
                executed += 1
                buying_power -= position_size['position_value']
                self.db.save_signal(signal)
                logger.info(f"Executed signal for {signal['symbol']}: {position_size['shares']} shares")

    def _calculate_signal_priority(self, signal: Dict, market_regime: Dict) -> float:
        """Calculate signal execution priority"""
        # Base priority from confidence and expected return
        base_priority = signal['confidence'] * abs(signal['expected_return'])

        # Adjust for risk
        risk_adjustment = 1 - signal['risk_score']

        # Market regime adjustment
        regime_multiplier = {
            'bull': 1.2,
            'bear': 0.8,
            'neutral': 1.0
        }.get(market_regime.get('regime', 'neutral'), 1.0)

        # Momentum bonus
        momentum_bonus = 0
        if signal['market_context'].get('regime') == 'strong_uptrend':
            momentum_bonus = 0.1

        return base_priority * risk_adjustment * regime_multiplier + momentum_bonus

    def _calculate_dynamic_position_size(self, signal: Dict,
                                         market_regime: Dict,
                                         buying_power: float) -> Dict:
        """Calculate position size with Kelly Criterion and regime adjustment"""
        # Get base position size
        base_size = self.position_sizer.calculate_position_size(
            signal, self.broker.get_positions(), market_regime
        )

        # Apply Kelly Criterion for optimal sizing
        win_probability = signal['probability']
        avg_win = 0.06  # 6% target
        avg_loss = 0.03  # 3% stop loss

        kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Adjust for market regime
        regime_adjustment = {
            'bull': 1.1,
            'bear': 0.7,
            'neutral': 0.9
        }.get(market_regime.get('regime', 'neutral'), 0.9)

        # Calculate final position value
        position_value = min(
            base_size['position_value'] * kelly_fraction * regime_adjustment,
            buying_power * 0.3,  # Max 30% of buying power
            Config.MAX_POSITION_SIZE
        )

        # Calculate shares
        current_price = signal['market_context']['current_price']
        shares = int(position_value / current_price)

        return {
            'shares': shares,
            'position_value': shares * current_price,
            'kelly_fraction': kelly_fraction,
            'regime_adjustment': regime_adjustment
        }

    async def _update_dashboard_async(self):
        """Update dashboard with real-time data"""
        dashboard_data = {
            'portfolio': self.broker.get_account_info(),
            'positions': self.broker.get_positions(),
            'performance': self.performance_tracker.get_current_metrics(),
            'gpu_metrics': self.gpu_monitor.get_metrics(),
            'active_signals': list(self.signal_queue.queue)[:10],  # Top 10 signals
            'market_regime': await self._check_market_regime_async(),
            'cycle_times': self.cycle_times[-20:],  # Last 20 cycles
            'prediction_times': self.prediction_times[-50:]  # Last 50 predictions
        }

        # Send update via WebSocket (implementation depends on dashboard setup)
        # self.dashboard.update(dashboard_data)

    async def run_async(self):
        """Main async run loop"""
        self.is_running = True
        logger.info("Enhanced Smart Swing Trader started")

        # Initial setup
        if not os.path.exists('models/saved'):
            logger.info("No saved models found, training new models...")
            success = await self.train_models_async()
            if not success:
                logger.error("Model training failed")
                return
        else:
            logger.info("Loading saved models...")
            self.ml_model.load_models('models/saved')

        # Create async tasks for different components
        tasks = []

        # Main trading loop
        async def trading_loop():
            cycle_count = 0
            while self.is_running:
                try:
                    await self.run_trading_cycle_async()

                    # Periodic tasks
                    if cycle_count % 12 == 0:  # Every minute
                        await self._update_order_status_async()

                    if cycle_count % 60 == 0:  # Every 5 minutes
                        await self._save_state_async()

                    if cycle_count % 720 == 0:  # Every hour
                        await self._generate_reports_async()
                        await self._retrain_models_if_needed()

                    cycle_count += 1
                    await asyncio.sleep(5)  # 5 second intervals

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    await asyncio.sleep(30)

        # News monitoring loop
        async def news_loop():
            while self.is_running:
                try:
                    await self._monitor_news_async()
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in news loop: {e}")
                    await asyncio.sleep(300)

        # Performance monitoring loop
        async def performance_loop():
            while self.is_running:
                try:
                    await self._monitor_performance_async()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in performance loop: {e}")
                    await asyncio.sleep(60)

        # Start all loops
        tasks = [
            asyncio.create_task(trading_loop()),
            asyncio.create_task(news_loop()),
            asyncio.create_task(performance_loop())
        ]

        # Wait for all tasks
        await asyncio.gather(*tasks)

    def run(self):
        """Main entry point"""
        asyncio.run(self.run_async())

    async def _check_market_status_async(self):
        """Check market status asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.broker.get_market_status
        )

    async def _update_positions_async(self):
        """Update positions asynchronously"""
        positions = self.broker.get_positions()
        if not positions:
            return None

        # Process position updates in parallel
        update_tasks = []
        for position in positions:
            task = self._process_position_update(position)
            update_tasks.append(task)

        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def _process_position_update(self, position):
        """Process individual position update"""
        symbol = position['symbol']

        # Get latest data
        bars = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.market_data.get_bars,
            symbol, '1Day', 100
        )

        if bars.empty:
            return None

        # Check for exit signals
        exit_signal = await asyncio.get_event_loop().run_in_executor(
            None,
            self.signal_generator.check_exit_conditions,
            position, bars
        )

        if exit_signal:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.order_manager.create_exit_order,
                position, exit_signal
            )

        # Update stop loss
        current_price = bars['close'].iloc[-1]
        stop_update = self.stop_loss_manager.update_position_stop(
            position, current_price
        )

        if stop_update.get('stop_moved'):
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.order_manager.create_stop_loss_order,
                position, stop_update['stop_loss_price']
            )

        return {
            'symbol': symbol,
            'exit_signal': exit_signal,
            'stop_update': stop_update
        }

    async def _monitor_news_async(self):
        """Monitor news sentiment asynchronously"""
        # Get news for watchlist symbols
        news_tasks = []
        for symbol in WATCHLIST[:50]:  # Top 50 symbols
            task = asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.news_analyzer.analyze_symbol_news,
                symbol, 24
            )
            news_tasks.append(task)

        results = await asyncio.gather(*news_tasks, return_exceptions=True)

        # Process high-impact news
        for result in results:
            if isinstance(result, Exception):
                continue

            if result.get('recommendation') in ['strong_buy', 'strong_sell']:
                logger.info(f"High impact news for {result['symbol']}: "
                            f"{result['recommendation']} ({result['confidence']:.2f})")

    async def _monitor_performance_async(self):
        """Monitor performance metrics asynchronously"""
        metrics = self.performance_tracker.get_current_metrics()

        # Check for risk limits
        if metrics.get('daily_loss', 0) > Config.MAX_DAILY_LOSS:
            logger.warning(f"Daily loss limit reached: ${metrics['daily_loss']}")
            self.risk_manager.halt_trading()

        if metrics.get('drawdown', 0) > Config.MAX_DRAWDOWN:
            logger.warning(f"Max drawdown reached: {metrics['drawdown']:.2%}")
            self.risk_manager.reduce_position_sizes()

    async def _retrain_models_if_needed(self):
        """Check if models need retraining"""
        last_train_date = self.ml_model.training_history[-1]['date'] if self.ml_model.training_history else None

        if not last_train_date or (datetime.now() - last_train_date).days > Config.RETRAIN_DAYS:
            logger.info("Initiating model retraining...")
            await self.train_models_async()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.is_running = False

    async def _shutdown_async(self):
        """Graceful async shutdown"""
        logger.info("Shutting down Enhanced Smart Swing Trader...")

        # Cancel all open orders
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.order_manager.shutdown
        )

        # Save final state
        await self._save_state_async()

        # Generate final reports
        await self._generate_reports_async()

        # Cleanup
        self.executor.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Smart Swing Trading System')
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest', 'train', 'dashboard'],
        default='paper',
        help='Trading mode'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Force GPU usage'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to trade'
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/backups', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)

    # Override GPU setting
    if args.gpu:
        Config.USE_GPU = True

    # Initialize trader
    trader = EnhancedSmartSwingTrader(mode=args.mode)

    # Override watchlist if specified
    if args.symbols:
        global WATCHLIST
        WATCHLIST = args.symbols

    # Run based on mode
    if args.mode == 'train':
        asyncio.run(trader.train_models_async())
    elif args.mode == 'dashboard':
        trader.dashboard.run()
    elif args.mode == 'backtest':
        logger.info("Backtest mode - coming soon")
    else:
        trader.run()


if __name__ == "__main__":
    main()