# config/settings.py
# COMPLETE FILE - REPLACE YOUR ENTIRE settings.py WITH THIS

import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Complete trading system configuration"""

    # ==================== ACCOUNT SETTINGS ====================
    ACCOUNT_SIZE = 100000  # Starting account size in USD
    MAX_POSITIONS = 10  # Maximum number of concurrent positions
    MIN_POSITION_SIZE = 1000  # Minimum position size in USD
    MAX_POSITION_SIZE_PCT = 0.1  # Maximum 10% of portfolio per position

    # ==================== TRADING PARAMETERS ====================
    MIN_CONFIDENCE = 0.65  # Minimum ML model confidence for trades
    MIN_EXPECTED_RETURN = 0.03  # Minimum 3% expected return
    PREDICTION_HORIZON = 3  # Days ahead to predict
    SWING_TRADE_DURATION = 3  # Target holding period in days

    # ==================== RISK MANAGEMENT ====================
    MAX_DAILY_LOSS = 0.03  # 3% daily loss limit
    MAX_WEEKLY_LOSS = 0.05  # 5% weekly loss limit
    MAX_MONTHLY_LOSS = 0.10  # 10% monthly loss limit
    MAX_DRAWDOWN = 0.20  # 20% maximum drawdown
    DEFAULT_STOP_LOSS = 0.05  # 5% stop loss
    DEFAULT_TAKE_PROFIT = 0.15  # 15% take profit
    TRAILING_STOP_ACTIVATION = 0.05  # Activate trailing stop at 5% profit
    TRAILING_STOP_DISTANCE = 0.03  # Trail by 3%

    # Risk per trade
    RISK_PER_TRADE = 0.02  # Risk 2% of account per trade
    MAX_CORRELATION = 0.7  # Maximum correlation between positions
    MAX_SECTOR_EXPOSURE = 0.3  # Maximum 30% in one sector

    # ==================== ML MODEL SETTINGS ====================
    SEQUENCE_LENGTH = 60  # Days of history for LSTM
    PREDICTION_THRESHOLD = 0.6  # Threshold for positive prediction
    FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Minimum feature importance

    # Model retraining
    RETRAIN_FREQUENCY = 'weekly'  # 'daily', 'weekly', 'monthly'
    MIN_TRAINING_SAMPLES = 10000  # Minimum samples for training
    VALIDATION_SPLIT = 0.2  # 20% validation split

    # ==================== GPU SETTINGS ====================
    USE_GPU = True  # Enable GPU usage
    REQUIRE_GPU = True  # If True, will fail if GPU not available (no CPU fallback)
    GPU_DEVICE_ID = 0  # Which GPU to use (if multiple)
    GPU_MEMORY_FRACTION = 0.8  # Use up to 80% of GPU memory

    # GPU Monitoring
    LOG_GPU_STATS_INTERVAL = 100  # Log GPU stats every N batches
    GPU_MEMORY_WARNING_THRESHOLD = 0.9  # Warn when GPU memory usage exceeds 90%
    GPU_MEMORY_CRITICAL_THRESHOLD = 0.95  # Critical alert at 95%

    # ==================== ALERT SETTINGS ====================
    ALERT_EMAIL = os.getenv('ALERT_EMAIL', None)  # Email for critical alerts
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')  # Use app-specific password

    # Alert thresholds
    SEND_EMAIL_ALERTS = True  # Enable email alerts
    ALERT_ON_GPU_FAILURE = True
    ALERT_ON_LARGE_LOSS = True
    ALERT_ON_SYSTEM_ERROR = True

    # ==================== DATA SETTINGS ====================
    DATA_PROVIDER = 'alpaca'  # 'alpaca', 'polygon', 'yfinance'
    DATA_TIMEFRAME = '1Day'  # Primary timeframe for analysis
    INTRADAY_TIMEFRAME = '5Min'  # For intraday analysis

    # Historical data requirements
    MIN_HISTORY_DAYS = 252  # 1 year minimum
    PREFERRED_HISTORY_DAYS = 504  # 2 years preferred

    # Data quality
    MAX_MISSING_DATA_PCT = 0.05  # Maximum 5% missing data
    MAX_PRICE_SPIKE_PCT = 0.20  # Flag 20% price spikes as anomalies

    # ==================== BROKER SETTINGS ====================
    # Alpaca API (get from https://alpaca.markets/)
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Order settings
    ORDER_TIME_IN_FORCE = 'day'  # 'day', 'gtc', 'ioc', 'fok'
    USE_LIMIT_ORDERS = True  # Use limit orders vs market orders
    LIMIT_ORDER_BUFFER = 0.001  # 0.1% buffer for limit orders

    # ==================== DATABASE SETTINGS ====================
    DATABASE_PATH = 'data/trading_system.db'
    BACKUP_FREQUENCY = 'daily'
    MAX_BACKUPS = 30  # Keep 30 days of backups

    # Cache settings
    CACHE_DIR = 'cache'
    CACHE_EXPIRY = 300  # 5 minutes
    MAX_CACHE_SIZE_MB = 1000  # 1GB max cache

    # ==================== MONITORING & LOGGING ====================
    LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_DIR = 'logs'
    LOG_ROTATION = 'daily'
    LOG_RETENTION_DAYS = 30

    # Performance tracking
    TRACK_SLIPPAGE = True
    TRACK_FEES = True
    BENCHMARK_SYMBOL = 'SPY'  # Benchmark for comparison

    # ==================== DASHBOARD SETTINGS ====================
    DASHBOARD_PORT = 8501
    DASHBOARD_REFRESH_INTERVAL = 5  # seconds
    DASHBOARD_THEME = 'dark'
    SHOW_ADVANCED_METRICS = True

    # ==================== NEWS & SENTIMENT ====================
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    ENABLE_SENTIMENT_ANALYSIS = True
    SENTIMENT_WEIGHT = 0.2  # 20% weight in trading decisions
    MIN_NEWS_RELEVANCE = 0.7  # Minimum relevance score

    # OpenAI settings (for advanced sentiment)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    USE_GPT_SENTIMENT = False  # Enable GPT-based sentiment
    GPT_MODEL = 'gpt-4'

    # ==================== STRATEGY SETTINGS ====================
    # Technical indicators
    USE_TECHNICAL_INDICATORS = True
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MACD_SIGNAL_THRESHOLD = 0

    # Market regime
    DETECT_MARKET_REGIME = True
    BULL_MARKET_THRESHOLD = 0.2  # 20% above 200 SMA
    BEAR_MARKET_THRESHOLD = -0.2  # 20% below 200 SMA

    # Position sizing
    USE_KELLY_CRITERION = False  # Use Kelly for position sizing
    KELLY_FRACTION = 0.25  # Use 25% of Kelly size

    # ==================== EXECUTION SETTINGS ====================
    ENABLE_PAPER_TRADING = True  # Start in paper trading mode
    REQUIRE_MANUAL_APPROVAL = False  # Require approval for trades
    MAX_SLIPPAGE_PCT = 0.01  # Cancel if slippage > 1%

    # Order routing
    USE_SMART_ROUTING = True  # Use smart order routing
    SPLIT_LARGE_ORDERS = True  # Split orders > 10% of avg volume
    MAX_ORDER_PCT_OF_VOLUME = 0.1  # Max 10% of average volume

    # ==================== ADVANCED FEATURES ====================
    # Pairs trading
    ENABLE_PAIRS_TRADING = False
    PAIRS_CORRELATION_MIN = 0.8
    PAIRS_ZSCORE_ENTRY = 2.0
    PAIRS_ZSCORE_EXIT = 0.5

    # Options strategies
    ENABLE_OPTIONS = False
    OPTIONS_STRATEGIES = ['covered_call', 'protective_put']
    MIN_OPTIONS_VOLUME = 100

    # Crypto trading
    ENABLE_CRYPTO = False
    CRYPTO_SYMBOLS = ['BTC/USD', 'ETH/USD']
    CRYPTO_ALLOCATION = 0.1  # Max 10% in crypto

    # ==================== SYSTEM SETTINGS ====================
    # Threading and performance
    MAX_WORKERS = 4  # Thread pool size
    ASYNC_ORDERS = True  # Process orders asynchronously

    # Update frequencies (seconds)
    POSITION_UPDATE_FREQ = 5
    MARKET_DATA_UPDATE_FREQ = 1
    ACCOUNT_UPDATE_FREQ = 60

    # System limits
    MAX_SIGNALS_PER_DAY = 50
    MAX_ORDERS_PER_MINUTE = 10
    MAX_API_CALLS_PER_MINUTE = 200

    # ==================== DEVELOPMENT SETTINGS ====================
    DEBUG_MODE = False
    SAVE_DEBUG_DATA = True
    PROFILE_PERFORMANCE = False
    TEST_MODE = False  # For unit tests

    # Simulation settings
    SIMULATION_START_DATE = '2023-01-01'
    SIMULATION_END_DATE = '2024-01-01'
    SIMULATION_SLIPPAGE_PCT = 0.001
    SIMULATION_COMMISSION = 0.001  # $1 per trade

    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []

        # Check API keys
        if not cls.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY not set")
        if not cls.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY not set")

        # Check GPU settings
        if cls.REQUIRE_GPU and not cls.USE_GPU:
            errors.append("REQUIRE_GPU is True but USE_GPU is False")

        # Check risk settings
        if cls.MAX_POSITION_SIZE_PCT > 0.25:
            errors.append("MAX_POSITION_SIZE_PCT > 25% is too risky")

        # Check data settings
        if cls.MIN_HISTORY_DAYS < 100:
            errors.append("MIN_HISTORY_DAYS should be at least 100")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        return True

    @classmethod
    def get_env_path(cls):
        """Get the path to the .env file"""
        return os.path.join(os.path.dirname(__file__), '..', '.env')

    @classmethod
    def create_env_template(cls):
        """Create a template .env file"""
        template = """# Trading System Environment Variables
# Copy this to .env and fill in your values

# Alpaca API
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Email Alerts
ALERT_EMAIL=your-email@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password

# News API
NEWS_API_KEY=your_news_api_key

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key
"""

        env_path = cls.get_env_path()
        if not os.path.exists(env_path):
            with open(env_path, 'w') as f:
                f.write(template)
            print(f"Created .env template at {env_path}")
            print("Please edit it with your API keys")

    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("\n" + "=" * 60)
        print("TRADING SYSTEM CONFIGURATION")
        print("=" * 60)

        # Group settings by category
        categories = {
            'Account': ['ACCOUNT_SIZE', 'MAX_POSITIONS', 'MIN_CONFIDENCE'],
            'Risk': ['MAX_DAILY_LOSS', 'DEFAULT_STOP_LOSS', 'RISK_PER_TRADE'],
            'GPU': ['USE_GPU', 'REQUIRE_GPU', 'GPU_DEVICE_ID'],
            'Model': ['SEQUENCE_LENGTH', 'PREDICTION_THRESHOLD', 'RETRAIN_FREQUENCY'],
            'Alerts': ['ALERT_EMAIL', 'SEND_EMAIL_ALERTS'],
        }

        for category, settings in categories.items():
            print(f"\n{category} Settings:")
            for setting in settings:
                value = getattr(cls, setting, 'Not Set')
                # Hide sensitive data
                if 'KEY' in setting or 'PASSWORD' in setting:
                    value = '***' if value else 'Not Set'
                print(f"  {setting}: {value}")

        print("\n" + "=" * 60 + "\n")


# Validate config on import
if __name__ != "__main__":
    try:
        Config.validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        Config.create_env_template()