# config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Main configuration class for the trading system"""

    # Account settings
    ACCOUNT_SIZE = 10000  # $10,000 starting capital
    CASH_RESERVE_PCT = 0.10  # Keep 10% in cash

    # Position management
    MAX_POSITIONS = 4  # Maximum concurrent positions for $10k account
    MIN_POSITION_SIZE = 2000  # Minimum $2,000 per position
    MAX_POSITION_SIZE = 3000  # Maximum $3,000 per position
    TARGET_POSITION_SIZE = 2500  # Target $2,500 per position

    # Risk management
    STOP_LOSS_PCT = 0.03  # 3% stop loss
    PROFIT_TARGET_PCT = 0.06  # 6% primary profit target
    STRETCH_TARGET_PCT = 0.09  # 9% stretch target
    TRAILING_STOP_ACTIVATION_PCT = 0.04  # Activate trailing stop at 4% profit
    MAX_DAILY_LOSS = 300  # Maximum $300 daily loss
    MAX_DRAWDOWN = 0.15  # 15% maximum drawdown
    RISK_PER_TRADE = 0.0075  # Risk 0.75% of account per trade

    # ML model settings
    MIN_CONFIDENCE = 0.65  # Minimum 65% confidence for trades
    RETRAIN_DAYS = 90  # Retrain models every 90 days
    LOOKBACK_DAYS = 500  # Minimum training data
    SEQUENCE_LENGTH = 60  # LSTM sequence length

    # API settings
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # API rate limits
    MAX_API_CALLS_PER_MINUTE = 190  # Leave buffer under 200
    API_CALL_DELAY = 0.5  # Seconds between calls

    # Data settings
    DATA_FETCH_INTERVAL = 300  # Fetch new data every 5 minutes
    CACHE_DURATION = 3600  # Cache data for 1 hour

    # Database settings
    DB_PATH = Path('data/trading_system.db')
    DB_BACKUP_PATH = Path('data/backups')

    # Logging settings
    LOG_DIR = Path('logs')
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Trading hours (Eastern Time)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0

    # Backtesting settings
    BACKTEST_START_DATE = '2019-01-01'
    BACKTEST_END_DATE = '2023-12-31'
    WALK_FORWARD_MONTHS = 3

    # GPU settings
    USE_GPU = True
    GPU_DEVICE_ID = 0
    GPU_MEMORY_FRACTION = 0.8

    # Notification settings
    ENABLE_NOTIFICATIONS = True
    NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL')

    # Performance thresholds
    MIN_SHARPE_RATIO = 1.5
    MIN_WIN_RATE = 0.55
    MIN_PROFIT_FACTOR = 1.5

    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.ALPACA_API_KEY or not cls.ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API credentials not set")

        if not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not set")

        if cls.MAX_POSITIONS * cls.MIN_POSITION_SIZE > cls.ACCOUNT_SIZE * (1 - cls.CASH_RESERVE_PCT):
            raise ValueError("Position sizing exceeds available capital")

        # Create directories if they don't exist
        cls.LOG_DIR.mkdir(exist_ok=True)
        cls.DB_PATH.parent.mkdir(exist_ok=True)
        cls.DB_BACKUP_PATH.mkdir(exist_ok=True)

        return True


# Validate on import
Config.validate()