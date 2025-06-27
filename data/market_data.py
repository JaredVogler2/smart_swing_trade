# data/market_data.py
"""
Market Data Manager using yfinance exclusively
Simplified and reliable data fetching for historical and recent data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import yfinance as yf
from datetime import datetime, timedelta
import logging
import time
from functools import lru_cache
import warnings

from config.settings import Config

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class MarketDataManager:
    """Market data manager using yfinance for all data needs"""

    def __init__(self):
        """Initialize market data manager"""
        self._cache = {}
        self._last_request_time = {}
        self._rate_limit_delay = 0.1  # 100ms between requests

        # Still initialize Alpaca for trading operations (not data)
        self.api = None
        try:
            import alpaca_trade_api as tradeapi
            if Config.ALPACA_API_KEY and Config.ALPACA_SECRET_KEY:
                self.api = tradeapi.REST(
                    Config.ALPACA_API_KEY,
                    Config.ALPACA_SECRET_KEY,
                    Config.ALPACA_BASE_URL,
                    api_version='v2'
                )
                logger.info("Alpaca API initialized for trading operations")
        except Exception as e:
            logger.info(f"Alpaca not initialized: {e}")

        logger.info("MarketDataManager initialized with yfinance")

    def get_bars(self, symbol: str, timeframe: str = '1Day', 
                 limit: int = 100, start: str = None, end: str = None) -> pd.DataFrame:
        """Get historical bars using yfinance"""
        try:
            # Apply rate limiting
            self._apply_rate_limit(symbol)

            # Convert timeframe
            interval = self._convert_timeframe(timeframe)

            # Get ticker
            ticker = yf.Ticker(symbol)

            # For daily data without specific dates, use period
            if timeframe == '1Day' and start is None and end is None:
                # Determine period based on limit
                if limit <= 5:
                    period = "5d"
                elif limit <= 30:
                    period = "1mo"
                elif limit <= 90:
                    period = "3mo"
                elif limit <= 180:
                    period = "6mo"
                elif limit <= 365:
                    period = "1y"
                elif limit <= 730:
                    period = "2y"
                else:
                    period = "5y"

                df = ticker.history(
                    period=period,
                    interval=interval,
                    actions=False,
                    auto_adjust=True,
                    repair=True
                )
            else:
                # Use date range
                if end is None:
                    end_date = datetime.now()
                else:
                    end_date = pd.to_datetime(end)

                if start is None:
                    # Calculate start based on limit
                    start_date = end_date - timedelta(days=int(limit * 1.5))
                else:
                    start_date = pd.to_datetime(start)

                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    actions=False,
                    auto_adjust=True,
                    repair=True
                )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Clean data
            df = self._clean_data(df)

            # Limit to requested bars
            if len(df) > limit:
                df = df.tail(limit)

            logger.debug(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        # Standardize column names
        df.columns = df.columns.str.lower()

        # Select OHLCV columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in cols if col in df.columns]
        df = df[available_cols]

        # Ensure volume is integer
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(int)

        # Drop NaN rows
        df = df.dropna()

        # Sort by date
        df = df.sort_index()

        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try multiple fields
            for field in ['currentPrice', 'regularMarketPrice', 'previousClose']:
                if field in info and info[field]:
                    return float(info[field])

            # Fallback to last close
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            return None

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_current_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols"""
        prices = {}

        try:
            # Download all at once
            data = yf.download(
                ' '.join(symbols),
                period='1d',
                interval='1d',
                progress=False,
                group_by='ticker'
            )

            if not data.empty:
                # Handle single vs multiple symbols
                if len(symbols) == 1:
                    prices[symbols[0]] = float(data['Close'].iloc[-1])
                else:
                    for symbol in symbols:
                        try:
                            if len(data.columns.levels) > 1:
                                price = data[symbol]['Close'].iloc[-1]
                            else:
                                price = data['Close'][symbol].iloc[-1]
                            prices[symbol] = float(price)
                        except:
                            pass

            # Fill missing prices
            for symbol in symbols:
                if symbol not in prices:
                    price = self.get_current_price(symbol)
                    if price:
                        prices[symbol] = price

        except Exception as e:
            logger.error(f"Batch price error: {e}")
            # Fallback to individual
            for symbol in symbols:
                price = self.get_current_price(symbol)
                if price:
                    prices[symbol] = price

        return prices

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote with bid/ask"""
        try:
            price = self.get_current_price(symbol)
            if not price:
                return None

            # Estimate spread
            spread = price * 0.0001  # 0.01%

            return {
                'bid': price - spread/2,
                'ask': price + spread/2,
                'bid_size': 100,
                'ask_size': 100,
                'spread': spread,
                'spread_pct': 0.01
            }

        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
            return None

    def get_market_status(self) -> Dict:
        """Get market status"""
        try:
            # Use Alpaca if available
            if self.api:
                try:
                    clock = self.api.get_clock()
                    return {
                        'is_open': clock.is_open,
                        'next_open': clock.next_open,
                        'next_close': clock.next_close
                    }
                except:
                    pass

            # Simple check
            now = datetime.now()
            weekday = now.weekday()
            hour = now.hour

            is_open = weekday < 5 and 9 <= hour < 16

            return {'is_open': is_open}

        except:
            return {'is_open': False}

    def get_snapshot(self, symbols: List[str]) -> Dict:
        """Get snapshot for symbols"""
        snapshot = {}

        try:
            prices = self.get_current_prices_batch(symbols)

            for symbol in symbols:
                if symbol in prices:
                    bars = self.get_bars(symbol, '1Day', limit=2)
                    if len(bars) >= 2:
                        prev_close = bars.iloc[-2]['close']
                        current = prices[symbol]

                        snapshot[symbol] = {
                            'price': current,
                            'change': current - prev_close,
                            'change_pct': ((current - prev_close) / prev_close) * 100,
                            'volume': bars.iloc[-1]['volume']
                        }

        except Exception as e:
            logger.error(f"Snapshot error: {e}")

        return snapshot

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators"""
        try:
            import talib

            if len(df) >= 20:
                df['sma_20'] = talib.SMA(df['close'].values, 20)
                df['rsi'] = talib.RSI(df['close'].values, 14)

                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(df['close'].values, 20)
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower

                df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            return df

        except:
            return df

    def get_asset_info(self, symbol: str) -> Dict:
        """Get asset info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'tradable': True,
                'marginable': True,
                'shortable': False,
                'easy_to_borrow': False
            }

        except:
            return {'symbol': symbol, 'tradable': True}

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe"""
        conversions = {
            '1Min': '1m',
            '5Min': '5m',
            '15Min': '15m',
            '1Hour': '1h',
            '1Day': '1d'
        }
        return conversions.get(timeframe, '1d')

    def _apply_rate_limit(self, symbol: str):
        """Rate limiting"""
        current_time = time.time()

        if symbol in self._last_request_time:
            elapsed = current_time - self._last_request_time[symbol]
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)

        self._last_request_time[symbol] = time.time()

    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()

    def get_api_usage(self) -> Dict:
        """API usage"""
        return {
            'calls_last_minute': 0,
            'calls_limit': 2000,
            'usage_pct': 0
        }
