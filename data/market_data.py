# data/market_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import alpaca_trade_api as tradeapi
import yfinance as yf
import logging
import time
from functools import lru_cache
import threading
from collections import defaultdict
import requests

from config.settings import Config

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Manages market data fetching with efficient API usage"""

    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        """Initialize market data manager"""
        self.api_key = api_key or Config.ALPACA_API_KEY
        self.secret_key = secret_key or Config.ALPACA_SECRET_KEY
        self.base_url = base_url or Config.ALPACA_BASE_URL

        # Initialize Alpaca API
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url)

        # Rate limiting
        self.api_calls = []
        self.rate_limit_lock = threading.Lock()
        self.max_calls_per_minute = Config.MAX_API_CALLS_PER_MINUTE

        # Data cache
        self.cache = {}
        self.cache_expiry = {}

    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        with self.rate_limit_lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.api_calls = [t for t in self.api_calls if now - t < 60]

            # If at limit, wait
            if len(self.api_calls) >= self.max_calls_per_minute:
                sleep_time = 60 - (now - self.api_calls[0]) + 0.1
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)

            # Record this call
            self.api_calls.append(now)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        cache_key = f"price_{symbol}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_price = self.cache[cache_key]
            if time.time() - cached_time < 60:  # 1 minute cache
                return cached_price

        self._check_rate_limit()

        try:
            trade = self.api.get_latest_trade(symbol)
            price = float(trade.price)

            # Cache the result
            self.cache[cache_key] = (time.time(), price)

            return price

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            # Try Yahoo Finance as backup
            try:
                ticker = yf.Ticker(symbol)
                price = ticker.info.get('regularMarketPrice', ticker.info.get('price'))
                if price:
                    self.cache[cache_key] = (time.time(), price)
                    return price
            except:
                pass

            return None

    def get_current_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols efficiently"""
        prices = {}
        symbols_to_fetch = []

        # Check cache first
        for symbol in symbols:
            cache_key = f"price_{symbol}"
            if cache_key in self.cache:
                cached_time, cached_price = self.cache[cache_key]
                if time.time() - cached_time < 60:
                    prices[symbol] = cached_price
                else:
                    symbols_to_fetch.append(symbol)
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            return prices

        # Batch fetch from Alpaca
        self._check_rate_limit()

        try:
            # Alpaca allows up to 100 symbols per request
            for i in range(0, len(symbols_to_fetch), 100):
                batch = symbols_to_fetch[i:i + 100]

                trades = self.api.get_latest_trades(batch)

                for symbol, trade in trades.items():
                    price = float(trade.price)
                    prices[symbol] = price
                    self.cache[f"price_{symbol}"] = (time.time(), price)

        except Exception as e:
            logger.error(f"Error in batch price fetch: {e}")
            # Fallback to individual fetches
            for symbol in symbols_to_fetch:
                price = self.get_current_price(symbol)
                if price:
                    prices[symbol] = price

        return prices

    def get_bars(self, symbol: str, timeframe: str = '1Day',
                 limit: int = 100, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Get historical bars for a symbol"""
        cache_key = f"bars_{symbol}_{timeframe}_{limit}_{start}_{end}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            cache_duration = 3600 if 'Day' in timeframe else 300  # 1 hour for daily, 5 min for intraday
            if time.time() - cached_time < cache_duration:
                return cached_data

        self._check_rate_limit()

        try:
            # Get bars from Alpaca
            if start and end:
                bars = self.api.get_bars(
                    symbol, timeframe,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    adjustment='all'
                ).df
            else:
                bars = self.api.get_bars(
                    symbol, timeframe,
                    limit=limit,
                    adjustment='all'
                ).df

            if not bars.empty:
                # Clean up the dataframe
                bars = bars.reset_index()
                bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
                bars.set_index('timestamp', inplace=True)
                bars = bars[['open', 'high', 'low', 'close', 'volume']]  # Keep only OHLCV

                # Cache the result
                self.cache[cache_key] = (time.time(), bars)

                return bars

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")

            # Try Yahoo Finance as backup
            try:
                ticker = yf.Ticker(symbol)

                if start and end:
                    bars = ticker.history(start=start, end=end)
                else:
                    # Convert limit to period
                    if 'Day' in timeframe:
                        period = f"{limit}d"
                    elif 'Min' in timeframe:
                        period = "1d" if limit <= 390 else "5d"
                    else:
                        period = "1mo"

                    bars = ticker.history(period=period)

                if not bars.empty:
                    bars.columns = [col.lower() for col in bars.columns]
                    bars = bars[['open', 'high', 'low', 'close', 'volume']]

                    self.cache[cache_key] = (time.time(), bars)
                    return bars

            except Exception as e2:
                logger.error(f"Yahoo Finance backup failed: {e2}")

        return pd.DataFrame()

    def get_quote(self, symbol: str) -> Dict:
        """Get current quote (bid/ask) for a symbol"""
        cache_key = f"quote_{symbol}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_quote = self.cache[cache_key]
            if time.time() - cached_time < 30:  # 30 second cache for quotes
                return cached_quote

        self._check_rate_limit()

        try:
            quote = self.api.get_latest_quote(symbol)

            quote_dict = {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'spread': float(quote.ask_price) - float(quote.bid_price),
                'spread_pct': (float(quote.ask_price) - float(quote.bid_price)) / float(quote.ask_price) * 100,
                'timestamp': quote.timestamp
            }

            # Cache the result
            self.cache[cache_key] = (time.time(), quote_dict)

            return quote_dict

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            # Return a simple quote based on last price
            price = self.get_current_price(symbol)
            if price:
                return {
                    'bid': price * 0.9999,
                    'ask': price * 1.0001,
                    'bid_size': 100,
                    'ask_size': 100,
                    'spread': price * 0.0002,
                    'spread_pct': 0.02,
                    'timestamp': datetime.now()
                }

            return {}

    def get_market_status(self) -> Dict:
        """Get current market status"""
        self._check_rate_limit()

        try:
            clock = self.api.get_clock()

            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'timestamp': clock.timestamp
            }

        except Exception as e:
            logger.error(f"Error getting market status: {e}")

            # Simple fallback based on time
            now = datetime.now()
            is_weekday = now.weekday() < 5

            # Convert to market hours (ET)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            is_open = is_weekday and market_open <= now <= market_close

            return {
                'is_open': is_open,
                'next_open': market_open + timedelta(days=1 if now > market_close else 0),
                'next_close': market_close,
                'timestamp': now
            }

    def get_market_calendar(self, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Get market calendar with trading days"""
        if start is None:
            start = datetime.now() - timedelta(days=30)
        if end is None:
            end = datetime.now() + timedelta(days=30)

        self._check_rate_limit()

        try:
            calendar = self.api.get_calendar(start=start.date(), end=end.date())

            cal_data = []
            for day in calendar:
                cal_data.append({
                    'date': day.date,
                    'open': day.open,
                    'close': day.close
                })

            return pd.DataFrame(cal_data)

        except Exception as e:
            logger.error(f"Error getting market calendar: {e}")
            return pd.DataFrame()

    def get_asset_info(self, symbol: str) -> Dict:
        """Get asset information"""
        cache_key = f"asset_{symbol}"

        # Check cache (asset info doesn't change often)
        if cache_key in self.cache:
            cached_time, cached_info = self.cache[cache_key]
            if time.time() - cached_time < 86400:  # 24 hour cache
                return cached_info

        self._check_rate_limit()

        try:
            asset = self.api.get_asset(symbol)

            asset_info = {
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'asset_class': asset.asset_class,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable
            }

            # Cache the result
            self.cache[cache_key] = (time.time(), asset_info)

            return asset_info

        except Exception as e:
            logger.error(f"Error getting asset info for {symbol}: {e}")

            # Return basic info
            return {
                'symbol': symbol,
                'tradable': True,
                'marginable': True,
                'shortable': False  # We don't short
            }

    def get_snapshot(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get snapshot of multiple symbols with key stats"""
        snapshots = {}

        # Get current prices
        prices = self.get_current_prices_batch(symbols)

        # Get additional data for each symbol
        for symbol in symbols:
            try:
                # Get daily bar for change calculation
                daily_bars = self.get_bars(symbol, '1Day', limit=2)

                if not daily_bars.empty and symbol in prices:
                    prev_close = daily_bars.iloc[-2]['close'] if len(daily_bars) > 1 else daily_bars.iloc[-1]['open']
                    current_price = prices[symbol]

                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100

                    # Get volume
                    volume = daily_bars.iloc[-1]['volume'] if not daily_bars.empty else 0

                    snapshots[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': volume,
                        'prev_close': prev_close,
                        'timestamp': datetime.now()
                    }

            except Exception as e:
                logger.error(f"Error getting snapshot for {symbol}: {e}")

        return snapshots

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df

        # Simple Moving Averages
        for period in [10, 20, 50, 200]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in [12, 26]:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        if len(df) >= 20:
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_mid'] = sma20

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # ATR (Average True Range)
        if len(df) >= 14:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()

        return df

    def get_intraday_bars(self, symbol: str, interval: str = '5Min', days: int = 1) -> pd.DataFrame:
        """Get intraday bars for detailed analysis"""
        end = datetime.now()
        start = end - timedelta(days=days)

        return self.get_bars(symbol, interval, start=start, end=end)

    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Market data cache cleared")

    def get_api_usage(self) -> Dict:
        """Get current API usage statistics"""
        with self.rate_limit_lock:
            now = time.time()
            recent_calls = [t for t in self.api_calls if now - t < 60]

            return {
                'calls_last_minute': len(recent_calls),
                'calls_limit': self.max_calls_per_minute,
                'usage_pct': len(recent_calls) / self.max_calls_per_minute * 100,
                'cache_size': len(self.cache)
            }