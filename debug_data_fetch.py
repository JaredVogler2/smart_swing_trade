# debug_data_fetch.py
# Test why data fetching isn't working

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.market_data import MarketDataManager
from config.settings import Config
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

print("Testing data fetching...")

# Test 1: Direct Alpaca API
print("\n1. Testing direct Alpaca API:")
api = tradeapi.REST(
    Config.ALPACA_API_KEY,
    Config.ALPACA_SECRET_KEY,
    Config.ALPACA_BASE_URL
)

try:
    # Test with AAPL
    symbol = 'AAPL'
    end = datetime.now()
    start = end - timedelta(days=1000)

    bars = api.get_bars(
        symbol,
        timeframe='1Day',
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        limit=1000,
        adjustment='all'
    ).df

    print(f"Direct API - {symbol}: {len(bars)} bars")
    print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
    print(f"Sample data:\n{bars.head()}")

except Exception as e:
    print(f"Direct API error: {e}")

# Test 2: MarketDataManager
print("\n\n2. Testing MarketDataManager:")
mdm = MarketDataManager()

try:
    bars2 = mdm.get_bars('AAPL', '1Day', limit=1000)
    print(f"MarketDataManager - AAPL: {len(bars2)} bars")
    if not bars2.empty:
        print(f"Date range: {bars2.index[0]} to {bars2.index[-1]}")
except Exception as e:
    print(f"MarketDataManager error: {e}")

# Test 3: Check the actual method being called
print("\n\n3. Checking get_bars implementation:")
print(f"MarketDataManager class location: {MarketDataManager.__module__}")

# Test 4: Try multiple symbols
print("\n\n4. Testing multiple symbols:")
test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
for sym in test_symbols:
    try:
        bars = mdm.get_bars(sym, '1Day', limit=1000)
        print(f"{sym}: {len(bars) if bars is not None and not bars.empty else 0} bars")
    except Exception as e:
        print(f"{sym}: Error - {str(e)[:50]}...")