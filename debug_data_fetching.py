# test_data_fetching.py
"""
Test script to debug data fetching issues
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Data Fetching\n" + "=" * 50)

# Test 1: Direct yfinance test
print("\n1. Testing direct yfinance...")
try:
    import yfinance as yf

    symbol = 'AAPL'
    ticker = yf.Ticker(symbol)

    # Try different methods
    print(f"\nTesting history method for {symbol}:")
    hist = ticker.history(period="1mo")
    print(f"  Shape: {hist.shape}")
    print(f"  Date range: {hist.index[0]} to {hist.index[-1]}" if not hist.empty else "  No data")

    print(f"\nTesting download method for {symbol}:")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    print(f"  Shape: {data.shape}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}" if not data.empty else "  No data")

except Exception as e:
    print(f"✗ yfinance test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 2: Test MarketDataManager
print("\n\n2. Testing MarketDataManager...")
try:
    from data.market_data import MarketDataManager

    mdm = MarketDataManager()

    # Test different limit values
    test_cases = [
        ('AAPL', 10),
        ('AAPL', 100),
        ('AAPL', 252),  # 1 year
        ('MSFT', 100),
    ]

    for symbol, limit in test_cases:
        print(f"\nFetching {limit} days for {symbol}:")
        bars = mdm.get_bars(symbol, '1Day', limit=limit)

        if bars.empty:
            print(f"  ✗ No data returned")
        else:
            print(f"  ✓ Shape: {bars.shape}")
            print(f"  Date range: {bars.index[0]} to {bars.index[-1]}")
            print(f"  Columns: {bars.columns.tolist()}")

            # Check for issues
            if len(bars) < limit * 0.5:  # Less than 50% of requested
                print(f"  ⚠ Only got {len(bars)}/{limit} bars requested")

except Exception as e:
    print(f"✗ MarketDataManager test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Test with date range
print("\n\n3. Testing with date range...")
try:
    from data.market_data import MarketDataManager

    mdm = MarketDataManager()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"Fetching AAPL from {start_date} to {end_date}:")
    bars = mdm.get_bars('AAPL', '1Day', start=start_date, end=end_date)

    if bars.empty:
        print("  ✗ No data returned")
    else:
        print(f"  ✓ Shape: {bars.shape}")
        print(f"  Actual data:")
        print(bars.tail())

except Exception as e:
    print(f"✗ Date range test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Check if it's an Alpaca vs yfinance issue
print("\n\n4. Checking data source...")
try:
    from data.market_data import MarketDataManager

    mdm = MarketDataManager()

    print(f"Using Alpaca: {mdm.use_alpaca}")
    print(f"API initialized: {mdm.api is not None}")

    if mdm.use_alpaca and mdm.api:
        print("\nTrying to force yfinance fallback...")
        # Temporarily disable Alpaca
        mdm.use_alpaca = False

        bars = mdm.get_bars('AAPL', '1Day', limit=100)
        print(f"yfinance result shape: {bars.shape if not bars.empty else 'No data'}")

except Exception as e:
    print(f"✗ Data source check failed: {e}")

print("\n" + "=" * 50)
print("DATA FETCHING TEST COMPLETE")