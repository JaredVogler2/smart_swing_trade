# test_environment.py
"""
Test script to verify environment setup and data access
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("\n" + "=" * 80)

# Test 1: Import modules
print("TEST 1: Checking module imports...")
try:
    from config.watchlist import WATCHLIST

    print("✓ Config imported successfully")
    print(f"  Watchlist has {len(WATCHLIST)} symbols")
    print(f"  First 5 symbols: {WATCHLIST[:5]}")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from data.market_data import MarketDataManager

    print("✓ MarketDataManager imported successfully")
except Exception as e:
    print(f"✗ MarketDataManager import failed: {e}")

try:
    from models.features import FeatureEngineer

    print("✓ FeatureEngineer imported successfully")
except Exception as e:
    print(f"✗ FeatureEngineer import failed: {e}")

try:
    from models.ensemble_gpu_windows import GPUEnsembleModel

    print("✓ GPUEnsembleModel imported successfully")
except Exception as e:
    print(f"✗ GPUEnsembleModel import failed: {e}")

# Test 2: Check GPU availability
print("\n" + "=" * 80)
print("TEST 2: Checking GPU availability...")
try:
    import torch

    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"✗ PyTorch check failed: {e}")

# Test 3: Check data access
print("\n" + "=" * 80)
print("TEST 3: Testing market data access...")
try:
    market_data = MarketDataManager()

    # Try fetching data for AAPL
    test_symbol = 'AAPL'
    print(f"\nFetching data for {test_symbol}...")

    df = market_data.get_bars(test_symbol, '1Day', limit=10)

    if df.empty:
        print("✗ No data returned!")
    else:
        print("✓ Data fetched successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print("\nFirst few rows:")
        print(df.head())

except Exception as e:
    print(f"✗ Market data fetch failed: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Test feature creation with minimal data
print("\n" + "=" * 80)
print("TEST 4: Testing feature creation...")
try:
    # Create minimal test data
    dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 250),
        'high': np.random.uniform(110, 120, 250),
        'low': np.random.uniform(90, 100, 250),
        'close': np.random.uniform(95, 115, 250),
        'volume': np.random.uniform(1000000, 5000000, 250)
    }, index=dates)

    # Ensure high >= low, close between high and low
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)

    print("Created test data:")
    print(test_data.head())

    # Try creating features
    feature_engineer = FeatureEngineer(enable_gpu=False)
    features = feature_engineer.create_features(test_data, 'TEST')

    if features.empty:
        print("✗ No features created!")
    else:
        print("✓ Features created successfully!")
        print(f"  Shape: {features.shape}")
        print(f"  Number of features: {len(features.columns)}")

        # Check for NaN
        nan_count = features.isna().sum().sum()
        print(f"  Total NaN values: {nan_count}")

        if nan_count > 0:
            nan_cols = features.columns[features.isna().any()].tolist()
            print(f"  Columns with NaN: {nan_cols[:10]}...")

except Exception as e:
    print(f"✗ Feature creation failed: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Check configuration
print("\n" + "=" * 80)
print("TEST 5: Checking configuration...")
try:
    from config.settings import Config

    print("Key configuration values:")
    print(f"  ACCOUNT_SIZE: ${Config.ACCOUNT_SIZE:,.2f}")
    print(f"  MAX_POSITIONS: {Config.MAX_POSITIONS}")
    print(f"  SEQUENCE_LENGTH: {Config.SEQUENCE_LENGTH}")
    print(f"  ALPACA_API_KEY: {'*' * 10 if Config.ALPACA_API_KEY else 'NOT SET'}")
    print(f"  ALPACA_BASE_URL: {Config.ALPACA_BASE_URL}")

except Exception as e:
    print(f"✗ Configuration check failed: {e}")

# Test 6: Check file structure
print("\n" + "=" * 80)
print("TEST 6: Checking file structure...")
required_dirs = ['data', 'models', 'config', 'logs', 'cache']
required_files = ['config/settings.py', 'config/watchlist.py', 'models/features.py']

for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"✓ Directory exists: {dir_name}")
    else:
        print(f"✗ Directory missing: {dir_name}")

for file_name in required_files:
    if os.path.exists(file_name):
        print(f"✓ File exists: {file_name}")
    else:
        print(f"✗ File missing: {file_name}")

print("\n" + "=" * 80)
print("ENVIRONMENT TEST COMPLETE")
print("=" * 80)