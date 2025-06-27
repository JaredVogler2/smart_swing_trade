# test_minimal_features.py
"""
Minimal test to debug feature engineering issues
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Feature Engineering Step by Step\n" + "=" * 50)

# Step 1: Create simple test data
print("\n1. Creating test data...")
dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
test_df = pd.DataFrame({
    'open': 100 + np.random.randn(300) * 5,
    'high': 105 + np.random.randn(300) * 5,
    'low': 95 + np.random.randn(300) * 5,
    'close': 100 + np.random.randn(300) * 5,
    'volume': 1000000 + np.random.randn(300) * 100000
}, index=dates)

# Fix high/low logic
test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1)
test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1)
test_df['volume'] = test_df['volume'].abs()

print(f"Test data shape: {test_df.shape}")
print(f"Columns: {test_df.columns.tolist()}")
print("\nFirst 5 rows:")
print(test_df.head())

# Step 2: Test feature engineering
print("\n2. Testing FeatureEngineer...")
try:
    from models.features import FeatureEngineer

    fe = FeatureEngineer(enable_gpu=False)

    # Test if the create_features method exists
    if hasattr(fe, 'create_features'):
        print("✓ create_features method exists")

        # Try to create features
        features = fe.create_features(test_df, 'TEST')

        if features is None:
            print("✗ create_features returned None")
        elif features.empty:
            print("✗ create_features returned empty DataFrame")

            # Test individual feature creation methods
            print("\n3. Testing individual feature methods...")

            # Test price features
            if hasattr(fe, '_create_price_features'):
                try:
                    price_features = fe._create_price_features(test_df)
                    print(f"✓ Price features: {len(price_features)} created")
                    print(f"  Sample: {list(price_features.keys())[:5]}")
                except Exception as e:
                    print(f"✗ Price features failed: {e}")

            # Test volume features
            if hasattr(fe, '_create_volume_features'):
                try:
                    volume_features = fe._create_volume_features(test_df)
                    print(f"✓ Volume features: {len(volume_features)} created")
                    print(f"  Sample: {list(volume_features.keys())[:5]}")
                except Exception as e:
                    print(f"✗ Volume features failed: {e}")

            # Test technical indicators
            if hasattr(fe, '_create_technical_indicators'):
                try:
                    tech_features = fe._create_technical_indicators(test_df)
                    print(f"✓ Technical features: {len(tech_features)} created")
                    print(f"  Sample: {list(tech_features.keys())[:5]}")
                except Exception as e:
                    print(f"✗ Technical features failed: {e}")

        else:
            print(f"✓ Features created successfully!")
            print(f"  Shape: {features.shape}")
            print(f"  Number of features: {len(features.columns)}")
            print(f"  NaN count: {features.isna().sum().sum()}")

            # Show feature categories
            price_features = [f for f in features.columns if 'price' in f or 'return' in f or 'sma' in f]
            volume_features = [f for f in features.columns if 'volume' in f or 'obv' in f]
            tech_features = [f for f in features.columns if 'rsi' in f or 'macd' in f]

            print(f"\n  Price features: {len(price_features)}")
            print(f"  Volume features: {len(volume_features)}")
            print(f"  Technical features: {len(tech_features)}")

    else:
        print("✗ create_features method not found!")
        print(f"  Available methods: {[m for m in dir(fe) if not m.startswith('_')]}")

except ImportError as e:
    print(f"✗ Failed to import FeatureEngineer: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback

    traceback.print_exc()

# Step 3: Test with actual market data
print("\n4. Testing with actual market data...")
try:
    from data.market_data import MarketDataManager

    mdm = MarketDataManager()

    # Try to get data
    symbol = 'AAPL'
    bars = mdm.get_bars(symbol, '1Day', limit=300)

    if bars.empty:
        print(f"✗ No market data retrieved for {symbol}")
    else:
        print(f"✓ Market data retrieved: {bars.shape}")

        # Try features on real data
        if 'fe' in locals():
            real_features = fe.create_features(bars, symbol)
            if real_features is not None and not real_features.empty:
                print(f"✓ Features from market data: {real_features.shape}")
            else:
                print("✗ Failed to create features from market data")

except Exception as e:
    print(f"✗ Market data test failed: {e}")

# Step 4: Manual feature creation test
print("\n5. Testing manual feature creation...")
try:
    # Simple manual features
    manual_features = pd.DataFrame(index=test_df.index)

    # Returns
    manual_features['return_1d'] = test_df['close'].pct_change()
    manual_features['return_5d'] = test_df['close'].pct_change(5)

    # Moving averages
    manual_features['sma_20'] = test_df['close'].rolling(20).mean()
    manual_features['price_to_sma_20'] = test_df['close'] / manual_features['sma_20']

    # Volume
    manual_features['volume_ratio'] = test_df['volume'] / test_df['volume'].rolling(20).mean()

    # Volatility
    manual_features['volatility_20d'] = test_df['close'].pct_change().rolling(20).std()

    print(f"✓ Manual features created: {manual_features.shape}")
    print(f"  NaN count: {manual_features.isna().sum().sum()}")
    print("\nSample features:")
    print(manual_features.tail())

except Exception as e:
    print(f"✗ Manual feature creation failed: {e}")

print("\n" + "=" * 50)
print("FEATURE ENGINEERING TEST COMPLETE")