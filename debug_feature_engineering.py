# debug_feature_engineering.py
"""
Detailed diagnostic script to debug feature engineering and NaN issues
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os
import json
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.watchlist import WATCHLIST
from data.market_data import MarketDataManager
from models.features import FeatureEngineer
from models.ensemble_gpu_windows import GPUEnsembleModel

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_feature_creation():
    """Test feature creation step by step"""
    print("\n" + "=" * 80)
    print("TESTING FEATURE CREATION")
    print("=" * 80)

    # Initialize
    market_data = MarketDataManager()
    feature_engineer = FeatureEngineer(enable_gpu=False)

    # Test with one symbol
    test_symbol = 'AAPL'

    print(f"\n1. Testing with symbol: {test_symbol}")

    try:
        # Get data
        print("   Fetching market data...")
        df = market_data.get_bars(test_symbol, '1Day', limit=300)

        if df.empty:
            print("   ERROR: No data retrieved!")
            return

        print(f"   Data shape: {df.shape}")
        print(f"   Data columns: {df.columns.tolist()}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Sample data:")
        print(df.head())

        # Check for basic data issues
        print("\n2. Checking data quality:")
        print(f"   NaN values: {df.isna().sum().sum()}")
        print(f"   Zero prices: {(df[['open', 'high', 'low', 'close']] == 0).sum().sum()}")
        print(f"   Zero volume days: {(df['volume'] == 0).sum()}")

        # Create features
        print("\n3. Creating features...")
        features = feature_engineer.create_features(df, test_symbol)

        if features.empty:
            print("   ERROR: No features created!")
            print("   Checking feature creation methods...")

            # Test individual feature methods
            test_methods = [
                '_create_price_features',
                '_create_volume_features',
                '_create_volatility_features',
                '_create_technical_indicators',
                '_create_microstructure_features',
                '_create_temporal_features',
                '_create_statistical_features'
            ]

            for method in test_methods:
                if hasattr(feature_engineer, method):
                    try:
                        print(f"\n   Testing {method}...")
                        method_func = getattr(feature_engineer, method)
                        result = method_func(df)
                        print(f"   {method} created {len(result)} features")
                    except Exception as e:
                        print(f"   ERROR in {method}: {str(e)}")
                        traceback.print_exc()

            return

        print(f"   Features shape: {features.shape}")
        print(f"   Number of features: {len(features.columns)}")

        # Analyze NaN values
        print("\n4. Analyzing NaN values in features:")
        nan_counts = features.isna().sum()
        nan_pct = (nan_counts / len(features)) * 100

        features_with_nan = nan_counts[nan_counts > 0]
        if len(features_with_nan) > 0:
            print(f"   Features with NaN: {len(features_with_nan)}/{len(features.columns)}")
            print("\n   Top 20 features with most NaN (%):")
            for feat, pct in nan_pct[features_with_nan.index].nlargest(20).items():
                print(f"   - {feat}: {pct:.1f}%")
        else:
            print("   No NaN values found in features!")

        # Check for infinite values
        print("\n5. Checking for infinite values:")
        numeric_features = features.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_features).sum()
        features_with_inf = inf_counts[inf_counts > 0]

        if len(features_with_inf) > 0:
            print(f"   Features with infinity: {len(features_with_inf)}")
            for feat, count in features_with_inf.items():
                print(f"   - {feat}: {count} infinite values")
        else:
            print("   No infinite values found!")

        # Feature statistics
        print("\n6. Feature statistics:")
        print(f"   Total features created: {len(features.columns)}")

        # Group features by type
        feature_groups = {
            'price': [],
            'volume': [],
            'technical': [],
            'volatility': [],
            'other': []
        }

        for col in features.columns:
            if any(x in col for x in ['return', 'sma', 'ema', 'price']):
                feature_groups['price'].append(col)
            elif any(x in col for x in ['volume', 'obv', 'ad']):
                feature_groups['volume'].append(col)
            elif any(x in col for x in ['rsi', 'macd', 'stoch']):
                feature_groups['technical'].append(col)
            elif any(x in col for x in ['atr', 'bb_', 'volatility']):
                feature_groups['volatility'].append(col)
            else:
                feature_groups['other'].append(col)

        for group, feats in feature_groups.items():
            print(f"   {group}: {len(feats)} features")

        # Sample feature values
        print("\n7. Sample feature values (last 5 rows):")
        sample_features = ['return_5d', 'rsi_14', 'volume_ratio_20', 'volatility_20d']
        available_samples = [f for f in sample_features if f in features.columns]

        if available_samples:
            print(features[available_samples].tail())

        return features

    except Exception as e:
        print(f"\nERROR in feature creation: {str(e)}")
        traceback.print_exc()
        return None


def test_model_target_creation():
    """Test model target creation for class balance"""
    print("\n" + "=" * 80)
    print("TESTING MODEL TARGET CREATION")
    print("=" * 80)

    model = GPUEnsembleModel()
    market_data = MarketDataManager()

    # Test with multiple symbols
    test_symbols = WATCHLIST[:10]

    all_targets = []

    for symbol in test_symbols:
        try:
            # Get data
            df = market_data.get_bars(symbol, '1Day', limit=500)

            if df.empty or len(df) < 200:
                continue

            # Create target
            target = model._create_advanced_target(df)

            # Remove NaN
            target_clean = target.dropna()

            if len(target_clean) > 0:
                all_targets.append(target_clean)

                # Show distribution for this symbol
                positive_pct = (target_clean == 1).sum() / len(target_clean) * 100
                print(f"\n{symbol}:")
                print(f"  Total samples: {len(target_clean)}")
                print(f"  Positive class: {(target_clean == 1).sum()} ({positive_pct:.1f}%)")
                print(f"  Negative class: {(target_clean == 0).sum()} ({100 - positive_pct:.1f}%)")
                print(f"  Ratio: {(target_clean == 0).sum() / max((target_clean == 1).sum(), 1):.1f}:1")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Combined statistics
    if all_targets:
        combined_target = pd.concat(all_targets)

        print("\n" + "-" * 50)
        print("COMBINED TARGET STATISTICS:")
        print(f"Total samples: {len(combined_target)}")
        print(
            f"Positive class: {(combined_target == 1).sum()} ({(combined_target == 1).sum() / len(combined_target) * 100:.1f}%)")
        print(
            f"Negative class: {(combined_target == 0).sum()} ({(combined_target == 0).sum() / len(combined_target) * 100:.1f}%)")
        print(f"Imbalance ratio: {(combined_target == 0).sum() / max((combined_target == 1).sum(), 1):.1f}:1")


def test_full_pipeline():
    """Test the full data preparation pipeline"""
    print("\n" + "=" * 80)
    print("TESTING FULL PIPELINE")
    print("=" * 80)

    model = GPUEnsembleModel()
    market_data = MarketDataManager()

    # Prepare training data
    train_data = {}

    print("\n1. Fetching data for training...")
    for symbol in WATCHLIST[:5]:
        try:
            df = market_data.get_bars(symbol, '1Day', limit=500)
            if not df.empty and len(df) >= 300:
                train_data[symbol] = df
                print(f"   {symbol}: {len(df)} days")
        except Exception as e:
            print(f"   {symbol}: Error - {str(e)}")

    if len(train_data) < 2:
        print("ERROR: Insufficient data for training!")
        return

    print(f"\n2. Preparing features and targets...")
    X_all = []
    y_all = []

    for symbol, df in train_data.items():
        try:
            # Create features
            features = model.feature_engineer.create_features(df, symbol)

            if features.empty:
                print(f"   {symbol}: No features created!")
                continue

            # Create target
            target = model._create_advanced_target(df)

            # Align
            min_len = min(len(features), len(target))
            features = features.iloc[:min_len]
            target = target.iloc[:min_len]

            # Remove last prediction_horizon rows
            features = features[:-model.prediction_horizon]
            target = target[:-model.prediction_horizon]

            # Check validity
            valid_idx = ~(features.isna().any(axis=1) | target.isna())

            print(f"   {symbol}:")
            print(f"     Features shape: {features.shape}")
            print(f"     Valid samples: {valid_idx.sum()}/{len(valid_idx)}")
            print(f"     Target distribution: {target[valid_idx].value_counts().to_dict()}")

            if valid_idx.sum() > 50:
                X_all.append(features[valid_idx])
                y_all.append(target[valid_idx])

        except Exception as e:
            print(f"   {symbol}: Error - {str(e)}")
            traceback.print_exc()

    if X_all:
        X_combined = pd.concat(X_all)
        y_combined = pd.concat(y_all)

        print(f"\n3. Combined data statistics:")
        print(f"   Total samples: {len(X_combined)}")
        print(f"   Total features: {len(X_combined.columns)}")
        print(f"   Class distribution: {y_combined.value_counts().to_dict()}")
        print(f"   Positive class ratio: {(y_combined == 1).sum() / len(y_combined) * 100:.1f}%")

        # Test scaling
        print(f"\n4. Testing feature scaling...")
        try:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_combined)

            nan_after_scaling = np.isnan(X_scaled).sum()
            inf_after_scaling = np.isinf(X_scaled).sum()

            print(f"   NaN after scaling: {nan_after_scaling}")
            print(f"   Inf after scaling: {inf_after_scaling}")

        except Exception as e:
            print(f"   Scaling error: {str(e)}")


def main():
    """Run all diagnostic tests"""
    # Test 1: Feature Creation
    features = test_feature_creation()

    # Test 2: Target Creation
    test_model_target_creation()

    # Test 3: Full Pipeline
    test_full_pipeline()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Save results
    results = {
        'timestamp': str(datetime.now()),
        'tests_run': ['feature_creation', 'target_creation', 'full_pipeline'],
        'recommendations': [
            'Check if FeatureEngineer is properly creating features',
            'Verify market data is being fetched correctly',
            'Ensure target thresholds are appropriate for your data',
            'Consider using the fixed ensemble model with better NaN handling'
        ]
    }

    with open('detailed_diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to detailed_diagnostic_results.json")


if __name__ == "__main__":
    main()