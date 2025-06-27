# watchlist_feature_engineering.py
"""
Feature Engineering for entire watchlist with GPU acceleration
Processes multiple symbols in parallel for efficiency
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
import time
import os
import pickle

# Import your custom modules
from advanced_feature_engineering import AdvancedFeatureEngineer
from config.watchlist import WATCHLIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WatchlistFeatureProcessor:
    """Process features for entire watchlist efficiently"""

    def __init__(self, use_gpu: bool = True, num_workers: int = 4):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_workers = num_workers
        self.feature_engineer = AdvancedFeatureEngineer(use_gpu=self.use_gpu)

        if self.use_gpu:
            logger.info(f"GPU Feature Engineering enabled: {torch.cuda.get_device_name()}")
        else:
            logger.info("Using CPU for feature engineering")

    def fetch_data_for_symbol(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch historical data for a single symbol"""
        try:
            logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")

            if len(df) < 500:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def process_single_symbol(self, symbol: str, df: pd.DataFrame = None) -> Dict:
        """Process features for a single symbol"""
        try:
            # Fetch data if not provided
            if df is None or df.empty:
                df = self.fetch_data_for_symbol(symbol)

            if df.empty:
                return {'symbol': symbol, 'features': None, 'error': 'No data available'}

            # Create features
            start_time = time.time()
            features = self.feature_engineer.create_all_features(df, symbol)
            processing_time = time.time() - start_time

            logger.info(f"Created {len(features.columns)} features for {symbol} in {processing_time:.2f}s")

            return {
                'symbol': symbol,
                'features': features,
                'raw_data': df,
                'processing_time': processing_time,
                'feature_count': len(features.columns),
                'error': None
            }

        except Exception as e:
            logger.error(f"Error processing features for {symbol}: {e}")
            return {'symbol': symbol, 'features': None, 'error': str(e)}

    def process_watchlist_parallel(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Process entire watchlist in parallel"""
        if symbols is None:
            symbols = WATCHLIST

        logger.info(f"Processing {len(symbols)} symbols with {self.num_workers} workers...")
        start_time = time.time()

        results = {}

        # Use ThreadPoolExecutor for I/O bound tasks (data fetching)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.process_single_symbol, symbol): symbol
                for symbol in symbols
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    completed += 1

                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(symbols)} symbols processed")

                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    results[symbol] = {'symbol': symbol, 'features': None, 'error': str(e)}

        total_time = time.time() - start_time
        logger.info(f"Completed processing {len(symbols)} symbols in {total_time:.2f}s")

        # Summary statistics
        successful = sum(1 for r in results.values() if r.get('features') is not None)
        failed = len(results) - successful

        logger.info(f"Success: {successful}, Failed: {failed}")

        return results

    def process_watchlist_batch(self, symbols: List[str] = None, batch_size: int = 10) -> Dict[str, Dict]:
        """Process watchlist in batches (useful for memory management)"""
        if symbols is None:
            symbols = WATCHLIST

        results = {}

        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}")

            batch_results = self.process_watchlist_parallel(batch)
            results.update(batch_results)

            # Clear GPU cache if using GPU
            if self.use_gpu:
                torch.cuda.empty_cache()

        return results

    def save_features(self, results: Dict, output_dir: str = "features_cache"):
        """Save processed features to disk"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for symbol, result in results.items():
            if result.get('features') is not None:
                # Save features as parquet for efficiency
                features_path = os.path.join(output_dir, f"{symbol}_features_{timestamp}.parquet")
                result['features'].to_parquet(features_path)

                # Save metadata
                metadata = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'feature_count': result['feature_count'],
                    'processing_time': result['processing_time'],
                    'data_shape': result['features'].shape
                }

                metadata_path = os.path.join(output_dir, f"{symbol}_metadata_{timestamp}.pkl")
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)

        logger.info(f"Features saved to {output_dir}")

    def load_features(self, symbol: str, output_dir: str = "features_cache") -> pd.DataFrame:
        """Load previously saved features"""
        # Find most recent file for symbol
        files = [f for f in os.listdir(output_dir) if f.startswith(f"{symbol}_features_") and f.endswith('.parquet')]

        if not files:
            logger.warning(f"No saved features found for {symbol}")
            return pd.DataFrame()

        # Get most recent file
        files.sort()
        latest_file = files[-1]

        features_path = os.path.join(output_dir, latest_file)
        features = pd.read_parquet(features_path)

        logger.info(f"Loaded {len(features.columns)} features for {symbol} from {latest_file}")
        return features

    def get_feature_statistics(self, results: Dict) -> pd.DataFrame:
        """Get statistics about processed features"""
        stats = []

        for symbol, result in results.items():
            if result.get('features') is not None:
                features = result['features']

                stats.append({
                    'symbol': symbol,
                    'feature_count': len(features.columns),
                    'data_points': len(features),
                    'missing_values': features.isna().sum().sum(),
                    'missing_pct': features.isna().sum().sum() / (features.shape[0] * features.shape[1]) * 100,
                    'processing_time': result.get('processing_time', 0),
                    'golden_cross_signals': features.get('golden_cross',
                                                         pd.Series()).sum() if 'golden_cross' in features else 0,
                    'death_cross_signals': features.get('death_cross',
                                                        pd.Series()).sum() if 'death_cross' in features else 0
                })

        return pd.DataFrame(stats)


# Example usage functions
def process_full_watchlist():
    """Process the entire watchlist"""
    processor = WatchlistFeatureProcessor(use_gpu=True, num_workers=8)

    # Process all symbols
    results = processor.process_watchlist_parallel(WATCHLIST)

    # Save features
    processor.save_features(results)

    # Get statistics
    stats = processor.get_feature_statistics(results)
    print("\nFeature Processing Statistics:")
    print(stats.head(20))

    # Show summary
    print(f"\nTotal symbols processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('features') is not None)}")
    print(f"Failed: {sum(1 for r in results.values() if r.get('error') is not None)}")

    return results


def process_subset_example():
    """Process a subset of symbols as an example"""
    processor = WatchlistFeatureProcessor(use_gpu=True)

    # Process first 5 symbols
    subset = WATCHLIST[:5]
    results = processor.process_watchlist_parallel(subset)

    # Show detailed results for each symbol
    for symbol, result in results.items():
        if result.get('features') is not None:
            features = result['features']
            print(f"\n{symbol}:")
            print(f"  Features: {len(features.columns)}")
            print(f"  Data points: {len(features)}")

            # Show some key features
            if 'golden_cross' in features:
                golden_crosses = features['golden_cross'].sum()
                print(f"  Golden crosses: {golden_crosses}")

            if 'rsi_14' in features:
                current_rsi = features['rsi_14'].iloc[-1]
                print(f"  Current RSI(14): {current_rsi:.2f}")

            if 'volume_ratio_20' in features:
                current_vol_ratio = features['volume_ratio_20'].iloc[-1]
                print(f"  Current Volume Ratio: {current_vol_ratio:.2f}")


def check_feature_consistency():
    """Check that all symbols produce the same features"""
    processor = WatchlistFeatureProcessor(use_gpu=True)

    # Process a few symbols
    test_symbols = WATCHLIST[:3]
    results = processor.process_watchlist_parallel(test_symbols)

    # Compare feature names
    feature_sets = []
    for symbol, result in results.items():
        if result.get('features') is not None:
            feature_sets.append(set(result['features'].columns))

    if feature_sets:
        # Check if all symbols have the same features
        common_features = set.intersection(*feature_sets)
        print(f"\nCommon features across all symbols: {len(common_features)}")

        # Check for any differences
        for i, symbol in enumerate(test_symbols):
            if results[symbol].get('features') is not None:
                unique_features = feature_sets[i] - common_features
                if unique_features:
                    print(f"{symbol} has {len(unique_features)} unique features")


def load_and_update_features(symbol: str):
    """Load existing features and update with latest data"""
    processor = WatchlistFeatureProcessor(use_gpu=True)

    # Try to load existing features
    existing_features = processor.load_features(symbol)

    if existing_features.empty:
        print(f"No existing features for {symbol}, creating new...")
        result = processor.process_single_symbol(symbol)
    else:
        print(f"Loaded existing features for {symbol}")
        # Fetch latest data and update
        latest_data = processor.fetch_data_for_symbol(symbol, period="1mo")
        result = processor.process_single_symbol(symbol, latest_data)

    return result


if __name__ == "__main__":
    # Process full watchlist
    results = process_full_watchlist()

    # Or process a subset
    # process_subset_example()

    # Check consistency
    # check_feature_consistency()