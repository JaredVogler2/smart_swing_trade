# debug_nan_issues.py
"""
Diagnostic script to identify and fix NaN issues in the trading system
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.watchlist import WATCHLIST
from data.market_data import MarketDataManager
from models.features import FeatureEngineer
from models.ensemble_gpu_windows_fixed import GPUEnsembleModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaNDiagnostic:
    """Diagnose and fix NaN issues in the trading system"""

    def __init__(self):
        self.market_data = MarketDataManager()
        self.feature_engineer = FeatureEngineer(enable_gpu=False)
        self.nan_report = {
            'data_issues': {},
            'feature_issues': {},
            'model_issues': {},
            'recommendations': []
        }

    def run_full_diagnostic(self, symbols: list = None, days: int = 100):
        """Run comprehensive NaN diagnostic"""
        symbols = symbols or WATCHLIST[:10]  # Test with first 10 symbols

        logger.info(f"Running NaN diagnostic for {len(symbols)} symbols over {days} days")

        # Step 1: Check raw data
        self._check_raw_data(symbols, days)

        # Step 2: Check feature engineering
        self._check_feature_engineering(symbols, days)

        # Step 3: Check model training data
        self._check_model_training_data(symbols, days)

        # Step 4: Generate report
        self._generate_report()

        return self.nan_report

    def _check_raw_data(self, symbols: list, days: int):
        """Check for NaN in raw market data"""
        logger.info("Checking raw market data for NaN values...")

        for symbol in symbols:
            try:
                # Fetch data
                df = self.market_data.get_bars(symbol, '1Day', limit=days)

                if df.empty:
                    self.nan_report['data_issues'][symbol] = "No data retrieved"
                    continue

                # Check for NaN
                nan_counts = df.isna().sum()
                if nan_counts.sum() > 0:
                    self.nan_report['data_issues'][symbol] = {
                        'nan_columns': nan_counts[nan_counts > 0].to_dict(),
                        'total_nans': nan_counts.sum(),
                        'nan_percentage': (nan_counts.sum() / (len(df) * len(df.columns))) * 100
                    }

                # Check for data quality issues
                quality_issues = []

                # Check for zero/negative prices
                if (df['close'] <= 0).any():
                    quality_issues.append("Zero or negative close prices")

                # Check for extreme price jumps
                returns = df['close'].pct_change()
                if (returns.abs() > 0.5).any():  # 50% daily move
                    quality_issues.append("Extreme price jumps detected")

                # Check for zero volume
                if (df['volume'] == 0).any():
                    zero_vol_days = (df['volume'] == 0).sum()
                    quality_issues.append(f"Zero volume on {zero_vol_days} days")

                if quality_issues:
                    if symbol not in self.nan_report['data_issues']:
                        self.nan_report['data_issues'][symbol] = {}
                    self.nan_report['data_issues'][symbol]['quality_issues'] = quality_issues

            except Exception as e:
                self.nan_report['data_issues'][symbol] = f"Error: {str(e)}"

        logger.info(f"Found data issues in {len(self.nan_report['data_issues'])} symbols")

    def _check_feature_engineering(self, symbols: list, days: int):
        """Check for NaN in engineered features"""
        logger.info("Checking feature engineering for NaN values...")

        feature_nan_summary = {
            'total_features': 0,
            'features_with_nan': 0,
            'worst_features': {},
            'symbol_issues': {}
        }

        for symbol in symbols[:5]:  # Test on subset
            try:
                # Get data
                df = self.market_data.get_bars(symbol, '1Day', limit=days + 50)  # Extra for features

                if df.empty or len(df) < 50:
                    continue

                # Create features
                features = self.feature_engineer.create_features(df, symbol)

                if features.empty:
                    feature_nan_summary['symbol_issues'][symbol] = "No features created"
                    continue

                # Analyze NaN in features
                nan_counts = features.isna().sum()
                nan_pct = (nan_counts / len(features)) * 100

                # Track total features
                if feature_nan_summary['total_features'] == 0:
                    feature_nan_summary['total_features'] = len(features.columns)

                # Features with NaN
                features_with_nan = nan_counts[nan_counts > 0]
                if len(features_with_nan) > 0:
                    feature_nan_summary['features_with_nan'] = max(
                        feature_nan_summary['features_with_nan'],
                        len(features_with_nan)
                    )

                    # Track worst features
                    for feat, count in features_with_nan.items():
                        pct = nan_pct[feat]
                        if feat not in feature_nan_summary['worst_features'] or \
                                pct > feature_nan_summary['worst_features'][feat]:
                            feature_nan_summary['worst_features'][feat] = pct

                    feature_nan_summary['symbol_issues'][symbol] = {
                        'features_with_nan': len(features_with_nan),
                        'max_nan_pct': nan_pct[features_with_nan.index].max(),
                        'most_problematic': nan_pct[features_with_nan.index].nlargest(5).to_dict()
                    }

                # Check for infinite values
                inf_counts = np.isinf(features.select_dtypes(include=[np.number])).sum()
                inf_features = inf_counts[inf_counts > 0]
                if len(inf_features) > 0:
                    if symbol not in feature_nan_summary['symbol_issues']:
                        feature_nan_summary['symbol_issues'][symbol] = {}
                    feature_nan_summary['symbol_issues'][symbol]['infinite_values'] = inf_features.to_dict()

            except Exception as e:
                feature_nan_summary['symbol_issues'][symbol] = f"Error: {str(e)}"

        self.nan_report['feature_issues'] = feature_nan_summary

        # Sort worst features
        if feature_nan_summary['worst_features']:
            worst_sorted = sorted(
                feature_nan_summary['worst_features'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            self.nan_report['feature_issues']['worst_features_sorted'] = worst_sorted

        logger.info(f"Found {feature_nan_summary['features_with_nan']} features with NaN "
                    f"out of {feature_nan_summary['total_features']} total features")

    def _check_model_training_data(self, symbols: list, days: int):
        """Check for NaN in model training pipeline"""
        logger.info("Checking model training data preparation...")

        # Prepare training data like the model would
        train_data = {}

        for symbol in symbols[:10]:
            try:
                df = self.market_data.get_bars(symbol, '1Day', limit=days + 200)
                if not df.empty and len(df) >= 200:
                    train_data[symbol] = df
            except:
                pass

        if len(train_data) < 5:
            self.nan_report['model_issues']['error'] = "Insufficient symbols for training test"
            return

        # Test the training data preparation
        model = GPUEnsembleModel()

        # Collect all features and targets
        all_features = []
        all_targets = []

        for symbol, df in train_data.items():
            try:
                # Create features
                features = model.feature_engineer.create_features(df, symbol)

                if features.empty:
                    continue

                # Create target
                target = model._create_advanced_target(df)

                # Align
                features = features[:-model.prediction_horizon]
                target = target[:-model.prediction_horizon]

                # Check validity
                valid_idx = ~(features.isna().any(axis=1) | target.isna())

                nan_features_count = features.isna().any(axis=1).sum()
                nan_target_count = target.isna().sum()
                invalid_count = (~valid_idx).sum()

                if invalid_count > 0:
                    self.nan_report['model_issues'][symbol] = {
                        'nan_features_rows': nan_features_count,
                        'nan_target_rows': nan_target_count,
                        'invalid_rows': invalid_count,
                        'valid_percentage': (valid_idx.sum() / len(valid_idx)) * 100
                    }

                # Add valid data
                features_valid = features[valid_idx]
                target_valid = target[valid_idx]

                if len(features_valid) >= 50:
                    all_features.append(features_valid)
                    all_targets.append(target_valid)

            except Exception as e:
                self.nan_report['model_issues'][symbol] = f"Error: {str(e)}"

        # Check combined data
        if all_features:
            X_combined = pd.concat(all_features)
            y_combined = pd.concat(all_targets)

            # Class distribution
            class_dist = y_combined.value_counts()
            class_ratio = class_dist[1] / len(y_combined) if 1 in class_dist else 0

            self.nan_report['model_issues']['combined_data'] = {
                'total_samples': len(X_combined),
                'total_features': len(X_combined.columns),
                'class_distribution': class_dist.to_dict(),
                'positive_class_ratio': class_ratio,
                'imbalance_ratio': max(class_dist) / min(class_dist) if len(class_dist) == 2 else 'N/A'
            }

            # Check for NaN after scaling
            try:
                scaler = model.scaler
                X_scaled = scaler.fit_transform(X_combined)
                nan_after_scaling = np.isnan(X_scaled).sum()

                self.nan_report['model_issues']['scaling'] = {
                    'nan_after_scaling': nan_after_scaling,
                    'inf_after_scaling': np.isinf(X_scaled).sum()
                }
            except Exception as e:
                self.nan_report['model_issues']['scaling'] = f"Error: {str(e)}"

    def _generate_report(self):
        """Generate recommendations based on findings"""
        recommendations = []

        # Data quality recommendations
        if self.nan_report['data_issues']:
            recommendations.append("DATA QUALITY ISSUES FOUND:")

            zero_volume_symbols = [s for s, issues in self.nan_report['data_issues'].items()
                                   if isinstance(issues, dict) and 'quality_issues' in issues
                                   and any('Zero volume' in issue for issue in issues['quality_issues'])]

            if zero_volume_symbols:
                recommendations.append(f"- Remove symbols with zero volume days: {zero_volume_symbols[:5]}")

            extreme_jump_symbols = [s for s, issues in self.nan_report['data_issues'].items()
                                    if isinstance(issues, dict) and 'quality_issues' in issues
                                    and any('Extreme price jumps' in issue for issue in issues['quality_issues'])]

            if extreme_jump_symbols:
                recommendations.append(f"- Investigate extreme price jumps in: {extreme_jump_symbols[:5]}")

        # Feature engineering recommendations
        if self.nan_report['feature_issues'].get('worst_features_sorted'):
            recommendations.append("\nFEATURE ENGINEERING ISSUES:")
            recommendations.append("- Top features with NaN values:")

            for feat, pct in self.nan_report['feature_issues']['worst_features_sorted'][:10]:
                recommendations.append(f"  * {feat}: {pct:.1f}% NaN")

                # Specific recommendations
                if 'hurst' in feat:
                    recommendations.append("    → Hurst exponent needs more data points")
                elif 'autocorr' in feat:
                    recommendations.append("    → Autocorrelation needs sufficient lag data")
                elif 'efficiency_ratio' in feat:
                    recommendations.append("    → Efficiency ratio needs volatility handling")

        # Model training recommendations
        if self.nan_report['model_issues'].get('combined_data'):
            data = self.nan_report['model_issues']['combined_data']

            recommendations.append("\nMODEL TRAINING ISSUES:")

            if data['positive_class_ratio'] < 0.1:
                recommendations.append(f"- Severe class imbalance: {data['positive_class_ratio']:.1%} positive")
                recommendations.append("  → Reduce target thresholds for more balanced classes")
                recommendations.append("  → Use SMOTE or other oversampling techniques")

            if data['imbalance_ratio'] != 'N/A' and data['imbalance_ratio'] > 10:
                recommendations.append(f"- Imbalance ratio {data['imbalance_ratio']:.1f}:1 is too high")
                recommendations.append("  → Current fix: Use class weights and weighted sampling")

        # General recommendations
        recommendations.extend([
            "\nGENERAL RECOMMENDATIONS:",
            "1. Use the fixed ensemble model (ensemble_gpu_windows_fixed.py)",
            "2. Ensure minimum 200 days of data for each symbol",
            "3. Filter out symbols with poor data quality before training",
            "4. Monitor NaN values during feature creation",
            "5. Use robust scaling and NaN handling in preprocessing",
            "6. Implement gradient clipping in LSTM training",
            "7. Use mixed precision training with proper error handling"
        ])

        self.nan_report['recommendations'] = recommendations

    def fix_common_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply common fixes for data issues"""
        df_fixed = df.copy()

        # Fix zero/negative prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df_fixed.columns:
                df_fixed[col] = df_fixed[col].replace(0, np.nan)
                df_fixed[col] = df_fixed[col].fillna(method='ffill')

        # Fix zero volume
        df_fixed['volume'] = df_fixed['volume'].replace(0, np.nan)
        df_fixed['volume'] = df_fixed['volume'].fillna(df_fixed['volume'].rolling(20, min_periods=1).median())

        # Remove extreme price jumps (likely data errors)
        returns = df_fixed['close'].pct_change()
        extreme_moves = returns.abs() > 0.5  # 50% daily move

        if extreme_moves.any():
            # Replace with NaN and forward fill
            for col in ['open', 'high', 'low', 'close']:
                df_fixed.loc[extreme_moves, col] = np.nan
            df_fixed = df_fixed.fillna(method='ffill')

        return df_fixed


def main():
    """Run the diagnostic"""
    diagnostic = NaNDiagnostic()

    # Run diagnostic on a subset of symbols
    test_symbols = WATCHLIST[:20]  # Test with first 20 symbols

    report = diagnostic.run_full_diagnostic(test_symbols, days=250)

    # Print report
    print("\n" + "=" * 80)
    print("NaN DIAGNOSTIC REPORT")
    print("=" * 80)

    # Data issues
    if report['data_issues']:
        print("\nDATA ISSUES:")
        for symbol, issues in list(report['data_issues'].items())[:10]:
            print(f"\n{symbol}:")
            print(f"  {issues}")

    # Feature issues
    if report['feature_issues']:
        print("\nFEATURE ENGINEERING ISSUES:")
        print(f"Total features: {report['feature_issues'].get('total_features', 0)}")
        print(f"Features with NaN: {report['feature_issues'].get('features_with_nan', 0)}")

        if report['feature_issues'].get('worst_features_sorted'):
            print("\nWorst features (% NaN):")
            for feat, pct in report['feature_issues']['worst_features_sorted'][:15]:
                print(f"  {feat}: {pct:.1f}%")

    # Model issues
    if report['model_issues']:
        print("\nMODEL TRAINING ISSUES:")
        if 'combined_data' in report['model_issues']:
            data = report['model_issues']['combined_data']
            print(f"Total samples: {data['total_samples']}")
            print(f"Class distribution: {data['class_distribution']}")
            print(f"Positive class ratio: {data['positive_class_ratio']:.2%}")
            print(f"Imbalance ratio: {data['imbalance_ratio']}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(rec)

    # Save detailed report
    import json
    with open('nan_diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\nDetailed report saved to nan_diagnostic_report.json")


if __name__ == "__main__":
    main()