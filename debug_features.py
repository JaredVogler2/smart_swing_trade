# debug_features.py
import yfinance as yf
import pandas as pd
import numpy as np
from models.features import FeatureEngineer

# Test with one symbol
symbol = 'AAPL'
print(f"Testing feature creation for {symbol}")

# Get data
ticker = yf.Ticker(symbol)
df = ticker.history(period="2y")
df.columns = [col.lower() for col in df.columns]
df = df[['open', 'high', 'low', 'close', 'volume']]

print(f"Data shape: {df.shape}")
print(f"Data range: {df.index[0]} to {df.index[-1]}")
print(f"Data columns: {df.columns.tolist()}")
print(f"First few rows:\n{df.head()}")
print(f"Any NaN values: {df.isna().any().any()}")

# Create feature engineer
fe = FeatureEngineer(enable_gpu=False)

# Try creating features
print("\nCreating features...")
features = fe.create_features(df, symbol)

print(f"\nFeatures shape: {features.shape}")
if not features.empty:
    print(f"Feature columns ({len(features.columns)}): {features.columns.tolist()[:10]}...")
    print(f"NaN counts per feature:\n{features.isna().sum().sort_values(ascending=False).head(20)}")
else:
    print("Features are empty!")

# Debug the feature creation process
print("\nDebugging individual feature groups...")

# Test price features
try:
    price_features = fe._create_price_features(df)
    print(f"Price features: {len(price_features)} created")
except Exception as e:
    print(f"Price features error: {e}")

# Test volume features
try:
    volume_features = fe._create_volume_features(df)
    print(f"Volume features: {len(volume_features)} created")
except Exception as e:
    print(f"Volume features error: {e}")

# Test technical indicators
try:
    technical_features = fe._create_technical_indicators(df)
    print(f"Technical features: {len(technical_features)} created")
except Exception as e:
    print(f"Technical features error: {e}")