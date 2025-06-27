# test_fix.py
"""
Test the boolean operations fix
"""

import numpy as np
import pandas as pd

print("\nTesting boolean operations fix...")

# Test basic boolean operations
try:
    test_series = pd.Series([True, False, True, False, True])
    bearish = pd.Series([True, False, False, True, False])
    bullish = pd.Series([False, True, True, False, True])

    # This should work now
    result = bullish.astype(float) - bearish.astype(float)
    print("SUCCESS: Boolean operations working correctly")
    print(f"  Result: {result.tolist()}")
except Exception as e:
    print(f"FAILED: Boolean operations still failing: {e}")

# Test feature import
try:
    from models.features import FeatureEngineer

    print("SUCCESS: FeatureEngineer imported successfully")
except Exception as e:
    print(f"FAILED: Failed to import FeatureEngineer: {e}")

print("\nFix appears to be working. Try running the model training again.")