with open("models/enhanced_features.py", "r") as f:
    content = f.read()

# Make the row requirement adaptive based on available data
# Replace the hard-coded 500 check
import re

# Find and update the create_all_features method
old_check = "if len(df) < 500:"
new_check = """if len(df) < 100:  # Absolute minimum
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows (need 100+ minimum)")
            return pd.DataFrame()
        
        # Warn if less than ideal amount of data
        if len(df) < 500:
            logger.info(f"Limited data for {symbol}: {len(df)} rows - some features will be unavailable")"""

content = content.replace(old_check, new_check)

# Also make long-period features conditional
# Find moving average loops and make them adaptive
ma_pattern = r"(for period in \[[^\]]+\]:)"
ma_replacement = r"\1\n            if len(df) < period + 10:  # Need extra data for calculation\n                continue"

content = re.sub(ma_pattern, ma_replacement, content)

with open("models/enhanced_features.py", "w") as f:
    f.write(content)

print("Updated feature engineering to be adaptive:")
print("- Minimum 100 rows (absolute minimum)")
print("- Features requiring long lookbacks will be skipped if insufficient data")
print("- System will work with available data rather than failing")
