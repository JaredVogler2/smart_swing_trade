with open("advanced_backtesting.py", "r") as f:
    content = f.read()

# Find the BacktestConfig class and enhance it
import re

# Add purging and embargo to the config
config_addition = """    
    # Sophisticated quant-style settings
    purge_days: int = 3          # Remove days around train/test boundary
    embargo_days: int = 2        # Wait days before trading after training
    
    # Multi-timeframe settings (for future enhancement)
    use_multi_timeframe: bool = False
    timeframes: dict = None"""

# Find the class definition and add the new fields
pattern = r"(class BacktestConfig:.*?)(# Walk-forward)"
replacement = r"\1" + config_addition + r"\n    \2"

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("advanced_backtesting.py", "w") as f:
    f.write(content)

print("Added sophisticated config options")
