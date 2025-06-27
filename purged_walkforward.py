import numpy as np
import pandas as pd
from typing import List, Tuple

class PurgedWalkForward:
    """
    Sophisticated walk-forward with purging and embargo
    Used by professional quant funds
    """
    
    def __init__(self, purge_days: int = 3, embargo_days: int = 2):
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def generate_splits(self, 
                       data: pd.DataFrame,
                       train_days: int,
                       test_days: int,
                       retrain_freq: int) -> List[Tuple]:
        """
        Generate train/test splits with purging and embargo
        """
        splits = []
        dates = data.index
        
        # Start after we have enough training data
        start_idx = train_days + self.purge_days
        
        while start_idx + test_days < len(dates):
            # Training period
            train_start = start_idx - train_days - self.purge_days
            train_end = start_idx - self.purge_days - 1
            
            # Embargo period (skip these days)
            embargo_end = start_idx + self.embargo_days
            
            # Test period  
            test_start = embargo_end
            test_end = min(test_start + test_days, len(dates) - 1)
            
            if test_end > test_start:
                train_indices = list(range(train_start, train_end + 1))
                test_indices = list(range(test_start, test_end + 1))
                
                splits.append({
                    "train": train_indices,
                    "test": test_indices,
                    "train_dates": (dates[train_start], dates[train_end]),
                    "test_dates": (dates[test_start], dates[test_end])
                })
            
            # Move forward by retrain frequency
            start_idx += retrain_freq
        
        return splits

# Example usage
print("Purged Walk-Forward Cross-Validation Example:")
print("=" * 50)

# Simulate data
dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
data = pd.DataFrame(index=dates)

# Create splitter
splitter = PurgedWalkForward(purge_days=3, embargo_days=2)
splits = splitter.generate_splits(data, train_days=252, test_days=63, retrain_freq=21)

print(f"Generated {len(splits)} train/test splits")
print(f"\\nFirst split:")
print(f"  Train: {splits[0]['train_dates'][0].date()} to {splits[0]['train_dates'][1].date()}")
print(f"  Test:  {splits[0]['test_dates'][0].date()} to {splits[0]['test_dates'][1].date()}")
print(f"  Gap:   {(splits[0]['test_dates'][0] - splits[0]['train_dates'][1]).days} days (purge + embargo)")
