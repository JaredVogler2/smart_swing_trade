# sophisticated_config.py
"""
Sophisticated quant-style configuration
"""

class SophisticatedBacktestConfig:
    """Professional quant hedge fund style configuration"""
    
    # Capital and Risk
    initial_capital: float = 100000
    max_portfolio_heat: float = 0.06  # 6% max portfolio risk
    position_size_method: str = "kelly_criterion"  # Not fixed percentage
    
    # Multi-timeframe ensemble
    timeframes = {
        "short": {
            "train_days": 63,    # 3 months
            "retrain_days": 5,   # Weekly
            "weight": 0.3,
            "min_data": 126      # 6 months
        },
        "medium": {
            "train_days": 252,   # 1 year  
            "retrain_days": 21,  # Monthly
            "weight": 0.5,
            "min_data": 378      # 1.5 years
        },
        "long": {
            "train_days": 504,   # 2 years
            "retrain_days": 63,  # Quarterly  
            "weight": 0.2,
            "min_data": 756      # 3 years
        }
    }
    
    # Purging and Embargo (prevent leakage)
    purge_days: int = 3      # Remove 3 days around train/test split
    embargo_days: int = 2    # Skip 2 days after training period
    
    # Regime-based adaptation
    volatility_lookback: int = 20
    volatility_threshold: float = 0.02  # 2% daily vol = high regime
    high_vol_train_reduction: float = 0.5  # Use 50% less data in high vol
    
    # Feature engineering
    feature_selection_method: str = "boruta"  # Advanced selection
    max_features_per_model: int = 50  # Prevent overfitting
    feature_importance_threshold: float = 0.001
    
    # Execution realism
    slippage_model: str = "square_root"  # Slippage = base + k * sqrt(size)
    market_impact_const: float = 0.1  # 10bps for $1M traded
    fill_probability: dict = {
        "limit_aggressive": 0.3,   # 30% fill rate
        "limit_passive": 0.7,      # 70% fill rate  
        "market": 0.95            # 95% fill rate
    }
    
    # Model validation
    min_sharpe_to_trade: float = 1.5
    min_samples_per_class: int = 100
    walk_forward_optimization: bool = True
    use_combinatorial_purging: bool = True
    
    # Risk limits
    max_leverage: float = 1.0  # No leverage for retail
    max_sector_concentration: float = 0.4  # 40% max in one sector
    correlation_limit: float = 0.6  # Max correlation between positions
    
    # Advanced ML settings
    use_gpu: bool = True
    ensemble_method: str = "stacking"  # Not simple averaging
    meta_learner: str = "lightgbm"     # Meta model to combine predictions

print("Sophisticated configuration that quant funds would use:")
print("- Multiple timeframes with different weights")
print("- Purging and embargo to prevent leakage")  
print("- Regime-based adaptation")
print("- Realistic execution modeling")
print("- Advanced feature selection")
print("- Strict risk limits")
