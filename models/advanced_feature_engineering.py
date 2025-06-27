# advanced_feature_engineering.py
"""
Advanced Feature Engineering with 200+ features including feature interactions
Optimized for GPU acceleration when available
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
import torch
import logging
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# GPU acceleration imports
try:
    import cupy as cp
    import cudf

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Creates 200+ advanced features including:
    - Technical indicators with multiple timeframes
    - Feature interactions (golden cross, death cross, etc.)
    - Market microstructure features
    - Statistical and ML-derived features
    - Regime detection features
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE and torch.cuda.is_available()
        self.scaler = StandardScaler()
        self.feature_names = []

        if self.use_gpu:
            logger.info("GPU-accelerated feature engineering enabled")
            self.device = torch.device('cuda')
        else:
            logger.info("Using CPU for feature engineering")
            self.device = torch.device('cpu')

    def create_all_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Create comprehensive feature set (200+ features)"""

        if len(df) < 500:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows (need 500+)")
            return pd.DataFrame()

        # Ensure we have lowercase column names
        df.columns = df.columns.str.lower()

        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index)

        # 1. Price Action Features (30+ features)
        features.update(self._create_price_features(df))

        # 2. Volume Features (25+ features)
        features.update(self._create_volume_features(df))

        # 3. Volatility Features (25+ features)
        features.update(self._create_volatility_features(df))

        # 4. Technical Indicators (40+ features)
        features.update(self._create_technical_indicators(df))

        # 5. Moving Average Features & Crosses (20+ features)
        features.update(self._create_ma_features(df))

        # 6. Market Microstructure (15+ features)
        features.update(self._create_microstructure_features(df))

        # 7. Statistical Features (20+ features)
        features.update(self._create_statistical_features(df))

        # 8. Pattern Recognition (15+ features)
        features.update(self._create_pattern_features(df))

        # 9. Interaction Features (20+ features) - INCLUDING GOLDEN CROSS
        features.update(self._create_interaction_features(df, features))

        # 10. Regime Features (10+ features)
        features.update(self._create_regime_features(df))

        # 11. Advanced ML Features (10+ features)
        features.update(self._create_ml_features(df, features))

        # Store feature names
        self.feature_names = features.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} features for {symbol}")

        # Handle missing values
        features = self._handle_missing_values(features)

        return features

    def _create_price_features(self, df: pd.DataFrame) -> Dict:
        """Create price-based features"""
        features = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values

        # Returns at multiple horizons
        for period in [1, 2, 3, 5, 7, 10, 14, 20, 30, 60, 120]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

        # Price relative to various anchors
        features['close_to_open'] = close / open_price
        features['close_to_high'] = close / high
        features['close_to_low'] = close / low
        features['high_low_spread'] = (high - low) / close
        features['body_size'] = abs(close - open_price) / close

        # Gaps
        prev_close = df['close'].shift(1)
        features['gap'] = open_price / prev_close
        features['gap_size'] = abs(features['gap'] - 1)
        features['gap_up'] = (features['gap'] > 1.002).astype(int)
        features['gap_down'] = (features['gap'] < 0.998).astype(int)

        # Price momentum
        for period in [5, 10, 20, 60]:
            features[f'momentum_{period}d'] = close / df['close'].shift(period).values

        # Price acceleration
        for period in [5, 10, 20]:
            mom1 = df['close'].pct_change(period)
            mom2 = mom1.shift(period)
            features[f'acceleration_{period}d'] = mom1 - mom2

        # Support and Resistance levels
        for period in [10, 20, 50, 100, 200]:
            resistance = df['high'].rolling(period).max()
            support = df['low'].rolling(period).min()

            features[f'resistance_{period}d'] = resistance
            features[f'support_{period}d'] = support
            features[f'dist_to_resistance_{period}d'] = (resistance - close) / close
            features[f'dist_to_support_{period}d'] = (close - support) / close
            features[f'sr_width_{period}d'] = (resistance - support) / close

        # Price channels
        for period in [10, 20, 50]:
            highest = df['high'].rolling(period).max()
            lowest = df['low'].rolling(period).min()
            features[f'channel_position_{period}d'] = (close - lowest) / (highest - lowest + 1e-10)

        # New highs/lows
        for period in [20, 50, 100, 252]:
            features[f'new_high_{period}d'] = (close == df['close'].rolling(period).max()).astype(int)
            features[f'new_low_{period}d'] = (close == df['close'].rolling(period).min()).astype(int)

        return features

    def _create_volume_features(self, df: pd.DataFrame) -> Dict:
        """Create volume-based features"""
        features = {}
        volume = df['volume'].values
        close = df['close'].values

        # Volume moving averages and ratios
        for period in [5, 10, 20, 50, 100]:
            vol_ma = talib.SMA(volume, timeperiod=period)
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = volume / (vol_ma + 1e-10)

        # Volume momentum
        for period in [5, 10, 20]:
            features[f'volume_momentum_{period}'] = talib.ROC(volume, timeperiod=period)

        # Price-Volume indicators
        features['obv'] = talib.OBV(close, volume)
        features['obv_ma20'] = talib.SMA(features['obv'], timeperiod=20)
        features['obv_divergence'] = self._calculate_divergence(close, features['obv'])

        # Accumulation/Distribution
        features['ad_line'] = talib.AD(df['high'].values, df['low'].values, close, volume)
        features['ad_oscillator'] = talib.ADOSC(df['high'].values, df['low'].values,
                                                close, volume, fastperiod=3, slowperiod=10)

        # Chaikin Money Flow
        for period in [10, 20]:
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            mfv = mfm * volume
            features[f'cmf_{period}'] = mfv.rolling(period).sum() / volume.rolling(period).sum()

        # Money Flow Index
        for period in [14, 20]:
            features[f'mfi_{period}'] = talib.MFI(df['high'].values, df['low'].values,
                                                  close, volume, timeperiod=period)

        # Volume-weighted average price (VWAP)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['price_to_vwap'] = close / features['vwap']

        # Volume profile
        features['volume_mean_20'] = volume.rolling(20).mean()
        features['volume_std_20'] = volume.rolling(20).std()
        features['volume_skew_20'] = volume.rolling(20).skew()
        features['volume_kurt_20'] = volume.rolling(20).kurt()

        # Volume spikes
        vol_zscore = (volume - features['volume_mean_20']) / (features['volume_std_20'] + 1e-10)
        features['volume_spike'] = (vol_zscore > 2).astype(int)
        features['volume_dry_up'] = (vol_zscore < -1).astype(int)

        # Force Index
        for period in [13, 20]:
            features[f'force_index_{period}'] = close.diff() * volume
            features[f'force_index_ma_{period}'] = features[f'force_index_{period}'].rolling(period).mean()

        return features

    def _create_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Create volatility features"""
        features = {}
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_price = df['open'].values

        # ATR variations
        for period in [7, 14, 20, 30, 50]:
            atr = talib.ATR(high, low, close, timeperiod=period)
            features[f'atr_{period}'] = atr
            features[f'atr_pct_{period}'] = atr / close
            features[f'atr_ma_{period}'] = talib.SMA(atr, timeperiod=period)

        # Bollinger Bands with different parameters
        for period in [10, 20, 30]:
            for num_std in [1.5, 2.0, 2.5, 3.0]:
                upper, middle, lower = talib.BBANDS(close, timeperiod=period,
                                                    nbdevup=num_std, nbdevdn=num_std)
                suffix = f'{period}_{int(num_std * 10)}'

                features[f'bb_upper_{suffix}'] = upper
                features[f'bb_lower_{suffix}'] = lower
                features[f'bb_width_{suffix}'] = (upper - lower) / middle
                features[f'bb_position_{suffix}'] = (close - lower) / (upper - lower + 1e-10)

                # Bollinger Band squeeze
                atr = talib.ATR(high, low, close, timeperiod=period)
                kc_upper = middle + (2 * atr)
                kc_lower = middle - (2 * atr)
                features[f'bb_squeeze_{suffix}'] = ((upper < kc_upper) & (lower > kc_lower)).astype(int)

        # Historical volatility
        returns = df['close'].pct_change()
        for period in [5, 10, 20, 30, 60, 100]:
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_ma_{period}d'] = features[f'volatility_{period}d'].rolling(20).mean()

        # Parkinson volatility
        features['parkinson_vol_20'] = np.sqrt(
            252 / (4 * np.log(2)) * (np.log(high / low) ** 2).rolling(20).mean()
        )

        # Garman-Klass volatility
        features['garman_klass_vol_20'] = np.sqrt(
            252 * (0.5 * (np.log(high / low) ** 2).rolling(20).mean() -
                   (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2).rolling(20).mean())
        )

        # Average True Range Percent Rank
        for period in [14, 20]:
            atr = talib.ATR(high, low, close, timeperiod=period)
            features[f'atr_percentile_{period}'] = atr.rolling(252).rank(pct=True)

        return features

    def _create_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Create technical indicators"""
        features = {}
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        # RSI variations
        for period in [7, 14, 21, 28, 50]:
            rsi = talib.RSI(close, timeperiod=period)
            features[f'rsi_{period}'] = rsi
            features[f'rsi_{period}_ma'] = talib.SMA(rsi, timeperiod=10)
            features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
            features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)

        # MACD variations
        macd_params = [(12, 26, 9), (5, 35, 5), (8, 17, 9)]
        for fast, slow, signal in macd_params:
            macd, macd_signal, macd_hist = talib.MACD(close, fast, slow, signal)
            suffix = f'{fast}_{slow}_{signal}'

            features[f'macd_{suffix}'] = macd
            features[f'macd_signal_{suffix}'] = macd_signal
            features[f'macd_hist_{suffix}'] = macd_hist
            features[f'macd_cross_up_{suffix}'] = ((macd > macd_signal) &
                                                   (macd.shift(1) <= macd_signal.shift(1))).astype(int)
            features[f'macd_cross_down_{suffix}'] = ((macd < macd_signal) &
                                                     (macd.shift(1) >= macd_signal.shift(1))).astype(int)

        # Stochastic oscillators
        for k_period in [14, 21]:
            for d_period in [3, 5]:
                k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowd_period=d_period)
                suffix = f'{k_period}_{d_period}'

                features[f'stoch_k_{suffix}'] = k
                features[f'stoch_d_{suffix}'] = d
                features[f'stoch_cross_{suffix}'] = ((k > d) & (k.shift(1) <= d.shift(1))).astype(int)

        # ADX and directional indicators
        for period in [14, 20, 30]:
            features[f'adx_{period}'] = talib.ADX(high, low, close, timeperiod=period)
            features[f'plus_di_{period}'] = talib.PLUS_DI(high, low, close, timeperiod=period)
            features[f'minus_di_{period}'] = talib.MINUS_DI(high, low, close, timeperiod=period)
            features[f'di_diff_{period}'] = features[f'plus_di_{period}'] - features[f'minus_di_{period}']

        # Williams %R
        for period in [10, 14, 20]:
            features[f'williams_r_{period}'] = talib.WILLR(high, low, close, timeperiod=period)

        # CCI
        for period in [14, 20, 30]:
            features[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)

        # Ultimate Oscillator
        features['ultimate_osc'] = talib.ULTOSC(high, low, close)

        # ROC
        for period in [5, 10, 20, 30]:
            features[f'roc_{period}'] = talib.ROC(close, timeperiod=period)

        # Aroon
        for period in [14, 25]:
            aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
            features[f'aroon_up_{period}'] = aroon_up
            features[f'aroon_down_{period}'] = aroon_down
            features[f'aroon_osc_{period}'] = aroon_up - aroon_down

        return features

    def _create_ma_features(self, df: pd.DataFrame) -> Dict:
        """Create moving average features including crosses"""
        features = {}
        close = df['close'].values

        # Simple Moving Averages
        sma_periods = [5, 10, 20, 50, 100, 200]
        smas = {}

        for period in sma_periods:
            sma = talib.SMA(close, timeperiod=period)
            smas[period] = sma
            features[f'sma_{period}'] = sma
            features[f'price_to_sma_{period}'] = close / sma
            features[f'sma_{period}_slope'] = (sma - sma.shift(5)) / 5

        # Exponential Moving Averages
        ema_periods = [8, 12, 21, 26, 50, 100, 200]
        emas = {}

        for period in ema_periods:
            ema = talib.EMA(close, timeperiod=period)
            emas[period] = ema
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = close / ema

        # GOLDEN CROSS and DEATH CROSS
        # Classic Golden Cross: 50-day SMA crosses above 200-day SMA
        if 50 in smas and 200 in smas:
            sma50 = pd.Series(smas[50])
            sma200 = pd.Series(smas[200])

            # Golden Cross
            features['golden_cross'] = ((sma50 > sma200) &
                                        (sma50.shift(1) <= sma200.shift(1))).astype(int)

            # Death Cross
            features['death_cross'] = ((sma50 < sma200) &
                                       (sma50.shift(1) >= sma200.shift(1))).astype(int)

            # Distance from golden/death cross
            features['sma50_sma200_ratio'] = sma50 / sma200
            features['sma50_sma200_diff'] = (sma50 - sma200) / close

        # EMA crosses
        if 12 in emas and 26 in emas:
            ema12 = pd.Series(emas[12])
            ema26 = pd.Series(emas[26])

            features['ema_golden_cross'] = ((ema12 > ema26) &
                                            (ema12.shift(1) <= ema26.shift(1))).astype(int)
            features['ema_death_cross'] = ((ema12 < ema26) &
                                           (ema12.shift(1) >= ema26.shift(1))).astype(int)

        # Multiple timeframe crosses
        ma_cross_pairs = [(20, 50), (10, 30), (5, 20)]
        for fast, slow in ma_cross_pairs:
            if fast in smas and slow in smas:
                fast_ma = pd.Series(smas[fast])
                slow_ma = pd.Series(smas[slow])

                features[f'sma_cross_up_{fast}_{slow}'] = ((fast_ma > slow_ma) &
                                                           (fast_ma.shift(1) <= slow_ma.shift(1))).astype(int)
                features[f'sma_cross_down_{fast}_{slow}'] = ((fast_ma < slow_ma) &
                                                             (fast_ma.shift(1) >= slow_ma.shift(1))).astype(int)
                features[f'sma_ratio_{fast}_{slow}'] = fast_ma / slow_ma

        # MA Ribbon
        if all(p in smas for p in [10, 20, 30, 40, 50]):
            # Check if MAs are in order (bullish alignment)
            ma_values = np.column_stack([smas[p] for p in [10, 20, 30, 40, 50]])
            sorted_ma = np.sort(ma_values, axis=1)
            features['ma_ribbon_bullish'] = (ma_values == sorted_ma[:, ::-1]).all(axis=1).astype(int)
            features['ma_ribbon_bearish'] = (ma_values == sorted_ma).all(axis=1).astype(int)

        # Hull Moving Average
        for period in [20, 50]:
            wma_half = talib.WMA(close, timeperiod=period // 2)
            wma_full = talib.WMA(close, timeperiod=period)
            hull_ma = talib.WMA(2 * wma_half - wma_full, timeperiod=int(np.sqrt(period)))
            features[f'hull_ma_{period}'] = hull_ma
            features[f'price_to_hull_{period}'] = close / hull_ma

        return features

    def _create_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Create market microstructure features"""
        features = {}

        # Bid-Ask Spread proxies
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        features['co_spread'] = abs(df['close'] - df['open']) / df['close']

        # Intraday momentum
        features['intraday_momentum'] = (df['close'] - df['open']) / df['open']
        features['intraday_range'] = (df['high'] - df['low']) / df['open']

        # Candle patterns
        body = df['close'] - df['open']
        body_abs = abs(body)
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']

        features['body_to_range'] = body_abs / (df['high'] - df['low'] + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (body_abs + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (body_abs + 1e-10)

        # Order flow imbalance proxy
        features['order_flow_imbalance'] = (df['close'] - df['open']) * df['volume']
        features['ofi_ma'] = features['order_flow_imbalance'].rolling(20).mean()

        # Microstructure noise
        features['noise_ratio'] = features['hl_spread'].rolling(20).std() / features['hl_spread'].rolling(20).mean()

        # Amihud illiquidity
        returns_abs = abs(df['close'].pct_change())
        features['amihud_illiquidity'] = returns_abs / (df['volume'] + 1)
        features['amihud_ma'] = features['amihud_illiquidity'].rolling(20).mean()

        # Roll's implied spread
        returns = df['close'].pct_change()
        features['roll_spread'] = 2 * np.sqrt(abs(returns.rolling(20).cov(returns.shift(1))))

        return features

    def _create_statistical_features(self, df: pd.DataFrame) -> Dict:
        """Create statistical features"""
        features = {}
        returns = df['close'].pct_change()

        # Rolling statistics
        for period in [5, 10, 20, 50, 100]:
            rolling_returns = returns.rolling(period)

            features[f'return_mean_{period}'] = rolling_returns.mean()
            features[f'return_std_{period}'] = rolling_returns.std()
            features[f'return_skew_{period}'] = rolling_returns.skew()
            features[f'return_kurt_{period}'] = rolling_returns.kurt()

            # Sharpe ratio
            features[f'sharpe_{period}'] = features[f'return_mean_{period}'] / (
                        features[f'return_std_{period}'] + 1e-10)

        # Z-score
        for period in [20, 50]:
            price_mean = df['close'].rolling(period).mean()
            price_std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - price_mean) / (price_std + 1e-10)

        # Autocorrelation
        for lag in [1, 2, 5, 10]:
            features[f'autocorr_lag{lag}'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=lag))

        # Hurst exponent
        for period in [50, 100]:
            features[f'hurst_{period}'] = df['close'].rolling(period).apply(
                lambda x: self._calculate_hurst_exponent(x.values)
            )

        # Efficiency ratio
        for period in [10, 20]:
            change = abs(df['close'] - df['close'].shift(period))
            volatility = abs(df['close'].diff()).rolling(period).sum()
            features[f'efficiency_ratio_{period}'] = change / (volatility + 1e-10)

        # Percentile rank
        for period in [20, 50, 100]:
            features[f'percentile_rank_{period}'] = df['close'].rolling(period).rank(pct=True)

        return features

    def _create_pattern_features(self, df: pd.DataFrame) -> Dict:
        """Create candlestick pattern features"""
        features = {}

        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Major candlestick patterns
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'inverted_hammer': talib.CDLINVERTEDHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing': talib.CDLENGULFING,
            'harami': talib.CDLHARAMI,
            'piercing': talib.CDLPIERCING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'spinning_top': talib.CDLSPINNINGTOP,
            'marubozu': talib.CDLMARUBOZU
        }

        for name, func in patterns.items():
            pattern_signal = func(open_price, high, low, close)
            features[f'cdl_{name}'] = pattern_signal / 100  # Normalize to -1, 0, 1

            # Pattern strength (time since last occurrence)
            pattern_series = pd.Series(pattern_signal)
            last_signal = pattern_series.replace(0, np.nan).fillna(method='ffill')
            features[f'cdl_{name}_days_since'] = pattern_series.index - last_signal.index

        # Custom patterns
        body = close - open_price
        body_abs = abs(body)
        range_hl = high - low

        # Pin bar
        features['pin_bar_bull'] = ((low - df['low'].shift(1).values < -range_hl * 0.66) &
                                    (body > 0) &
                                    (body_abs < range_hl * 0.3)).astype(int)

        features['pin_bar_bear'] = ((high - df['high'].shift(1).values > range_hl * 0.66) &
                                    (body < 0) &
                                    (body_abs < range_hl * 0.3)).astype(int)

        # Inside/Outside bars
        features['inside_bar'] = ((high < df['high'].shift(1)) &
                                  (low > df['low'].shift(1))).astype(int)

        features['outside_bar'] = ((high > df['high'].shift(1)) &
                                   (low < df['low'].shift(1))).astype(int)

        return features

    def _create_interaction_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create interaction features between indicators"""
        features = {}

        # GOLDEN CROSS INTERACTIONS
        if 'golden_cross' in base_features and 'volume_ratio_20' in base_features:
            # Golden cross with high volume confirmation
            features['golden_cross_volume_confirm'] = (
                    (base_features['golden_cross'] == 1) &
                    (base_features['volume_ratio_20'] > 1.5)
            ).astype(int)

        if 'golden_cross' in base_features and 'rsi_14' in base_features:
            # Golden cross with RSI confirmation
            features['golden_cross_rsi_confirm'] = (
                    (base_features['golden_cross'] == 1) &
                    (base_features['rsi_14'] > 50) &
                    (base_features['rsi_14'] < 70)
            ).astype(int)

        if 'death_cross' in base_features and 'rsi_14' in base_features:
            # Death cross with RSI confirmation
            features['death_cross_rsi_confirm'] = (
                    (base_features['death_cross'] == 1) &
                    (base_features['rsi_14'] < 50)
            ).astype(int)

        # Price-Volume interactions
        if 'return_5d' in base_features and 'volume_ratio_20' in base_features:
            features['price_volume_divergence'] = (
                    ((base_features['return_5d'] > 0.02) & (base_features['volume_ratio_20'] < 0.8)) |
                    ((base_features['return_5d'] < -0.02) & (base_features['volume_ratio_20'] > 1.2))
            ).astype(int)

            features['volume_price_momentum'] = (
                    base_features['return_5d'] * base_features['volume_ratio_20']
            )

        # RSI-MACD confluence
        if 'rsi_14' in base_features and 'macd_hist_12_26_9' in base_features:
            features['rsi_macd_bullish'] = (
                    (base_features['rsi_14'] > 50) &
                    (base_features['rsi_14'] < 70) &
                    (base_features['macd_hist_12_26_9'] > 0)
            ).astype(int)

            features['rsi_macd_bearish'] = (
                    (base_features['rsi_14'] < 50) &
                    (base_features['rsi_14'] > 30) &
                    (base_features['macd_hist_12_26_9'] < 0)
            ).astype(int)

        # Support/Resistance with indicators
        if 'dist_to_support_20d' in base_features and 'rsi_14' in base_features:
            features['support_bounce_setup'] = (
                    (base_features['dist_to_support_20d'] < 0.02) &
                    (base_features['rsi_14'] < 40)
            ).astype(int)

        if 'dist_to_resistance_20d' in base_features and 'rsi_14' in base_features:
            features['resistance_test_setup'] = (
                    (base_features['dist_to_resistance_20d'] < 0.02) &
                    (base_features['rsi_14'] > 60)
            ).astype(int)

        # Bollinger Band squeezes with ADX
        if 'bb_squeeze_20_20' in base_features and 'adx_14' in base_features:
            features['bb_squeeze_trending'] = (
                    (base_features['bb_squeeze_20_20'] == 1) &
                    (base_features['adx_14'] > 25)
            ).astype(int)

        # Multi-timeframe momentum alignment
        momentum_features = [col for col in base_features.columns if col.startswith('momentum_')]
        if len(momentum_features) >= 3:
            mom_values = base_features[momentum_features].values
            features['momentum_alignment'] = (mom_values > 1).sum(axis=1) / len(momentum_features)
            features['strong_momentum'] = (features['momentum_alignment'] > 0.8).astype(int)

        # Volatility regime interactions
        if 'volatility_20d' in base_features and 'volume_ratio_20' in base_features:
            vol_percentile = base_features['volatility_20d'].rolling(100).rank(pct=True)

            features['high_vol_high_volume'] = (
                    (vol_percentile > 0.8) &
                    (base_features['volume_ratio_20'] > 1.5)
            ).astype(int)

            features['low_vol_breakout'] = (
                    (vol_percentile < 0.2) &
                    (base_features['volume_ratio_20'] > 2)
            ).astype(int)

        return features

    def _create_regime_features(self, df: pd.DataFrame) -> Dict:
        """Create market regime features"""
        features = {}

        # Trend regime
        sma20 = talib.SMA(df['close'].values, 20)
        sma50 = talib.SMA(df['close'].values, 50)
        sma200 = talib.SMA(df['close'].values, 200)

        features['bull_regime'] = (
                (df['close'] > sma20) & (sma20 > sma50) & (sma50 > sma200)
        ).astype(int)

        features['bear_regime'] = (
                (df['close'] < sma20) & (sma20 < sma50) & (sma50 < sma200)
        ).astype(int)

        # Volatility regime
        returns = df['close'].pct_change()
        vol = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = vol.rolling(252).rank(pct=True)

        features['low_vol_regime'] = (vol_percentile < 0.3).astype(int)
        features['high_vol_regime'] = (vol_percentile > 0.7).astype(int)

        # Volume regime
        volume_ma = df['volume'].rolling(20).mean()
        volume_percentile = volume_ma.rolling(252).rank(pct=True)

        features['high_volume_regime'] = (volume_percentile > 0.7).astype(int)
        features['low_volume_regime'] = (volume_percentile < 0.3).astype(int)

        # Market efficiency
        efficiency_ratio = self._calculate_efficiency_ratio(df['close'], 20)
        features['trending_regime'] = (efficiency_ratio > 0.7).astype(int)
        features['ranging_regime'] = (efficiency_ratio < 0.3).astype(int)

        # Regime duration
        for regime in ['bull_regime', 'bear_regime', 'high_vol_regime', 'trending_regime']:
            if regime in features:
                regime_series = pd.Series(features[regime])
                regime_changes = regime_series.diff().fillna(0)
                regime_starts = (regime_changes == 1).cumsum()
                features[f'{regime}_duration'] = regime_series.groupby(regime_starts).cumsum()

        return features

    def _create_ml_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create ML-derived features"""
        features = {}

        # Non-linear transformations of key features
        key_features = ['rsi_14', 'macd_hist_12_26_9', 'volatility_20d', 'volume_ratio_20']

        for feat in key_features:
            if feat in base_features:
                # Polynomial features
                features[f'{feat}_squared'] = base_features[feat] ** 2
                features[f'{feat}_cubed'] = base_features[feat] ** 3
                features[f'{feat}_sqrt'] = np.sqrt(np.abs(base_features[feat]))
                features[f'{feat}_log'] = np.log(np.abs(base_features[feat]) + 1)

        # Interaction polynomials
        if 'rsi_14' in base_features and 'volume_ratio_20' in base_features:
            features['rsi_volume_poly2'] = base_features['rsi_14'] * base_features['volume_ratio_20']
            features['rsi_volume_poly3'] = features['rsi_volume_poly2'] * base_features['rsi_14']

        # PCA features (if enough base features)
        if len(base_features.columns) > 50:
            # Select numerical features for PCA
            pca_features = base_features.select_dtypes(include=[np.number]).fillna(0)

            if len(pca_features) > 0:
                pca = PCA(n_components=5)
                pca_components = pca.fit_transform(StandardScaler().fit_transform(pca_features))

                for i in range(5):
                    features[f'pca_component_{i + 1}'] = pca_components[:, i]

        # Fractal dimension
        for window in [30, 60]:
            features[f'fractal_dim_{window}'] = df['close'].rolling(window).apply(
                lambda x: self._calculate_fractal_dimension(x.values) if len(x) == window else 1.5
            )

        return features

    def _calculate_hurst_exponent(self, series):
        """Calculate Hurst exponent"""
        if len(series) < 20:
            return 0.5

        # R/S calculation
        mean = np.mean(series)
        std = np.std(series)

        if std == 0:
            return 0.5

        cumsum = np.cumsum(series - mean)
        R = np.max(cumsum) - np.min(cumsum)
        S = std

        if S == 0:
            return 0.5

        return np.log(R / S) / np.log(len(series))

    def _calculate_fractal_dimension(self, series):
        """Calculate fractal dimension using box counting"""
        if len(series) < 10:
            return 1.5

        # Normalize series
        series_norm = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-10)

        # Box counting
        n = len(series)
        max_box_size = n // 2
        box_sizes = [2 ** i for i in range(1, int(np.log2(max_box_size)))]
        counts = []

        for box_size in box_sizes:
            count = 0
            for i in range(0, n, box_size):
                if i + box_size <= n:
                    box_range = np.ptp(series_norm[i:i + box_size])
                    if box_range > 0:
                        count += 1
            counts.append(count)

        if len(counts) > 1 and len(box_sizes) > 1:
            # Calculate slope
            log_sizes = np.log(box_sizes)
            log_counts = np.log(counts)

            # Linear regression
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            return -coeffs[0]
        else:
            return 1.5

    def _calculate_divergence(self, price_series, indicator_series, lookback=14):
        """Calculate divergence between price and indicator"""
        price_highs = price_series.rolling(lookback).max()
        price_lows = price_series.rolling(lookback).min()

        ind_highs = indicator_series.rolling(lookback).max()
        ind_lows = indicator_series.rolling(lookback).min()

        # Bearish divergence
        bearish_div = (
                (price_highs > price_highs.shift(lookback)) &
                (ind_highs < ind_highs.shift(lookback))
        ).astype(int)

        # Bullish divergence
        bullish_div = (
                (price_lows < price_lows.shift(lookback)) &
                (ind_lows > ind_lows.shift(lookback))
        ).astype(int)

        return bullish_div.astype(float) - bearish_div.astype(float)

    def _calculate_efficiency_ratio(self, prices, period=10):
        """Calculate Kaufman's Efficiency Ratio"""
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        change = abs(prices - prices.shift(period))
        volatility = abs(prices.diff()).rolling(period).sum()

        return change / (volatility + 1e-10)

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        # Forward fill for time series continuity
        features = features.fillna(method='ffill', limit=5)

        # Backward fill
        features = features.fillna(method='bfill', limit=5)

        # Fill remaining with appropriate defaults
        for col in features.columns:
            if features[col].isna().any():
                if 'ratio' in col or 'pct' in col:
                    features[col].fillna(1, inplace=True)
                elif 'rsi' in col:
                    features[col].fillna(50, inplace=True)
                else:
                    features[col].fillna(0, inplace=True)

        return features

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features organized by importance groups"""
        groups = {
            'critical': [],  # Golden cross, major signals
            'primary': [],  # Core technical indicators
            'secondary': [],  # Supporting indicators
            'auxiliary': []  # Additional features
        }

        # Critical features (highest importance)
        critical_keywords = ['golden_cross', 'death_cross', 'macd_cross', 'support_bounce',
                             'resistance_test', 'breakout', 'reversal']

        # Primary features
        primary_keywords = ['rsi_', 'macd_', 'bb_', 'sma_', 'ema_', 'volume_ratio',
                            'momentum_', 'volatility_', 'atr_']

        # Secondary features
        secondary_keywords = ['stoch_', 'cci_', 'williams_', 'aroon_', 'adx_', 'force_']

        for feature in self.feature_names:
            if any(keyword in feature for keyword in critical_keywords):
                groups['critical'].append(feature)
            elif any(keyword in feature for keyword in primary_keywords):
                groups['primary'].append(feature)
            elif any(keyword in feature for keyword in secondary_keywords):
                groups['secondary'].append(feature)
            else:
                groups['auxiliary'].append(feature)

        return groups


# Example usage
if __name__ == "__main__":
    # Test the feature engineer
    import yfinance as yf

    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y", interval="1d")

    # Create features
    engineer = AdvancedFeatureEngineer(use_gpu=torch.cuda.is_available())
    features = engineer.create_all_features(df, "AAPL")

    print(f"Created {len(features.columns)} features")
    print(f"Feature shape: {features.shape}")

    # Show feature groups
    groups = engineer.get_feature_importance_groups()
    print(f"\nCritical features ({len(groups['critical'])}): {groups['critical'][:5]}")
    print(f"Primary features ({len(groups['primary'])}): {groups['primary'][:5]}")
    print(f"Secondary features ({len(groups['secondary'])}): {groups['secondary'][:5]}")
    print(f"Auxiliary features ({len(groups['auxiliary'])}): {groups['auxiliary'][:5]}")