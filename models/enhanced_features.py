# models/enhanced_features.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from numba import cuda, jit, prange
import warnings

# Try to import GPU libraries
try:
    import cupy as cp
    import cudf

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cudf = None

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """GPU-accelerated feature engineering with advanced techniques"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.feature_cache = {}
        self.scaler = RobustScaler()
        self.feature_names = []

        if self.use_gpu:
            logger.info("GPU-accelerated feature engineering enabled")
        else:
            logger.info("Using CPU for feature engineering")

    @staticmethod
    @cuda.jit
    def _gpu_rolling_stats(arr, window, out_mean, out_std, out_skew, out_kurt):
        """GPU kernel for rolling statistics"""
        idx = cuda.grid(1)
        n = arr.shape[0]

        if idx < n - window + 1:
            # Calculate mean
            sum_val = 0.0
            for i in range(window):
                sum_val += arr[idx + i]
            mean = sum_val / window
            out_mean[idx] = mean

            # Calculate variance and higher moments
            sum_sq = 0.0
            sum_cube = 0.0
            sum_quad = 0.0

            for i in range(window):
                diff = arr[idx + i] - mean
                diff_sq = diff * diff
                sum_sq += diff_sq
                sum_cube += diff_sq * diff
                sum_quad += diff_sq * diff_sq

            variance = sum_sq / window
            std = cuda.sqrt(variance)
            out_std[idx] = std

            # Skewness and kurtosis
            if std > 0:
                out_skew[idx] = (sum_cube / window) / (std ** 3)
                out_kurt[idx] = (sum_quad / window) / (variance ** 2) - 3
            else:
                out_skew[idx] = 0
                out_kurt[idx] = 0

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_rolling_stats(arr, window):
        """CPU-optimized rolling statistics using Numba"""
        n = len(arr)
        means = np.empty(n - window + 1)
        stds = np.empty(n - window + 1)

        for i in prange(n - window + 1):
            window_data = arr[i:i + window]
            means[i] = np.mean(window_data)
            stds[i] = np.std(window_data)

        return means, stds

    def create_all_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set with GPU acceleration"""

        if len(df) < 200:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            return pd.DataFrame()

        if self.use_gpu and CUPY_AVAILABLE:
            try:
                # Convert to GPU dataframe
                df_gpu = cudf.from_pandas(df)
                features = self._create_features_gpu(df_gpu, symbol)
                # Convert back to pandas
                return features.to_pandas()
            except Exception as e:
                logger.warning(f"GPU feature creation failed, falling back to CPU: {e}")
                return self._create_features_cpu(df, symbol)
        else:
            return self._create_features_cpu(df, symbol)

    def _add_features_to_dataframe(self, features_df: pd.DataFrame, new_features: Dict,
                                   index: pd.Index) -> None:
        """Helper method to properly add features to DataFrame"""
        for key, value in new_features.items():
            if isinstance(value, pd.Series):
                features_df[key] = value
            elif isinstance(value, pd.DataFrame):
                for col in value.columns:
                    features_df[col] = value[col]
            elif isinstance(value, np.ndarray):
                if len(value) == len(index):
                    features_df[key] = value
                else:
                    # Handle mismatched lengths by creating a Series with NaN
                    series = pd.Series(index=index, dtype=float)
                    series.iloc[:len(value)] = value
                    features_df[key] = series
            else:
                # Scalar value
                features_df[key] = value

    def _create_features_cpu(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """CPU-based feature creation"""
        features = pd.DataFrame(index=df.index)

        # FIX: Ensure all price/volume data is float64 for TA-Lib
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype('float64')

        # 1. Price-based features
        self._add_features_to_dataframe(features, self._create_price_features(df), df.index)

        # 2. Volume features
        self._add_features_to_dataframe(features, self._create_volume_features(df), df.index)

        # 3. Volatility features
        self._add_features_to_dataframe(features, self._create_volatility_features(df), df.index)

        # 4. Technical indicators
        self._add_features_to_dataframe(features, self._create_technical_indicators(df), df.index)

        # 5. Market microstructure
        self._add_features_to_dataframe(features, self._create_microstructure_features(df), df.index)

        # 6. Statistical features
        self._add_features_to_dataframe(features, self._create_statistical_features(df), df.index)

        # 7. Pattern recognition features
        self._add_features_to_dataframe(features, self._create_pattern_features(df), df.index)

        # 8. Interaction features
        self._add_features_to_dataframe(features, self._create_interaction_features(df, features), df.index)

        # 9. Market regime features
        self._add_features_to_dataframe(features, self._create_regime_features(df), df.index)

        # 10. Advanced ML features
        self._add_features_to_dataframe(features, self._create_ml_features(df, features), df.index)

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Handle missing values
        features = self._handle_missing_values(features)

        return features

    def _create_features_gpu(self, df, symbol: str):
        """GPU-accelerated feature creation"""
        if CUPY_AVAILABLE:
            features = cudf.DataFrame(index=df.index)
        else:
            # Fallback to pandas if GPU not available
            return self._create_features_cpu(df, symbol)

        # Convert to CuPy arrays for faster computation
        if CUPY_AVAILABLE and cp is not None:
            close = cp.asarray(df['close'].values)
            high = cp.asarray(df['high'].values)
            low = cp.asarray(df['low'].values)
            open_price = cp.asarray(df['open'].values)
            volume = cp.asarray(df['volume'].values)

            # Price features using GPU
            features = self._create_price_features_gpu(features, close, high, low, open_price)

            # Volume features using GPU
            features = self._create_volume_features_gpu(features, close, volume)

            # More GPU-accelerated features...
            # (Implementation continues similar to CPU version but using CuPy)

            return features
        else:
            # Fallback to CPU
            return self._create_features_cpu(df, symbol)

    def _create_price_features(self, df: pd.DataFrame) -> Dict:
        """Create price-based features"""
        features = {}

        # Ensure float64 for TA-Lib
        close = df['close'].astype('float64').values
        high = df['high'].astype('float64').values
        low = df['low'].astype('float64').values
        open_price = df['open'].astype('float64').values

        # Returns at multiple timeframes
        for period in [1, 2, 3, 5, 10, 20, 60]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                try:
                    sma = talib.SMA(close, timeperiod=period)
                    features[f'sma_{period}'] = sma
                    features[f'price_to_sma_{period}'] = close / sma

                    # MA slopes
                    features[f'sma_{period}_slope'] = (sma - talib.SMA(sma.astype('float64'), 5)) / 5
                except Exception as e:
                    logger.warning(f"Error calculating SMA {period}: {e}")
                    features[f'sma_{period}'] = close
                    features[f'price_to_sma_{period}'] = 1.0
                    features[f'sma_{period}_slope'] = 0.0

        # Exponential moving averages
        for period in [8, 12, 21, 26, 50]:
            if len(df) >= period:
                try:
                    ema = talib.EMA(close, timeperiod=period)
                    features[f'ema_{period}'] = ema
                    features[f'price_to_ema_{period}'] = close / ema
                except Exception as e:
                    logger.warning(f"Error calculating EMA {period}: {e}")
                    features[f'ema_{period}'] = close
                    features[f'price_to_ema_{period}'] = 1.0

        # VWAP approximation
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * df['volume'].astype('float64')).rolling(20).sum() / df['volume'].astype(
            'float64').rolling(20).sum()
        features['price_to_vwap'] = close / features['vwap']

        # Price positions and ranges
        features['close_to_high'] = close / high
        features['close_to_low'] = close / low
        features['high_low_range'] = (high - low) / close
        features['close_to_open'] = close / open_price
        features['body_size'] = abs(close - open_price) / close

        # Gaps
        features['gap'] = open_price / df['close'].shift(1).values
        features['gap_size'] = abs(features['gap'] - 1)
        features['gap_up'] = (features['gap'] > 1.01).astype(int)
        features['gap_down'] = (features['gap'] < 0.99).astype(int)

        # Support/Resistance levels
        for period in [10, 20, 50, 100]:
            if len(df) >= period:
                resistance = df['high'].rolling(period).max()
                support = df['low'].rolling(period).min()

                features[f'resistance_{period}d'] = resistance
                features[f'support_{period}d'] = support
                features[f'dist_from_resistance_{period}d'] = (resistance - close) / close
                features[f'dist_from_support_{period}d'] = (close - support) / close
                features[f'sr_range_{period}d'] = (resistance - support) / close

        # Price channels
        for period in [20, 50]:
            if len(df) >= period:
                highest = df['high'].rolling(period).max()
                lowest = df['low'].rolling(period).min()
                features[f'price_channel_pos_{period}'] = (close - lowest) / (highest - lowest + 1e-10)

        # Fibonacci retracements
        if len(df) >= 100:
            high_100 = df['high'].rolling(100).max()
            low_100 = df['low'].rolling(100).min()
            fib_range = high_100 - low_100

            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for level in fib_levels:
                fib_price = low_100 + fib_range * level
                features[f'dist_from_fib_{int(level * 1000)}'] = (close - fib_price) / close

        return features

    def _create_volume_features(self, df: pd.DataFrame) -> Dict:
        """Create volume-based features"""
        features = {}

        # FIX: Ensure float64 for TA-Lib
        volume = df['volume'].astype('float64').values
        close = df['close'].astype('float64').values

        # Volume moving averages and ratios
        for period in [5, 10, 20, 50]:
            try:
                vol_ma = talib.SMA(volume, timeperiod=period)
                features[f'volume_ma_{period}'] = vol_ma
                features[f'volume_ratio_{period}'] = volume / (vol_ma + 1e-10)

                # Volume trend
                vol_ma_double = talib.SMA(volume, timeperiod=period * 2)
                features[f'volume_trend_{period}'] = vol_ma / (vol_ma_double + 1e-10)
            except Exception as e:
                logger.warning(f"Error calculating volume MA {period}: {e}")
                features[f'volume_ma_{period}'] = volume
                features[f'volume_ratio_{period}'] = 1.0
                features[f'volume_trend_{period}'] = 1.0

        # Volume rate of change
        try:
            features['volume_roc_5'] = talib.ROC(volume, timeperiod=5)
            features['volume_roc_10'] = talib.ROC(volume, timeperiod=10)
        except Exception as e:
            logger.warning(f"Error calculating volume ROC: {e}")
            features['volume_roc_5'] = 0.0
            features['volume_roc_10'] = 0.0

        # On Balance Volume
        try:
            obv = talib.OBV(close, volume)
            features['obv'] = obv
            obv_series = pd.Series(obv)
            features['obv_ma'] = talib.SMA(obv.astype('float64'), timeperiod=20)
            features['obv_signal'] = (obv_series > features['obv_ma']).astype(int)
            features['obv_divergence'] = self._calculate_divergence(close, obv_series)
        except Exception as e:
            logger.warning(f"Error calculating OBV: {e}")
            features['obv'] = 0.0
            features['obv_ma'] = 0.0
            features['obv_signal'] = 0
            features['obv_divergence'] = 0

        # Accumulation/Distribution
        try:
            ad = talib.AD(df['high'].astype('float64').values,
                          df['low'].astype('float64').values,
                          close, volume)
            features['ad'] = ad
            features['ad_ma'] = talib.SMA(ad.astype('float64'), timeperiod=20)
            features['ad_signal'] = (features['ad'] > features['ad_ma']).astype(int)
        except Exception as e:
            logger.warning(f"Error calculating AD: {e}")
            features['ad'] = 0.0
            features['ad_ma'] = 0.0
            features['ad_signal'] = 0

        # Chaikin Money Flow
        try:
            features['cmf'] = talib.ADOSC(df['high'].astype('float64').values,
                                          df['low'].astype('float64').values,
                                          close, volume,
                                          fastperiod=3, slowperiod=10)
        except Exception as e:
            logger.warning(f"Error calculating CMF: {e}")
            features['cmf'] = 0.0

        # Volume Price Trend
        features['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        features['vpt_ma'] = features['vpt'].rolling(20).mean()

        # Money Flow
        typical_price = (df['high'].astype('float64') + df['low'].astype('float64') + df['close'].astype('float64')) / 3
        money_flow = typical_price * df['volume'].astype('float64')
        features['money_flow_ratio'] = money_flow / money_flow.rolling(20).mean()

        # Volume profile
        features['volume_concentration'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

        # Price-Volume correlation
        features['pv_correlation'] = df['close'].rolling(20).corr(df['volume'])

        # Volume spike detection
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        features['volume_spike'] = ((df['volume'] - vol_mean) / (vol_std + 1e-10) > 2).astype(int)

        # Volume-weighted price momentum
        features['vw_momentum'] = (df['close'] * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
        features['vw_momentum_change'] = features['vw_momentum'].pct_change(5)

        return features

    def _create_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Create volatility features"""
        features = {}

        # Ensure float64 for TA-Lib
        high = df['high'].astype('float64').values
        low = df['low'].astype('float64').values
        close = df['close'].astype('float64').values

        # ATR at multiple timeframes
        for period in [7, 14, 20, 30]:
            try:
                atr = talib.ATR(high, low, close, timeperiod=period)
                features[f'atr_{period}'] = atr
                features[f'atr_pct_{period}'] = atr / close

                # ATR bands
                features[f'atr_upper_{period}'] = close + 2 * atr
                features[f'atr_lower_{period}'] = close - 2 * atr
            except Exception as e:
                logger.warning(f"Error calculating ATR {period}: {e}")
                features[f'atr_{period}'] = 0.0
                features[f'atr_pct_{period}'] = 0.0

        # Bollinger Bands with multiple parameters
        for period in [10, 20, 30]:
            for std_dev in [1.5, 2, 2.5]:
                try:
                    upper, middle, lower = talib.BBANDS(close, timeperiod=period,
                                                        nbdevup=std_dev, nbdevdn=std_dev)
                    suffix = f'{period}_{int(std_dev * 10)}'

                    features[f'bb_upper_{suffix}'] = upper
                    features[f'bb_lower_{suffix}'] = lower
                    features[f'bb_middle_{suffix}'] = middle
                    features[f'bb_width_{suffix}'] = (upper - lower) / (middle + 1e-10)
                    features[f'bb_position_{suffix}'] = (close - lower) / (upper - lower + 1e-10)

                    # Bollinger Band squeeze
                    bb_width_series = pd.Series((upper - lower) / (middle + 1e-10))
                    features[f'bb_squeeze_{suffix}'] = bb_width_series / bb_width_series.rolling(50).mean()
                except Exception as e:
                    logger.warning(f"Error calculating BB {period}_{std_dev}: {e}")
                    suffix = f'{period}_{int(std_dev * 10)}'
                    features[f'bb_upper_{suffix}'] = close
                    features[f'bb_lower_{suffix}'] = close
                    features[f'bb_middle_{suffix}'] = close
                    features[f'bb_width_{suffix}'] = 0.0
                    features[f'bb_position_{suffix}'] = 0.5
                    features[f'bb_squeeze_{suffix}'] = 1.0

        # Keltner Channels
        for period in [10, 20]:
            for mult in [1.5, 2, 2.5]:
                try:
                    ma = talib.EMA(close, timeperiod=period)
                    atr = talib.ATR(high, low, close, timeperiod=period)

                    features[f'kc_upper_{period}_{int(mult * 10)}'] = ma + (mult * atr)
                    features[f'kc_lower_{period}_{int(mult * 10)}'] = ma - (mult * atr)
                    features[f'kc_position_{period}_{int(mult * 10)}'] = (close - features[
                        f'kc_lower_{period}_{int(mult * 10)}']) / \
                                                                         (features[
                                                                              f'kc_upper_{period}_{int(mult * 10)}'] -
                                                                          features[
                                                                              f'kc_lower_{period}_{int(mult * 10)}'] + 1e-10)
                except Exception as e:
                    logger.warning(f"Error calculating KC {period}_{mult}: {e}")

        # Historical volatility
        returns = df['close'].pct_change()
        for period in [5, 10, 20, 30, 60]:
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)

            # Volatility of volatility
            features[f'vol_of_vol_{period}d'] = features[f'volatility_{period}d'].rolling(period).std()

        # Parkinson volatility
        features['parkinson_vol'] = np.sqrt(
            252 * (1 / (4 * np.log(2))) *
            pd.Series(np.log(high / low) ** 2).rolling(20).mean()
        )

        # Garman-Klass volatility
        open_prices = df['open'].astype('float64').values
        features['garman_klass_vol'] = np.sqrt(
            252 * (
                    0.5 * pd.Series(np.log(high / low) ** 2).rolling(20).mean() -
                    (2 * np.log(2) - 1) * pd.Series(np.log(close / open_prices) ** 2).rolling(20).mean()
            )
        )

        # Rogers-Satchell volatility
        features['rogers_satchell_vol'] = np.sqrt(
            252 * pd.Series(
                np.log(high / close) * np.log(high / open_prices) +
                np.log(low / close) * np.log(low / open_prices)
            ).rolling(20).mean()
        )

        # Yang-Zhang volatility
        overnight_var = pd.Series(np.log(df['open'] / df['close'].shift(1)) ** 2).rolling(20).mean()
        open_close_var = pd.Series(np.log(close / open_prices) ** 2).rolling(20).mean()
        features['yang_zhang_vol'] = np.sqrt(252 * (overnight_var + 0.383 * open_close_var +
                                                    0.641 * features['rogers_satchell_vol'] ** 2))

        # Volatility regime
        vol_median = features['volatility_20d'].rolling(100).median()
        features['vol_regime'] = (features['volatility_20d'] > vol_median * 1.5).astype(int)

        return features

    def _create_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Create technical indicators"""
        features = {}

        # Ensure float64 for TA-Lib
        high = df['high'].astype('float64').values
        low = df['low'].astype('float64').values
        close = df['close'].astype('float64').values
        volume = df['volume'].astype('float64').values

        # RSI with multiple periods
        for period in [7, 14, 21, 28]:
            try:
                rsi = talib.RSI(close, timeperiod=period)
                features[f'rsi_{period}'] = rsi
                features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
                features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)

                # RSI divergence
                features[f'rsi_{period}_divergence'] = self._calculate_divergence(close, rsi)
            except Exception as e:
                logger.warning(f"Error calculating RSI {period}: {e}")
                features[f'rsi_{period}'] = 50.0
                features[f'rsi_{period}_oversold'] = 0
                features[f'rsi_{period}_overbought'] = 0
                features[f'rsi_{period}_divergence'] = 0

        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
            try:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=fast,
                                                          slowperiod=slow, signalperiod=signal)
                suffix = f'{fast}_{slow}_{signal}'

                features[f'macd_{suffix}'] = macd
                features[f'macd_signal_{suffix}'] = macd_signal
                features[f'macd_hist_{suffix}'] = macd_hist
                features[f'macd_cross_{suffix}'] = ((macd > macd_signal) &
                                                    (pd.Series(macd).shift(1) <= pd.Series(macd_signal).shift(
                                                        1))).astype(int)

                # MACD momentum
                features[f'macd_momentum_{suffix}'] = pd.Series(macd_hist) - pd.Series(macd_hist).shift(1)
            except Exception as e:
                logger.warning(f"Error calculating MACD {fast}_{slow}_{signal}: {e}")
                suffix = f'{fast}_{slow}_{signal}'
                features[f'macd_{suffix}'] = 0.0
                features[f'macd_signal_{suffix}'] = 0.0
                features[f'macd_hist_{suffix}'] = 0.0
                features[f'macd_cross_{suffix}'] = 0
                features[f'macd_momentum_{suffix}'] = 0.0

        # Stochastic variations
        for k_period, d_period in [(14, 3), (21, 5), (5, 3)]:
            try:
                k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowd_period=d_period)
                suffix = f'{k_period}_{d_period}'

                features[f'stoch_k_{suffix}'] = k
                features[f'stoch_d_{suffix}'] = d
                features[f'stoch_cross_{suffix}'] = ((k > d) & (pd.Series(k).shift(1) <= pd.Series(d).shift(1))).astype(
                    int)
                features[f'stoch_oversold_{suffix}'] = ((k < 20) & (d < 20)).astype(int)
                features[f'stoch_overbought_{suffix}'] = ((k > 80) & (d > 80)).astype(int)
            except Exception as e:
                logger.warning(f"Error calculating Stochastic {k_period}_{d_period}: {e}")
                suffix = f'{k_period}_{d_period}'
                features[f'stoch_k_{suffix}'] = 50.0
                features[f'stoch_d_{suffix}'] = 50.0
                features[f'stoch_cross_{suffix}'] = 0
                features[f'stoch_oversold_{suffix}'] = 0
                features[f'stoch_overbought_{suffix}'] = 0

        # Williams %R
        for period in [10, 14, 20]:
            try:
                features[f'williams_r_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
            except Exception as e:
                logger.warning(f"Error calculating Williams %R {period}: {e}")
                features[f'williams_r_{period}'] = -50.0

        # CCI
        for period in [14, 20, 30]:
            try:
                features[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)
            except Exception as e:
                logger.warning(f"Error calculating CCI {period}: {e}")
                features[f'cci_{period}'] = 0.0

        # MFI
        for period in [14, 20]:
            try:
                features[f'mfi_{period}'] = talib.MFI(high, low, close, volume, timeperiod=period)
            except Exception as e:
                logger.warning(f"Error calculating MFI {period}: {e}")
                features[f'mfi_{period}'] = 50.0

        # ADX and directional indicators
        for period in [14, 20]:
            try:
                features[f'adx_{period}'] = talib.ADX(high, low, close, timeperiod=period)
                features[f'plus_di_{period}'] = talib.PLUS_DI(high, low, close, timeperiod=period)
                features[f'minus_di_{period}'] = talib.MINUS_DI(high, low, close, timeperiod=period)
                features[f'di_diff_{period}'] = features[f'plus_di_{period}'] - features[f'minus_di_{period}']

                # Trend strength
                features[f'trend_strength_{period}'] = features[f'adx_{period}'] * np.sign(
                    features[f'di_diff_{period}'])
            except Exception as e:
                logger.warning(f"Error calculating ADX {period}: {e}")
                features[f'adx_{period}'] = 0.0
                features[f'plus_di_{period}'] = 0.0
                features[f'minus_di_{period}'] = 0.0
                features[f'di_diff_{period}'] = 0.0
                features[f'trend_strength_{period}'] = 0.0

        # Aroon
        for period in [14, 25]:
            try:
                aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
                features[f'aroon_up_{period}'] = aroon_up
                features[f'aroon_down_{period}'] = aroon_down
                features[f'aroon_osc_{period}'] = aroon_up - aroon_down
            except Exception as e:
                logger.warning(f"Error calculating Aroon {period}: {e}")
                features[f'aroon_up_{period}'] = 50.0
                features[f'aroon_down_{period}'] = 50.0
                features[f'aroon_osc_{period}'] = 0.0

        # Ultimate Oscillator
        try:
            features['ultimate_osc'] = talib.ULTOSC(high, low, close)
        except Exception as e:
            logger.warning(f"Error calculating Ultimate Oscillator: {e}")
            features['ultimate_osc'] = 50.0

        # ROC
        for period in [5, 10, 20]:
            try:
                features[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
            except Exception as e:
                logger.warning(f"Error calculating ROC {period}: {e}")
                features[f'roc_{period}'] = 0.0

        # CMO
        for period in [14, 20]:
            try:
                features[f'cmo_{period}'] = talib.CMO(close, timeperiod=period)
            except Exception as e:
                logger.warning(f"Error calculating CMO {period}: {e}")
                features[f'cmo_{period}'] = 0.0

        # PPO
        try:
            features['ppo'] = talib.PPO(close)
        except Exception as e:
            logger.warning(f"Error calculating PPO: {e}")
            features['ppo'] = 0.0

        # TRIX
        for period in [14, 20]:
            try:
                features[f'trix_{period}'] = talib.TRIX(close, timeperiod=period)
            except Exception as e:
                logger.warning(f"Error calculating TRIX {period}: {e}")
                features[f'trix_{period}'] = 0.0

        return features

    def _create_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Create market microstructure features"""
        features = {}

        # Spread proxies
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        features['co_spread'] = abs(df['close'] - df['open']) / df['close']
        features['oc_spread'] = (df['close'] - df['open']) / df['open']

        # Intraday patterns
        features['intraday_momentum'] = (df['close'] - df['open']) / df['open']
        features['intraday_volatility'] = (df['high'] - df['low']) / df['open']

        # Shadows
        body_high = df[['close', 'open']].max(axis=1)
        body_low = df[['close', 'open']].min(axis=1)

        features['upper_shadow'] = (df['high'] - body_high) / df['close']
        features['lower_shadow'] = (body_low - df['low']) / df['close']
        features['shadow_ratio'] = features['upper_shadow'] / (features['lower_shadow'] + 1e-10)

        # Volume at price levels
        features['volume_at_high'] = df['volume'] * (df['close'] == df['high']).astype(int)
        features['volume_at_low'] = df['volume'] * (df['close'] == df['low']).astype(int)
        features['volume_at_close'] = df['volume'] * (
                abs(df['close'] - df['high']) < abs(df['close'] - df['low'])).astype(int)

        # Order flow imbalance proxy
        features['order_flow_imbalance'] = (df['close'] - df['open']) * df['volume']
        features['ofi_ma'] = features['order_flow_imbalance'].rolling(20).mean()
        features['ofi_std'] = features['order_flow_imbalance'].rolling(20).std()
        features['ofi_zscore'] = (features['order_flow_imbalance'] - features['ofi_ma']) / (features['ofi_std'] + 1e-10)

        # Microstructure noise - calculate volatility here if needed
        volatility_20d = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        features['noise_ratio'] = features['hl_spread'] / (volatility_20d + 1e-10)

        # Time-weighted average price
        features['twap'] = (df['high'] + df['low'] + df['close']) / 3
        features['price_to_twap'] = df['close'] / features['twap']

        # Amihud illiquidity
        features['amihud_illiquidity'] = abs(df['close'].pct_change()) / (df['volume'] + 1)
        features['amihud_ma'] = features['amihud_illiquidity'].rolling(20).mean()

        # Kyle's lambda (simplified)
        price_change = df['close'].pct_change()
        signed_volume = df['volume'] * np.sign(price_change)
        features['kyle_lambda'] = abs(price_change) / (abs(signed_volume) + 1)

        return features

    def _create_statistical_features(self, df: pd.DataFrame) -> Dict:
        """Create statistical features"""
        features = {}

        # Rolling statistics
        for period in [5, 10, 20, 50]:
            rolling_returns = df['close'].pct_change().rolling(period)

            # Higher moments
            features[f'skew_{period}d'] = rolling_returns.skew()
            features[f'kurtosis_{period}d'] = rolling_returns.kurt()

            # Jarque-Bera test statistic
            jb_stat = (period / 6) * (features[f'skew_{period}d'] ** 2 +
                                      0.25 * (features[f'kurtosis_{period}d'] - 3) ** 2)
            features[f'jarque_bera_{period}d'] = jb_stat

        # Auto-correlation
        returns = df['close'].pct_change()
        for lag in [1, 2, 5, 10, 20]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )

        # Hurst exponent
        for period in [20, 50, 100]:
            features[f'hurst_{period}'] = df['close'].rolling(period).apply(
                lambda x: self._calculate_hurst_exponent(x.values) if len(x) == period else 0.5
            )

        # Entropy
        for period in [20, 50]:
            features[f'entropy_{period}'] = returns.rolling(period).apply(
                lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-10)
            )

        # Price efficiency (Variance Ratio Test)
        for period in [10, 20]:
            features[f'efficiency_ratio_{period}'] = self._calculate_efficiency_ratio(df['close'], period)

        # Z-score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)

        # Percentile rank
        for period in [20, 50, 100]:
            features[f'percentile_rank_{period}'] = df['close'].rolling(period).rank(pct=True)

        # Mean reversion indicators
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            features[f'mean_reversion_score_{period}'] = -abs(df['close'] - ma) / ma

        # Trend consistency
        for period in [10, 20]:
            positive_days = (df['close'].pct_change() > 0).rolling(period).sum()
            features[f'trend_consistency_{period}'] = positive_days / period

        return features

    def _create_pattern_features(self, df: pd.DataFrame) -> Dict:
        """Create candlestick pattern features"""
        features = {}

        # Ensure float64 for TA-Lib
        open_price = df['open'].astype('float64').values
        high = df['high'].astype('float64').values
        low = df['low'].astype('float64').values
        close = df['close'].astype('float64').values

        # Basic candle properties
        body = close - open_price
        body_abs = abs(body)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low

        # Candlestick patterns using TA-Lib
        pattern_functions = [
            ('cdl_doji', talib.CDLDOJI),
            ('cdl_hammer', talib.CDLHAMMER),
            ('cdl_hanging_man', talib.CDLHANGINGMAN),
            ('cdl_engulfing', talib.CDLENGULFING),
            ('cdl_harami', talib.CDLHARAMI),
            ('cdl_morning_star', talib.CDLMORNINGSTAR),
            ('cdl_evening_star', talib.CDLEVENINGSTAR),
            ('cdl_3_white_soldiers', talib.CDL3WHITESOLDIERS),
            ('cdl_3_black_crows', talib.CDL3BLACKCROWS),
            ('cdl_spinning_top', talib.CDLSPINNINGTOP),
            ('cdl_shooting_star', talib.CDLSHOOTINGSTAR),
            ('cdl_marubozu', talib.CDLMARUBOZU),
            ('cdl_abandoned_baby', talib.CDLABANDONEDBABY),
            ('cdl_breakaway', talib.CDLBREAKAWAY),
            ('cdl_3_inside', talib.CDL3INSIDE),
            ('cdl_3_outside', talib.CDL3OUTSIDE)
        ]

        for name, func in pattern_functions:
            try:
                pattern = func(open_price, high, low, close)
                features[name] = pattern / 100  # Normalize to -1, 0, 1

                # Pattern strength (how many periods since last occurrence)
                pattern_series = pd.Series(pattern, index=df.index)
                last_pattern = pattern_series.replace(0, np.nan).fillna(method='ffill')
                features[f'{name}_strength'] = (pd.Series(range(len(pattern_series)), index=df.index) -
                                                pd.Series(range(len(last_pattern)), index=df.index))
            except Exception as e:
                logger.warning(f"Error calculating pattern {name}: {e}")
                features[name] = 0
                features[f'{name}_strength'] = 0

        # Custom patterns

        # Pin bar
        pin_bar_bull = ((lower_shadow > body_abs * 2) &
                        (upper_shadow < body_abs * 0.5) &
                        (close > open_price))
        features['pin_bar_bull'] = pin_bar_bull.astype(int)

        pin_bar_bear = ((upper_shadow > body_abs * 2) &
                        (lower_shadow < body_abs * 0.5) &
                        (close < open_price))
        features['pin_bar_bear'] = pin_bar_bear.astype(int)

        # Inside bar
        inside_bar = ((high < df['high'].shift(1)) &
                      (low > df['low'].shift(1)))
        features['inside_bar'] = inside_bar.astype(int)

        # Outside bar
        outside_bar = ((high > df['high'].shift(1)) &
                       (low < df['low'].shift(1)))
        features['outside_bar'] = outside_bar.astype(int)

        # Consecutive patterns
        for period in [3, 5]:
            features[f'consecutive_up_{period}'] = (df['close'] > df['open']).rolling(period).sum() == period
            features[f'consecutive_down_{period}'] = (df['close'] < df['open']).rolling(period).sum() == period

        # Pattern combinations
        features['reversal_pattern'] = (
                                               features.get('cdl_hammer', 0) +
                                               features.get('cdl_morning_star', 0) +
                                               features.get('pin_bar_bull', 0)
                                       ) > 0

        features['continuation_pattern'] = (
                                                   features.get('cdl_3_white_soldiers', 0) +
                                                   features.get('cdl_marubozu', 0)
                                           ) > 0

        return features

    def _create_interaction_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create interaction features between indicators"""
        features = {}

        # Price-Volume interactions
        if 'volume_ratio_20' in base_features and 'return_5d' in base_features:
            features['volume_momentum'] = base_features['volume_ratio_20'] * base_features['return_5d']
            features['volume_trend_alignment'] = (
                    (base_features['volume_ratio_20'] > 1) &
                    (base_features['return_5d'] > 0)
            ).astype(int)

        # RSI-MACD interaction
        if 'rsi_14' in base_features and 'macd_hist_12_26_9' in base_features:
            features['rsi_macd_confluence'] = (
                    ((base_features['rsi_14'] > 50) & (base_features['macd_hist_12_26_9'] > 0)) |
                    ((base_features['rsi_14'] < 50) & (base_features['macd_hist_12_26_9'] < 0))
            ).astype(int)

            features['rsi_macd_divergence'] = (
                    ((base_features['rsi_14'] > 70) & (base_features['macd_hist_12_26_9'] < 0)) |
                    ((base_features['rsi_14'] < 30) & (base_features['macd_hist_12_26_9'] > 0))
            ).astype(int)

        # Volatility-Volume interaction
        if 'volatility_20d' in base_features and 'volume_ratio_20' in base_features:
            features['high_vol_high_volume'] = (
                    (base_features['volatility_20d'] > base_features['volatility_20d'].rolling(50).median()) &
                    (base_features['volume_ratio_20'] > 1.5)
            ).astype(int)

        # Support/Resistance interaction with indicators
        if 'dist_from_resistance_20d' in base_features and 'rsi_14' in base_features:
            features['resistance_overbought'] = (
                    (base_features['dist_from_resistance_20d'] < 0.02) &
                    (base_features['rsi_14'] > 70)
            ).astype(int)

        if 'dist_from_support_20d' in base_features and 'rsi_14' in base_features:
            features['support_oversold'] = (
                    (base_features['dist_from_support_20d'] < 0.02) &
                    (base_features['rsi_14'] < 30)
            ).astype(int)

        # Moving average interactions
        ma_periods = [col for col in base_features.columns if col.startswith('sma_') and col.count('_') == 1]
        if len(ma_periods) >= 2:
            # MA alignment score
            ma_values = base_features[ma_periods].values
            ma_sorted = np.sort(ma_values, axis=1)
            features['ma_alignment_score'] = np.mean(ma_values == ma_sorted, axis=1)

            # MA compression
            features['ma_compression'] = base_features[ma_periods].std(axis=1) / base_features[ma_periods].mean(axis=1)

        # Bollinger Band and RSI interaction
        if 'bb_position_20_20' in base_features and 'rsi_14' in base_features:
            features['bb_rsi_squeeze'] = (
                                                 (base_features['bb_position_20_20'] > 0.8) &
                                                 (base_features['rsi_14'] > 70)
                                         ).astype(int) - (
                                                 (base_features['bb_position_20_20'] < 0.2) &
                                                 (base_features['rsi_14'] < 30)
                                         ).astype(int)

        # Multi-timeframe momentum
        momentum_cols = [col for col in base_features.columns if col.startswith('return_') and 'd' in col]
        if len(momentum_cols) >= 3:
            # Momentum alignment across timeframes
            momentum_signs = np.sign(base_features[momentum_cols])
            features['momentum_alignment'] = momentum_signs.sum(axis=1) / len(momentum_cols)

            # Momentum acceleration
            short_momentum = base_features[[col for col in momentum_cols if int(col.split('_')[1][:-1]) <= 5]].mean(
                axis=1)
            long_momentum = base_features[[col for col in momentum_cols if int(col.split('_')[1][:-1]) >= 10]].mean(
                axis=1)
            features['momentum_acceleration'] = short_momentum - long_momentum

        return features

    def _create_regime_features(self, df: pd.DataFrame) -> Dict:
        """Create market regime features"""
        features = {}

        # Ensure float64 for TA-Lib
        close = df['close'].astype('float64').values

        # Trend regime
        try:
            sma_20 = talib.SMA(close, 20)
            sma_50 = talib.SMA(close, 50)
            sma_200 = talib.SMA(close, 200)

            features['trend_regime_bullish'] = (
                    (df['close'] > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
            ).astype(int)

            features['trend_regime_bearish'] = (
                    (df['close'] < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
            ).astype(int)
        except Exception as e:
            logger.warning(f"Error calculating trend regime: {e}")
            features['trend_regime_bullish'] = 0
            features['trend_regime_bearish'] = 0

        # Volatility regime (using rolling percentile)
        vol = df['close'].pct_change().rolling(20).std()
        vol_percentile = vol.rolling(252).rank(pct=True)

        features['low_vol_regime'] = (vol_percentile < 0.3).astype(int)
        features['high_vol_regime'] = (vol_percentile > 0.7).astype(int)

        # Volume regime
        vol_ma = df['volume'].rolling(20).mean()
        vol_percentile = vol_ma.rolling(252).rank(pct=True)

        features['high_volume_regime'] = (vol_percentile > 0.7).astype(int)
        features['low_volume_regime'] = (vol_percentile < 0.3).astype(int)

        # Momentum regime
        try:
            roc_20 = talib.ROC(close, 20)
            momentum_percentile = pd.Series(roc_20).rolling(252).rank(pct=True)

            features['strong_momentum_regime'] = (momentum_percentile > 0.8).astype(int)
            features['weak_momentum_regime'] = (momentum_percentile < 0.2).astype(int)
        except Exception as e:
            logger.warning(f"Error calculating momentum regime: {e}")
            features['strong_momentum_regime'] = 0
            features['weak_momentum_regime'] = 0

        # Market efficiency regime
        efficiency_ratio = self._calculate_efficiency_ratio(df['close'], 20)
        features['trending_market'] = (efficiency_ratio > 0.7).astype(int)
        features['ranging_market'] = (efficiency_ratio < 0.3).astype(int)

        return features

    def _create_ml_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create ML-discovered features"""
        features = {}

        # Polynomial features for key indicators
        key_features = ['rsi_14', 'macd_hist_12_26_9', 'bb_position_20_20', 'volume_ratio_20']

        for feat in key_features:
            if feat in base_features:
                # Non-linear transformations
                features[f'{feat}_squared'] = base_features[feat] ** 2
                features[f'{feat}_cubed'] = base_features[feat] ** 3
                features[f'{feat}_sqrt'] = np.sqrt(np.abs(base_features[feat]))
                features[f'{feat}_log'] = np.log(np.abs(base_features[feat]) + 1)

        # Interaction polynomials
        if 'rsi_14' in base_features and 'volume_ratio_20' in base_features:
            features['rsi_volume_interaction'] = base_features['rsi_14'] * base_features['volume_ratio_20']
            features['rsi_volume_poly'] = (base_features['rsi_14'] ** 2) * base_features['volume_ratio_20']

        # Fourier features for cyclical patterns
        if len(df) >= 252:  # At least 1 year of data
            close_detrended = df['close'] - df['close'].rolling(252).mean()

            # FFT to find dominant frequencies
            fft = np.fft.fft(close_detrended.fillna(0).values)
            frequencies = np.fft.fftfreq(len(close_detrended))

            # Get top 3 frequencies
            power = np.abs(fft)
            top_freq_idx = np.argsort(power)[-4:-1]  # Exclude DC component

            for i, idx in enumerate(top_freq_idx):
                if frequencies[idx] > 0:  # Only positive frequencies
                    period = int(1 / frequencies[idx])
                    if 5 <= period <= 100:  # Reasonable period range
                        features[f'cycle_{period}d_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / period)
                        features[f'cycle_{period}d_cos'] = np.cos(2 * np.pi * np.arange(len(df)) / period)

        # Wavelet features (simplified)
        if len(df) >= 64:
            close_array = df['close'].values

            # Simple wavelet-like decomposition using differences at multiple scales
            for scale in [2, 4, 8, 16]:
                if len(close_array) >= scale * 2:
                    features[f'wavelet_d{scale}'] = pd.Series(close_array).diff(scale).values
                    features[f'wavelet_smooth{scale}'] = pd.Series(close_array).rolling(scale).mean().values

        # Entropy-based features
        for window in [20, 50]:
            returns = df['close'].pct_change()

            # Sample entropy
            features[f'sample_entropy_{window}'] = returns.rolling(window).apply(
                lambda x: self._sample_entropy(x.values, 2, 0.2 * x.std()) if len(x) == window else 0
            )

        # Fractal dimension
        for window in [30, 60]:
            features[f'fractal_dim_{window}'] = df['close'].rolling(window).apply(
                lambda x: self._calculate_fractal_dimension(x.values) if len(x) == window else 1.5
            )

        return features

    def _calculate_divergence(self, price_series, indicator_series, lookback=14):
        """Calculate divergence between price and indicator"""
        # Ensure we're working with pandas Series
        if isinstance(price_series, np.ndarray):
            price_series = pd.Series(price_series)
        if isinstance(indicator_series, np.ndarray):
            indicator_series = pd.Series(indicator_series)

        price_highs = price_series.rolling(lookback).max()
        price_lows = price_series.rolling(lookback).min()

        ind_highs = indicator_series.rolling(lookback).max()
        ind_lows = indicator_series.rolling(lookback).min()

        # Bearish divergence: price makes higher high, indicator makes lower high
        bearish_div = (
                (price_highs > price_highs.shift(lookback)) &
                (ind_highs < ind_highs.shift(lookback))
        ).astype(int)

        # Bullish divergence: price makes lower low, indicator makes higher low
        bullish_div = (
                (price_lows < price_lows.shift(lookback)) &
                (ind_lows > ind_lows.shift(lookback))
        ).astype(int)

        return bullish_div.astype(int).astype(float) - bearish_div.astype(int).astype(float)

    def _calculate_hurst_exponent(self, series):
        """Calculate Hurst exponent for a series"""
        if len(series) < 20:
            return 0.5

        # Create range series
        lags = range(2, min(20, len(series) // 2))
        tau = []

        for lag in lags:
            # Calculate standard deviation of differences
            diff = np.diff(series, lag)
            if len(diff) > 0:
                tau.append(np.std(diff))

        if len(tau) < 2:
            return 0.5

        # Calculate Hurst exponent
        lags = np.array(list(lags))
        tau = np.array(tau)

        # Avoid zero or negative values
        valid_mask = tau > 0
        if np.sum(valid_mask) < 2:
            return 0.5

        poly = np.polyfit(np.log(lags[valid_mask]), np.log(tau[valid_mask]), 1)
        hurst = poly[0] / 2.0

        return np.clip(hurst, 0, 1)

    def _calculate_efficiency_ratio(self, prices, period=10):
        """Calculate Kaufman's Efficiency Ratio"""
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        change = abs(prices - prices.shift(period))
        volatility = abs(prices.diff()).rolling(period).sum()

        return change / (volatility + 1e-10)

    def _sample_entropy(self, series, m, r):
        """Calculate sample entropy"""
        N = len(series)

        def _maxdist(xi, xj, m):
            """Maximum distance between patterns"""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([series[i:i + m] for i in range(N - m + 1)])
            C = 0

            for i in range(N - m + 1):
                matching = 0
                for j in range(N - m + 1):
                    if i != j and _maxdist(patterns[i], patterns[j], m) <= r:
                        matching += 1
                if matching > 0:
                    C += np.log(matching / (N - m))

            return C / (N - m + 1)

        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0

    def _calculate_fractal_dimension(self, series):
        """Calculate fractal dimension using box-counting method"""
        if len(series) < 10:
            return 1.5

        # Normalize series
        series = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-10)

        # Box sizes
        n = len(series)
        box_sizes = [2 ** i for i in range(1, int(np.log2(n)))]
        box_counts = []

        for box_size in box_sizes:
            # Count non-empty boxes
            boxes = np.array_split(series, n // box_size)
            non_empty = sum(1 for box in boxes if len(box) > 0 and np.ptp(box) > 0)
            box_counts.append(non_empty)

        if len(box_counts) < 2:
            return 1.5

        # Calculate fractal dimension
        coeffs = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)

        return abs(coeffs[0])

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Forward fill first (for time series continuity)
        features = features.fillna(method='ffill', limit=5)

        # Then backward fill
        features = features.fillna(method='bfill', limit=5)

        # Fill remaining with 0 or median based on feature type
        for col in features.columns:
            if features[col].isna().any():
                # For ratio/percentage features, fill with 1 or 0
                if 'ratio' in col or 'pct' in col or 'position' in col:
                    features[col].fillna(1 if 'ratio' in col else 0, inplace=True)
                # For indicator features, fill with neutral values
                elif 'rsi' in col:
                    features[col].fillna(50, inplace=True)
                elif 'stoch' in col:
                    features[col].fillna(50, inplace=True)
                else:
                    # Fill with 0 for most other features
                    features[col].fillna(0, inplace=True)

        # Remove features with too many missing values
        missing_pct = features.isna().sum() / len(features)
        valid_features = missing_pct[missing_pct < 0.1].index

        return features[valid_features]

    def scale_features(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        if fit:
            scaled = self.scaler.fit_transform(features)
        else:
            scaled = self.scaler.transform(features)

        return pd.DataFrame(scaled, index=features.index, columns=features.columns)

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by groups"""
        groups = {
            'price': [],
            'volume': [],
            'volatility': [],
            'technical': [],
            'microstructure': [],
            'statistical': [],
            'pattern': [],
            'interaction': [],
            'regime': [],
            'ml': []
        }

        for feature in self.feature_names:
            if any(x in feature for x in ['return', 'sma', 'ema', 'price', 'resistance', 'support']):
                groups['price'].append(feature)
            elif any(x in feature for x in ['volume', 'obv', 'ad', 'cmf', 'mfi']):
                groups['volume'].append(feature)
            elif any(x in feature for x in ['volatility', 'atr', 'bb_', 'kc_']):
                groups['volatility'].append(feature)
            elif any(x in feature for x in ['rsi', 'macd', 'stoch', 'cci', 'willr', 'adx']):
                groups['technical'].append(feature)
            elif any(x in feature for x in ['spread', 'shadow', 'noise', 'amihud']):
                groups['microstructure'].append(feature)
            elif any(x in feature for x in ['skew', 'kurt', 'hurst', 'entropy', 'zscore']):
                groups['statistical'].append(feature)
            elif any(x in feature for x in ['cdl_', 'pin_bar', 'inside_bar']):
                groups['pattern'].append(feature)
            elif any(x in feature for x in ['interaction', 'confluence', 'divergence']):
                groups['interaction'].append(feature)
            elif any(x in feature for x in ['regime', 'trending', 'ranging']):
                groups['regime'].append(feature)
            elif any(x in feature for x in ['squared', 'cubed', 'poly', 'cycle', 'wavelet']):
                groups['ml'].append(feature)

        return groups