# models/features.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature engineering with interaction features"""

    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu = enable_gpu
        self.feature_names = []
        self.scaler = RobustScaler()  # Robust to outliers
        self.interaction_cache = {}

        # Try to import GPU libraries
        if enable_gpu:
            try:
                import cupy as cp
                import cudf
                self.cp = cp
                self.cudf = cudf
                self.gpu_available = True
                logger.info("GPU acceleration enabled for feature engineering")
            except ImportError:
                self.gpu_available = False
                logger.warning("GPU libraries not available, using CPU")
        else:
            self.gpu_available = False

    def _ensure_float64(self, data):
        """Ensure data is float64 for TA-Lib compatibility"""
        if isinstance(data, pd.Series):
            return data.astype(np.float64).values
        elif isinstance(data, np.ndarray):
            return data.astype(np.float64)
        else:
            return np.array(data, dtype=np.float64)

    def create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Create all features including interactions"""
        if df.empty or len(df) < 200:  # Need sufficient history
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            return pd.DataFrame()

        # Ensure all columns are float64 for TA-Lib compatibility
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        # Start with base features - using DataFrame from the beginning
        features = pd.DataFrame(index=df.index)

        logger.debug(f"Creating features for {symbol}: starting with {len(df)} rows")

        # 1. Price-based features
        try:
            price_dict = self._create_price_features(df)
            for name, values in price_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(price_dict)} price features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in price features: {e}")

        # 2. Volume features
        try:
            volume_dict = self._create_volume_features(df)
            for name, values in volume_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(volume_dict)} volume features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in volume features: {e}")

        # 3. Volatility features
        try:
            volatility_dict = self._create_volatility_features(df)
            for name, values in volatility_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(volatility_dict)} volatility features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in volatility features: {e}")

        # 4. Technical indicators
        try:
            technical_dict = self._create_technical_indicators(df)
            for name, values in technical_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(technical_dict)} technical features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in technical features: {e}")

        # 5. Market microstructure
        try:
            microstructure_dict = self._create_microstructure_features(df)
            for name, values in microstructure_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(microstructure_dict)} microstructure features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in microstructure features: {e}")

        # 6. Temporal features
        try:
            temporal_dict = self._create_temporal_features(df)
            for name, values in temporal_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(temporal_dict)} temporal features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in temporal features: {e}")

        # 7. Interaction features
        try:
            interaction_dict = self._create_interaction_features(df, features)
            for name, values in interaction_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(interaction_dict)} interaction features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in interaction features: {e}")

        # 8. Statistical features
        try:
            statistical_dict = self._create_statistical_features(df)
            for name, values in statistical_dict.items():
                features[name] = values
            logger.debug(f"{symbol}: {len(statistical_dict)} statistical features created")
        except Exception as e:
            logger.error(f"{symbol}: Error in statistical features: {e}")

        logger.info(f"{symbol}: Total features before cleaning: {features.shape[1]}")

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Handle missing values
        features = self._handle_missing_values(features)

        logger.info(f"{symbol}: Final features shape: {features.shape}")

        return features

    def _create_price_features(self, df: pd.DataFrame) -> Dict:
        """Create price-based features"""
        features = {}

        # Ensure all data is float64 for TA-Lib
        close = self._ensure_float64(df['close'])
        high = self._ensure_float64(df['high'])
        low = self._ensure_float64(df['low'])
        open_price = self._ensure_float64(df['open'])

        # Returns
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

        # Moving averages
        for period in [10, 20, 50, 200]:
            if len(df) >= period:
                sma = talib.SMA(close, timeperiod=period)
                features[f'sma_{period}'] = sma
                features[f'price_to_sma_{period}'] = close / sma

        # Exponential moving averages
        for period in [12, 26]:
            if len(df) >= period:
                ema = talib.EMA(close, timeperiod=period)
                features[f'ema_{period}'] = ema
                features[f'price_to_ema_{period}'] = close / ema

        # Price positions
        features['close_to_high'] = close / high
        features['close_to_low'] = close / low
        features['high_low_range'] = (high - low) / close
        features['close_to_open'] = close / open_price

        # Gaps
        features['gap'] = open_price / np.roll(close, 1)
        features['gap_up'] = (features['gap'] > 1.01).astype(int)
        features['gap_down'] = (features['gap'] < 0.99).astype(int)

        # Support/Resistance levels
        for period in [20, 50]:
            features[f'dist_from_high_{period}d'] = close / pd.Series(high).rolling(period).max().values
            features[f'dist_from_low_{period}d'] = close / pd.Series(low).rolling(period).min().values

        return features

    def _create_volume_features(self, df: pd.DataFrame) -> Dict:
        """Create volume-based features"""
        features = {}

        # Ensure all data is float64 for TA-Lib
        volume = self._ensure_float64(df['volume'])
        close = self._ensure_float64(df['close'])

        # Volume moving averages
        for period in [10, 20, 50]:
            vol_ma = talib.SMA(volume, timeperiod=period)
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = volume / vol_ma

        # Volume rate of change
        features['volume_roc'] = talib.ROC(volume, timeperiod=10)

        # On Balance Volume
        features['obv'] = talib.OBV(close, volume)
        features['obv_ma'] = talib.SMA(features['obv'], timeperiod=20)
        features['obv_signal'] = features['obv'] > features['obv_ma']

        # Accumulation/Distribution
        features['ad'] = talib.AD(
            self._ensure_float64(df['high']),
            self._ensure_float64(df['low']),
            close,
            volume
        )
        features['ad_ma'] = talib.SMA(features['ad'], timeperiod=20)

        # Volume Price Trend
        features['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()

        # Money Flow
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        features['money_flow_ratio'] = money_flow / money_flow.rolling(20).mean()

        return features

    def _create_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Create volatility features"""
        features = {}

        high = self._ensure_float64(df['high'])
        low = self._ensure_float64(df['low'])
        close = self._ensure_float64(df['close'])

        # ATR
        for period in [14, 20]:
            features[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / close

        # Bollinger Bands
        for period in [20, 30]:
            upper, middle, lower = talib.BBANDS(close, timeperiod=period)
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_middle_{period}'] = middle
            features[f'bb_width_{period}'] = (upper - lower) / middle
            features[f'bb_position_{period}'] = (close - lower) / (upper - lower)

        # Keltner Channels
        for period in [20]:
            ma = talib.EMA(close, timeperiod=period)
            atr = talib.ATR(high, low, close, timeperiod=period)
            features[f'kc_upper_{period}'] = ma + (2 * atr)
            features[f'kc_lower_{period}'] = ma - (2 * atr)
            features[f'kc_position_{period}'] = (close - features[f'kc_lower_{period}']) / \
                                                (features[f'kc_upper_{period}'] - features[f'kc_lower_{period}'])

        # Historical volatility
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}d'] = returns.rolling(period).std()

        # Parkinson volatility
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        )

        return features

    def _create_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Create technical indicators"""
        features = {}

        high = self._ensure_float64(df['high'])
        low = self._ensure_float64(df['low'])
        close = self._ensure_float64(df['close'])
        volume = self._ensure_float64(df['volume'])

        # RSI
        for period in [14, 21]:
            features[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)

        # MACD
        macd, signal, hist = talib.MACD(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_cross'] = (macd > signal).astype(int)

        # Stochastic
        k, d = talib.STOCH(high, low, close)
        features['stoch_k'] = k
        features['stoch_d'] = d
        features['stoch_cross'] = (k > d).astype(int)

        # Williams %R
        features['williams_r'] = talib.WILLR(high, low, close)

        # CCI
        features['cci'] = talib.CCI(high, low, close)

        # MFI
        features['mfi'] = talib.MFI(high, low, close, volume)

        # ADX
        features['adx'] = talib.ADX(high, low, close)
        features['plus_di'] = talib.PLUS_DI(high, low, close)
        features['minus_di'] = talib.MINUS_DI(high, low, close)

        # Aroon
        aroon_up, aroon_down = talib.AROON(high, low)
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_osc'] = aroon_up - aroon_down

        return features

    def _create_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Create market microstructure features"""
        features = {}

        # Spread proxies
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        features['co_spread'] = abs(df['close'] - df['open']) / df['close']

        # Intraday patterns
        features['intraday_momentum'] = (df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']

        # Volume at price levels
        features['volume_at_high'] = df['volume'] * (df['close'] == df['high']).astype(int)
        features['volume_at_low'] = df['volume'] * (df['close'] == df['low']).astype(int)

        # Order flow imbalance proxy
        features['order_flow_imbalance'] = (df['close'] - df['open']) * df['volume']
        features['ofi_ma'] = features['order_flow_imbalance'].rolling(20).mean()

        # Time-weighted returns
        features['twap'] = (df['high'] + df['low'] + df['close']) / 3
        features['vwap_proxy'] = (features['twap'] * df['volume']).rolling(20).sum() / \
                                 df['volume'].rolling(20).sum()

        return features

    def _create_temporal_features(self, df: pd.DataFrame) -> Dict:
        """Create time-based features"""
        features = {}

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Day of week
        features['day_of_week'] = df.index.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)

        # Month
        features['month'] = df.index.month
        features['is_january'] = (features['month'] == 1).astype(int)
        features['is_december'] = (features['month'] == 12).astype(int)

        # Quarter
        features['quarter'] = df.index.quarter
        features['is_quarter_end'] = df.index.is_quarter_end.astype(int)

        # Trading day of month
        features['day_of_month'] = df.index.day
        features['is_month_start'] = (features['day_of_month'] <= 5).astype(int)
        features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)

        return features

    def _create_interaction_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create interaction features - the secret sauce!"""
        features = {}

        # 1. Classic Technical Patterns
        features.update(self._create_ma_crossover_features(df, base_features))
        features.update(self._create_support_resistance_features(df, base_features))

        # 2. Momentum-Volume Interactions
        features.update(self._create_momentum_volume_features(df, base_features))

        # 3. Volatility-Trend Interactions
        features.update(self._create_volatility_trend_features(df, base_features))

        # 4. Multi-Indicator Confluence
        features.update(self._create_confluence_features(df, base_features))

        # 5. Market Structure Interactions
        features.update(self._create_market_structure_features(df, base_features))

        # 6. Advanced Pattern Recognition
        features.update(self._create_advanced_patterns(df, base_features))

        return features

    def _create_ma_crossover_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Moving average crossover patterns"""
        features = {}

        # Golden/Death Cross
        if 'sma_50' in base_features and 'sma_200' in base_features:
            sma_50 = base_features['sma_50']
            sma_200 = base_features['sma_200']

            features['golden_cross'] = ((sma_50 > sma_200) &
                                        (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
            features['death_cross'] = ((sma_50 < sma_200) &
                                       (sma_50.shift(1) >= sma_200.shift(1))).astype(int)

            # Days since cross
            golden_cross_dates = df.index[features['golden_cross'] == 1]
            death_cross_dates = df.index[features['death_cross'] == 1]

            features['days_since_golden_cross'] = self._days_since_event(
                df.index, golden_cross_dates
            )
            features['days_since_death_cross'] = self._days_since_event(
                df.index, death_cross_dates
            )

            # Distance to cross
            features['distance_to_cross'] = (sma_50 - sma_200) / sma_200

        # EMA crossovers
        if 'ema_12' in base_features and 'ema_26' in base_features:
            ema_12 = base_features['ema_12']
            ema_26 = base_features['ema_26']

            features['ema_cross_up'] = ((ema_12 > ema_26) &
                                        (ema_12.shift(1) <= ema_26.shift(1))).astype(int)
            features['ema_cross_down'] = ((ema_12 < ema_26) &
                                          (ema_12.shift(1) >= ema_26.shift(1))).astype(int)

        # MA alignment
        ma_columns = [col for col in base_features.columns if 'sma_' in col]
        if len(ma_columns) >= 3:
            ma_values = base_features[ma_columns]
            features['ma_alignment_score'] = self._calculate_ma_alignment(ma_values)

            # MA compression (when MAs converge)
            ma_std = ma_values.std(axis=1)
            features['ma_compression'] = ma_std / df['close']

        return features

    def _create_support_resistance_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Support and resistance interaction features"""
        features = {}

        # Identify support/resistance levels
        for period in [20, 50]:
            if f'dist_from_high_{period}d' in base_features:
                resistance = df['high'].rolling(period).max()
                support = df['low'].rolling(period).min()

                # Price action at levels
                features[f'at_resistance_{period}'] = (
                        (df['high'] >= resistance * 0.99) &
                        (df['close'] < resistance)
                ).astype(int)

                features[f'at_support_{period}'] = (
                        (df['low'] <= support * 1.01) &
                        (df['close'] > support)
                ).astype(int)

                # Breakout/breakdown
                features[f'resistance_break_{period}'] = (
                        (df['close'] > resistance) &
                        (df['close'].shift(1) <= resistance)
                ).astype(int)

                features[f'support_break_{period}'] = (
                        (df['close'] < support) &
                        (df['close'].shift(1) >= support)
                ).astype(int)

                # Failed breakout recovery
                features[f'failed_breakdown_{period}'] = (
                        (df['low'].shift(1) < support) &
                        (df['close'] > support)
                ).astype(int)

        return features

    def _create_momentum_volume_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Momentum and volume interaction features"""
        features = {}

        # RSI-Volume interactions
        if 'rsi_14' in base_features and 'volume_ratio_20' in base_features:
            rsi = base_features['rsi_14']
            vol_ratio = base_features['volume_ratio_20']

            # Volume-confirmed momentum
            features['high_rsi_high_volume'] = (
                    (rsi > 60) & (vol_ratio > 1.5)
            ).astype(int)

            features['low_rsi_high_volume'] = (
                    (rsi < 40) & (vol_ratio > 1.5)
            ).astype(int)

            # Divergences
            features['rsi_volume_divergence'] = self._detect_divergence(
                rsi, df['volume'], lookback=14
            )

        # MACD-Volume interactions
        if 'macd_hist' in base_features and 'volume_ratio_20' in base_features:
            macd_hist = base_features['macd_hist']
            vol_ratio = base_features['volume_ratio_20']

            features['macd_volume_thrust'] = (
                    (macd_hist > 0) &
                    (macd_hist > macd_hist.shift(1)) &
                    (vol_ratio > 1.2)
            ).astype(int)

            # Weighted MACD
            features['volume_weighted_macd'] = macd_hist * vol_ratio

        # Accumulation/Distribution patterns
        if 'ad' in base_features:
            ad = base_features['ad']
            price_change = df['close'].pct_change()

            features['accumulation_pattern'] = (
                    (price_change < 0) &
                    (ad > ad.shift(1)) &
                    (df['volume'] > base_features.get('volume_ma_20', df['volume'].rolling(20).mean()))
            ).astype(int)

            features['distribution_pattern'] = (
                    (price_change > 0) &
                    (ad < ad.shift(1)) &
                    (df['volume'] > base_features.get('volume_ma_20', df['volume'].rolling(20).mean()))
            ).astype(int)

        return features

    def _create_volatility_trend_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Volatility and trend interaction features"""
        features = {}

        # Bollinger Band patterns
        if 'bb_width_20' in base_features and 'bb_position_20' in base_features:
            bb_width = base_features['bb_width_20']
            bb_position = base_features['bb_position_20']

            # Bollinger squeeze
            bb_squeeze = bb_width < bb_width.rolling(50).mean() * 0.5
            features['bb_squeeze'] = bb_squeeze.astype(int)

            # Squeeze breakout
            features['bb_squeeze_breakout'] = (
                    bb_squeeze.shift(1) &
                    (bb_position > 1)  # Price above upper band
            ).astype(int)

            # Mean reversion setup
            features['bb_mean_reversion'] = (
                    (bb_position < 0) &  # Below lower band
                    (base_features.get('rsi_14', 50) < 30)
            ).astype(int)

        # ATR-Trend interactions
        if 'atr_14' in base_features and 'adx' in base_features:
            atr = base_features['atr_14']
            adx = base_features['adx']
            trend_strength = abs(df['close'] - base_features.get('sma_20', df['close'].rolling(20).mean()))

            features['trend_volatility_ratio'] = trend_strength / (atr + 1e-7)
            features['strong_trend_low_vol'] = (
                    (adx > 25) &
                    (atr < atr.rolling(20).mean())
            ).astype(int)

        # Keltner-Bollinger squeeze
        if 'kc_upper_20' in base_features and 'bb_upper_20' in base_features:
            features['ttm_squeeze'] = (
                    (base_features['bb_upper_20'] < base_features['kc_upper_20']) &
                    (base_features['bb_lower_20'] > base_features['kc_lower_20'])
            ).astype(int)

        return features

    def _create_confluence_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Multi-indicator confluence patterns"""
        features = {}

        # Bullish confluence
        bullish_conditions = []
        if 'rsi_14' in base_features:
            bullish_conditions.append(base_features['rsi_14'] > 50)
        if 'macd_cross' in base_features:
            bullish_conditions.append(base_features['macd_cross'] == 1)
        if 'price_to_sma_20' in base_features:
            bullish_conditions.append(base_features['price_to_sma_20'] > 1)
        if 'volume_ratio_20' in base_features:
            bullish_conditions.append(base_features['volume_ratio_20'] > 1)

        if len(bullish_conditions) >= 3:
            features['bullish_confluence'] = (
                    sum(bullish_conditions) >= 3
            ).astype(int)

            features['bullish_confluence_score'] = sum(bullish_conditions) / len(bullish_conditions)

        # Oversold bounce setup
        if all(ind in base_features for ind in ['rsi_14', 'bb_position_20', 'stoch_k']):
            features['oversold_bounce'] = (
                    (base_features['rsi_14'] < 30) &
                    (base_features['bb_position_20'] < 0) &
                    (base_features['stoch_k'] < 20) &
                    (df['close'] > df['open'])  # Bullish candle
            ).astype(int)

        # Momentum alignment
        momentum_indicators = []
        if 'rsi_14' in base_features:
            momentum_indicators.append(base_features['rsi_14'] > 50)
        if 'macd' in base_features:
            momentum_indicators.append(base_features['macd'] > 0)
        if 'stoch_k' in base_features:
            momentum_indicators.append(base_features['stoch_k'] > 50)

        if len(momentum_indicators) >= 2:
            features['momentum_alignment'] = (
                    sum(momentum_indicators) == len(momentum_indicators)
            ).astype(int)

        return features

    def _create_market_structure_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Market structure interaction features"""
        features = {}

        # Trend regime detection
        if 'sma_20' in base_features and 'sma_50' in base_features:
            sma_20 = base_features['sma_20']
            sma_50 = base_features['sma_50']

            features['uptrend'] = (
                    (df['close'] > sma_20) &
                    (sma_20 > sma_50)
            ).astype(int)

            features['downtrend'] = (
                    (df['close'] < sma_20) &
                    (sma_20 < sma_50)
            ).astype(int)

            features['ranging'] = (
                    ~features['uptrend'].astype(bool) & ~features['downtrend'].astype(bool)
            ).astype(int)

        # Higher highs/lower lows
        lookback = 20
        rolling_high = df['high'].rolling(lookback).max()
        rolling_low = df['low'].rolling(lookback).min()

        features['higher_high'] = (
                df['high'] > rolling_high.shift(1)
        ).astype(int)

        features['lower_low'] = (
                df['low'] < rolling_low.shift(1)
        ).astype(int)

        # Swing failure pattern - fix the type issue here
        # Convert to boolean Series first, then do the AND operation
        lower_low_shifted = pd.Series(features['lower_low'].shift(1) == 1, index=df.index)
        close_comparison = pd.Series(df['close'] > df['close'].shift(1), index=df.index)

        features['swing_failure'] = (
                lower_low_shifted & close_comparison
        ).astype(int)

        return features

    def _create_advanced_patterns(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Advanced pattern recognition"""
        features = {}

        # Price action patterns
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']

        # Hammer pattern
        features['hammer'] = (
                (lower_shadow > body * 2) &
                (upper_shadow < body * 0.1) &
                (df['close'].rolling(5).mean() < df['close'].rolling(20).mean())  # In downtrend
        ).astype(int)

        # Engulfing pattern
        features['bullish_engulfing'] = (
                (df['close'] > df['open']) &  # Green candle
                (df['open'] < df['close'].shift(1)) &  # Open below previous close
                (df['close'] > df['open'].shift(1)) &  # Close above previous open
                (body > body.shift(1))  # Larger body
        ).astype(int)

        # Morning star pattern (simplified)
        features['morning_star'] = (
                (df['close'].shift(2) < df['open'].shift(2)) &  # First candle bearish
                (body.shift(1) < body.shift(2) * 0.3) &  # Small middle candle
                (df['close'] > df['open']) &  # Third candle bullish
                (df['close'] > df['open'].shift(2))  # Close above first candle open
        ).astype(int)

        # Gap patterns
        if 'gap_up' in base_features:
            features['gap_and_go'] = (
                    base_features['gap_up'] &
                    (df['close'] > df['open']) &
                    (base_features.get('volume_ratio_20', 1) > 1.5)
            ).astype(int)

        # Fractal patterns
        features['fractal_high'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['high'] > df['high'].shift(2)) &
                (df['high'] > df['high'].shift(-1)) &
                (df['high'] > df['high'].shift(-2))
        ).astype(int)

        # Machine learning discovered patterns
        if all(ind in base_features for ind in ['rsi_14', 'volume_ratio_20', 'bb_position_20']):
            # Example of ML-discovered pattern
            features['ml_pattern_1'] = (
                    (base_features['rsi_14'] > 45) &
                    (base_features['rsi_14'] < 55) &
                    (base_features['volume_ratio_20'] > 1.2) &
                    (base_features['bb_position_20'] > 0.6) &
                    (base_features['bb_position_20'] < 0.8)
            ).astype(int)

        return features

    def _create_statistical_features(self, df: pd.DataFrame) -> Dict:
        """Statistical features"""
        features = {}

        # Rolling correlations
        if len(df) > 100:
            # Auto-correlation
            for lag in [1, 5, 10]:
                features[f'autocorr_lag_{lag}'] = df['close'].rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                )

        # Price efficiency
        features['efficiency_ratio'] = self._calculate_efficiency_ratio(df['close'])

        # Hurst exponent (simplified)
        features['hurst'] = self._calculate_hurst_exponent(df['close'])

        # Z-score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-7)

        # Percentile rank
        for period in [20, 50]:
            features[f'percentile_rank_{period}'] = df['close'].rolling(period).rank(pct=True)

        return features

    def _detect_divergence(self, series1: pd.Series, series2: pd.Series,
                           lookback: int = 14) -> pd.Series:
        """Detect divergence between two series"""
        # Find local extrema
        highs1 = series1.rolling(lookback).max()
        highs2 = series2.rolling(lookback).max()

        lows1 = series1.rolling(lookback).min()
        lows2 = series2.rolling(lookback).min()

        # Bearish divergence: series1 making higher highs, series2 making lower highs
        bearish_div = (
                (highs1 > highs1.shift(lookback)) &
                (highs2 < highs2.shift(lookback))
        )

        # Bullish divergence: series1 making lower lows, series2 making higher lows
        bullish_div = (
                (lows1 < lows1.shift(lookback)) &
                (lows2 > lows2.shift(lookback))
        )

        return bullish_div.astype(int) - bearish_div.astype(int)

    def _days_since_event(self, index: pd.DatetimeIndex,
                          event_dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate days since last event"""
        days_since = pd.Series(index=index, dtype=float)
        days_since[:] = np.inf

        for date in index:
            past_events = event_dates[event_dates <= date]
            if len(past_events) > 0:
                last_event = past_events[-1]
                days_since[date] = (date - last_event).days

        return days_since

    def _calculate_ma_alignment(self, ma_values: pd.DataFrame) -> pd.Series:
        """Calculate moving average alignment score"""
        # Check if MAs are in order (shortest to longest)
        ma_sorted = ma_values.values
        ma_sorted.sort(axis=1)

        alignment = (ma_values.values == ma_sorted).all(axis=1)

        # Calculate distance between MAs
        ma_std = ma_values.std(axis=1)
        ma_mean = ma_values.mean(axis=1)

        # Score: 1 when perfectly aligned and close together
        score = alignment.astype(float) * (1 - ma_std / ma_mean)

        return score

    def _calculate_efficiency_ratio(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio"""
        change = abs(prices - prices.shift(period))
        volatility = prices.diff().abs().rolling(period).sum()

        return change / (volatility + 1e-7)

    def _calculate_hurst_exponent(self, prices: pd.Series, lags: int = 20) -> pd.Series:
        """Simplified Hurst exponent calculation"""
        # This is a simplified version for speed
        returns = prices.pct_change().dropna()

        if len(returns) < lags * 2:
            return pd.Series(index=prices.index, data=0.5)

        hurst = pd.Series(index=prices.index, dtype=float)

        for i in range(lags * 2, len(prices)):
            subset = returns.iloc[i - lags * 2:i].values

            # Calculate R/S statistic
            mean = np.mean(subset)
            deviations = subset - mean
            Z = np.cumsum(deviations)
            R = np.max(Z) - np.min(Z)
            S = np.std(subset, ddof=1)

            if S != 0:
                RS = R / S
                hurst.iloc[i] = np.log(RS) / np.log(lags)
            else:
                hurst.iloc[i] = 0.5

        return hurst.fillna(0.5)

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        if features.empty:
            return features

        # First, forward fill then backward fill
        features = features.ffill(limit=5)
        features = features.bfill(limit=5)

        # Fill remaining NaN with 0
        features = features.fillna(0)

        # Remove features with too many missing values BEFORE filling
        # This is more lenient - we check the original missing values
        # but only after we've tried to fill them
        original_shape = features.shape[1]

        # Don't remove features unless they have extreme missing values
        # Technical indicators often have NaN in the first 20-50 rows due to lookback
        # So we check missing values after row 50
        if len(features) > 50:
            missing_pct = features.iloc[50:].isna().sum() / len(features.iloc[50:])
            valid_features = missing_pct[missing_pct < 0.3].index  # Allow up to 30% missing
            features = features[valid_features]

        # Remove constant features (all same value)
        features = features.loc[:, (features != features.iloc[0]).any()]

        # Remove features with near-zero variance
        if len(features) > 0:
            feature_std = features.std()
            features = features.loc[:, feature_std > 1e-8]

        logger.debug(f"Features after handling missing values: {original_shape} -> {features.shape[1]}")

        return features

    def scale_features(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        if fit:
            scaled = self.scaler.fit_transform(features)
        else:
            scaled = self.scaler.transform(features)

        return pd.DataFrame(scaled, index=features.index, columns=features.columns)

    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Calculate feature importance using mutual information"""
        from sklearn.feature_selection import mutual_info_classif

        # Handle missing values
        features_clean = features.fillna(0)
        target_clean = target.fillna(0)

        # Calculate mutual information
        mi_scores = mutual_info_classif(features_clean, target_clean)

        # Create series with feature names
        importance = pd.Series(mi_scores, index=features.columns)

        return importance.sort_values(ascending=False)