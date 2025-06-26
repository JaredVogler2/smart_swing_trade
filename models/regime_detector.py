# models/regime_detector.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import talib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detects market regime to adjust trading strategies"""

    def __init__(self):
        self.regimes = ['bull_quiet', 'bull_volatile', 'bear_quiet', 'bear_volatile', 'ranging']
        self.regime_history = []
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=5, random_state=42)
        self.is_fitted = False

    def detect_regime(self, market_data: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        Detect current market regime

        Args:
            market_data: DataFrame with OHLCV data (typically SPY or market index)
            lookback: Days to look back for regime detection

        Returns:
            Dictionary with regime information
        """
        if len(market_data) < lookback:
            return {
                'regime': 'unknown',
                'confidence': 0,
                'characteristics': {}
            }

        # Calculate regime features
        features = self._calculate_regime_features(market_data, lookback)

        # Classify regime
        regime_info = self._classify_regime(features)

        # Store in history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime_info['regime'],
            'confidence': regime_info['confidence']
        })

        return regime_info

    def _calculate_regime_features(self, df: pd.DataFrame, lookback: int) -> Dict:
        """Calculate features for regime detection"""
        features = {}

        # Use recent data
        recent_data = df.tail(lookback).copy()
        close = recent_data['close'].values
        high = recent_data['high'].values
        low = recent_data['low'].values
        volume = recent_data['volume'].values

        # Trend features
        sma_20 = talib.SMA(close, timeperiod=min(20, len(close)))[-1]
        sma_50 = talib.SMA(close, timeperiod=min(50, len(close)))[-1]
        current_price = close[-1]

        features['trend_strength'] = (current_price - sma_50) / sma_50
        features['trend_consistency'] = np.sum(close > sma_20) / len(close)

        # Volatility features
        returns = pd.Series(close).pct_change().dropna()
        features['volatility'] = returns.std()
        features['volatility_of_volatility'] = returns.rolling(5).std().std()

        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        features['atr_ratio'] = atr / current_price

        # Market breadth (using volume as proxy)
        features['volume_trend'] = np.mean(volume[-5:]) / np.mean(volume)

        # Momentum
        features['rsi'] = talib.RSI(close, timeperiod=14)[-1]
        features['momentum_10d'] = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0

        # Volatility regime
        features['high_low_ratio'] = np.mean((high - low) / close)

        # Trend quality
        if len(close) >= 20:
            # Calculate R-squared of linear regression
            x = np.arange(20)
            y = close[-20:]
            correlation = np.corrcoef(x, y)[0, 1]
            features['trend_quality'] = correlation ** 2
        else:
            features['trend_quality'] = 0

        # Fear/Greed indicators
        features['fear_greed_score'] = self._calculate_fear_greed(features)

        return features

    def _classify_regime(self, features: Dict) -> Dict:
        """Classify market regime based on features"""
        trend = features['trend_strength']
        volatility = features['volatility']
        trend_quality = features['trend_quality']
        fear_greed = features['fear_greed_score']

        # Rule-based classification
        if trend > 0.05 and volatility < 0.015:
            regime = 'bull_quiet'
            confidence = min(trend * 10, 1.0) * (1 - volatility * 50)
        elif trend > 0.05 and volatility >= 0.015:
            regime = 'bull_volatile'
            confidence = min(trend * 10, 1.0) * min(volatility * 30, 1.0)
        elif trend < -0.05 and volatility < 0.015:
            regime = 'bear_quiet'
            confidence = min(abs(trend) * 10, 1.0) * (1 - volatility * 50)
        elif trend < -0.05 and volatility >= 0.015:
            regime = 'bear_volatile'
            confidence = min(abs(trend) * 10, 1.0) * min(volatility * 30, 1.0)
        else:
            regime = 'ranging'
            confidence = 1 - abs(trend) * 10

        # Adjust confidence based on trend quality
        confidence *= (0.5 + 0.5 * trend_quality)

        # Create characteristics
        characteristics = {
            'trend_direction': 'up' if trend > 0 else 'down' if trend < 0 else 'flat',
            'trend_strength': abs(trend),
            'volatility_level': 'high' if volatility > 0.02 else 'medium' if volatility > 0.01 else 'low',
            'market_sentiment': 'fearful' if fear_greed < 30 else 'greedy' if fear_greed > 70 else 'neutral',
            'tradability_score': self._calculate_tradability(features)
        }

        return {
            'regime': regime,
            'confidence': confidence,
            'characteristics': characteristics,
            'features': features
        }

    def _calculate_fear_greed(self, features: Dict) -> float:
        """Calculate fear/greed score (0-100)"""
        score = 50  # Neutral start

        # RSI component
        if features['rsi'] > 70:
            score += (features['rsi'] - 70) * 0.5  # Greed
        elif features['rsi'] < 30:
            score -= (30 - features['rsi']) * 0.5  # Fear

        # Momentum component
        if features['momentum_10d'] > 0.05:
            score += features['momentum_10d'] * 100  # Greed
        elif features['momentum_10d'] < -0.05:
            score += features['momentum_10d'] * 100  # Fear (negative adds fear)

        # Volatility component (high volatility = fear)
        if features['volatility'] > 0.02:
            score -= (features['volatility'] - 0.02) * 500

        # Volume component
        if features['volume_trend'] > 1.5:
            if features['trend_strength'] > 0:
                score += 10  # High volume in uptrend = greed
            else:
                score -= 10  # High volume in downtrend = fear

        # Clamp to 0-100
        return max(0, min(100, score))

    def _calculate_tradability(self, features: Dict) -> float:
        """Calculate how suitable the market is for trading (0-1)"""
        score = 1.0

        # Reduce score for extreme volatility
        if features['volatility'] > 0.03:
            score *= 0.7
        elif features['volatility'] < 0.005:
            score *= 0.8  # Too quiet is also bad

        # Reduce score for poor trend quality
        score *= (0.3 + 0.7 * features['trend_quality'])

        # Reduce score for extreme RSI
        if features['rsi'] > 80 or features['rsi'] < 20:
            score *= 0.8

        # Boost score for strong trends with reasonable volatility
        if abs(features['trend_strength']) > 0.03 and features['volatility'] < 0.02:
            score *= 1.2

        return min(1.0, score)

    def fit_gmm_model(self, historical_data: pd.DataFrame):
        """Train GMM model on historical data for regime detection"""
        # Calculate features for all historical periods
        all_features = []

        window_size = 60
        for i in range(window_size, len(historical_data)):
            window = historical_data.iloc[i - window_size:i]
            features = self._calculate_regime_features(window, window_size)

            # Convert to feature vector
            feature_vector = [
                features['trend_strength'],
                features['volatility'],
                features['trend_quality'],
                features['atr_ratio'],
                features['volume_trend']
            ]
            all_features.append(feature_vector)

        # Fit GMM
        X = np.array(all_features)
        X_scaled = self.scaler.fit_transform(X)
        self.gmm.fit(X_scaled)
        self.is_fitted = True

        logger.info("GMM regime model fitted")

    def get_regime_specific_params(self, regime: str) -> Dict:
        """Get trading parameters specific to each regime"""
        params = {
            'bull_quiet': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.2,
                'min_confidence': 0.60,
                'preferred_strategies': ['momentum', 'breakout'],
                'avoid_strategies': ['mean_reversion']
            },
            'bull_volatile': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.5,
                'min_confidence': 0.70,
                'preferred_strategies': ['momentum', 'volatility_breakout'],
                'avoid_strategies': ['mean_reversion']
            },
            'bear_quiet': {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.8,
                'min_confidence': 0.75,
                'preferred_strategies': ['oversold_bounce', 'mean_reversion'],
                'avoid_strategies': ['momentum', 'breakout']
            },
            'bear_volatile': {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.0,
                'min_confidence': 0.80,
                'preferred_strategies': ['oversold_bounce'],
                'avoid_strategies': ['momentum', 'breakout']
            },
            'ranging': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'min_confidence': 0.65,
                'preferred_strategies': ['mean_reversion', 'range_breakout'],
                'avoid_strategies': ['momentum']
            }
        }

        return params.get(regime, params['ranging'])

    def analyze_regime_transition(self, lookback_days: int = 30) -> Dict:
        """Analyze recent regime transitions"""
        if len(self.regime_history) < 2:
            return {
                'transitioning': False,
                'from_regime': None,
                'to_regime': None,
                'stability': 1.0
            }

        # Get recent regimes
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_regimes = [
            r for r in self.regime_history
            if r['timestamp'] > cutoff
        ]

        if not recent_regimes:
            return {
                'transitioning': False,
                'stability': 1.0
            }

        # Count regime occurrences
        regime_counts = {}
        for r in recent_regimes:
            regime = r['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate stability
        total_observations = len(recent_regimes)
        max_count = max(regime_counts.values())
        stability = max_count / total_observations

        # Check for transition
        transitioning = stability < 0.7

        result = {
            'transitioning': transitioning,
            'stability': stability,
            'regime_distribution': regime_counts
        }

        # If transitioning, identify from/to
        if transitioning and len(recent_regimes) >= 10:
            first_half = recent_regimes[:len(recent_regimes) // 2]
            second_half = recent_regimes[len(recent_regimes) // 2:]

            # Most common regime in each half
            from_regime = max(set(r['regime'] for r in first_half),
                              key=lambda x: sum(1 for r in first_half if r['regime'] == x))
            to_regime = max(set(r['regime'] for r in second_half),
                            key=lambda x: sum(1 for r in second_half if r['regime'] == x))

            if from_regime != to_regime:
                result['from_regime'] = from_regime
                result['to_regime'] = to_regime

        return result

    def get_historical_performance(self, regime: str,
                                   performance_data: pd.DataFrame) -> Dict:
        """Get historical performance statistics for a specific regime"""
        # This would analyze historical performance during different regimes
        # For now, return placeholder
        return {
            'avg_return': 0.02,
            'win_rate': 0.55,
            'avg_holding_days': 3.5,
            'best_performing_sectors': ['Technology', 'Healthcare'],
            'worst_performing_sectors': ['Energy', 'Utilities']
        }