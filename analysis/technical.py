# analysis/technical.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import talib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Technical analysis for swing trading
    Identifies high-probability setups based on price action and indicators
    """

    def __init__(self):
        # Pattern configurations
        self.min_pattern_strength = 0.7
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }

        # Setup types for swing trading
        self.setup_types = [
            'breakout',
            'pullback',
            'reversal',
            'momentum',
            'range_bound'
        ]

    def analyze(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """
        Comprehensive technical analysis

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            Dictionary with technical analysis results
        """
        if len(df) < 50:
            return {
                'symbol': symbol,
                'setup': None,
                'confidence': 0,
                'error': 'Insufficient data'
            }

        # Calculate all indicators
        indicators = self._calculate_indicators(df)

        # Identify patterns
        patterns = self._identify_patterns(df, indicators)

        # Find best setup
        setup = self._find_best_setup(df, indicators, patterns)

        # Calculate support/resistance
        levels = self._calculate_support_resistance(df)

        # Generate signal
        signal = self._generate_signal(setup, indicators, levels)

        return {
            'symbol': symbol,
            'setup': setup,
            'signal': signal,
            'indicators': self._summarize_indicators(indicators),
            'patterns': patterns,
            'levels': levels,
            'timestamp': datetime.now()
        }

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        indicators = {}

        # Moving averages
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)
        indicators['sma_200'] = talib.SMA(close, timeperiod=200) if len(close) >= 200 else None
        indicators['ema_20'] = talib.EMA(close, timeperiod=20)

        # Momentum indicators
        indicators['rsi'] = talib.RSI(close, timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
        indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)

        # Volatility indicators
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']

        # Volume indicators
        indicators['obv'] = talib.OBV(close, volume)
        indicators['ad'] = talib.AD(high, low, close, volume)
        indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

        # Trend indicators
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
        indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        # Additional calculations
        indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)
        indicators['volume_ratio'] = volume / indicators['volume_sma']

        return indicators

    def _identify_patterns(self, df: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Identify chart patterns"""
        patterns = []

        # Candlestick patterns
        candle_patterns = self._identify_candlestick_patterns(df)
        patterns.extend(candle_patterns)

        # Chart patterns
        chart_patterns = self._identify_chart_patterns(df, indicators)
        patterns.extend(chart_patterns)

        # Indicator patterns
        indicator_patterns = self._identify_indicator_patterns(indicators)
        patterns.extend(indicator_patterns)

        return patterns

    def _identify_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Identify candlestick patterns"""
        patterns = []

        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Bullish patterns
        hammer = talib.CDLHAMMER(open_price, high, low, close)
        if hammer[-1] != 0:
            patterns.append({
                'type': 'candlestick',
                'name': 'hammer',
                'signal': 'bullish',
                'strength': abs(hammer[-1]) / 100
            })

        engulfing = talib.CDLENGULFING(open_price, high, low, close)
        if engulfing[-1] > 0:
            patterns.append({
                'type': 'candlestick',
                'name': 'bullish_engulfing',
                'signal': 'bullish',
                'strength': 0.8
            })

        morning_star = talib.CDLMORNINGSTAR(open_price, high, low, close)
        if morning_star[-1] != 0:
            patterns.append({
                'type': 'candlestick',
                'name': 'morning_star',
                'signal': 'bullish',
                'strength': 0.9
            })

        # Bearish patterns
        shooting_star = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        if shooting_star[-1] != 0:
            patterns.append({
                'type': 'candlestick',
                'name': 'shooting_star',
                'signal': 'bearish',
                'strength': abs(shooting_star[-1]) / 100
            })

        if engulfing[-1] < 0:
            patterns.append({
                'type': 'candlestick',
                'name': 'bearish_engulfing',
                'signal': 'bearish',
                'strength': 0.8
            })

        return patterns

    def _identify_chart_patterns(self, df: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Identify chart patterns like triangles, flags, etc."""
        patterns = []

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Double bottom
        if self._is_double_bottom(low, close):
            patterns.append({
                'type': 'chart',
                'name': 'double_bottom',
                'signal': 'bullish',
                'strength': 0.8
            })

        # Ascending triangle
        if self._is_ascending_triangle(high, low, close):
            patterns.append({
                'type': 'chart',
                'name': 'ascending_triangle',
                'signal': 'bullish',
                'strength': 0.7
            })

        # Bull flag
        if self._is_bull_flag(high, low, close, indicators):
            patterns.append({
                'type': 'chart',
                'name': 'bull_flag',
                'signal': 'bullish',
                'strength': 0.75
            })

        return patterns

    def _identify_indicator_patterns(self, indicators: Dict) -> List[Dict]:
        """Identify patterns in indicators"""
        patterns = []

        # RSI patterns
        rsi = indicators['rsi']
        if rsi[-1] < 30 and rsi[-2] < rsi[-1]:  # Oversold bounce
            patterns.append({
                'type': 'indicator',
                'name': 'rsi_oversold_bounce',
                'signal': 'bullish',
                'strength': 0.7
            })
        elif rsi[-1] > 70 and rsi[-2] > rsi[-1]:  # Overbought reversal
            patterns.append({
                'type': 'indicator',
                'name': 'rsi_overbought_reversal',
                'signal': 'bearish',
                'strength': 0.7
            })

        # MACD patterns
        macd = indicators['macd']
        signal = indicators['macd_signal']

        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:  # Bullish cross
            patterns.append({
                'type': 'indicator',
                'name': 'macd_bullish_cross',
                'signal': 'bullish',
                'strength': 0.8
            })
        elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:  # Bearish cross
            patterns.append({
                'type': 'indicator',
                'name': 'macd_bearish_cross',
                'signal': 'bearish',
                'strength': 0.8
            })

        # Moving average patterns
        if indicators['sma_50'] is not None:
            close = indicators['bb_middle']  # Using close from BB calculation

            # Golden cross
            if (indicators['sma_50'][-1] > indicators['sma_200'][-1] and
                    indicators['sma_50'][-2] <= indicators['sma_200'][-2]):
                patterns.append({
                    'type': 'indicator',
                    'name': 'golden_cross',
                    'signal': 'bullish',
                    'strength': 0.9
                })

        return patterns

    def _find_best_setup(self, df: pd.DataFrame, indicators: Dict,
                         patterns: List[Dict]) -> Dict:
        """Find the best trading setup"""
        setups = []

        # Breakout setup
        breakout = self._check_breakout_setup(df, indicators, patterns)
        if breakout['confidence'] > 0:
            setups.append(breakout)

        # Pullback setup
        pullback = self._check_pullback_setup(df, indicators, patterns)
        if pullback['confidence'] > 0:
            setups.append(pullback)

        # Reversal setup
        reversal = self._check_reversal_setup(df, indicators, patterns)
        if reversal['confidence'] > 0:
            setups.append(reversal)

        # Momentum setup
        momentum = self._check_momentum_setup(df, indicators, patterns)
        if momentum['confidence'] > 0:
            setups.append(momentum)

        # Range-bound setup
        range_bound = self._check_range_setup(df, indicators)
        if range_bound['confidence'] > 0:
            setups.append(range_bound)

        # Return best setup
        if setups:
            return max(setups, key=lambda x: x['confidence'])
        else:
            return {
                'type': 'none',
                'confidence': 0,
                'description': 'No clear setup identified'
            }

    def _check_breakout_setup(self, df: pd.DataFrame, indicators: Dict,
                              patterns: List[Dict]) -> Dict:
        """Check for breakout setup"""
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values

        confidence = 0
        reasons = []

        # Recent high
        recent_high = high[-20:].max()

        # Check if breaking out
        if close[-1] > recent_high * 0.99:
            confidence += 0.3
            reasons.append("Price near 20-day high")

        # Volume confirmation
        if indicators['volume_ratio'][-1] > 1.5:
            confidence += 0.2
            reasons.append("High volume")

        # ADX trending
        if indicators['adx'][-1] > 25:
            confidence += 0.2
            reasons.append("Strong trend (ADX > 25)")

        # Bollinger Band breakout
        if close[-1] > indicators['bb_upper'][-1]:
            confidence += 0.1
            reasons.append("Breaking upper Bollinger Band")

        # Pattern confirmation
        bullish_patterns = [p for p in patterns if p['signal'] == 'bullish']
        if bullish_patterns:
            confidence += 0.2
            reasons.append(f"Bullish pattern: {bullish_patterns[0]['name']}")

        return {
            'type': 'breakout',
            'confidence': min(confidence, 1.0),
            'direction': 'long',
            'entry': close[-1],
            'stop': recent_high * 0.97,
            'target': close[-1] * 1.06,
            'reasons': reasons,
            'description': 'Breakout from consolidation'
        }

    def _check_pullback_setup(self, df: pd.DataFrame, indicators: Dict,
                              patterns: List[Dict]) -> Dict:
        """Check for pullback setup"""
        close = df['close'].values

        confidence = 0
        reasons = []

        # Trend check
        if indicators['sma_20'][-1] > indicators['sma_50'][-1]:
            confidence += 0.2
            reasons.append("Uptrend intact")

        # Pullback to support
        if abs(close[-1] - indicators['sma_20'][-1]) / close[-1] < 0.02:
            confidence += 0.3
            reasons.append("Pullback to 20-SMA support")

        # RSI not oversold
        if 40 < indicators['rsi'][-1] < 60:
            confidence += 0.2
            reasons.append("RSI in healthy range")

        # Decreasing volume on pullback
        if indicators['volume_ratio'][-1] < 0.8:
            confidence += 0.1
            reasons.append("Low volume pullback")

        # Bouncing pattern
        if close[-1] > close[-2] and close[-2] < close[-3]:
            confidence += 0.2
            reasons.append("Potential bounce pattern")

        return {
            'type': 'pullback',
            'confidence': min(confidence, 1.0),
            'direction': 'long',
            'entry': close[-1],
            'stop': indicators['sma_20'][-1] * 0.97,
            'target': close[-1] * 1.05,
            'reasons': reasons,
            'description': 'Pullback to support in uptrend'
        }

    def _check_reversal_setup(self, df: pd.DataFrame, indicators: Dict,
                              patterns: List[Dict]) -> Dict:
        """Check for reversal setup"""
        close = df['close'].values
        low = df['low'].values

        confidence = 0
        reasons = []

        # Oversold bounce
        if indicators['rsi'][-1] < 30:
            confidence += 0.3
            reasons.append("RSI oversold")

        # Bullish divergence
        if self._check_divergence(close[-20:], indicators['rsi'][-20:], 'bullish'):
            confidence += 0.3
            reasons.append("Bullish divergence")

        # Support level
        recent_low = low[-20:].min()
        if abs(low[-1] - recent_low) / recent_low < 0.01:
            confidence += 0.2
            reasons.append("At support level")

        # Reversal pattern
        reversal_patterns = [p for p in patterns
                             if p['signal'] == 'bullish' and 'reversal' in p.get('name', '')]
        if reversal_patterns:
            confidence += 0.2
            reasons.append(f"Reversal pattern: {reversal_patterns[0]['name']}")

        return {
            'type': 'reversal',
            'confidence': min(confidence, 1.0),
            'direction': 'long',
            'entry': close[-1],
            'stop': recent_low * 0.98,
            'target': close[-1] * 1.08,
            'reasons': reasons,
            'description': 'Potential trend reversal'
        }

    def _check_momentum_setup(self, df: pd.DataFrame, indicators: Dict,
                              patterns: List[Dict]) -> Dict:
        """Check for momentum setup"""
        close = df['close'].values

        confidence = 0
        reasons = []

        # Strong momentum
        returns_5d = (close[-1] / close[-5] - 1) * 100
        if returns_5d > 5:
            confidence += 0.3
            reasons.append(f"Strong 5-day momentum: {returns_5d:.1f}%")

        # All MAs aligned
        if (indicators['sma_20'][-1] > indicators['sma_50'][-1] and
                close[-1] > indicators['sma_20'][-1]):
            confidence += 0.2
            reasons.append("Price above all MAs")

        # RSI strong but not overbought
        if 55 < indicators['rsi'][-1] < 70:
            confidence += 0.2
            reasons.append("RSI showing strength")

        # Volume confirmation
        if indicators['volume_ratio'][-1] > 1.2:
            confidence += 0.2
            reasons.append("Above average volume")

        # ADX trending
        if indicators['adx'][-1] > 30:
            confidence += 0.1
            reasons.append("Very strong trend")

        return {
            'type': 'momentum',
            'confidence': min(confidence, 1.0),
            'direction': 'long',
            'entry': close[-1],
            'stop': indicators['sma_20'][-1],
            'target': close[-1] * 1.06,
            'reasons': reasons,
            'description': 'Momentum continuation play'
        }

    def _check_range_setup(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Check for range-bound setup"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        confidence = 0
        reasons = []

        # Define range
        recent_high = high[-20:].max()
        recent_low = low[-20:].min()
        range_size = (recent_high - recent_low) / recent_low

        # Check if in range
        if 0.05 < range_size < 0.15:  # 5-15% range
            confidence += 0.2
            reasons.append(f"Trading in {range_size * 100:.1f}% range")

        # Near range boundary
        distance_to_low = (close[-1] - recent_low) / recent_low
        distance_to_high = (recent_high - close[-1]) / recent_high

        if distance_to_low < 0.02:
            confidence += 0.3
            reasons.append("Near range support")
            direction = 'long'
            entry = close[-1]
            stop = recent_low * 0.98
            target = recent_low + (recent_high - recent_low) * 0.8

        elif distance_to_high < 0.02:
            confidence += 0.3
            reasons.append("Near range resistance")
            direction = 'short'  # Note: we don't short, so this would be exit signal
            entry = close[-1]
            stop = recent_high * 1.02
            target = recent_high - (recent_high - recent_low) * 0.8
        else:
            return {'type': 'range', 'confidence': 0}

        # Low ADX confirms range
        if indicators['adx'][-1] < 20:
            confidence += 0.2
            reasons.append("Low ADX confirms range")

        # RSI in middle zone
        if 40 < indicators['rsi'][-1] < 60:
            confidence += 0.1
            reasons.append("RSI neutral")

        return {
            'type': 'range',
            'confidence': min(confidence, 1.0),
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'reasons': reasons,
            'description': 'Range-bound trading opportunity'
        }

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Recent highs and lows
        resistance_1 = high[-20:].max()
        support_1 = low[-20:].min()

        resistance_2 = high[-50:].max() if len(high) >= 50 else resistance_1
        support_2 = low[-50:].min() if len(low) >= 50 else support_1

        # Pivot points
        pivot = (high[-1] + low[-1] + close[-1]) / 3
        r1 = 2 * pivot - low[-1]
        s1 = 2 * pivot - high[-1]

        # Psychological levels (round numbers)
        psychological_levels = []
        current = close[-1]

        for level in [10, 25, 50, 75, 100]:
            psych_level = round(current / level) * level
            if abs(psych_level - current) / current < 0.05:  # Within 5%
                psychological_levels.append(psych_level)

        return {
            'resistance_1': round(resistance_1, 2),
            'resistance_2': round(resistance_2, 2),
            'support_1': round(support_1, 2),
            'support_2': round(support_2, 2),
            'pivot': round(pivot, 2),
            'r1': round(r1, 2),
            's1': round(s1, 2),
            'psychological': psychological_levels
        }

    def _generate_signal(self, setup: Dict, indicators: Dict, levels: Dict) -> Dict:
        """Generate trading signal based on analysis"""
        if setup['confidence'] < 0.6:
            return {
                'action': 'hold',
                'confidence': setup['confidence'],
                'reason': 'Insufficient confidence'
            }

        # Check trend alignment
        trend_aligned = True
        if setup['direction'] == 'long':
            # Don't buy if clearly downtrending
            if (indicators['sma_20'][-1] < indicators['sma_50'][-1] and
                    indicators['adx'][-1] > 25):
                trend_aligned = False

        if not trend_aligned:
            return {
                'action': 'hold',
                'confidence': setup['confidence'] * 0.5,
                'reason': 'Setup against primary trend'
            }

        # Generate signal
        signal = {
            'action': 'buy' if setup['direction'] == 'long' else 'hold',
            'confidence': setup['confidence'],
            'setup_type': setup['type'],
            'entry': setup['entry'],
            'stop_loss': setup['stop'],
            'target': setup['target'],
            'risk_reward': (setup['target'] - setup['entry']) / (setup['entry'] - setup['stop']),
            'reasons': setup['reasons'],
            'levels': levels
        }

        return signal

    def _summarize_indicators(self, indicators: Dict) -> Dict:
        """Create summary of indicator states"""
        summary = {}

        # Trend
        if indicators['sma_20'][-1] > indicators['sma_50'][-1]:
            summary['trend'] = 'up'
        else:
            summary['trend'] = 'down'

        # Momentum
        rsi = indicators['rsi'][-1]
        if rsi > 70:
            summary['momentum'] = 'overbought'
        elif rsi < 30:
            summary['momentum'] = 'oversold'
        elif rsi > 60:
            summary['momentum'] = 'strong'
        elif rsi < 40:
            summary['momentum'] = 'weak'
        else:
            summary['momentum'] = 'neutral'

        # Volatility
        bb_width = indicators['bb_width'][-1]
        if bb_width > indicators['bb_width'][-20:].mean() * 1.5:
            summary['volatility'] = 'high'
        elif bb_width < indicators['bb_width'][-20:].mean() * 0.5:
            summary['volatility'] = 'low'
        else:
            summary['volatility'] = 'normal'

        # Volume
        if indicators['volume_ratio'][-1] > 1.5:
            summary['volume'] = 'high'
        elif indicators['volume_ratio'][-1] < 0.5:
            summary['volume'] = 'low'
        else:
            summary['volume'] = 'average'

        # Key levels
        summary['current_price'] = indicators['bb_middle'][-1]
        summary['rsi'] = round(rsi, 1)
        summary['adx'] = round(indicators['adx'][-1], 1)

        return summary

    def _is_double_bottom(self, low: np.ndarray, close: np.ndarray) -> bool:
        """Check if double bottom pattern exists"""
        if len(low) < 40:
            return False

        # Find two recent lows
        first_low_idx = np.argmin(low[-40:-20])
        second_low_idx = np.argmin(low[-20:]) + len(low) - 20

        first_low = low[-40:-20][first_low_idx]
        second_low = low[second_low_idx]

        # Check if lows are similar (within 2%)
        if abs(first_low - second_low) / first_low > 0.02:
            return False

        # Check if there was a peak between them
        peak_between = max(low[-40 + first_low_idx:second_low_idx])
        if (peak_between - first_low) / first_low < 0.05:  # At least 5% move up
            return False

        # Check if price is breaking above neckline
        neckline = peak_between
        if close[-1] > neckline * 0.98:
            return True

        return False

    def _is_ascending_triangle(self, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray) -> bool:
        """Check if ascending triangle pattern exists"""
        if len(high) < 30:
            return False

        # Check for flat resistance
        recent_highs = high[-30:]
        high_std = np.std(recent_highs[-10:])
        high_mean = np.mean(recent_highs[-10:])

        if high_std / high_mean > 0.02:  # Too much variance
            return False

        # Check for rising support
        recent_lows = low[-30:]

        # Simple linear regression on lows
        x = np.arange(len(recent_lows))
        slope = np.polyfit(x, recent_lows, 1)[0]

        # Positive slope indicates ascending triangle
        if slope > 0 and close[-1] > high_mean * 0.99:
            return True

        return False

    def _is_bull_flag(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, indicators: Dict) -> bool:
        """Check if bull flag pattern exists"""
        if len(close) < 30:
            return False

        # Need strong move up first (pole)
        pole_start = close[-30]
        pole_end = max(close[-20:-10])
        pole_gain = (pole_end - pole_start) / pole_start

        if pole_gain < 0.10:  # Need at least 10% move
            return False

        # Then consolidation (flag)
        flag_highs = high[-10:]
        flag_lows = low[-10:]

        # Flag should be tight consolidation
        flag_range = (max(flag_highs) - min(flag_lows)) / min(flag_lows)
        if flag_range > 0.05:  # Too wide
            return False

        # Slightly descending is okay
        flag_slope = np.polyfit(np.arange(len(flag_highs)), flag_highs, 1)[0]

        # Volume should be decreasing in flag
        if 'volume_ratio' in indicators:
            recent_vol = indicators['volume_ratio'][-10:].mean()
            if recent_vol > 1.0:  # Volume too high
                return False

        return True

    def _check_divergence(self, price: np.ndarray, indicator: np.ndarray,
                          divergence_type: str) -> bool:
        """Check for divergence between price and indicator"""
        if len(price) < 10 or len(indicator) < 10:
            return False

        # Find peaks and troughs
        price_peaks = []
        indicator_peaks = []

        for i in range(2, len(price) - 2):
            # Price peaks
            if price[i] > price[i - 1] and price[i] > price[i - 2] and \
                    price[i] > price[i + 1] and price[i] > price[i + 2]:
                price_peaks.append((i, price[i]))

            # Indicator peaks
            if indicator[i] > indicator[i - 1] and indicator[i] > indicator[i - 2] and \
                    indicator[i] > indicator[i + 1] and indicator[i] > indicator[i + 2]:
                indicator_peaks.append((i, indicator[i]))

        if divergence_type == 'bullish':
            # Price making lower lows, indicator making higher lows
            if len(price_peaks) >= 2:
                if (price_peaks[-1][1] < price_peaks[-2][1] and
                        len(indicator_peaks) >= 2 and
                        indicator_peaks[-1][1] > indicator_peaks[-2][1]):
                    return True

        return False