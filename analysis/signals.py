# analysis/signals.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from models.ensemble import EnsembleModel
from models.regime_detector import MarketRegimeDetector
from analysis.technical import TechnicalAnalyzer
from analysis.news_sentiment import NewsSentimentAnalyzer

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Combines ML predictions, technical analysis, and news sentiment
    to generate high-confidence trading signals
    """

    def __init__(self, ml_model: EnsembleModel = None,
                 news_analyzer: NewsSentimentAnalyzer = None):
        """Initialize signal generator"""
        self.ml_model = ml_model or EnsembleModel()
        self.news_analyzer = news_analyzer or NewsSentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()

        # Signal thresholds
        self.min_confidence = 0.65
        self.min_ml_confidence = 0.60
        self.min_technical_confidence = 0.70
        self.min_combined_score = 0.65

        # Signal history for tracking
        self.signal_history = []

    def generate_signals(self, watchlist: List[str], market_data: Dict[str, pd.DataFrame],
                         market_regime: str = None) -> List[Dict]:
        """
        Generate trading signals for watchlist

        Args:
            watchlist: List of symbols to analyze
            market_data: Dictionary of symbol -> price data
            market_regime: Current market regime

        Returns:
            List of trading signals
        """
        signals = []

        # Get market regime if not provided
        if market_regime is None and 'SPY' in market_data:
            regime_info = self.regime_detector.detect_regime(market_data['SPY'])
            market_regime = regime_info['regime']
        else:
            market_regime = market_regime or 'ranging'

        logger.info(f"Generating signals for {len(watchlist)} symbols in {market_regime} market")

        # Batch process news sentiment
        news_sentiments = self._batch_analyze_news(watchlist[:50])  # Limit for API

        # Analyze each symbol
        for symbol in watchlist:
            if symbol not in market_data:
                continue

            price_data = market_data[symbol]
            if len(price_data) < 100:
                continue

            try:
                # Generate signal
                signal = self._analyze_symbol(
                    symbol, price_data,
                    news_sentiments.get(symbol, {}),
                    market_regime
                )

                if signal and signal['confidence'] >= self.min_confidence:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Rank signals by combined score
        signals = self._rank_signals(signals)

        # Store in history
        self.signal_history.extend(signals[:10])  # Keep top 10

        logger.info(f"Generated {len(signals)} signals")

        return signals

    def _analyze_symbol(self, symbol: str, price_data: pd.DataFrame,
                        news_sentiment: Dict, market_regime: str) -> Optional[Dict]:
        """Analyze individual symbol"""

        # 1. ML Prediction
        ml_signal = self._get_ml_signal(symbol, price_data)

        # 2. Technical Analysis
        technical_signal = self._get_technical_signal(symbol, price_data)

        # 3. News Sentiment (already fetched)
        news_signal = self._process_news_sentiment(news_sentiment)

        # 4. Combine signals
        combined_signal = self._combine_signals(
            symbol, ml_signal, technical_signal, news_signal, market_regime
        )

        return combined_signal

    def _get_ml_signal(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Get ML model prediction"""
        if not self.ml_model.is_trained:
            return {
                'confidence': 0,
                'prediction': 0,
                'source': 'ml_model'
            }

        prediction = self.ml_model.predict(symbol, price_data)

        return {
            'confidence': prediction['confidence'],
            'prediction': prediction['prediction'],
            'probability': prediction['probability'],
            'expected_return': prediction.get('expected_return', 0),
            'source': 'ml_model',
            'details': prediction
        }

    def _get_technical_signal(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Get technical analysis signal"""
        analysis = self.technical_analyzer.analyze(price_data, symbol)

        signal = analysis.get('signal', {})
        setup = analysis.get('setup', {})

        # Convert to standard format
        if signal.get('action') == 'buy':
            prediction = 1
            confidence = signal.get('confidence', 0)
        else:
            prediction = 0
            confidence = 0

        return {
            'confidence': confidence,
            'prediction': prediction,
            'setup_type': setup.get('type', 'none'),
            'entry': signal.get('entry'),
            'stop_loss': signal.get('stop_loss'),
            'target': signal.get('target'),
            'risk_reward': signal.get('risk_reward', 0),
            'reasons': signal.get('reasons', []),
            'source': 'technical',
            'details': analysis
        }

    def _batch_analyze_news(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch analyze news for multiple symbols"""
        news_sentiments = {}

        # Get market sentiment first
        market_sentiment = self.news_analyzer.get_market_sentiment()

        # Analyze individual symbols
        for symbol in symbols:
            try:
                sentiment = self.news_analyzer.analyze_symbol_news(symbol, hours=48)
                news_sentiments[symbol] = sentiment
            except Exception as e:
                logger.error(f"Error analyzing news for {symbol}: {e}")
                news_sentiments[symbol] = {
                    'sentiment': 'neutral',
                    'confidence': 0
                }

        return news_sentiments

    def _process_news_sentiment(self, sentiment: Dict) -> Dict:
        """Process news sentiment into signal format"""
        if not sentiment or sentiment.get('confidence', 0) == 0:
            return {
                'confidence': 0,
                'prediction': 0,
                'source': 'news'
            }

        # Convert sentiment to prediction
        if sentiment['sentiment'] == 'bullish':
            prediction = 1
            confidence = sentiment['confidence']
        elif sentiment['sentiment'] == 'bearish':
            prediction = 0
            confidence = sentiment['confidence']
        else:
            prediction = 0
            confidence = 0

        return {
            'confidence': confidence,
            'prediction': prediction,
            'sentiment': sentiment['sentiment'],
            'article_count': sentiment.get('article_count', 0),
            'key_events': sentiment.get('key_events', []),
            'source': 'news',
            'details': sentiment
        }

    def _combine_signals(self, symbol: str, ml_signal: Dict, technical_signal: Dict,
                         news_signal: Dict, market_regime: str) -> Optional[Dict]:
        """Combine multiple signals into final signal"""

        # Check minimum requirements
        if ml_signal['confidence'] < 0.3 and technical_signal['confidence'] < 0.3:
            return None

        # Calculate weighted confidence
        weights = self._get_signal_weights(market_regime)

        combined_confidence = (
                ml_signal['confidence'] * weights['ml'] +
                technical_signal['confidence'] * weights['technical'] +
                news_signal['confidence'] * weights['news']
        )

        # Determine action
        predictions = [
            ml_signal['prediction'],
            technical_signal['prediction'],
            news_signal['prediction']
        ]

        # Weighted vote
        weighted_prediction = (
                                      ml_signal['prediction'] * ml_signal['confidence'] * weights['ml'] +
                                      technical_signal['prediction'] * technical_signal['confidence'] * weights[
                                          'technical'] +
                                      news_signal['prediction'] * news_signal['confidence'] * weights['news']
                              ) / (
                                      ml_signal['confidence'] * weights['ml'] +
                                      technical_signal['confidence'] * weights['technical'] +
                                      news_signal['confidence'] * weights['news'] + 1e-6
                              )

        # Final prediction
        final_prediction = 1 if weighted_prediction > 0.6 else 0

        # Compile reasons
        reasons = []

        if ml_signal['confidence'] > 0.6:
            reasons.append(f"ML model confident ({ml_signal['confidence']:.1%})")

        if technical_signal['confidence'] > 0.6:
            reasons.append(f"Technical setup: {technical_signal.get('setup_type', 'unknown')}")
            reasons.extend(technical_signal.get('reasons', [])[:2])

        if news_signal['confidence'] > 0.6:
            reasons.append(f"News sentiment: {news_signal.get('sentiment', 'unknown')}")

        # Build final signal
        signal = {
            'symbol': symbol,
            'action': 'buy' if final_prediction == 1 else 'hold',
            'confidence': combined_confidence,
            'timestamp': datetime.now(),

            # Component scores
            'ml_confidence': ml_signal['confidence'],
            'technical_confidence': technical_signal['confidence'],
            'news_confidence': news_signal['confidence'],

            # Trading parameters
            'entry_price': technical_signal.get('entry'),
            'stop_loss': technical_signal.get('stop_loss'),
            'target_price': technical_signal.get('target'),
            'expected_return': ml_signal.get('expected_return', 0),
            'risk_reward': technical_signal.get('risk_reward', 0),

            # Details
            'setup_type': technical_signal.get('setup_type'),
            'reasons': reasons,
            'market_regime': market_regime,

            # Source data
            'ml_signal': ml_signal,
            'technical_signal': technical_signal,
            'news_signal': news_signal
        }

        # Calculate combined score for ranking
        signal['combined_score'] = self._calculate_combined_score(signal)

        return signal

    def _get_signal_weights(self, market_regime: str) -> Dict[str, float]:
        """Get signal weights based on market regime"""
        weights = {
            'bull_quiet': {'ml': 0.5, 'technical': 0.3, 'news': 0.2},
            'bull_volatile': {'ml': 0.4, 'technical': 0.4, 'news': 0.2},
            'bear_quiet': {'ml': 0.3, 'technical': 0.5, 'news': 0.2},
            'bear_volatile': {'ml': 0.3, 'technical': 0.4, 'news': 0.3},
            'ranging': {'ml': 0.4, 'technical': 0.4, 'news': 0.2}
        }

        return weights.get(market_regime, weights['ranging'])

    def _calculate_combined_score(self, signal: Dict) -> float:
        """Calculate combined score for signal ranking"""
        # Base score is confidence
        score = signal['confidence']

        # Bonus for agreement between signals
        agreement_count = sum([
            signal['ml_confidence'] > 0.6,
            signal['technical_confidence'] > 0.6,
            signal['news_confidence'] > 0.6
        ])

        if agreement_count >= 3:
            score *= 1.3
        elif agreement_count >= 2:
            score *= 1.15

        # Bonus for good risk/reward
        risk_reward = signal.get('risk_reward', 0)
        if risk_reward > 3:
            score *= 1.2
        elif risk_reward > 2:
            score *= 1.1

        # Penalty for volatile markets
        if 'volatile' in signal.get('market_regime', ''):
            score *= 0.9

        # Expected return factor
        expected_return = signal.get('expected_return', 0)
        if expected_return > 0.08:
            score *= 1.15
        elif expected_return > 0.05:
            score *= 1.05

        return min(score, 1.0)

    def _rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by quality"""
        # Sort by combined score
        ranked = sorted(signals, key=lambda x: x['combined_score'], reverse=True)

        # Add ranking
        for i, signal in enumerate(ranked):
            signal['rank'] = i + 1

        return ranked

    def get_exit_signals(self, positions: List[Dict], market_data: Dict[str, pd.DataFrame],
                         market_regime: str = None) -> List[Dict]:
        """Generate exit signals for existing positions"""
        exit_signals = []

        for position in positions:
            symbol = position['symbol']

            if symbol not in market_data:
                continue

            price_data = market_data[symbol]
            current_price = price_data['close'].iloc[-1]

            # Calculate metrics
            entry_price = position['entry_price']
            current_return = (current_price - entry_price) / entry_price
            days_held = (datetime.now() - position['entry_time']).days

            # Exit conditions
            exit_reasons = []
            confidence = 0

            # 1. Target reached
            if current_return >= 0.06:  # 6% target
                exit_reasons.append("Target reached")
                confidence = 0.9

            # 2. Stop loss
            elif current_return <= -0.03:  # 3% stop
                exit_reasons.append("Stop loss triggered")
                confidence = 1.0

            # 3. Time stop
            elif days_held >= 5 and current_return < 0.01:
                exit_reasons.append("Time stop (5 days)")
                confidence = 0.8

            # 4. Technical reversal
            else:
                tech_analysis = self.technical_analyzer.analyze(price_data, symbol)
                if tech_analysis.get('signal', {}).get('action') == 'sell':
                    exit_reasons.append("Technical reversal")
                    confidence = tech_analysis['signal'].get('confidence', 0.5)

                # 5. ML model reversal
                ml_pred = self.ml_model.predict(symbol, price_data)
                if ml_pred['prediction'] == 0 and ml_pred['confidence'] > 0.7:
                    exit_reasons.append("ML model reversal")
                    confidence = max(confidence, ml_pred['confidence'])

            if exit_reasons and confidence > 0.5:
                exit_signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': confidence,
                    'current_price': current_price,
                    'current_return': current_return,
                    'days_held': days_held,
                    'reasons': exit_reasons,
                    'position': position,
                    'timestamp': datetime.now()
                })

        return exit_signals

    def validate_signal(self, signal: Dict, current_positions: List[Dict],
                        account_value: float) -> Dict:
        """Validate signal against portfolio constraints"""
        validation = {
            'valid': True,
            'warnings': [],
            'adjustments': {}
        }

        symbol = signal['symbol']

        # Check if already in position
        if any(pos['symbol'] == symbol for pos in current_positions):
            validation['valid'] = False
            validation['warnings'].append("Already in position")
            return validation

        # Check correlation with existing positions
        correlated_positions = self._check_correlations(symbol, current_positions)
        if len(correlated_positions) > 1:
            validation['warnings'].append(f"Correlated with: {correlated_positions}")

        # Check sector concentration
        sector_exposure = self._calculate_sector_exposure(symbol, current_positions)
        if sector_exposure > 0.4:
            validation['warnings'].append(f"High sector concentration: {sector_exposure:.1%}")

        # Validate entry/stop/target
        if signal.get('stop_loss') and signal.get('entry_price'):
            stop_distance = abs(signal['stop_loss'] - signal['entry_price']) / signal['entry_price']
            if stop_distance > 0.05:
                validation['warnings'].append(f"Wide stop loss: {stop_distance:.1%}")

        # Check if signal is stale
        if 'timestamp' in signal:
            age = (datetime.now() - signal['timestamp']).total_seconds() / 60
            if age > 30:
                validation['warnings'].append(f"Stale signal: {age:.0f} minutes old")
                validation['valid'] = False

        return validation

    def _check_correlations(self, symbol: str, positions: List[Dict]) -> List[str]:
        """Check for correlated positions"""
        # Simple correlation check - could be enhanced with actual correlation data
        correlated = []

        # Define correlation groups
        correlation_groups = [
            ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],  # Big tech
            ['JPM', 'BAC', 'WFC', 'GS', 'MS'],  # Banks
            ['XOM', 'CVX', 'COP', 'SLB'],  # Energy
        ]

        for group in correlation_groups:
            if symbol in group:
                for pos in positions:
                    if pos['symbol'] in group:
                        correlated.append(pos['symbol'])

        return correlated

    def _calculate_sector_exposure(self, symbol: str, positions: List[Dict]) -> float:
        """Calculate sector exposure"""
        # This would use actual sector mapping
        # For now, return a dummy value
        return 0.2

    def get_signal_performance(self, lookback_days: int = 30) -> Dict:
        """Analyze historical signal performance"""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'performance_metrics': {}
            }

        # Filter recent signals
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_signals = [s for s in self.signal_history
                          if s.get('timestamp', datetime.min) > cutoff]

        if not recent_signals:
            return {
                'total_signals': 0,
                'performance_metrics': {}
            }

        # Analyze by confidence level
        confidence_buckets = {
            'high': [s for s in recent_signals if s['confidence'] > 0.8],
            'medium': [s for s in recent_signals if 0.65 <= s['confidence'] <= 0.8],
            'low': [s for s in recent_signals if s['confidence'] < 0.65]
        }

        # Analyze by signal source
        source_performance = {
            'ml_dominant': [],
            'technical_dominant': [],
            'news_dominant': [],
            'consensus': []
        }

        for signal in recent_signals:
            # Determine dominant source
            ml_conf = signal.get('ml_confidence', 0)
            tech_conf = signal.get('technical_confidence', 0)
            news_conf = signal.get('news_confidence', 0)

            max_conf = max(ml_conf, tech_conf, news_conf)

            if ml_conf == max_conf and ml_conf > 0.6:
                source_performance['ml_dominant'].append(signal)
            elif tech_conf == max_conf and tech_conf > 0.6:
                source_performance['technical_dominant'].append(signal)
            elif news_conf == max_conf and news_conf > 0.6:
                source_performance['news_dominant'].append(signal)

            # Check for consensus
            if all(conf > 0.6 for conf in [ml_conf, tech_conf, news_conf]):
                source_performance['consensus'].append(signal)

        # Calculate metrics
        metrics = {
            'total_signals': len(recent_signals),
            'avg_confidence': np.mean([s['confidence'] for s in recent_signals]),
            'confidence_distribution': {
                k: len(v) for k, v in confidence_buckets.items()
            },
            'source_distribution': {
                k: len(v) for k, v in source_performance.items()
            },
            'top_setups': self._get_top_setups(recent_signals),
            'regime_distribution': self._get_regime_distribution(recent_signals)
        }

        return metrics

    def _get_top_setups(self, signals: List[Dict]) -> List[Dict]:
        """Get most common setup types"""
        setup_counts = defaultdict(int)

        for signal in signals:
            setup = signal.get('setup_type', 'unknown')
            setup_counts[setup] += 1

        top_setups = sorted(
            setup_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return [{'setup': setup, 'count': count} for setup, count in top_setups]

    def _get_regime_distribution(self, signals: List[Dict]) -> Dict[str, int]:
        """Get distribution of signals by market regime"""
        regime_counts = defaultdict(int)

        for signal in signals:
            regime = signal.get('market_regime', 'unknown')
            regime_counts[regime] += 1

        return dict(regime_counts)

    def generate_signal_report(self) -> str:
        """Generate comprehensive signal report"""
        performance = self.get_signal_performance()

        report = ["=== Signal Generation Report ==="]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if performance['total_signals'] == 0:
            report.append("No signals generated in the last 30 days")
            return "\n".join(report)

        report.append(f"Total Signals (30d): {performance['total_signals']}")
        report.append(f"Average Confidence: {performance['avg_confidence']:.1%}\n")

        report.append("Confidence Distribution:")
        for level, count in performance['confidence_distribution'].items():
            report.append(f"  {level.capitalize()}: {count}")

        report.append("\nSignal Sources:")
        for source, count in performance['source_distribution'].items():
            report.append(f"  {source.replace('_', ' ').title()}: {count}")

        report.append("\nTop Setup Types:")
        for setup_info in performance['top_setups']:
            report.append(f"  {setup_info['setup']}: {setup_info['count']}")

        report.append("\nMarket Regime Distribution:")
        for regime, count in performance['regime_distribution'].items():
            report.append(f"  {regime.replace('_', ' ').title()}: {count}")

        return "\n".join(report)