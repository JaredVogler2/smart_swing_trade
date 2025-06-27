# risk/position_sizer.py
# COMPLETE FILE - Replace your entire position_sizer.py with this

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats

from config.settings import Config

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Advanced position sizing for $10K account
    Uses Kelly Criterion with safety adjustments
    """

    def __init__(self, account_value: float = None):
        self.account_value = account_value or Config.ACCOUNT_SIZE
        self.min_position = Config.MIN_POSITION_SIZE
        self.max_position = Config.MAX_POSITION_SIZE_PCT * self.account_value  # Fixed: was MAX_POSITION_SIZE
        self.target_position = 0.05 * self.account_value  # Fixed: TARGET_POSITION_SIZE doesn't exist, using 5%
        self.max_positions = Config.MAX_POSITIONS
        self.kelly_fraction = Config.KELLY_FRACTION if hasattr(Config, 'KELLY_FRACTION') else 0.25  # Use 25% of Kelly

        # Track position history for adaptive sizing
        self.position_history = []
        self.performance_history = []

    def calculate_position_size(self, signal: Dict,
                                current_positions: List[Dict],
                                market_regime: str = 'ranging') -> Dict:
        """
        Calculate optimal position size for a trade signal

        Args:
            signal: Trading signal with confidence, expected_return, etc.
            current_positions: List of current positions
            market_regime: Current market regime

        Returns:
            Dictionary with position sizing details
        """
        # Extract signal components
        symbol = signal['symbol']
        confidence = signal.get('confidence', 0.5)
        expected_return = signal.get('expected_return', 0.06)
        volatility = signal.get('volatility', 0.02)
        current_price = signal.get('current_price', 100)

        # Get available capital
        available_capital = self._calculate_available_capital(current_positions)

        # Calculate base position size using Kelly
        kelly_size = self._calculate_kelly_position(
            confidence, expected_return, volatility
        )

        # Apply regime adjustments
        regime_adjusted = self._apply_regime_adjustment(
            kelly_size, market_regime
        )

        # Apply portfolio constraints
        constrained_size = self._apply_constraints(
            regime_adjusted, available_capital, len(current_positions)
        )

        # Apply volatility adjustment
        final_size = self._apply_volatility_adjustment(
            constrained_size, volatility
        )

        # Convert to shares
        shares = int(final_size / current_price)

        # Ensure minimum position
        if shares * current_price < self.min_position:
            if available_capital >= self.min_position:
                shares = int(self.min_position / current_price)
            else:
                shares = 0  # Don't trade if can't meet minimum

        # Final position value
        position_value = shares * current_price

        # Calculate risk metrics
        stop_loss_price = current_price * (1 - Config.DEFAULT_STOP_LOSS)  # Fixed: was STOP_LOSS_PCT
        dollar_risk = shares * (current_price - stop_loss_price)
        portfolio_risk_pct = dollar_risk / self.account_value

        return {
            'symbol': symbol,
            'shares': shares,
            'position_value': position_value,
            'position_pct': position_value / self.account_value,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': current_price * (1 + Config.DEFAULT_TAKE_PROFIT),  # Fixed: was PROFIT_TARGET_PCT
            'dollar_risk': dollar_risk,
            'portfolio_risk_pct': portfolio_risk_pct,
            'kelly_fraction_used': self.kelly_fraction,
            'confidence': confidence,
            'sizing_method': 'kelly_constrained'
        }

    def _calculate_available_capital(self, current_positions: List[Dict]) -> float:
        """Calculate available capital for new positions"""
        # Sum current position values
        current_exposure = sum(pos.get('position_value', 0) for pos in current_positions)

        # Reserve cash (default to 10% if not in config)
        cash_reserve_pct = getattr(Config, 'CASH_RESERVE_PCT', 0.1)
        cash_reserve = self.account_value * cash_reserve_pct

        # Available capital
        available = self.account_value - current_exposure - cash_reserve

        return max(0, available)

    def _calculate_kelly_position(self, win_prob: float,
                                  avg_win: float, avg_loss: float = None) -> float:
        """
        Calculate position size using Kelly Criterion

        Kelly formula: f = (p*b - q)/b
        where:
            f = fraction of capital to bet
            p = probability of winning
            b = ratio of win amount to loss amount
            q = probability of losing (1-p)
        """
        if avg_loss is None:
            avg_loss = Config.DEFAULT_STOP_LOSS

        # Ensure valid inputs
        if win_prob <= 0 or win_prob >= 1 or avg_loss <= 0:
            return self.target_position

        # Calculate Kelly fraction
        q = 1 - win_prob
        b = avg_win / avg_loss

        kelly_full = (win_prob * b - q) / b

        # Apply safety factor (use 25% of Kelly)
        kelly_fraction = max(0, kelly_full * self.kelly_fraction)

        # Convert to dollar amount
        position_size = self.account_value * kelly_fraction

        return position_size

    def _apply_regime_adjustment(self, base_size: float, regime: str) -> float:
        """Adjust position size based on market regime"""
        regime_multipliers = {
            'bull_quiet': 1.2,  # Increase size in calm bull markets
            'bull_volatile': 0.8,  # Reduce in volatile bulls
            'bear_quiet': 0.7,  # Reduce in bear markets
            'bear_volatile': 0.5,  # Significantly reduce in volatile bears
            'ranging': 1.0  # Normal size in ranging markets
        }

        multiplier = regime_multipliers.get(regime, 1.0)
        return base_size * multiplier

    def _apply_constraints(self, size: float, available_capital: float,
                           num_positions: int) -> float:
        """Apply portfolio constraints"""
        # Don't exceed available capital
        size = min(size, available_capital)

        # Apply maximum position size
        size = min(size, self.max_position)

        # Apply minimum position size
        if size < self.min_position:
            # Either meet minimum or don't trade
            if available_capital >= self.min_position:
                size = self.min_position
            else:
                size = 0

        # Adjust for number of positions
        if num_positions >= self.max_positions - 1:
            # Last allowed position - be more conservative
            size = min(size, self.target_position * 0.8)

        return size

    def _apply_volatility_adjustment(self, size: float, volatility: float) -> float:
        """Adjust position size based on volatility"""
        # Normal volatility assumed to be 2%
        normal_volatility = 0.02

        if volatility > normal_volatility * 1.5:
            # High volatility - reduce size
            adjustment = normal_volatility / volatility
            size = size * adjustment
        elif volatility < normal_volatility * 0.5:
            # Very low volatility - slight increase
            size = size * 1.1

        return size

    def calculate_portfolio_heat(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio heat (total risk)"""
        total_risk = 0
        position_risks = []

        for pos in positions:
            # Calculate risk for each position
            entry_price = pos.get('entry_price', pos.get('current_price', 0))
            stop_loss = pos.get('stop_loss_price', entry_price * 0.97)
            shares = pos.get('shares', 0)

            position_risk = shares * (entry_price - stop_loss)
            total_risk += position_risk

            position_risks.append({
                'symbol': pos['symbol'],
                'risk': position_risk,
                'risk_pct': position_risk / self.account_value
            })

        return {
            'total_risk': total_risk,
            'total_risk_pct': total_risk / self.account_value,
            'positions_at_risk': len(positions),
            'position_risks': position_risks,
            'heat_level': 'high' if total_risk / self.account_value > 0.06 else
            'medium' if total_risk / self.account_value > 0.03 else 'low'
        }

    def update_performance(self, trade_result: Dict):
        """Update performance history for adaptive sizing"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'symbol': trade_result['symbol'],
            'return': trade_result['return'],
            'win': trade_result['return'] > 0,
            'confidence': trade_result.get('entry_confidence', 0.5)
        })

        # Keep only recent history (last 100 trades)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_adaptive_kelly_fraction(self) -> float:
        """Calculate adaptive Kelly fraction based on recent performance"""
        if len(self.performance_history) < 20:
            return self.kelly_fraction

        # Calculate recent win rate
        recent_trades = self.performance_history[-50:]
        wins = [t for t in recent_trades if t['win']]
        win_rate = len(wins) / len(recent_trades)

        # Calculate average win/loss
        winning_returns = [t['return'] for t in wins]
        losing_returns = [t['return'] for t in recent_trades if not t['win']]

        if winning_returns and losing_returns:
            avg_win = np.mean(winning_returns)
            avg_loss = abs(np.mean(losing_returns))

            # Recalculate Kelly
            kelly_full = (win_rate * (avg_win / avg_loss) - (1 - win_rate)) / (avg_win / avg_loss)

            # Adjust safety factor based on consistency
            returns = [t['return'] for t in recent_trades]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)

            if sharpe > 1.5:
                # Good performance - can use up to 30% Kelly
                self.kelly_fraction = min(0.30, max(0.15, kelly_full * 0.30))
            elif sharpe > 0.5:
                # Decent performance - use 25% Kelly
                self.kelly_fraction = min(0.25, max(0.10, kelly_full * 0.25))
            else:
                # Poor performance - reduce to 15% Kelly
                self.kelly_fraction = min(0.15, max(0.05, kelly_full * 0.15))

        return self.kelly_fraction

    def calculate_correlation_adjustment(self, new_position: Dict,
                                         current_positions: List[Dict],
                                         correlation_matrix: pd.DataFrame) -> float:
        """Adjust position size based on correlation with existing positions"""
        if not current_positions or correlation_matrix.empty:
            return 1.0

        new_symbol = new_position['symbol']
        correlation_sum = 0

        for pos in current_positions:
            symbol = pos['symbol']
            if symbol in correlation_matrix.columns and new_symbol in correlation_matrix.index:
                correlation = correlation_matrix.loc[new_symbol, symbol]
                correlation_sum += abs(correlation) * (pos['position_value'] / self.account_value)

        # Reduce size if highly correlated
        if correlation_sum > 0.5:
            adjustment = 1 - (correlation_sum - 0.5) * 0.5
            return max(0.5, adjustment)  # Don't reduce by more than 50%

        return 1.0

    def optimize_portfolio_allocation(self, signals: List[Dict],
                                      risk_budget: float = None) -> List[Dict]:
        """
        Optimize allocation across multiple signals

        Args:
            signals: List of trading signals
            risk_budget: Maximum portfolio risk (default from config)

        Returns:
            List of optimized position sizes
        """
        if risk_budget is None:
            risk_budget = Config.RISK_PER_TRADE * Config.MAX_POSITIONS

        # Sort signals by expected Sharpe ratio
        for signal in signals:
            expected_return = signal.get('expected_return', 0.06)
            volatility = signal.get('volatility', 0.02)
            signal['expected_sharpe'] = expected_return / volatility

        signals_sorted = sorted(signals, key=lambda x: x['expected_sharpe'], reverse=True)

        # Allocate to best opportunities first
        allocations = []
        total_risk = 0
        cash_reserve_pct = getattr(Config, 'CASH_RESERVE_PCT', 0.1)
        available_capital = self.account_value * (1 - cash_reserve_pct)

        for signal in signals_sorted:
            if len(allocations) >= Config.MAX_POSITIONS:
                break

            position = self.calculate_position_size(signal, allocations)

            if position['shares'] > 0:
                # Check if within risk budget
                position_risk = position['portfolio_risk_pct']
                if total_risk + position_risk <= risk_budget:
                    allocations.append(position)
                    total_risk += position_risk
                    available_capital -= position['position_value']

        return allocations

    def rebalance_positions(self, positions: List[Dict], target_weights: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate trades needed to rebalance portfolio

        Args:
            positions: Current positions
            target_weights: Target weights by symbol

        Returns:
            Dictionary of symbols to share changes
        """
        rebalance_trades = {}

        # Current weights
        current_weights = {}
        total_value = sum(p.get('market_value', p.get('position_value', 0)) for p in positions)

        if total_value == 0:
            return rebalance_trades

        for position in positions:
            symbol = position['symbol']
            value = position.get('market_value', position.get('position_value', 0))
            weight = value / total_value
            current_weights[symbol] = weight

        # Calculate required trades
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight

            # Calculate share difference
            target_value = self.account_value * target_weight
            current_value = self.account_value * current_weight
            value_diff = target_value - current_value

            # Get current price (would need market data in practice)
            current_price = next((p.get('current_price', p.get('avg_price', 0))
                                  for p in positions if p['symbol'] == symbol), 0)

            if current_price > 0:
                share_diff = int(value_diff / current_price)
                if abs(share_diff) > 0:
                    rebalance_trades[symbol] = share_diff

        return rebalance_trades

    def update_account_size(self, new_size: float):
        """Update account size for position calculations"""
        old_size = self.account_value
        self.account_value = new_size

        # Update position limits based on new account size
        self.max_position = Config.MAX_POSITION_SIZE_PCT * self.account_value
        self.target_position = 0.05 * self.account_value  # 5% target

        logger.info(f"Account size updated: ${old_size:,.2f} -> ${new_size:,.2f}")

    def get_position_metrics(self) -> Dict:
        """Get current position sizing metrics"""
        return {
            'account_size': self.account_value,
            'max_position': self.max_position,
            'max_position_pct': self.max_position / self.account_value,
            'target_position': self.target_position,
            'target_position_pct': self.target_position / self.account_value,
            'min_position': self.min_position,
            'risk_per_trade': Config.RISK_PER_TRADE,
            'max_positions': self.max_positions,
            'kelly_fraction': self.kelly_fraction,
            'use_kelly': Config.USE_KELLY_CRITERION if hasattr(Config, 'USE_KELLY_CRITERION') else True
        }