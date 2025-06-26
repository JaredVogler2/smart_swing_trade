# risk/stop_loss.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from config.settings import Config

logger = logging.getLogger(__name__)


class StopLossManager:
    """
    Advanced stop loss management system
    Handles initial stops, trailing stops, and time-based exits
    """

    def __init__(self):
        self.stop_types = ['fixed', 'trailing', 'volatility', 'time', 'breakeven']
        self.default_stop_pct = Config.STOP_LOSS_PCT
        self.profit_target_pct = Config.PROFIT_TARGET_PCT
        self.trailing_activation_pct = Config.TRAILING_STOP_ACTIVATION_PCT

    def calculate_initial_stops(self, entry_price: float,
                                volatility: float,
                                support_level: float = None,
                                timeframe: str = 'swing') -> Dict:
        """
        Calculate initial stop loss and take profit levels

        Args:
            entry_price: Entry price for the position
            volatility: Stock volatility (ATR or standard deviation)
            support_level: Technical support level if available
            timeframe: Trading timeframe (swing, day, etc.)

        Returns:
            Dictionary with stop loss and target levels
        """
        # Base stop loss
        stop_loss = entry_price * (1 - self.default_stop_pct)

        # Volatility-adjusted stop
        volatility_stop = entry_price - (2 * volatility * entry_price)

        # Use wider of the two
        stop_loss = min(stop_loss, volatility_stop)

        # If support level is provided, consider it
        if support_level and support_level > stop_loss * 0.95:
            # Place stop just below support
            stop_loss = support_level * 0.99

        # Calculate targets
        risk = entry_price - stop_loss

        # Primary target (2:1 risk/reward)
        target_1 = entry_price + (risk * 2)

        # Stretch target (3:1 risk/reward)
        target_2 = entry_price + (risk * 3)

        # Breakeven level (move stop here when target_1 is approached)
        breakeven_level = entry_price * 1.002  # Entry + 0.2% for costs

        return {
            'stop_loss': round(stop_loss, 2),
            'stop_type': 'volatility_adjusted',
            'target_1': round(target_1, 2),
            'target_2': round(target_2, 2),
            'breakeven_level': round(breakeven_level, 2),
            'risk_amount': round(risk, 2),
            'risk_pct': (risk / entry_price) * 100,
            'reward_risk_ratio': 2.0
        }

    def update_trailing_stop(self, position: Dict, current_price: float,
                             high_since_entry: float = None) -> Dict:
        """
        Update trailing stop for a position

        Args:
            position: Current position dictionary
            current_price: Current market price
            high_since_entry: Highest price since entry

        Returns:
            Updated stop information
        """
        entry_price = position['entry_price']
        current_stop = position.get('stop_loss_price', entry_price * 0.97)

        # Calculate profit
        profit_pct = (current_price - entry_price) / entry_price

        # Use high since entry if provided
        if high_since_entry:
            peak_profit_pct = (high_since_entry - entry_price) / entry_price
        else:
            peak_profit_pct = profit_pct

        new_stop = current_stop
        stop_type = position.get('stop_type', 'fixed')

        # Breakeven stop
        if profit_pct >= 0.01 and current_stop < entry_price:
            new_stop = position['breakeven_level']
            stop_type = 'breakeven'
            logger.info(f"Moving {position['symbol']} stop to breakeven")

        # Trailing stop activation
        elif profit_pct >= self.trailing_activation_pct:
            # Calculate trailing distance based on profit level
            trail_distance = self._calculate_trail_distance(profit_pct)

            # New trailing stop
            if high_since_entry:
                trailing_stop = high_since_entry * (1 - trail_distance)
            else:
                trailing_stop = current_price * (1 - trail_distance)

            # Only move stop up, never down
            if trailing_stop > new_stop:
                new_stop = trailing_stop
                stop_type = 'trailing'

        # Profit target based stops
        if 'target_1' in position:
            # If we hit first target, tighten stop
            if current_price >= position['target_1'] * 0.98:
                profit_stop = entry_price + (0.5 * (position['target_1'] - entry_price))
                if profit_stop > new_stop:
                    new_stop = profit_stop
                    stop_type = 'profit_protection'

        return {
            'stop_loss_price': round(new_stop, 2),
            'stop_type': stop_type,
            'stop_moved': new_stop != current_stop,
            'profit_pct': profit_pct * 100,
            'distance_to_stop': (current_price - new_stop) / current_price * 100
        }

    def _calculate_trail_distance(self, profit_pct: float) -> float:
        """Calculate trailing stop distance based on profit level"""
        if profit_pct < 0.02:
            return 0.03  # 3% trail
        elif profit_pct < 0.04:
            return 0.02  # 2% trail
        elif profit_pct < 0.06:
            return 0.015  # 1.5% trail
        else:
            return 0.01  # 1% trail for high profits

    def check_stop_conditions(self, position: Dict, current_price: float,
                              current_time: datetime = None) -> Dict:
        """
        Check if any stop conditions are met

        Returns:
            Dictionary with stop status and exit details
        """
        if current_time is None:
            current_time = datetime.now()

        stop_hit = False
        exit_reason = None
        exit_details = {}

        # Price-based stops
        stop_price = position.get('stop_loss_price', 0)
        if current_price <= stop_price:
            stop_hit = True
            exit_reason = 'stop_loss'
            exit_details['stop_price'] = stop_price
            exit_details['slippage'] = (stop_price - current_price) / stop_price

        # Time-based stop
        entry_time = position.get('entry_time', current_time)
        days_held = (current_time - entry_time).days

        if days_held >= 5:  # 5-day time stop
            profit_pct = (current_price - position['entry_price']) / position['entry_price']

            if profit_pct < 0.005:  # Less than 0.5% profit after 5 days
                stop_hit = True
                exit_reason = 'time_stop'
                exit_details['days_held'] = days_held
                exit_details['profit_pct'] = profit_pct * 100

        # Profit target hit
        if 'target_1' in position and current_price >= position['target_1']:
            stop_hit = True
            exit_reason = 'target_reached'
            exit_details['target'] = position['target_1']
            exit_details['profit_pct'] = ((current_price - position['entry_price']) /
                                          position['entry_price'] * 100)

        return {
            'stop_hit': stop_hit,
            'exit_reason': exit_reason,
            'exit_price': current_price,
            'exit_details': exit_details
        }

    def calculate_position_risk_metrics(self, position: Dict,
                                        current_price: float) -> Dict:
        """Calculate current risk metrics for a position"""
        entry_price = position['entry_price']
        stop_price = position.get('stop_loss_price', entry_price * 0.97)
        shares = position['shares']

        # Current risk
        price_risk = current_price - stop_price
        dollar_risk = price_risk * shares
        risk_pct = price_risk / current_price

        # Profit/Loss
        unrealized_pnl = (current_price - entry_price) * shares
        unrealized_pnl_pct = (current_price - entry_price) / entry_price

        # Risk/Reward
        if 'target_1' in position:
            potential_profit = (position['target_1'] - current_price) * shares
            current_rr_ratio = potential_profit / dollar_risk if dollar_risk > 0 else 0
        else:
            potential_profit = 0
            current_rr_ratio = 0

        # Maximum favorable/adverse excursion
        high_since_entry = position.get('high_since_entry', current_price)
        low_since_entry = position.get('low_since_entry', current_price)

        mfe = (high_since_entry - entry_price) / entry_price  # Max favorable excursion
        mae = (entry_price - low_since_entry) / entry_price  # Max adverse excursion

        return {
            'current_risk': dollar_risk,
            'risk_pct': risk_pct * 100,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct * 100,
            'distance_to_stop': ((current_price - stop_price) / current_price) * 100,
            'current_rr_ratio': current_rr_ratio,
            'mfe': mfe * 100,
            'mae': mae * 100,
            'stop_price': stop_price,
            'risk_adjusted_return': unrealized_pnl_pct / risk_pct if risk_pct > 0 else 0
        }

    def optimize_stop_placement(self, price_data: pd.DataFrame,
                                entry_price: float,
                                lookback: int = 20) -> Dict:
        """
        Optimize stop placement using historical price action

        Args:
            price_data: Historical price data
            entry_price: Intended entry price
            lookback: Days to analyze

        Returns:
            Optimized stop levels
        """
        recent_data = price_data.tail(lookback)

        # Calculate key levels
        support_levels = self._find_support_levels(recent_data)
        volatility = recent_data['close'].pct_change().std()
        atr = self._calculate_atr(recent_data)

        # Find optimal stop below support
        optimal_stop = entry_price * (1 - self.default_stop_pct)

        # Check each support level
        for support in support_levels:
            if support < entry_price * 0.97 and support > entry_price * 0.90:
                # Good support level for stop
                optimal_stop = support * 0.995  # Just below support
                break

        # Volatility adjustment
        vol_stop = entry_price - (2.5 * atr)

        # Use the more conservative stop
        final_stop = min(optimal_stop, vol_stop)

        # Calculate win rate at this stop level
        historical_win_rate = self._backtest_stop_level(
            recent_data, final_stop, entry_price
        )

        return {
            'optimal_stop': round(final_stop, 2),
            'stop_distance': (entry_price - final_stop) / entry_price * 100,
            'support_based': optimal_stop == final_stop,
            'historical_win_rate': historical_win_rate,
            'volatility': volatility,
            'atr': atr
        }

    def _find_support_levels(self, price_data: pd.DataFrame) -> List[float]:
        """Find potential support levels in price data"""
        lows = price_data['low'].values

        # Find local minima
        support_levels = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                    lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
                support_levels.append(lows[i])

        # Also add round numbers near current price
        current_price = price_data['close'].iloc[-1]
        round_levels = [
            round(current_price * 0.95, 0),  # 5% below
            round(current_price * 0.93, 0),  # 7% below
        ]

        support_levels.extend(round_levels)

        # Sort and remove duplicates
        support_levels = sorted(list(set(support_levels)), reverse=True)

        return support_levels[:5]  # Return top 5 levels

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )

        if len(tr) >= period:
            atr = np.mean(tr[-period:])
        else:
            atr = np.mean(tr)

        return atr

    def _backtest_stop_level(self, price_data: pd.DataFrame,
                             stop_price: float, entry_price: float) -> float:
        """Backtest a stop level to see historical win rate"""
        # Simplified backtest
        wins = 0
        losses = 0

        # Simulate trades at this price level
        for i in range(len(price_data) - 5):
            # Assume entry at close
            entry = price_data['close'].iloc[i]

            # Check next 5 days
            future_prices = price_data.iloc[i + 1:i + 6]

            # Check if stop hit
            min_price = future_prices['low'].min()
            max_price = future_prices['high'].max()

            stop_distance = (entry - stop_price) / entry

            if min_price <= entry * (1 - stop_distance):
                losses += 1
            elif max_price >= entry * 1.03:  # 3% profit target
                wins += 1

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.5

        return win_rate

    def create_stop_order(self, position: Dict, broker_api) -> Dict:
        """
        Create stop loss order with broker

        Args:
            position: Position dictionary
            broker_api: Broker API instance

        Returns:
            Order details
        """
        try:
            # Determine order type
            if position.get('stop_type') == 'trailing':
                # Create trailing stop order
                order = broker_api.submit_order(
                    symbol=position['symbol'],
                    qty=position['shares'],
                    side='sell',
                    type='trailing_stop',
                    trail_percent=position.get('trail_percent', 2.0),
                    time_in_force='gtc'
                )
            else:
                # Create regular stop order
                order = broker_api.submit_order(
                    symbol=position['symbol'],
                    qty=position['shares'],
                    side='sell',
                    type='stop',
                    stop_price=position['stop_loss_price'],
                    time_in_force='gtc'
                )

            return {
                'success': True,
                'order_id': order.id,
                'order_type': order.order_type,
                'stop_price': position['stop_loss_price']
            }

        except Exception as e:
            logger.error(f"Error creating stop order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def update_position_stops_batch(self, positions: List[Dict],
                                    market_data: Dict[str, Dict]) -> List[Dict]:
        """Update stops for multiple positions efficiently"""
        updated_positions = []

        for position in positions:
            symbol = position['symbol']
            if symbol in market_data:
                current_price = market_data[symbol].get('price', position['entry_price'])
                high_since_entry = market_data[symbol].get('high_since_entry', current_price)

                # Update trailing stop
                stop_update = self.update_trailing_stop(
                    position, current_price, high_since_entry
                )

                # Update position with new stop info
                position.update(stop_update)

                # Check if stop conditions met
                stop_check = self.check_stop_conditions(position, current_price)
                if stop_check['stop_hit']:
                    position['exit_signal'] = True
                    position['exit_reason'] = stop_check['exit_reason']
                    position['exit_price'] = stop_check['exit_price']

                updated_positions.append(position)

        return updated_positions

    def generate_stop_report(self, positions: List[Dict]) -> str:
        """Generate stop loss report for all positions"""
        report_lines = ["=== Stop Loss Report ===\n"]

        total_risk = 0

        for pos in positions:
            risk_metrics = self.calculate_position_risk_metrics(
                pos, pos.get('current_price', pos['entry_price'])
            )

            total_risk += risk_metrics['current_risk']

            report_lines.append(f"\n{pos['symbol']}:")
            report_lines.append(f"  Entry: ${pos['entry_price']:.2f}")
            report_lines.append(f"  Stop: ${risk_metrics['stop_price']:.2f} ({risk_metrics['risk_pct']:.1f}%)")
            report_lines.append(
                f"  P&L: ${risk_metrics['unrealized_pnl']:.2f} ({risk_metrics['unrealized_pnl_pct']:.1f}%)")
            report_lines.append(f"  Distance to Stop: {risk_metrics['distance_to_stop']:.1f}%")
            report_lines.append(f"  Stop Type: {pos.get('stop_type', 'fixed')}")

        report_lines.append(f"\n\nTotal Portfolio Risk: ${total_risk:.2f}")

        return "\n".join(report_lines)