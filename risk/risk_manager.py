# risk/risk_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from config.settings import Config
from config.watchlist import CORRELATION_PAIRS, get_sector

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management for trading system
    Protects capital and enforces risk limits
    """

    def __init__(self, account_value: float = None):
        self.account_value = account_value or Config.ACCOUNT_SIZE
        self.max_drawdown = Config.MAX_DRAWDOWN
        self.max_daily_loss = Config.MAX_DAILY_LOSS

        # Risk tracking
        self.peak_value = account_value
        self.daily_pnl = 0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.risk_events = []

        # Position tracking
        self.open_positions = {}
        self.closed_trades_today = []

        # Risk limits
        self.limits = {
            'max_positions': Config.MAX_POSITIONS,
            'max_position_size': Config.MAX_POSITION_SIZE,
            'max_sector_exposure': 0.4,  # 40% in one sector
            'max_correlated_positions': 2,  # Max 2 highly correlated positions
            'max_daily_trades': 10,  # Prevent overtrading
            'max_portfolio_heat': 0.06,  # 6% total portfolio risk
        }

    def check_trade_approval(self, trade_signal: Dict,
                             current_positions: List[Dict]) -> Dict:
        """
        Check if a trade should be approved based on risk rules

        Returns:
            Dictionary with approval status and reasons
        """
        approval = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'risk_score': 0
        }

        # Check 1: Daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            approval['approved'] = False
            approval['reasons'].append(f"Daily loss limit reached: ${self.daily_pnl:.2f}")

        # Check 2: Maximum positions
        if len(current_positions) >= self.limits['max_positions']:
            approval['approved'] = False
            approval['reasons'].append(f"Maximum positions ({self.limits['max_positions']}) reached")

        # Check 3: Position size limits
        position_value = trade_signal.get('position_value', 0)
        if position_value > self.limits['max_position_size']:
            approval['approved'] = False
            approval['reasons'].append(f"Position too large: ${position_value:.2f}")

        # Check 4: Sector concentration
        sector_check = self._check_sector_concentration(trade_signal, current_positions)
        if not sector_check['passed']:
            approval['warnings'].append(sector_check['message'])
            approval['risk_score'] += 20

        # Check 5: Correlation risk
        correlation_check = self._check_correlation_risk(trade_signal, current_positions)
        if not correlation_check['passed']:
            approval['warnings'].append(correlation_check['message'])
            approval['risk_score'] += 15

        # Check 6: Portfolio heat
        heat_check = self._check_portfolio_heat(trade_signal, current_positions)
        if not heat_check['passed']:
            approval['approved'] = False
            approval['reasons'].append(heat_check['message'])

        # Check 7: Consecutive losses
        if self.consecutive_losses >= 3:
            approval['warnings'].append(f"Consecutive losses: {self.consecutive_losses}")
            approval['risk_score'] += 25

        # Check 8: Daily trade limit
        if self.daily_trades >= self.limits['max_daily_trades']:
            approval['approved'] = False
            approval['reasons'].append("Daily trade limit reached")

        # Check 9: Drawdown
        current_drawdown = (self.peak_value - self.account_value) / self.peak_value
        if current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
            approval['warnings'].append(f"Approaching max drawdown: {current_drawdown:.1%}")
            approval['risk_score'] += 30

        # Final risk score assessment
        if approval['risk_score'] > 50 and approval['approved']:
            approval['approved'] = False
            approval['reasons'].append(f"Risk score too high: {approval['risk_score']}")

        return approval

    def _check_sector_concentration(self, trade_signal: Dict,
                                    current_positions: List[Dict]) -> Dict:
        """Check sector concentration limits"""
        symbol = trade_signal['symbol']
        sector = get_sector(symbol)

        if sector == 'Unknown':
            return {'passed': True, 'message': ''}

        # Calculate current sector exposure
        sector_exposure = defaultdict(float)
        for pos in current_positions:
            pos_sector = get_sector(pos['symbol'])
            sector_exposure[pos_sector] += pos['position_value']

        # Add new position
        sector_exposure[sector] += trade_signal['position_value']

        # Check limit
        total_exposure = sum(pos['position_value'] for pos in current_positions) + \
                         trade_signal['position_value']

        sector_pct = sector_exposure[sector] / total_exposure if total_exposure > 0 else 0

        if sector_pct > self.limits['max_sector_exposure']:
            return {
                'passed': False,
                'message': f"Sector concentration too high: {sector} = {sector_pct:.1%}"
            }

        return {'passed': True, 'message': ''}

    def _check_correlation_risk(self, trade_signal: Dict,
                                current_positions: List[Dict]) -> Dict:
        """Check correlation with existing positions"""
        symbol = trade_signal['symbol']
        correlated_count = 0

        for pos in current_positions:
            # Check predefined correlation pairs
            for pair in CORRELATION_PAIRS:
                if (symbol in pair and pos['symbol'] in pair):
                    correlated_count += 1
                    break

        if correlated_count >= self.limits['max_correlated_positions']:
            return {
                'passed': False,
                'message': f"Too many correlated positions: {correlated_count}"
            }

        return {'passed': True, 'message': ''}

    def _check_portfolio_heat(self, trade_signal: Dict,
                              current_positions: List[Dict]) -> Dict:
        """Check total portfolio risk (heat)"""
        total_risk = 0

        # Calculate risk for existing positions
        for pos in current_positions:
            position_risk = self._calculate_position_risk(pos)
            total_risk += position_risk

        # Add new position risk
        new_risk = trade_signal.get('dollar_risk', 0) / self.account_value
        total_risk += new_risk

        if total_risk > self.limits['max_portfolio_heat']:
            return {
                'passed': False,
                'message': f"Portfolio heat too high: {total_risk:.1%}"
            }

        return {'passed': True, 'message': ''}

    def _calculate_position_risk(self, position: Dict) -> float:
        """Calculate risk for a single position as % of account"""
        entry_price = position.get('entry_price', position.get('current_price', 0))
        stop_loss = position.get('stop_loss_price', entry_price * 0.97)
        shares = position.get('shares', 0)

        dollar_risk = shares * (entry_price - stop_loss)
        return dollar_risk / self.account_value

    def update_position_stops(self, positions: List[Dict],
                              market_data: Dict[str, float]) -> List[Dict]:
        """Update stop losses for all positions"""
        updated_positions = []

        for position in positions:
            symbol = position['symbol']
            entry_price = position['entry_price']
            current_price = market_data.get(symbol, entry_price)

            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price

            # Update stop loss
            new_stop = self._calculate_trailing_stop(
                entry_price, current_price,
                position.get('stop_loss_price', entry_price * 0.97),
                profit_pct
            )

            position['stop_loss_price'] = new_stop
            position['current_price'] = current_price
            position['unrealized_pnl'] = (current_price - entry_price) * position['shares']
            position['unrealized_pnl_pct'] = profit_pct

            # Check if stop hit
            if current_price <= new_stop:
                position['stop_hit'] = True
                position['exit_reason'] = 'stop_loss'

            # Check time stop
            days_held = (datetime.now() - position['entry_time']).days
            if days_held >= 5 and profit_pct < 0.005:  # 5 days with < 0.5% profit
                position['stop_hit'] = True
                position['exit_reason'] = 'time_stop'

            updated_positions.append(position)

        return updated_positions

    def _calculate_trailing_stop(self, entry_price: float, current_price: float,
                                 current_stop: float, profit_pct: float) -> float:
        """Calculate trailing stop based on profit level"""
        if profit_pct <= 0:
            # Not profitable, keep original stop
            return current_stop

        elif profit_pct < 0.02:  # Less than 2% profit
            # Move stop to breakeven if we're up 1%
            if profit_pct > 0.01:
                return max(current_stop, entry_price * 1.001)  # Breakeven + commission
            return current_stop

        elif profit_pct < 0.04:  # 2-4% profit
            # Trail at 2% below current
            return max(current_stop, current_price * 0.98)

        elif profit_pct < 0.06:  # 4-6% profit
            # Trail at 1.5% below current
            return max(current_stop, current_price * 0.985)

        else:  # > 6% profit
            # Tight trail at 1% below current
            return max(current_stop, current_price * 0.99)

    def record_trade_result(self, trade: Dict):
        """Record completed trade for risk tracking"""
        self.closed_trades_today.append(trade)

        # Update daily P&L
        self.daily_pnl += trade.get('pnl', 0)
        self.daily_trades += 1

        # Update consecutive losses
        if trade.get('pnl', 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update peak value
        self.account_value += trade.get('pnl', 0)
        if self.account_value > self.peak_value:
            self.peak_value = self.account_value

        # Log risk event if significant loss
        if trade.get('pnl', 0) < -100:  # Loss > $100
            self.risk_events.append({
                'timestamp': datetime.now(),
                'type': 'large_loss',
                'symbol': trade['symbol'],
                'amount': trade['pnl'],
                'details': trade
            })

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        current_drawdown = (self.peak_value - self.account_value) / self.peak_value

        # Calculate risk statistics
        if self.closed_trades_today:
            trade_pnls = [t['pnl'] for t in self.closed_trades_today]
            avg_trade = np.mean(trade_pnls)

            wins = [pnl for pnl in trade_pnls if pnl > 0]
            losses = [pnl for pnl in trade_pnls if pnl < 0]

            win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0

            profit_factor = abs(sum(wins) / sum(losses)) if losses else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            avg_trade = 0

        return {
            'account_value': self.account_value,
            'peak_value': self.peak_value,
            'current_drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'risk_events_today': len([e for e in self.risk_events
                                      if e['timestamp'].date() == datetime.now().date()]),
            'portfolio_heat': self._calculate_current_heat()
        }

    def _calculate_current_heat(self) -> float:
        """Calculate current portfolio heat"""
        total_risk = 0
        for position in self.open_positions.values():
            total_risk += self._calculate_position_risk(position)
        return total_risk

    def should_stop_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted"""
        # Daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return True, f"Daily loss limit reached: ${self.daily_pnl:.2f}"

        # Consecutive losses
        if self.consecutive_losses >= 5:
            return True, f"Too many consecutive losses: {self.consecutive_losses}"

        # Drawdown approaching limit
        current_drawdown = (self.peak_value - self.account_value) / self.peak_value
        if current_drawdown >= self.max_drawdown:
            return True, f"Maximum drawdown reached: {current_drawdown:.1%}"

        # Account below minimum
        if self.account_value < 5000:  # Half of starting capital
            return True, f"Account below minimum: ${self.account_value:.2f}"

        return False, ""

    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.closed_trades_today = []
        logger.info("Daily risk metrics reset")

    def generate_risk_report(self) -> str:
        """Generate daily risk report"""
        metrics = self.get_risk_metrics()

        report = f"""
=== Daily Risk Report ===
Date: {datetime.now().strftime('%Y-%m-%d')}

Account Status:
- Current Value: ${metrics['account_value']:,.2f}
- Peak Value: ${metrics['peak_value']:,.2f}
- Drawdown: {metrics['current_drawdown']:.1%}

Daily Performance:
- P&L: ${metrics['daily_pnl']:,.2f}
- Trades: {metrics['daily_trades']}
- Win Rate: {metrics['win_rate']:.1%}
- Avg Trade: ${metrics['avg_trade']:,.2f}

Risk Metrics:
- Consecutive Losses: {metrics['consecutive_losses']}
- Portfolio Heat: {metrics['portfolio_heat']:.1%}
- Risk Events: {metrics['risk_events_today']}

Trading Status: {'ACTIVE' if not self.should_stop_trading()[0] else 'HALTED - ' + self.should_stop_trading()[1]}
"""
        return report
    