# execution/broker_interface.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import alpaca_trade_api as tradeapi
import logging
from datetime import datetime, timedelta
import time
from decimal import Decimal

from config.settings import Config

logger = logging.getLogger(__name__)


class BrokerInterface:
    """
    Unified interface to Alpaca broker
    Handles all broker interactions with error handling and retry logic
    """

    def __init__(self, api_key: str = None, secret_key: str = None,
                 base_url: str = None):
        """Initialize broker interface"""
        self.api_key = api_key or Config.ALPACA_API_KEY
        self.secret_key = secret_key or Config.ALPACA_SECRET_KEY
        self.base_url = base_url or Config.ALPACA_BASE_URL

        # Initialize API connection
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )

        # Connection status
        self.is_connected = False
        self.last_connection_check = None

        # Verify connection
        self._verify_connection()

    def _verify_connection(self) -> bool:
        """Verify broker connection"""
        try:
            account = self.api.get_account()
            self.is_connected = True
            self.last_connection_check = datetime.now()
            logger.info(f"Broker connection verified. Account: {account.account_number}")
            return True
        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            self.is_connected = False
            return False

    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.api.get_account()

            return {
                'account_number': account.account_number,
                'status': account.status,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'long_market_value': float(account.long_market_value),
                'maintenance_margin': float(account.maintenance_margin),
                'pattern_day_trader': account.pattern_day_trader,
                'day_trade_count': int(account.daytrade_count),
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'sma': float(account.sma) if account.sma else None
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        try:
            positions = self.api.list_positions()

            position_list = []
            for pos in positions:
                position_dict = {
                    'symbol': pos.symbol,
                    'shares': int(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'lastday_price': float(pos.lastday_price),
                    'change_today': float(pos.change_today),
                    'asset_id': pos.asset_id,
                    'exchange': pos.exchange,
                    'asset_class': pos.asset_class,
                    'side': pos.side
                }
                position_list.append(position_dict)

            return position_list

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        try:
            pos = self.api.get_position(symbol)

            return {
                'symbol': pos.symbol,
                'shares': int(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price),
                'lastday_price': float(pos.lastday_price),
                'change_today': float(pos.change_today)
            }

        except Exception as e:
            if "position does not exist" not in str(e).lower():
                logger.error(f"Error getting position for {symbol}: {e}")
            return None

    def place_order(self, symbol: str, quantity: int, side: str,
                    order_type: str = 'market', limit_price: float = None,
                    stop_price: float = None, time_in_force: str = 'day',
                    extended_hours: bool = False) -> Optional[Dict]:
        """
        Place an order with the broker

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Allow extended hours trading

        Returns:
            Order details or None if failed
        """
        try:
            # Validate inputs
            if quantity <= 0:
                raise ValueError("Quantity must be positive")

            if side not in ['buy', 'sell']:
                raise ValueError("Side must be 'buy' or 'sell'")

            # Build order parameters
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }

            # Add price parameters
            if order_type in ['limit', 'stop_limit'] and limit_price:
                order_params['limit_price'] = round(limit_price, 2)

            if order_type in ['stop', 'stop_limit'] and stop_price:
                order_params['stop_price'] = round(stop_price, 2)

            # Extended hours only for limit orders
            if extended_hours and order_type == 'limit':
                order_params['extended_hours'] = True

            # Submit order
            order = self.api.submit_order(**order_params)

            # Return order details
            return {
                'order_id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side,
                'order_type': order.order_type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'status': order.status,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at
            }

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        try:
            cancelled = self.api.cancel_all_orders()
            count = len(cancelled)
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order details"""
        try:
            order = self.api.get_order(order_id)

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'side': order.side,
                'order_type': order.order_type,
                'status': order.status,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'expired_at': order.expired_at,
                'cancelled_at': order.canceled_at,
                'time_in_force': order.time_in_force
            }

        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None

    def get_orders(self, status: str = 'open', limit: int = 100,
                   symbols: List[str] = None) -> List[Dict]:
        """Get multiple orders"""
        try:
            # Build parameters
            params = {
                'status': status,
                'limit': limit,
                'direction': 'desc'
            }

            if symbols:
                params['symbols'] = ','.join(symbols)

            orders = self.api.list_orders(**params)

            order_list = []
            for order in orders:
                order_list.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'quantity': int(order.qty),
                    'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                    'side': order.side,
                    'order_type': order.order_type,
                    'status': order.status,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'created_at': order.created_at
                })

            return order_list

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            clock = self.api.get_clock()

            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'timestamp': clock.timestamp
            }

        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_open': False,
                'error': str(e)
            }

    def get_asset_info(self, symbol: str) -> Optional[Dict]:
        """Get asset information"""
        try:
            asset = self.api.get_asset(symbol)

            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'asset_class': asset.class_,
                'status': asset.status,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable,
                'id': asset.id
            }

        except Exception as e:
            logger.error(f"Error getting asset info for {symbol}: {e}")
            return None

    def is_tradable(self, symbol: str) -> bool:
        """Check if a symbol is tradable"""
        asset_info = self.get_asset_info(symbol)
        return asset_info.get('tradable', False) if asset_info else False

    def get_latest_trade(self, symbol: str) -> Optional[Dict]:
        """Get latest trade for a symbol"""
        try:
            trade = self.api.get_latest_trade(symbol)

            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': int(trade.size),
                'timestamp': trade.timestamp
            }

        except Exception as e:
            logger.error(f"Error getting latest trade for {symbol}: {e}")
            return None

    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for a symbol"""
        try:
            quote = self.api.get_latest_quote(symbol)

            return {
                'symbol': symbol,
                'bid_price': float(quote.bid_price),
                'bid_size': int(quote.bid_size),
                'ask_price': float(quote.ask_price),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp
            }

        except Exception as e:
            logger.error(f"Error getting latest quote for {symbol}: {e}")
            return None

    def check_pattern_day_trader_status(self) -> Dict:
        """Check PDT status and remaining day trades"""
        account = self.get_account_info()

        if not account:
            return {'error': 'Could not get account info'}

        return {
            'is_pattern_day_trader': account['pattern_day_trader'],
            'day_trade_count': account['day_trade_count'],
            'day_trades_remaining': max(0, 3 - account['day_trade_count']),
            'buying_power': account['buying_power'],
            'account_value': account['portfolio_value'],
            'can_day_trade': not account['pattern_day_trader'] or account['portfolio_value'] >= 25000
        }

    def wait_for_order_fill(self, order_id: str, timeout: int = 30) -> Optional[Dict]:
        """
        Wait for an order to fill

        Args:
            order_id: Order ID to monitor
            timeout: Maximum seconds to wait

        Returns:
            Filled order details or None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            order = self.get_order(order_id)

            if not order:
                return None

            if order['status'] == 'filled':
                return order
            elif order['status'] in ['cancelled', 'rejected', 'expired']:
                logger.warning(f"Order {order_id} {order['status']}")
                return None

            time.sleep(1)

        logger.warning(f"Order {order_id} fill timeout after {timeout} seconds")
        return None

    def calculate_buying_power_effect(self, symbol: str, quantity: int,
                                      side: str, price: float = None) -> Dict:
        """Calculate the effect of a trade on buying power"""
        try:
            # Get current price if not provided
            if not price:
                trade = self.get_latest_trade(symbol)
                if not trade:
                    return {'error': 'Could not get price'}
                price = trade['price']

            # Get asset info
            asset = self.get_asset_info(symbol)
            if not asset:
                return {'error': 'Could not get asset info'}

            # Calculate trade value
            trade_value = price * quantity

            # Get account info
            account = self.get_account_info()
            if not account:
                return {'error': 'Could not get account info'}

            # Calculate buying power effect
            if side == 'buy':
                # For margin accounts, typically 2:1 leverage
                if asset['marginable']:
                    buying_power_effect = trade_value * 0.5  # 50% margin requirement
                else:
                    buying_power_effect = trade_value  # 100% cash requirement

                new_buying_power = account['buying_power'] - buying_power_effect

            else:  # sell
                # Selling increases buying power
                if asset['marginable']:
                    buying_power_effect = trade_value * 0.5
                else:
                    buying_power_effect = trade_value

                new_buying_power = account['buying_power'] + buying_power_effect

            return {
                'current_buying_power': account['buying_power'],
                'trade_value': trade_value,
                'buying_power_effect': buying_power_effect,
                'new_buying_power': new_buying_power,
                'marginable': asset['marginable'],
                'sufficient_buying_power': new_buying_power >= 0
            }

        except Exception as e:
            logger.error(f"Error calculating buying power effect: {e}")
            return {'error': str(e)}

    def get_trade_history(self, days: int = 7) -> List[Dict]:
        """Get recent trade history"""
        try:
            # Get filled orders from the last N days
            since = (datetime.now() - timedelta(days=days)).isoformat()

            orders = self.api.list_orders(
                status='filled',
                since=since,
                direction='desc',
                limit=500
            )

            trades = []
            for order in orders:
                trades.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': int(order.filled_qty),
                    'price': float(order.filled_avg_price),
                    'value': float(order.filled_qty) * float(order.filled_avg_price),
                    'filled_at': order.filled_at,
                    'order_type': order.order_type
                })

            return trades

        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []