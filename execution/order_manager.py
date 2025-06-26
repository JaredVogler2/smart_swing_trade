# execution/order_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import time
from enum import Enum
import threading
from queue import Queue, Empty

from config.settings import Config

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderManager:
    """
    Manages order lifecycle, execution, and tracking
    Optimized for swing trading with Alpaca
    """

    def __init__(self, broker_api):
        self.broker = broker_api
        self.active_orders = {}
        self.order_history = []
        self.position_orders = {}  # Track orders by position

        # Order queue for batch processing
        self.order_queue = Queue()
        self.is_running = True

        # Start order processor thread
        self.processor_thread = threading.Thread(target=self._process_order_queue)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def create_entry_order(self, signal: Dict, position_size: Dict) -> Dict:
        """
        Create entry order based on signal and position sizing

        Args:
            signal: Trading signal
            position_size: Position sizing details

        Returns:
            Order result dictionary
        """
        symbol = signal['symbol']
        shares = position_size['shares']

        if shares == 0:
            return {
                'success': False,
                'reason': 'Position size is zero'
            }

        # Determine order type based on market conditions
        order_type = self._determine_order_type(signal)

        # Create order parameters
        order_params = {
            'symbol': symbol,
            'qty': shares,
            'side': 'buy',
            'type': order_type,
            'time_in_force': 'day',
            'extended_hours': False
        }

        # Add price parameters based on order type
        current_price = signal.get('current_price', 0)

        if order_type == 'limit':
            # Use limit order for better fill
            limit_price = self._calculate_limit_price(current_price, 'buy')
            order_params['limit_price'] = limit_price

        elif order_type == 'stop_limit':
            # Use stop limit for breakout entries
            stop_price = current_price * 1.001
            limit_price = current_price * 1.003
            order_params['stop_price'] = stop_price
            order_params['limit_price'] = limit_price

        # Submit order
        try:
            order = self.broker.submit_order(**order_params)

            # Track order
            order_info = {
                'order_id': order.id,
                'symbol': symbol,
                'side': 'buy',
                'quantity': shares,
                'order_type': order_type,
                'status': OrderStatus.SUBMITTED.value,
                'created_at': datetime.now(),
                'signal': signal,
                'position_size': position_size
            }

            self.active_orders[order.id] = order_info

            logger.info(f"Entry order submitted: {symbol} - {shares} shares @ "
                        f"{'market' if order_type == 'market' else f'${order_params.get('limit_price', 0):.2f}'}")

            return {
                'success': True,
                'order_id': order.id,
                'order_info': order_info
            }

        except Exception as e:
            logger.error(f"Error creating entry order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_exit_order(self, position: Dict, exit_signal: Dict) -> Dict:
        """
        Create exit order for a position

        Args:
            position: Current position details
            exit_signal: Exit signal details

        Returns:
            Order result dictionary
        """
        symbol = position['symbol']
        shares = position['shares']
        reason = exit_signal.get('reason', 'signal')

        # Cancel any existing orders for this position
        self.cancel_position_orders(symbol)

        # Determine exit order type
        if reason in ['stop_loss', 'trailing_stop']:
            order_type = 'market'  # Use market for stops
        else:
            order_type = 'limit'  # Use limit for targets

        # Create order parameters
        order_params = {
            'symbol': symbol,
            'qty': shares,
            'side': 'sell',
            'type': order_type,
            'time_in_force': 'day'
        }

        # Add price parameters
        current_price = exit_signal.get('current_price', position.get('current_price', 0))

        if order_type == 'limit':
            limit_price = self._calculate_limit_price(current_price, 'sell')
            order_params['limit_price'] = limit_price

        # Submit order
        try:
            order = self.broker.submit_order(**order_params)

            # Track order
            order_info = {
                'order_id': order.id,
                'symbol': symbol,
                'side': 'sell',
                'quantity': shares,
                'order_type': order_type,
                'status': OrderStatus.SUBMITTED.value,
                'created_at': datetime.now(),
                'exit_reason': reason,
                'position': position
            }

            self.active_orders[order.id] = order_info

            logger.info(f"Exit order submitted: {symbol} - {shares} shares - Reason: {reason}")

            return {
                'success': True,
                'order_id': order.id,
                'order_info': order_info
            }

        except Exception as e:
            logger.error(f"Error creating exit order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_stop_loss_order(self, position: Dict, stop_price: float) -> Dict:
        """Create or update stop loss order"""
        symbol = position['symbol']
        shares = position['shares']

        # Cancel existing stop orders
        self._cancel_stop_orders(symbol)

        # Create stop order
        try:
            order = self.broker.submit_order(
                symbol=symbol,
                qty=shares,
                side='sell',
                type='stop',
                stop_price=stop_price,
                time_in_force='gtc'  # Good till cancelled
            )

            order_info = {
                'order_id': order.id,
                'symbol': symbol,
                'side': 'sell',
                'quantity': shares,
                'order_type': 'stop',
                'stop_price': stop_price,
                'status': OrderStatus.SUBMITTED.value,
                'created_at': datetime.now(),
                'position': position
            }

            self.active_orders[order.id] = order_info

            # Track by position
            if symbol not in self.position_orders:
                self.position_orders[symbol] = []
            self.position_orders[symbol].append(order.id)

            logger.info(f"Stop loss order created: {symbol} @ ${stop_price:.2f}")

            return {
                'success': True,
                'order_id': order.id,
                'stop_price': stop_price
            }

        except Exception as e:
            logger.error(f"Error creating stop loss order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def update_order_status(self):
        """Update status of all active orders"""
        orders_to_remove = []

        for order_id, order_info in self.active_orders.items():
            try:
                # Get latest order status from broker
                broker_order = self.broker.get_order(order_id)

                # Update status
                old_status = order_info['status']
                new_status = broker_order.status

                if old_status != new_status:
                    order_info['status'] = new_status
                    order_info['updated_at'] = datetime.now()

                    # Handle status changes
                    if new_status == 'filled':
                        order_info['filled_at'] = broker_order.filled_at
                        order_info['filled_price'] = float(broker_order.filled_avg_price)
                        order_info['filled_qty'] = int(broker_order.filled_qty)

                        # Move to history
                        self.order_history.append(order_info)
                        orders_to_remove.append(order_id)

                        logger.info(f"Order filled: {order_info['symbol']} - "
                                    f"{order_info['quantity']} @ ${order_info['filled_price']:.2f}")

                    elif new_status in ['cancelled', 'rejected', 'expired']:
                        # Move to history
                        self.order_history.append(order_info)
                        orders_to_remove.append(order_id)

                        logger.warning(f"Order {new_status}: {order_info['symbol']}")

                    elif new_status == 'partially_filled':
                        order_info['filled_qty'] = int(broker_order.filled_qty)
                        order_info['filled_avg_price'] = float(broker_order.filled_avg_price)

            except Exception as e:
                logger.error(f"Error updating order {order_id}: {e}")

        # Remove completed orders
        for order_id in orders_to_remove:
            del self.active_orders[order_id]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            self.broker.cancel_order(order_id)

            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = OrderStatus.CANCELLED.value
                self.active_orders[order_id]['cancelled_at'] = datetime.now()

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_position_orders(self, symbol: str):
        """Cancel all orders for a specific position"""
        if symbol in self.position_orders:
            for order_id in self.position_orders[symbol]:
                self.cancel_order(order_id)

            self.position_orders[symbol] = []

    def _cancel_stop_orders(self, symbol: str):
        """Cancel existing stop orders for a symbol"""
        for order_id, order_info in list(self.active_orders.items()):
            if (order_info['symbol'] == symbol and
                    order_info['order_type'] in ['stop', 'trailing_stop'] and
                    order_info['side'] == 'sell'):
                self.cancel_order(order_id)

    def cancel_stale_orders(self, max_age_minutes: int = 60):
        """Cancel orders older than specified age"""
        current_time = datetime.now()

        for order_id, order_info in list(self.active_orders.items()):
            age = (current_time - order_info['created_at']).total_seconds() / 60

            if age > max_age_minutes and order_info['status'] == OrderStatus.SUBMITTED.value:
                logger.info(f"Cancelling stale order: {order_info['symbol']} - Age: {age:.0f} minutes")
                self.cancel_order(order_id)

    def _determine_order_type(self, signal: Dict) -> str:
        """Determine best order type based on signal and market conditions"""
        confidence = signal.get('confidence', 0.5)
        volatility = signal.get('volatility', 0.02)
        volume_ratio = signal.get('volume_ratio', 1.0)

        # High confidence and normal conditions - use limit
        if confidence > 0.7 and volatility < 0.03:
            return 'limit'

        # Breakout signal - use stop limit
        if signal.get('signal_type') == 'breakout':
            return 'stop_limit'

        # High volume/volatility - use market
        if volume_ratio > 2.0 or volatility > 0.04:
            return 'market'

        # Default to limit
        return 'limit'

    def _calculate_limit_price(self, current_price: float, side: str) -> float:
        """Calculate limit price with small buffer"""
        if side == 'buy':
            # Pay slightly more to ensure fill
            return round(current_price * 1.001, 2)
        else:
            # Accept slightly less to ensure fill
            return round(current_price * 0.999, 2)

    def _process_order_queue(self):
        """Process orders from queue (runs in separate thread)"""
        while self.is_running:
            try:
                # Get order from queue
                order_request = self.order_queue.get(timeout=1)

                # Process based on type
                if order_request['type'] == 'entry':
                    self.create_entry_order(
                        order_request['signal'],
                        order_request['position_size']
                    )
                elif order_request['type'] == 'exit':
                    self.create_exit_order(
                        order_request['position'],
                        order_request['exit_signal']
                    )

                # Small delay to respect rate limits
                time.sleep(0.5)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing order queue: {e}")

    def queue_order(self, order_type: str, **kwargs):
        """Add order to processing queue"""
        order_request = {'type': order_type}
        order_request.update(kwargs)
        self.order_queue.put(order_request)

    def get_order_summary(self) -> Dict:
        """Get summary of current orders"""
        summary = {
            'active_orders': len(self.active_orders),
            'pending': 0,
            'submitted': 0,
            'partial': 0,
            'by_symbol': {},
            'by_side': {'buy': 0, 'sell': 0}
        }

        for order_info in self.active_orders.values():
            status = order_info['status']
            symbol = order_info['symbol']
            side = order_info['side']

            # Count by status
            if status == OrderStatus.PENDING.value:
                summary['pending'] += 1
            elif status == OrderStatus.SUBMITTED.value:
                summary['submitted'] += 1
            elif status == OrderStatus.PARTIAL.value:
                summary['partial'] += 1

            # Count by symbol
            if symbol not in summary['by_symbol']:
                summary['by_symbol'][symbol] = 0
            summary['by_symbol'][symbol] += 1

            # Count by side
            summary['by_side'][side] += 1

        return summary

    def get_filled_orders(self, lookback_hours: int = 24) -> List[Dict]:
        """Get recently filled orders"""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        filled_orders = [
            order for order in self.order_history
            if (order.get('status') == 'filled' and
                order.get('filled_at', datetime.min) > cutoff)
        ]

        return filled_orders

    def calculate_order_metrics(self) -> Dict:
        """Calculate order execution metrics"""
        filled_orders = [o for o in self.order_history if o.get('status') == 'filled']

        if not filled_orders:
            return {
                'fill_rate': 0,
                'avg_fill_time': 0,
                'slippage': 0
            }

        # Calculate metrics
        total_orders = len(self.order_history)
        fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0

        # Average fill time
        fill_times = []
        for order in filled_orders:
            if 'filled_at' in order and 'created_at' in order:
                fill_time = (order['filled_at'] - order['created_at']).total_seconds()
                fill_times.append(fill_time)

        avg_fill_time = np.mean(fill_times) if fill_times else 0

        # Slippage (for limit orders)
        slippage_values = []
        for order in filled_orders:
            if order.get('order_type') == 'limit' and 'limit_price' in order:
                expected = order['limit_price']
                actual = order.get('filled_price', expected)

                if order['side'] == 'buy':
                    slippage = (actual - expected) / expected
                else:
                    slippage = (expected - actual) / expected

                slippage_values.append(slippage)

        avg_slippage = np.mean(slippage_values) if slippage_values else 0

        return {
            'fill_rate': fill_rate,
            'avg_fill_time_seconds': avg_fill_time,
            'avg_slippage_pct': avg_slippage * 100,
            'total_orders': total_orders,
            'filled_orders': len(filled_orders)
        }

    def shutdown(self):
        """Shutdown order manager"""
        self.is_running = False

        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            self.cancel_order(order_id)

        logger.info("Order manager shutdown complete")