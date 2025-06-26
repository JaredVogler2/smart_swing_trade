# data/database.py

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from contextlib import contextmanager
import threading
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class Database:
    """Thread-safe database manager for trading system"""

    def __init__(self, db_path: str = 'data/trading_system.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Market data table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS market_data
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             symbol
                             TEXT
                             NOT
                             NULL,
                             timestamp
                             DATETIME
                             NOT
                             NULL,
                             open
                             REAL
                             NOT
                             NULL,
                             high
                             REAL
                             NOT
                             NULL,
                             low
                             REAL
                             NOT
                             NULL,
                             close
                             REAL
                             NOT
                             NULL,
                             volume
                             INTEGER
                             NOT
                             NULL,
                             UNIQUE
                         (
                             symbol,
                             timestamp
                         )
                             )
                         ''')

            # Trades table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS trades
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             symbol
                             TEXT
                             NOT
                             NULL,
                             action
                             TEXT
                             NOT
                             NULL,
                             quantity
                             INTEGER
                             NOT
                             NULL,
                             price
                             REAL
                             NOT
                             NULL,
                             timestamp
                             DATETIME
                             NOT
                             NULL,
                             order_id
                             TEXT
                             UNIQUE,
                             status
                             TEXT
                             DEFAULT
                             'pending',
                             fill_price
                             REAL,
                             fill_time
                             DATETIME,
                             commission
                             REAL
                             DEFAULT
                             0,
                             pnl
                             REAL,
                             strategy
                             TEXT,
                             confidence
                             REAL,
                             metadata
                             TEXT
                         )
                         ''')

            # Positions table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS positions
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             symbol
                             TEXT
                             NOT
                             NULL
                             UNIQUE,
                             quantity
                             INTEGER
                             NOT
                             NULL,
                             entry_price
                             REAL
                             NOT
                             NULL,
                             entry_time
                             DATETIME
                             NOT
                             NULL,
                             current_price
                             REAL,
                             last_updated
                             DATETIME,
                             unrealized_pnl
                             REAL,
                             stop_loss
                             REAL,
                             take_profit
                             REAL,
                             trailing_stop_pct
                             REAL,
                             peak_price
                             REAL,
                             metadata
                             TEXT
                         )
                         ''')

            # Signals table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS signals
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             timestamp
                             DATETIME
                             NOT
                             NULL,
                             symbol
                             TEXT
                             NOT
                             NULL,
                             signal_type
                             TEXT
                             NOT
                             NULL,
                             strength
                             REAL
                             NOT
                             NULL,
                             confidence
                             REAL
                             NOT
                             NULL,
                             source
                             TEXT
                             NOT
                             NULL,
                             metadata
                             TEXT,
                             acted_upon
                             BOOLEAN
                             DEFAULT
                             FALSE
                         )
                         ''')

            # Model predictions table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS model_predictions
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             timestamp
                             DATETIME
                             NOT
                             NULL,
                             symbol
                             TEXT
                             NOT
                             NULL,
                             model_name
                             TEXT
                             NOT
                             NULL,
                             prediction
                             REAL
                             NOT
                             NULL,
                             confidence
                             REAL
                             NOT
                             NULL,
                             features_hash
                             TEXT,
                             actual_outcome
                             REAL,
                             metadata
                             TEXT
                         )
                         ''')

            # Performance metrics table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS performance_metrics
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             date
                             DATE
                             NOT
                             NULL
                             UNIQUE,
                             total_pnl
                             REAL
                             NOT
                             NULL,
                             daily_pnl
                             REAL
                             NOT
                             NULL,
                             win_rate
                             REAL,
                             sharpe_ratio
                             REAL,
                             max_drawdown
                             REAL,
                             total_trades
                             INTEGER,
                             winning_trades
                             INTEGER,
                             losing_trades
                             INTEGER,
                             avg_win
                             REAL,
                             avg_loss
                             REAL,
                             largest_win
                             REAL,
                             largest_loss
                             REAL,
                             portfolio_value
                             REAL,
                             metadata
                             TEXT
                         )
                         ''')

            # Backtest results table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS backtest_results
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             run_date
                             DATETIME
                             NOT
                             NULL,
                             strategy_name
                             TEXT
                             NOT
                             NULL,
                             start_date
                             DATE
                             NOT
                             NULL,
                             end_date
                             DATE
                             NOT
                             NULL,
                             initial_capital
                             REAL
                             NOT
                             NULL,
                             final_capital
                             REAL
                             NOT
                             NULL,
                             total_return
                             REAL
                             NOT
                             NULL,
                             sharpe_ratio
                             REAL,
                             max_drawdown
                             REAL,
                             win_rate
                             REAL,
                             profit_factor
                             REAL,
                             total_trades
                             INTEGER,
                             parameters
                             TEXT,
                             equity_curve
                             TEXT,
                             trade_log
                             TEXT,
                             metrics
                             TEXT
                         )
                         ''')

            # News sentiment table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS news_sentiment
                         (
                             id
                             INTEGER PRIMARY KEY AUTOINCREMENT,
                                                              timestamp DATETIME
                                                              NOT NULL,
                                                              symbol TEXT,
                                                              headline TEXT NOT NULL,
                                                              source TEXT,
                                                              sentiment TEXT,
                                                              sentiment_score REAL,
                                                              relevance_score REAL,
                                                              url TEXT,
                                                              metadata TEXT
                )
            ''')

            # System logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT
                )
            ''')

            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp ON model_predictions(symbol, timestamp)')

            conn.commit()

    @contextmanager

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_market_data(self, symbol: str, df: pd.DataFrame):
        """Save market data to database"""
        with self._lock:
            with self._get_connection() as conn:
                df = df.copy()
                df['symbol'] = symbol
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'timestamp'}, inplace=True)

                # Use INSERT OR REPLACE to handle duplicates
                df.to_sql('market_data', conn, if_exists='append', index=False, method='multi')

    def get_market_data(self, symbol: str, start_date: datetime = None,
                        end_date: datetime = None) -> pd.DataFrame:
        """Retrieve market data from database"""
        with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM market_data WHERE symbol = ?"
                params = [symbol]

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)

                query += " ORDER BY timestamp"

                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)

                return df

    def save_trade(self, trade: Dict):
        """Save trade to database"""
        with self._lock:
            with self._get_connection() as conn:
                metadata = trade.get('metadata', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                             INSERT INTO trades
                             (symbol, action, quantity, price, timestamp, order_id, status,
                              fill_price, fill_time, commission, pnl, strategy, confidence, metadata)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                             ''', (
                                 trade['symbol'], trade['action'], trade['quantity'],
                                 trade['price'], trade['timestamp'], trade.get('order_id'),
                                 trade.get('status', 'pending'), trade.get('fill_price'),
                                 trade.get('fill_time'), trade.get('commission', 0),
                                 trade.get('pnl'), trade.get('strategy'), trade.get('confidence'),
                                 metadata
                             ))
                conn.commit()

    def update_trade(self, order_id: str, updates: Dict):
        """Update trade status"""
        with self._lock:
            with self._get_connection() as conn:
                set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values()) + [order_id]

                conn.execute(f'''
                    UPDATE trades 
                    SET {set_clause}
                    WHERE order_id = ?
                ''', values)
                conn.commit()

    def save_position(self, position: Dict):
        """Save or update position"""
        with self._lock:
            with self._get_connection() as conn:
                metadata = position.get('metadata', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                    INSERT OR REPLACE INTO positions
                    (symbol, quantity, entry_price, entry_time, current_price,
                     last_updated, unrealized_pnl, stop_loss, take_profit,
                     trailing_stop_pct, peak_price, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position['symbol'], position['quantity'], position['entry_price'],
                    position['entry_time'], position.get('current_price'),
                    datetime.now(), position.get('unrealized_pnl'),
                    position.get('stop_loss'), position.get('take_profit'),
                    position.get('trailing_stop_pct'), position.get('peak_price'),
                    metadata
                ))
                conn.commit()

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute('SELECT * FROM positions WHERE quantity > 0')
                positions = []

                for row in cursor:
                    position = dict(row)
                    if position.get('metadata'):
                        try:
                            position['metadata'] = json.loads(position['metadata'])
                        except:
                            pass
                    positions.append(position)

                return positions

    def delete_position(self, symbol: str):
        """Remove closed position"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
                conn.commit()

    def save_signal(self, signal: Dict):
        """Save trading signal"""
        with self._lock:
            with self._get_connection() as conn:
                metadata = signal.get('metadata', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                             INSERT INTO signals
                             (timestamp, symbol, signal_type, strength, confidence, source, metadata)
                             VALUES (?, ?, ?, ?, ?, ?, ?)
                             ''', (
                                 signal['timestamp'], signal['symbol'], signal['signal_type'],
                                 signal['strength'], signal['confidence'], signal['source'],
                                 metadata
                             ))
                conn.commit()

    def save_prediction(self, prediction: Dict):
        """Save model prediction"""
        with self._lock:
            with self._get_connection() as conn:
                metadata = prediction.get('metadata', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                             INSERT INTO model_predictions
                             (timestamp, symbol, model_name, prediction, confidence,
                              features_hash, actual_outcome, metadata)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                             ''', (
                                 prediction['timestamp'], prediction['symbol'], prediction['model_name'],
                                 prediction['prediction'], prediction['confidence'],
                                 prediction.get('features_hash'), prediction.get('actual_outcome'),
                                 metadata
                             ))
                conn.commit()

    def update_prediction_outcome(self, prediction_id: int, actual_outcome: float):
        """Update prediction with actual outcome"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute('''
                             UPDATE model_predictions
                             SET actual_outcome = ?
                             WHERE id = ?
                             ''', (actual_outcome, prediction_id))
                conn.commit()

    def save_performance_metrics(self, metrics: Dict):
        """Save daily performance metrics"""
        with self._lock:
            with self._get_connection() as conn:
                metadata = metrics.get('metadata', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                    INSERT OR REPLACE INTO performance_metrics
                    (date, total_pnl, daily_pnl, win_rate, sharpe_ratio, max_drawdown,
                     total_trades, winning_trades, losing_trades, avg_win, avg_loss,
                     largest_win, largest_loss, portfolio_value, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics['date'], metrics['total_pnl'], metrics['daily_pnl'],
                    metrics.get('win_rate'), metrics.get('sharpe_ratio'),
                    metrics.get('max_drawdown'), metrics.get('total_trades'),
                    metrics.get('winning_trades'), metrics.get('losing_trades'),
                    metrics.get('avg_win'), metrics.get('avg_loss'),
                    metrics.get('largest_win'), metrics.get('largest_loss'),
                    metrics.get('portfolio_value'), metadata
                ))
                conn.commit()

    def get_performance_metrics(self, start_date: datetime = None,
                                end_date: datetime = None) -> pd.DataFrame:
        """Get performance metrics"""
        with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []

                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)

                query += " ORDER BY date"

                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

                return df

    def save_backtest_results(self, results: Dict):
        """Save backtest results"""
        with self._lock:
            with self._get_connection() as conn:
                # Convert complex objects to JSON
                parameters = json.dumps(results.get('parameters', {}))
                equity_curve = json.dumps(results.get('equity_curve', []))
                trade_log = json.dumps(results.get('trade_log', []))
                metrics = json.dumps(results.get('metrics', {}))

                conn.execute('''
                             INSERT INTO backtest_results
                             (run_date, strategy_name, start_date, end_date, initial_capital,
                              final_capital, total_return, sharpe_ratio, max_drawdown,
                              win_rate, profit_factor, total_trades, parameters,
                              equity_curve, trade_log, metrics)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                             ''', (
                                 datetime.now(), results['strategy_name'], results['start_date'],
                                 results['end_date'], results['initial_capital'],
                                 results['final_capital'], results['total_return'],
                                 results.get('sharpe_ratio'), results.get('max_drawdown'),
                                 results.get('win_rate'), results.get('profit_factor'),
                                 results.get('total_trades'), parameters, equity_curve,
                                 trade_log, metrics
                             ))
                conn.commit()

    def save_news_sentiment(self, news: Dict):
        """Save news sentiment analysis"""
        with self._lock:
            with self._get_connection() as conn:
                metadata = news.get('metadata', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                             INSERT INTO news_sentiment
                             (timestamp, symbol, headline, source, sentiment,
                              sentiment_score, relevance_score, url, metadata)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                             ''', (
                                 news['timestamp'], news.get('symbol'), news['headline'],
                                 news.get('source'), news.get('sentiment'),
                                 news.get('sentiment_score'), news.get('relevance_score'),
                                 news.get('url'), metadata
                             ))
                conn.commit()

    def log_system_event(self, level: str, component: str, message: str, metadata: Dict = None):
        """Log system events"""
        with self._lock:
            with self._get_connection() as conn:
                if metadata and isinstance(metadata, dict):
                    metadata = json.dumps(metadata)

                conn.execute('''
                             INSERT INTO system_logs
                                 (timestamp, level, component, message, metadata)
                             VALUES (?, ?, ?, ?, ?)
                             ''', (datetime.now(), level, component, message, metadata))
                conn.commit()

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to save space"""
        with self._lock:
            with self._get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)

                # Clean up old market data
                conn.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))

                # Clean up old signals
                conn.execute('DELETE FROM signals WHERE timestamp < ?', (cutoff_date,))

                # Clean up old predictions
                conn.execute('DELETE FROM model_predictions WHERE timestamp < ?', (cutoff_date,))

                # Clean up old logs
                conn.execute('DELETE FROM system_logs WHERE timestamp < ?', (cutoff_date,))

                # Vacuum to reclaim space
                conn.execute('VACUUM')
                conn.commit()

    def backup_database(self, backup_path: str = None):
        """Create database backup"""
        if backup_path is None:
            backup_path = f"data/backups/trading_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(exist_ok=True)

        with self._lock:
            with self._get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()

        logger.info(f"Database backed up to {backup_path}")
        return backup_path