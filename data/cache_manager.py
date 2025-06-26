# data/cache_manager.py

import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import threading
import logging
import hashlib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Efficient caching system for API data and computations"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory cache
        self.memory_cache = {}
        self.cache_metadata = {}
        self.lock = threading.Lock()

        # Cache durations (in seconds)
        self.durations = {
            'quote': 30,  # 30 seconds for quotes
            'price': 60,  # 1 minute for prices
            'bars_1min': 60,  # 1 minute for minute bars
            'bars_5min': 300,  # 5 minutes for 5-min bars
            'bars_1hour': 3600,  # 1 hour for hourly bars
            'bars_1day': 3600,  # 1 hour for daily bars
            'fundamental': 86400,  # 1 day for fundamentals
            'asset_info': 604800,  # 1 week for asset info
            'news': 1800,  # 30 minutes for news
            'sentiment': 3600,  # 1 hour for sentiment
            'features': 300,  # 5 minutes for computed features
            'predictions': 60,  # 1 minute for predictions
            'market_status': 60,  # 1 minute for market status
        }

        # Start cleanup thread
        self._start_cleanup_thread()

    def _get_cache_key(self, key: str, prefix: str = None) -> str:
        """Generate cache key with optional prefix"""
        if prefix:
            return f"{prefix}:{key}"
        return key

    def _get_file_path(self, key: str) -> Path:
        """Get file path for disk cache"""
        # Hash the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str, data_type: str = 'default') -> Optional[Any]:
        """Get data from cache"""
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                metadata = self.cache_metadata.get(key, {})
                expiry = metadata.get('expiry', 0)

                if time.time() < expiry:
                    metadata['hits'] = metadata.get('hits', 0) + 1
                    return self.memory_cache[key]
                else:
                    # Expired, remove from memory
                    del self.memory_cache[key]
                    del self.cache_metadata[key]

            # Check disk cache
            file_path = self._get_file_path(key)
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        data, timestamp = pickle.load(f)

                    # Check expiry
                    duration = self.durations.get(data_type, 300)
                    if time.time() - timestamp < duration:
                        # Load into memory cache
                        self.memory_cache[key] = data
                        self.cache_metadata[key] = {
                            'timestamp': timestamp,
                            'expiry': timestamp + duration,
                            'size': file_path.stat().st_size,
                            'hits': 1
                        }
                        return data
                    else:
                        # Expired, remove file
                        file_path.unlink()

                except Exception as e:
                    logger.error(f"Error reading cache file {file_path}: {e}")

        return None

    def set(self, key: str, data: Any, data_type: str = 'default',
            duration: int = None) -> bool:
        """Store data in cache"""
        with self.lock:
            try:
                # Determine duration
                if duration is None:
                    duration = self.durations.get(data_type, 300)

                timestamp = time.time()
                expiry = timestamp + duration

                # Store in memory
                self.memory_cache[key] = data
                self.cache_metadata[key] = {
                    'timestamp': timestamp,
                    'expiry': expiry,
                    'type': data_type,
                    'hits': 0
                }

                # Store on disk for persistence
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    pickle.dump((data, timestamp), f)

                # Track size
                self.cache_metadata[key]['size'] = file_path.stat().st_size

                return True

            except Exception as e:
                logger.error(f"Error setting cache for {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            # Remove from memory
            if key in self.memory_cache:
                del self.memory_cache[key]

            if key in self.cache_metadata:
                del self.cache_metadata[key]

            # Remove from disk
            file_path = self._get_file_path(key)
            if file_path.exists():
                try:
                    file_path.unlink()
                    return True
                except:
                    pass

        return False

    def clear(self, data_type: str = None):
        """Clear cache, optionally by data type"""
        with self.lock:
            if data_type:
                # Clear specific type
                keys_to_remove = [
                    k for k, v in self.cache_metadata.items()
                    if v.get('type') == data_type
                ]

                for key in keys_to_remove:
                    self.delete(key)
            else:
                # Clear everything
                self.memory_cache.clear()
                self.cache_metadata.clear()

                # Remove all cache files
                for file_path in self.cache_dir.glob('*.pkl'):
                    try:
                        file_path.unlink()
                    except:
                        pass

    def cleanup_expired(self):
        """Remove expired items from cache"""
        with self.lock:
            now = time.time()
            expired_keys = []

            # Find expired items
            for key, metadata in self.cache_metadata.items():
                if now > metadata.get('expiry', 0):
                    expired_keys.append(key)

            # Remove expired items
            for key in expired_keys:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.cache_metadata:
                    del self.cache_metadata[key]

                # Remove file
                file_path = self._get_file_path(key)
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except:
                        pass

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")

    def _cleanup_thread(self):
        """Background thread for cache cleanup"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self.cleanup_expired()
                self._manage_cache_size()
            except Exception as e:
                logger.error(f"Error in cache cleanup thread: {e}")

    def _start_cleanup_thread(self):
        """Start the cleanup thread"""
        thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        thread.start()

    def _manage_cache_size(self):
        """Manage cache size by removing least recently used items"""
        max_memory_size = 100 * 1024 * 1024  # 100 MB
        max_disk_size = 1024 * 1024 * 1024  # 1 GB

        with self.lock:
            # Check memory size
            memory_size = sum(
                self._estimate_size(v) for v in self.memory_cache.values()
            )

            if memory_size > max_memory_size:
                # Sort by last access (hits)
                sorted_keys = sorted(
                    self.cache_metadata.items(),
                    key=lambda x: x[1].get('hits', 0)
                )

                # Remove least used items
                while memory_size > max_memory_size * 0.8 and sorted_keys:
                    key, _ = sorted_keys.pop(0)
                    if key in self.memory_cache:
                        size = self._estimate_size(self.memory_cache[key])
                        del self.memory_cache[key]
                        memory_size -= size

            # Check disk size
            disk_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))

            if disk_size > max_disk_size:
                # Remove oldest files
                files = sorted(
                    self.cache_dir.glob('*.pkl'),
                    key=lambda f: f.stat().st_mtime
                )

                while disk_size > max_disk_size * 0.8 and files:
                    file = files.pop(0)
                    size = file.stat().st_size
                    try:
                        file.unlink()
                        disk_size -= size
                    except:
                        pass

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object"""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, dict):
            return sum(self._estimate_size(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(v) for v in obj)
        else:
            return len(pickle.dumps(obj))

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            memory_size = sum(
                self._estimate_size(v) for v in self.memory_cache.values()
            )

            disk_files = list(self.cache_dir.glob('*.pkl'))
            disk_size = sum(f.stat().st_size for f in disk_files)

            # Calculate hit rates by type
            type_stats = {}
            for key, metadata in self.cache_metadata.items():
                data_type = metadata.get('type', 'unknown')
                if data_type not in type_stats:
                    type_stats[data_type] = {'count': 0, 'hits': 0}
                type_stats[data_type]['count'] += 1
                type_stats[data_type]['hits'] += metadata.get('hits', 0)

            return {
                'memory_items': len(self.memory_cache),
                'memory_size_mb': memory_size / 1024 / 1024,
                'disk_files': len(disk_files),
                'disk_size_mb': disk_size / 1024 / 1024,
                'type_stats': type_stats,
                'total_hits': sum(m.get('hits', 0) for m in self.cache_metadata.values())
            }

    def cache_dataframe(self, key: str, df: pd.DataFrame, data_type: str = 'dataframe'):
        """Cache a pandas DataFrame efficiently"""
        # Use parquet format for better compression
        buffer = df.to_parquet()
        self.set(key, buffer, data_type)

    def get_dataframe(self, key: str, data_type: str = 'dataframe') -> Optional[pd.DataFrame]:
        """Retrieve a cached DataFrame"""
        buffer = self.get(key, data_type)
        if buffer:
            return pd.read_parquet(buffer)
        return None

    def cache_computation(self, func, *args, cache_key: str = None,
                          data_type: str = 'computation', **kwargs):
        """Cache expensive computation results"""
        # Generate cache key from function and arguments if not provided
        if cache_key is None:
            func_name = func.__name__
            args_str = str(args) + str(kwargs)
            cache_key = f"{func_name}:{hashlib.md5(args_str.encode()).hexdigest()}"

        # Check cache first
        result = self.get(cache_key, data_type)
        if result is not None:
            return result

        # Compute and cache
        result = func(*args, **kwargs)
        self.set(cache_key, result, data_type)

        return result
    