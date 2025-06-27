# models/ensemble.py
# COMPLETE FILE - REPLACE YOUR ENTIRE ensemble.py WITH THIS

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from datetime import datetime
import warnings
import os
import platform
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, log_loss
)
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Custom modules
from models.features import FeatureEngineer
from config.settings import Config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU status and send alerts"""

    def __init__(self, alert_email=None, require_gpu=True):
        self.alert_email = alert_email
        self.require_gpu = require_gpu
        self.gpu_status = self._check_gpu_status()

    def _check_gpu_status(self) -> Dict:
        """Comprehensive GPU status check"""
        status = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'driver_version': None,
            'cuda_version': None,
            'memory_info': {},
            'errors': []
        }

        try:
            import torch

            # Check CUDA availability
            status['cuda_available'] = torch.cuda.is_available()

            if status['cuda_available']:
                status['gpu_count'] = torch.cuda.device_count()

                # Get GPU names and memory
                for i in range(status['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    status['gpu_names'].append(gpu_name)

                    # Memory info
                    mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9

                    status['memory_info'][f'gpu_{i}'] = {
                        'name': gpu_name,
                        'allocated_gb': round(mem_allocated, 2),
                        'reserved_gb': round(mem_reserved, 2),
                        'total_gb': round(mem_total, 2),
                        'free_gb': round(mem_total - mem_reserved, 2)
                    }

                # Get CUDA version
                status['cuda_version'] = torch.version.cuda

                # Try to get driver version using nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
                                             '--format=csv,noheader'],
                                            capture_output=True, text=True)
                    if result.returncode == 0:
                        status['driver_version'] = result.stdout.strip()
                except:
                    pass

            else:
                status['errors'].append("CUDA is not available")

        except ImportError:
            status['errors'].append("PyTorch not installed")
        except Exception as e:
            status['errors'].append(f"GPU check error: {str(e)}")

        return status

    def send_alert(self, subject: str, message: str, critical: bool = False):
        """Send alert via console and optionally email"""
        # Console alert with color coding
        if critical:
            print(f"\n{'=' * 60}")
            print(f"🚨 CRITICAL GPU ALERT 🚨")
            print(f"{'=' * 60}")
            print(f"Subject: {subject}")
            print(f"Message: {message}")
            print(f"{'=' * 60}\n")

            # Log to file
            with open('gpu_alerts.log', 'a') as f:
                f.write(f"{datetime.now()} - CRITICAL - {subject}: {message}\n")
        else:
            print(f"\n⚠️  GPU WARNING: {subject}")
            print(f"   {message}\n")

        # Email alert if configured
        if self.alert_email and critical:
            self._send_email_alert(subject, message)

    def _send_email_alert(self, subject: str, message: str):
        """Send email alert (configure with your SMTP settings)"""
        # This is a template - configure with your email settings
        try:
            # Example using Gmail SMTP
            sender_email = "your_trading_bot@gmail.com"
            sender_password = "your_app_password"  # Use app-specific password

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = self.alert_email
            msg['Subject'] = f"Trading Bot GPU Alert: {subject}"

            body = f"""
            GPU ALERT from Trading System

            Time: {datetime.now()}
            Subject: {subject}

            Details:
            {message}

            Please check your trading system immediately.
            """

            msg.attach(MIMEText(body, 'plain'))

            # Uncomment and configure for actual email sending
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login(sender_email, sender_password)
            # server.send_message(msg)
            # server.quit()

        except Exception as e:
            print(f"Failed to send email alert: {e}")

    def validate_gpu_requirement(self) -> bool:
        """Validate GPU availability and alert if required"""
        if not self.gpu_status['cuda_available']:
            if self.require_gpu:
                self.send_alert(
                    "GPU REQUIRED BUT NOT AVAILABLE",
                    f"GPU is required but CUDA is not available.\n"
                    f"Errors: {', '.join(self.gpu_status['errors'])}",
                    critical=True
                )
                raise RuntimeError("GPU required but not available. Check gpu_alerts.log")
            else:
                self.send_alert(
                    "GPU Not Available - Using CPU",
                    "GPU was requested but is not available. Falling back to CPU.",
                    critical=False
                )
                return False

        # Check for low memory
        for gpu_id, mem_info in self.gpu_status['memory_info'].items():
            if mem_info['free_gb'] < 2.0:  # Less than 2GB free
                self.send_alert(
                    f"Low GPU Memory on {mem_info['name']}",
                    f"Only {mem_info['free_gb']}GB free out of {mem_info['total_gb']}GB",
                    critical=False
                )

        return True

    def print_gpu_status(self):
        """Print detailed GPU status"""
        print("\n" + "=" * 60)
        print("GPU STATUS REPORT")
        print("=" * 60)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"CUDA Available: {self.gpu_status['cuda_available']}")

        if self.gpu_status['cuda_available']:
            print(f"CUDA Version: {self.gpu_status['cuda_version']}")
            print(f"Driver Version: {self.gpu_status['driver_version'] or 'Unknown'}")
            print(f"GPU Count: {self.gpu_status['gpu_count']}")

            for i, gpu_name in enumerate(self.gpu_status['gpu_names']):
                print(f"\nGPU {i}: {gpu_name}")
                mem_info = self.gpu_status['memory_info'][f'gpu_{i}']
                print(f"  Memory: {mem_info['allocated_gb']}/{mem_info['total_gb']}GB allocated")
                print(f"  Free: {mem_info['free_gb']}GB")
        else:
            print("Errors:", ', '.join(self.gpu_status['errors']))

        print("=" * 60 + "\n")


class LSTMModel(nn.Module):
    """PyTorch LSTM model for GPU acceleration with proper output clipping"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True,
                             bidirectional=True, dropout=dropout if dropout > 0 else 0)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_dims[0] * 2, hidden_dims[1], batch_first=True,
                             bidirectional=True, dropout=dropout if dropout > 0 else 0)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_dims[1] * 2, hidden_dims[2], batch_first=True,
                             dropout=dropout if dropout > 0 else 0)
        self.dropout3 = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dims[2], num_heads=4,
                                               dropout=dropout, batch_first=True)

        # Output layers
        self.fc1 = nn.Linear(hidden_dims[2], 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(32, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)
        out = self.dropout3(out)

        # Attention
        attn_out, _ = self.attention(out, out, out)
        out = attn_out + out  # Residual connection

        # Take the last output
        out = out[:, -1, :]

        # Fully connected layers with batch norm
        out = self.fc1(out)
        if out.size(0) > 1:  # Only apply batch norm if batch size > 1
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout4(out)

        out = self.fc2(out)
        if out.size(0) > 1:  # Only apply batch norm if batch size > 1
            out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout5(out)

        # Final output
        out = self.fc3(out)

        # Use sigmoid to ensure output is in [0, 1]
        out = torch.sigmoid(out)

        # Extra safety: clamp output to avoid numerical issues
        out = torch.clamp(out, min=1e-7, max=1 - 1e-7)

        return out


class EnsembleModel:
    """GPU-accelerated ensemble model with enhanced GPU monitoring"""

    def __init__(self, use_gpu=True, require_gpu=False, alert_email=None):
        """
        Initialize ensemble model with GPU monitoring

        Args:
            use_gpu: Whether to attempt using GPU
            require_gpu: If True, will raise error if GPU not available
            alert_email: Email address for critical GPU alerts
        """
        self.use_gpu = use_gpu and Config.USE_GPU
        self.require_gpu = require_gpu
        self.models = {}
        self.feature_engineer = FeatureEngineer(enable_gpu=self.use_gpu)
        self.is_trained = False
        self.feature_importance = {}
        self.model_weights = {
            'lstm': 0.35,
            'xgboost': 0.3,
            'lightgbm': 0.25,
            'rf': 0.1
        }

        # Model performance tracking
        self.model_performance = {
            'lstm': {'accuracy': 0.5, 'sharpe': 0},
            'xgboost': {'accuracy': 0.5, 'sharpe': 0},
            'lightgbm': {'accuracy': 0.5, 'sharpe': 0},
            'rf': {'accuracy': 0.5, 'sharpe': 0}
        }

        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor(alert_email=alert_email, require_gpu=require_gpu)

        # Print GPU status on initialization
        self.gpu_monitor.print_gpu_status()

        # Initialize device with monitoring
        if self.use_gpu:
            if self.gpu_monitor.validate_gpu_requirement():
                self.device = torch.device('cuda')
                # Set CUDA settings for better stability
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

                # Test GPU with small tensor
                try:
                    test_tensor = torch.randn(10, 10).to(self.device)
                    _ = test_tensor * 2
                    logger.info(f"GPU initialized successfully: {torch.cuda.get_device_name()}")

                    # Monitor GPU memory after initialization
                    self._log_gpu_memory("After initialization")

                except Exception as e:
                    self.gpu_monitor.send_alert(
                        "GPU Initialization Failed",
                        f"Failed to run test computation on GPU: {str(e)}",
                        critical=True
                    )
                    if self.require_gpu:
                        raise
                    else:
                        logger.warning("Falling back to CPU due to GPU initialization failure")
                        self.device = torch.device('cpu')
                        self.use_gpu = False
            else:
                self.device = torch.device('cpu')
                self.use_gpu = False
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU as requested")

        # Model parameters
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.prediction_horizon = 3  # 3 days for swing trading

        # Feature cache for performance
        self.feature_cache = {}
        self.cache_size = 100

    def _log_gpu_memory(self, stage: str):
        """Log GPU memory usage at different stages"""
        if self.use_gpu and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9

                logger.info(f"GPU {i} Memory - {stage}:")
                logger.info(f"  Allocated: {allocated:.2f}GB")
                logger.info(f"  Reserved: {reserved:.2f}GB")
                logger.info(f"  Total: {total:.2f}GB")
                logger.info(f"  Free: {total - reserved:.2f}GB")

    def _prepare_lstm_data(self, features: pd.DataFrame, target: pd.Series = None) -> Tuple:
        """Prepare data for LSTM (sequences)"""
        sequences = []
        targets = [] if target is not None else None

        for i in range(self.sequence_length, len(features)):
            seq = features.iloc[i - self.sequence_length:i].values
            sequences.append(seq)

            if target is not None:
                targets.append(target.iloc[i])

        sequences = np.array(sequences)

        if target is not None:
            targets = np.array(targets)
            return sequences, targets
        else:
            return sequences

    def _create_lstm_model(self, input_shape):
        """Create LSTM model with GPU support"""
        model = LSTMModel(
            input_dim=input_shape[-1],
            hidden_dims=[128, 64, 32],
            dropout=0.2
        ).to(self.device)

        return model

    def _create_xgboost_model(self):
        """Create XGBoost model with GPU support"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1
        }

        # Remove GPU parameters if not using GPU
        if not self.use_gpu:
            params.pop('gpu_id', None)

        return xgb.XGBClassifier(**params)

    def _create_lightgbm_model(self):
        """Create LightGBM model with GPU support"""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1
        }

        # Remove GPU parameters if not using GPU
        if not self.use_gpu:
            params.pop('gpu_device_id', None)

        return lgb.LGBMClassifier(**params)

    def _create_random_forest_model(self):
        """Create Random Forest model"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

    def train(self, train_data: Dict[str, pd.DataFrame],
              validation_split: float = 0.2):
        """Enhanced training with GPU monitoring"""
        logger.info("Starting ensemble model training...")

        # Check GPU status before training
        if self.use_gpu:
            self.gpu_monitor.print_gpu_status()
            self._log_gpu_memory("Before training")

        # Clear GPU cache if using CUDA
        if self.use_gpu:
            torch.cuda.empty_cache()
            # Set environment for debugging
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # Prepare training data with better error handling
        try:
            X_train_all = []
            y_train_all = []
            symbols_processed = []

            logger.info("Creating features for training data...")
            logger.info(f"Processing {len(train_data)} symbols...")

            if self.use_gpu:
                self._log_gpu_memory("Before feature creation")

            # Process each symbol with detailed logging
            for idx, (symbol, df) in enumerate(train_data.items()):
                try:
                    logger.info(f"Processing {symbol} ({idx + 1}/{len(train_data)})")

                    if len(df) < self.sequence_length + 100:
                        logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                        continue

                    # Create features with error handling
                    features = self.feature_engineer.create_features(df, symbol)

                    if features.empty:
                        logger.warning(f"Skipping {symbol}: feature creation failed")
                        continue

                    # Create target (3-day forward return > 2.5% after costs)
                    forward_return = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)

                    # More conservative target to avoid extreme predictions
                    target = (forward_return > 0.025).astype(int)  # 2.5% threshold

                    # Remove last prediction_horizon rows
                    features = features[:-self.prediction_horizon]
                    target = target[:-self.prediction_horizon]

                    # Remove NaN and inf values
                    valid_idx = ~(features.isna().any(axis=1) |
                                  target.isna() |
                                  np.isinf(features).any(axis=1))

                    features = features[valid_idx]
                    target = target[valid_idx]

                    if len(features) < 100:
                        logger.warning(f"Skipping {symbol}: too few valid samples after cleaning")
                        continue

                    X_train_all.append(features)
                    y_train_all.append(target)
                    symbols_processed.append(symbol)

                    logger.info(f"Successfully processed {symbol}: {len(features)} samples")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue

            if len(X_train_all) < 5:
                raise ValueError(f"Insufficient symbols processed: {len(X_train_all)}")

            logger.info(f"Feature creation complete:")
            logger.info(f"- Successful: {len(symbols_processed)} symbols")
            logger.info(f"- Failed: {len(train_data) - len(symbols_processed)} symbols")

            if self.use_gpu:
                self._log_gpu_memory("After feature creation")

            # Combine all data
            logger.info("Combining training data...")
            X_train = pd.concat(X_train_all, ignore_index=True)
            y_train = pd.concat(y_train_all, ignore_index=True)

            # Final data cleaning
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_train = X_train.fillna(X_train.median())

            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Features: {X_train.shape[1]}")
            logger.info(f"Samples: {X_train.shape[0]}")
            logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")
            logger.info(f"Class balance: {y_train.mean():.2%} positive")

            # Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)

            # Create validation split
            val_size = int(len(X_train_scaled) * validation_split)
            X_val = X_train_scaled.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_scaled = X_train_scaled.iloc[:-val_size]
            y_train_final = y_train.iloc[:-val_size]

            # Train each model with error handling
            logger.info("Training individual models...")

            # Train LSTM
            try:
                logger.info("Training LSTM...")
                if self.use_gpu:
                    self._log_gpu_memory("Before LSTM training")
                self._train_lstm(X_train_scaled, y_train_final, X_val, y_val)
                if self.use_gpu:
                    self._log_gpu_memory("After LSTM training")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.gpu_monitor.send_alert(
                        "GPU Out of Memory",
                        f"LSTM training failed due to insufficient GPU memory. "
                        f"Error: {str(e)}\n"
                        f"Consider reducing batch size or model size.",
                        critical=True
                    )

                    if self.require_gpu:
                        raise
                    else:
                        # Try to recover by clearing cache and using smaller batch
                        logger.warning("Attempting recovery with smaller batch size...")
                        torch.cuda.empty_cache()
                        self._train_lstm(X_train_scaled, y_train_final, X_val, y_val, batch_size=32)

                elif "CUDA" in str(e):
                    self.gpu_monitor.send_alert(
                        "CUDA Error During Training",
                        f"CUDA error encountered: {str(e)}\n"
                        f"This may indicate a driver issue or hardware problem.",
                        critical=True
                    )

                    if self.require_gpu:
                        raise
                    else:
                        logger.info("Falling back to CPU training...")
                        self.device = torch.device('cpu')
                        self.use_gpu = False
                        self._train_lstm(X_train_scaled, y_train_final, X_val, y_val)
                else:
                    raise
            except Exception as e:
                logger.error(f"LSTM training failed: {str(e)}")
                if self.require_gpu and "GPU" in str(e):
                    raise
                logger.info("Continuing without LSTM...")

            # Train XGBoost
            try:
                logger.info("Training XGBoost...")
                self._train_xgboost(X_train_scaled, y_train_final, X_val, y_val)
            except Exception as e:
                logger.error(f"XGBoost training failed: {str(e)}")

            # Train LightGBM
            try:
                logger.info("Training LightGBM...")
                self._train_lightgbm(X_train_scaled, y_train_final, X_val, y_val)
            except Exception as e:
                logger.error(f"LightGBM training failed: {str(e)}")

            # Train Random Forest
            try:
                logger.info("Training Random Forest...")
                self._train_random_forest(X_train_scaled, y_train_final)
            except Exception as e:
                logger.error(f"Random Forest training failed: {str(e)}")

            # Calculate feature importance
            self._calculate_feature_importance(X_train_scaled, y_train_final)

            # Validate ensemble performance
            self._validate_ensemble(X_val, y_val)

            self.is_trained = True
            logger.info("Ensemble model training completed successfully")

            # Final GPU status report
            if self.use_gpu:
                self._log_gpu_memory("After training complete")
                self.gpu_monitor.print_gpu_status()

        except Exception as e:
            self.gpu_monitor.send_alert(
                "Training Failed",
                f"Training failed with error: {str(e)}",
                critical=True
            )
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    # Fix for _train_lstm method in ensemble.py
    # Replace your entire _train_lstm method with this:

    def _train_lstm(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series, batch_size: int = None):
        """Enhanced LSTM training with better memory management for 8GB GPU"""
        logger.info("Training LSTM...")
        logger.info(f"LSTM Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        # Clear GPU memory before starting
        if self.use_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"GPU memory before LSTM: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

        try:
            # Prepare sequences
            X_train_seq, y_train_seq = self._prepare_lstm_data(X_train, y_train)
            X_val_seq, y_val_seq = self._prepare_lstm_data(X_val, y_val)

            # Determine optimal batch size based on data size and GPU memory
            if batch_size is None:
                n_samples = len(X_train_seq)
                # For 8GB GPU with your data size, use very small batches
                if n_samples > 50000:
                    batch_size = 8
                elif n_samples > 20000:
                    batch_size = 16
                else:
                    batch_size = 32

            logger.info(f"Training samples: {len(X_train_seq)}, Validation samples: {len(X_val_seq)}")
            logger.info(f"Using batch size: {batch_size}")

            # Check data validity
            assert not np.isnan(X_train_seq).any(), "NaN values in training data"
            assert not np.isinf(X_train_seq).any(), "Inf values in training data"

            # Convert to tensors - DON'T put entire dataset on GPU at once!
            # We'll move batches to GPU as needed
            X_train_cpu = torch.FloatTensor(X_train_seq)
            y_train_cpu = torch.FloatTensor(y_train_seq)
            X_val_cpu = torch.FloatTensor(X_val_seq)
            y_val_cpu = torch.FloatTensor(y_val_seq)

            logger.info(f"Data prepared. Shape: {X_train_cpu.shape}")

            # Create model with smaller size for 8GB GPU
            input_shape = X_train_seq.shape

            # Reduce model size for memory constraints
            model = LSTMModel(
                input_dim=input_shape[-1],
                hidden_dims=[64, 32, 16],  # Reduced from [128, 64, 32]
                dropout=0.2
            ).to(self.device)

            logger.info(f"LSTM model created with reduced size")

            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5, verbose=True
            )

            # Training parameters
            n_epochs = 30  # Reduced from 50
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10

            # Create data loader for better batching
            train_dataset = TensorDataset(X_train_cpu, y_train_cpu)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            # Training loop
            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0
                n_batches = 0

                # Clear GPU cache at start of each epoch
                if self.use_gpu and epoch % 5 == 0:
                    torch.cuda.empty_cache()

                for batch_X, batch_y in train_loader:
                    # Move batch to GPU only when needed
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))

                    # Inside the batch loop, after forward pass:
                    if n_batches % 50 == 0:
                        logger.info(
                            f"Batch {n_batches}: GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                            f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.error("NaN loss detected, skipping batch")
                        continue

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                    # Clear batch from GPU
                    del batch_X, batch_y, outputs, loss

                if n_batches == 0:
                    logger.error("No valid batches in epoch")
                    break

                avg_train_loss = epoch_loss / n_batches

                # Validation - process in batches to avoid memory issues
                model.eval()
                val_loss = 0
                val_batches = 0

                with torch.no_grad():
                    # Process validation in smaller chunks
                    val_batch_size = min(batch_size * 2, 64)  # Slightly larger for validation

                    for i in range(0, len(X_val_cpu), val_batch_size):
                        batch_X_val = X_val_cpu[i:i + val_batch_size].to(self.device)
                        batch_y_val = y_val_cpu[i:i + val_batch_size].to(self.device)

                        val_outputs = model(batch_X_val)
                        batch_val_loss = criterion(val_outputs, batch_y_val.unsqueeze(1))

                        val_loss += batch_val_loss.item()
                        val_batches += 1

                        # Clear validation batch from GPU
                        del batch_X_val, batch_y_val, val_outputs

                val_loss_value = val_loss / val_batches if val_batches > 0 else float('inf')

                # Learning rate scheduling
                scheduler.step(val_loss_value)

                # Early stopping
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 5 == 0:
                    logger.info(f"LSTM Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                                f"Val Loss: {val_loss_value:.4f}")
                    if self.use_gpu:
                        logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")

            # Restore best model
            if 'best_model_state' in locals():
                model.load_state_dict(best_model_state)

            model.eval()
            self.models['lstm'] = model

            logger.info("LSTM training completed successfully")

            # Final cleanup
            if self.use_gpu:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"LSTM training error: {str(e)}", exc_info=True)
            if self.use_gpu:
                torch.cuda.empty_cache()
            raise

    # Also update the _predict_lstm method to handle memory better:

    def _predict_lstm(self, features: pd.DataFrame) -> Tuple[int, float]:
        """Get LSTM prediction with memory management"""
        model = self.models['lstm']
        model.eval()

        # Clear GPU cache before prediction
        if self.use_gpu:
            torch.cuda.empty_cache()

        # Prepare sequence
        X_seq = self._prepare_lstm_data(features)

        # Take only the last sequence
        last_seq = X_seq[-1:] if len(X_seq) > 0 else X_seq

        # Convert to tensor
        X_tensor = torch.FloatTensor(last_seq).to(self.device)

        # Predict
        with torch.no_grad():
            output = model(X_tensor)
            prob = output.cpu().numpy()[0, 0]

        # Clean up
        del X_tensor, output

        pred = 1 if prob > 0.5 else 0

        return pred, prob

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")

        model = self._create_xgboost_model()

        # Train with early stopping
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        self.models['xgboost'] = model

        # Store feature importance
        self.feature_importance['xgboost'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")

        model = self._create_lightgbm_model()

        # Train with early stopping
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        self.models['lightgbm'] = model

        # Store feature importance
        self.feature_importance['lightgbm'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")

        model = self._create_random_forest_model()
        model.fit(X, y)

        self.models['rf'] = model

        # Store feature importance
        self.feature_importance['rf'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    def _validate_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Validate ensemble performance and adjust weights"""
        logger.info("Validating ensemble performance...")

        predictions = {}
        probabilities = {}

        # Get predictions from each model
        for model_name in self.models:
            try:
                if model_name == 'lstm':
                    pred, prob = self._predict_lstm(X_val)
                else:
                    pred, prob = self._predict_tree_model(self.models[model_name], X_val)

                predictions[model_name] = pred
                probabilities[model_name] = prob

                # Calculate individual model performance
                if len(pred) == len(y_val):
                    accuracy = accuracy_score(y_val, pred)
                    self.model_performance[model_name]['accuracy'] = accuracy
                    logger.info(f"{model_name} validation accuracy: {accuracy:.4f}")

            except Exception as e:
                logger.error(f"Error validating {model_name}: {str(e)}")

        # Update weights based on performance
        self._update_model_weights_based_on_performance()

    def _update_model_weights_based_on_performance(self):
        """Update model weights based on validation performance"""
        total_accuracy = sum(perf['accuracy'] for perf in self.model_performance.values())

        if total_accuracy > 0:
            for model_name in self.model_weights:
                if model_name in self.model_performance:
                    # Weight based on accuracy
                    accuracy_weight = self.model_performance[model_name]['accuracy'] / total_accuracy

                    # Blend with existing weight
                    self.model_weights[model_name] = (
                            0.7 * self.model_weights[model_name] +
                            0.3 * accuracy_weight
                    )

            # Normalize weights
            total_weight = sum(self.model_weights.values())
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight

            logger.info(f"Updated model weights: {self.model_weights}")

    def predict(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Generate prediction for a symbol"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Create features
        features = self.feature_engineer.create_features(price_data, symbol)

        if features.empty or len(features) < self.sequence_length:
            return {
                'symbol': symbol,
                'prediction': 0,
                'confidence': 0,
                'error': 'Insufficient data'
            }

        # Scale features
        features_scaled = self.feature_engineer.scale_features(features, fit=False)

        # Get predictions from each model
        predictions = {}
        probabilities = {}

        # LSTM prediction
        if 'lstm' in self.models:
            try:
                lstm_pred, lstm_prob = self._predict_lstm(features_scaled)
                predictions['lstm'] = lstm_pred
                probabilities['lstm'] = lstm_prob
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")

        # XGBoost prediction
        if 'xgboost' in self.models:
            try:
                xgb_pred, xgb_prob = self._predict_tree_model(
                    self.models['xgboost'], features_scaled
                )
                predictions['xgboost'] = xgb_pred
                probabilities['xgboost'] = xgb_prob
            except Exception as e:
                logger.error(f"XGBoost prediction error: {e}")

        # LightGBM prediction
        if 'lightgbm' in self.models:
            try:
                lgb_pred, lgb_prob = self._predict_tree_model(
                    self.models['lightgbm'], features_scaled
                )
                predictions['lightgbm'] = lgb_pred
                probabilities['lightgbm'] = lgb_prob
            except Exception as e:
                logger.error(f"LightGBM prediction error: {e}")

        # Random Forest prediction
        if 'rf' in self.models:
            try:
                rf_pred, rf_prob = self._predict_tree_model(
                    self.models['rf'], features_scaled
                )
                predictions['rf'] = rf_pred
                probabilities['rf'] = rf_prob
            except Exception as e:
                logger.error(f"RF prediction error: {e}")

        # Weighted ensemble prediction
        ensemble_prob = 0
        total_weight = 0

        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 0.1)
            ensemble_prob += prob * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_prob /= total_weight
        else:
            ensemble_prob = 0.5

        ensemble_pred = 1 if ensemble_prob > 0.6 else 0  # Higher threshold for swing trades

        # Calculate confidence based on model agreement
        model_preds = list(predictions.values())
        agreement = sum(model_preds) / len(model_preds) if model_preds else 0

        # Adjust confidence
        confidence = ensemble_prob * (0.5 + 0.5 * abs(agreement - 0.5) * 2)

        # Get latest market data
        current_price = price_data['close'].iloc[-1]
        volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1]
        volume_ratio = price_data['volume'].iloc[-1] / price_data['volume'].rolling(20).mean().iloc[-1]

        return {
            'symbol': symbol,
            'prediction': ensemble_pred,
            'probability': ensemble_prob,
            'confidence': confidence,
            'model_predictions': predictions,
            'model_probabilities': probabilities,
            'current_price': current_price,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'expected_return': self._calculate_expected_return(
                ensemble_prob, volatility
            ),
            'timestamp': datetime.now()
        }

    def predict_with_confidence_interval(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Enhanced prediction with confidence intervals"""
        base_prediction = self.predict(symbol, price_data)

        # Add confidence interval based on model agreement and volatility
        model_probs = list(base_prediction.get('model_probabilities', {}).values())

        if model_probs:
            # Calculate standard deviation of predictions
            prob_std = np.std(model_probs)

            # Confidence interval (roughly 95%)
            lower_bound = max(0, base_prediction['probability'] - 2 * prob_std)
            upper_bound = min(1, base_prediction['probability'] + 2 * prob_std)

            base_prediction['confidence_interval'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }

            # Adjust confidence based on interval width
            base_prediction['confidence'] *= (1 - base_prediction['confidence_interval']['width'])

        return base_prediction

    def _predict_lstm(self, features: pd.DataFrame) -> Tuple[int, float]:
        """Get LSTM prediction"""
        model = self.models['lstm']
        model.eval()

        # Prepare sequence
        X_seq = self._prepare_lstm_data(features)

        # Take only the last sequence
        last_seq = X_seq[-1:] if len(X_seq) > 0 else X_seq

        # Convert to tensor
        X_tensor = torch.FloatTensor(last_seq).to(self.device)

        # Predict
        with torch.no_grad():
            output = model(X_tensor)
            prob = output.cpu().numpy()[0, 0]

        pred = 1 if prob > 0.5 else 0

        return pred, prob

    def _predict_tree_model(self, model, features: pd.DataFrame) -> Tuple[int, float]:
        """Get prediction from tree-based model"""
        # Use last row for prediction
        X = features.iloc[-1:].values

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0, 1]

        return int(pred), float(prob)

    def _calculate_expected_return(self, probability: float, volatility: float) -> float:
        """Calculate expected return based on probability and volatility"""
        # Base expected return for positive prediction
        base_return = 0.06  # 6% target

        # Adjust for probability
        prob_adjusted = base_return * (probability - 0.5) * 2

        # Adjust for volatility (higher vol = higher potential return but also risk)
        vol_adjusted = prob_adjusted * (1 + volatility * 10)

        return vol_adjusted

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate aggregated feature importance"""
        # Get importance from tree models
        importance_dfs = []

        for model_name in ['xgboost', 'lightgbm', 'rf']:
            if model_name in self.feature_importance:
                imp = self.feature_importance[model_name]
                importance_dfs.append(imp)

        if importance_dfs:
            # Average importance across models
            avg_importance = pd.concat(importance_dfs, axis=1).mean(axis=1)
            self.feature_importance['ensemble'] = avg_importance.sort_values(ascending=False)

            # Log top features
            logger.info("Top 20 features:")
            for feat, imp in self.feature_importance['ensemble'].head(20).items():
                logger.info(f"  {feat}: {imp:.4f}")

    def evaluate(self, test_data: Dict[str, pd.DataFrame]) -> Dict:
        """Evaluate model performance on test data"""
        all_predictions = []
        all_targets = []
        all_probabilities = []

        for symbol, df in test_data.items():
            # Create features
            features = self.feature_engineer.create_features(df, symbol)

            if features.empty:
                continue

            # Create target
            forward_return = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            target = (forward_return > 0.03).astype(int)

            # Get predictions for each valid point
            for i in range(self.sequence_length, len(features) - self.prediction_horizon):
                window_data = df.iloc[:i + 1]
                pred_result = self.predict(symbol, window_data)

                if pred_result['confidence'] > 0:
                    all_predictions.append(pred_result['prediction'])
                    all_probabilities.append(pred_result['probability'])
                    all_targets.append(target.iloc[i])

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions),
            'recall': recall_score(all_targets, all_predictions),
            'f1': f1_score(all_targets, all_predictions),
            'auc': roc_auc_score(all_targets, all_probabilities),
            'mcc': matthews_corrcoef(all_targets, all_predictions),
            'log_loss': log_loss(all_targets, all_probabilities),
            'total_predictions': len(all_predictions),
            'positive_rate': np.mean(all_predictions),
            'actual_positive_rate': np.mean(all_targets)
        }

        logger.info("Model evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def save_models(self, path: str):
        """Save all models to disk"""
        import os
        os.makedirs(path, exist_ok=True)

        # Save PyTorch LSTM
        if 'lstm' in self.models:
            torch.save(
                self.models['lstm'].state_dict(),
                os.path.join(path, 'lstm_model.pth')
            )

        # Save tree models
        for model_name in ['xgboost', 'lightgbm', 'rf']:
            if model_name in self.models:
                joblib.dump(
                    self.models[model_name],
                    os.path.join(path, f'{model_name}_model.pkl')
                )

        # Save feature scaler
        joblib.dump(
            self.feature_engineer.scaler,
            os.path.join(path, 'feature_scaler.pkl')
        )

        # Save feature importance
        pd.to_pickle(
            self.feature_importance,
            os.path.join(path, 'feature_importance.pkl')
        )

        # Save model weights
        joblib.dump(
            self.model_weights,
            os.path.join(path, 'model_weights.pkl')
        )

        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load models from disk"""
        import os

        # Load PyTorch LSTM
        lstm_path = os.path.join(path, 'lstm_model.pth')
        if os.path.exists(lstm_path):
            # Need to recreate model architecture first
            dummy_shape = (1, self.sequence_length, 100)  # Dummy shape
            model = self._create_lstm_model(dummy_shape)
            model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            model.eval()
            self.models['lstm'] = model

        # Load tree models
        for model_name in ['xgboost', 'lightgbm', 'rf']:
            model_path = os.path.join(path, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)

        # Load feature scaler
        scaler_path = os.path.join(path, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            self.feature_engineer.scaler = joblib.load(scaler_path)

        # Load feature importance
        importance_path = os.path.join(path, 'feature_importance.pkl')
        if os.path.exists(importance_path):
            self.feature_importance = pd.read_pickle(importance_path)

        # Load model weights
        weights_path = os.path.join(path, 'model_weights.pkl')
        if os.path.exists(weights_path):
            self.model_weights = joblib.load(weights_path)

        self.is_trained = True
        logger.info(f"Models loaded from {path}")

    def update_model_weights(self, performance_metrics: Dict[str, float]):
        """Dynamically update model weights based on recent performance"""
        # Calculate new weights based on model performance
        total_score = sum(performance_metrics.values())

        if total_score > 0:
            for model_name, score in performance_metrics.items():
                if model_name in self.model_weights:
                    # Blend old and new weights
                    old_weight = self.model_weights[model_name]
                    new_weight = score / total_score
                    self.model_weights[model_name] = 0.7 * old_weight + 0.3 * new_weight

            # Normalize weights
            total_weight = sum(self.model_weights.values())
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight

            logger.info(f"Updated model weights: {self.model_weights}")