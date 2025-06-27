# models/enhanced_ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from datetime import datetime
import warnings

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, log_loss
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# GPU acceleration
try:
    import cupy as cp
    import cudf
    from cuml.ensemble import RandomForestClassifier as cuRF

    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

# Advanced ML
from pytorch_lightning import LightningModule, Trainer
import optuna

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """Transformer encoder for time series prediction"""

    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.layer_norm(x)

        # Self-attention
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output network
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EnhancedLSTM(nn.Module):
    """Advanced LSTM with attention and residual connections"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super(EnhancedLSTM, self).__init__()

        # Multi-layer LSTM with skip connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.lstm_layers.append(
                nn.LSTM(prev_dim, hidden_dim, batch_first=True,
                        bidirectional=True, num_layers=2)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim * 2))
            self.dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim * 2

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1] * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Output layers with skip connection
        self.fc1 = nn.Linear(hidden_dims[-1] * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Store for skip connections
        skip_connections = []

        # Multi-layer LSTM processing
        for i, (lstm, ln, dropout) in enumerate(
                zip(self.lstm_layers, self.layer_norms, self.dropouts)
        ):
            lstm_out, _ = lstm(x)
            lstm_out = ln(lstm_out)
            lstm_out = dropout(lstm_out)

            # Residual connection for layers after the first
            if i > 0 and lstm_out.shape[-1] == x.shape[-1]:
                lstm_out = lstm_out + x

            skip_connections.append(lstm_out)
            x = lstm_out

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = attn_out + x  # Residual connection

        # Take the last output
        x = x[:, -1, :]

        # Output network
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))

        return x


class GPUAcceleratedEnsemble:
    """Advanced ensemble with GPU acceleration and AutoML"""

    def __init__(self, use_gpu=True, enable_automl=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_automl = enable_automl
        self.models = {}
        self.is_trained = False
        self.feature_importance = {}

        # Device setup
        if self.use_gpu:
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

            # Initialize RAPIDS
            self.use_rapids = RAPIDS_AVAILABLE
            if self.use_rapids:
                logger.info("RAPIDS acceleration enabled")
        else:
            self.device = torch.device('cpu')
            self.use_rapids = False

        # Model weights (will be optimized)
        self.model_weights = {
            'transformer': 0.35,
            'enhanced_lstm': 0.25,
            'xgboost': 0.20,
            'lightgbm': 0.15,
            'catboost': 0.05
        }

        # AutoML components
        if self.enable_automl:
            self.optuna_study = None
            self.best_params = {}

        # Mixed precision training
        self.scaler = GradScaler() if self.use_gpu else None

        # Feature engineering
        from models.enhanced_features import EnhancedFeatureEngineer
        self.feature_engineer = EnhancedFeatureEngineer(use_gpu=self.use_gpu)

    def train_with_automl(self, train_data: Dict[str, pd.DataFrame],
                          val_data: Dict[str, pd.DataFrame] = None,
                          n_trials: int = 100):
        """Train models with Optuna hyperparameter optimization"""

        if not self.enable_automl:
            return self.train(train_data, val_data)

        logger.info("Starting AutoML training with Optuna...")

        # Prepare data
        X_train, y_train, X_val, y_val = self._prepare_training_data(
            train_data, val_data
        )

        # Create Optuna study
        self.optuna_study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )

        # Optimize hyperparameters
        self.optuna_study.optimize(
            lambda trial: self._optuna_objective(
                trial, X_train, y_train, X_val, y_val
            ),
            n_trials=n_trials,
            n_jobs=1  # GPU doesn't benefit from parallel trials
        )

        # Train final models with best parameters
        self.best_params = self.optuna_study.best_params
        logger.info(f"Best parameters: {self.best_params}")

        # Train all models with optimized parameters
        self._train_final_models(X_train, y_train, X_val, y_val)

        self.is_trained = True

    def _optuna_objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for Optuna optimization"""

        # Suggest hyperparameters
        params = {
            'lstm_hidden_dims': [
                trial.suggest_int('lstm_hidden_1', 128, 512),
                trial.suggest_int('lstm_hidden_2', 64, 256),
                trial.suggest_int('lstm_hidden_3', 32, 128)
            ],
            'lstm_dropout': trial.suggest_float('lstm_dropout', 0.1, 0.5),
            'transformer_hidden': trial.suggest_int('transformer_hidden', 128, 512),
            'transformer_heads': trial.suggest_int('transformer_heads', 4, 16),
            'transformer_layers': trial.suggest_int('transformer_layers', 2, 8),
            'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'xgb_learning_rate': trial.suggest_float('xgb_lr', 0.001, 0.3, log=True),
            'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'lgb_num_leaves': trial.suggest_int('lgb_leaves', 15, 255),
            'lgb_learning_rate': trial.suggest_float('lgb_lr', 0.001, 0.3, log=True),
            'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000)
        }

        # Train small version of ensemble
        scores = []

        # Train LSTM
        lstm_score = self._train_trial_lstm(
            X_train, y_train, X_val, y_val,
            params['lstm_hidden_dims'], params['lstm_dropout']
        )
        scores.append(lstm_score)

        # Train XGBoost
        xgb_score = self._train_trial_xgboost(
            X_train, y_train, X_val, y_val,
            params['xgb_max_depth'], params['xgb_learning_rate'],
            params['xgb_n_estimators']
        )
        scores.append(xgb_score)

        # Return average score
        return np.mean(scores)

    def _prepare_training_data(self, train_data, val_data):
        """Prepare and preprocess training data"""

        X_train_list = []
        y_train_list = []

        for symbol, df in train_data.items():
            if len(df) < 200:
                continue

            # Create advanced features
            features = self.feature_engineer.create_all_features(df, symbol)

            if features.empty:
                continue

            # Create targets
            target = self._create_advanced_target(df)

            # Remove NaN
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_idx]
            target = target[valid_idx]

            X_train_list.append(features)
            y_train_list.append(target)

        # Combine and scale
        X_train = pd.concat(X_train_list)
        y_train = pd.concat(y_train_list)

        # Scale features
        X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)

        # Prepare validation data similarly
        X_val, y_val = None, None
        if val_data:
            X_val_list = []
            y_val_list = []

            for symbol, df in val_data.items():
                if len(df) < 200:
                    continue

                features = self.feature_engineer.create_all_features(df, symbol)
                if features.empty:
                    continue

                target = self._create_advanced_target(df)

                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                features = features[valid_idx]
                target = target[valid_idx]

                X_val_list.append(features)
                y_val_list.append(target)

            if X_val_list:
                X_val = pd.concat(X_val_list)
                y_val = pd.concat(y_val_list)
                X_val = self.feature_engineer.scale_features(X_val, fit=False)

        return X_train_scaled, y_train, X_val, y_val

    def _create_advanced_target(self, df):
        """Create sophisticated target variable"""

        # Multi-timeframe targets
        returns_3d = df['close'].pct_change(3).shift(-3)
        returns_5d = df['close'].pct_change(5).shift(-5)

        # Risk-adjusted returns (Sharpe-like)
        volatility = df['close'].pct_change().rolling(20).std()
        risk_adjusted_return = returns_3d / (volatility + 1e-6)

        # Transaction costs
        transaction_cost = 0.002  # 0.2% round trip
        min_return = 0.03  # 3% minimum

        # Target: Profitable trade after costs with good risk/reward
        target = (
                (returns_3d > min_return + transaction_cost) &
                (risk_adjusted_return > 2) &  # Sharpe > 2
                (returns_5d > min_return)  # Sustained move
        ).astype(int)

        return target

    def _train_final_models(self, X_train, y_train, X_val, y_val):
        """Train all models with optimized parameters"""

        # Train Transformer
        logger.info("Training Transformer...")
        self._train_transformer(X_train, y_train, X_val, y_val)

        # Train Enhanced LSTM
        logger.info("Training Enhanced LSTM...")
        self._train_enhanced_lstm(X_train, y_train, X_val, y_val)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self._train_xgboost(X_train, y_train, X_val, y_val)

        # Train LightGBM
        logger.info("Training LightGBM...")
        self._train_lightgbm(X_train, y_train, X_val, y_val)

        # Train CatBoost
        logger.info("Training CatBoost...")
        self._train_catboost(X_train, y_train, X_val, y_val)

        # Calculate feature importance
        self._calculate_feature_importance(X_train, y_train)

    def _train_transformer(self, X, y, X_val=None, y_val=None):
        """Train Transformer model"""

        # Prepare sequences
        X_seq, y_seq = self._prepare_lstm_data(X, y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create model
        input_dim = X_seq.shape[-1]
        model = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=self.best_params.get('transformer_hidden', 256),
            num_heads=self.best_params.get('transformer_heads', 8),
            num_layers=self.best_params.get('transformer_layers', 4)
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Training loop
        model.train()
        batch_size = 32
        n_epochs = 100
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0

            # Shuffle data
            indices = torch.randperm(len(X_tensor))

            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices].unsqueeze(1)

                optimizer.zero_grad()

                # Mixed precision training
                if self.use_gpu and self.scaler:
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            scheduler.step()

            # Validation
            if X_val is not None and epoch % 5 == 0:
                val_loss = self._evaluate_model(model, X_val, y_val, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                logger.info(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # Load best model
        if X_val is not None and 'best_model_state' in locals():
            model.load_state_dict(best_model_state)

        self.models['transformer'] = model

    def _train_enhanced_lstm(self, X, y, X_val=None, y_val=None):
        """Train Enhanced LSTM model"""

        # Similar to transformer training but with EnhancedLSTM
        X_seq, y_seq = self._prepare_lstm_data(X, y)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        model = EnhancedLSTM(
            input_dim=X_seq.shape[-1],
            hidden_dims=self.best_params.get('lstm_hidden_dims', [256, 128, 64]),
            dropout=self.best_params.get('lstm_dropout', 0.2)
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop (similar to transformer)
        model.train()
        batch_size = 32
        n_epochs = 100

        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0

            indices = torch.randperm(len(X_tensor))

            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices].unsqueeze(1)

                optimizer.zero_grad()

                if self.use_gpu and self.scaler:
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)

            if epoch % 10 == 0:
                logger.info(f"LSTM Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.models['enhanced_lstm'] = model

    def _train_xgboost(self, X, y, X_val=None, y_val=None):
        """Train XGBoost model with GPU support"""

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'n_estimators': self.best_params.get('xgb_n_estimators', 500),
            'max_depth': self.best_params.get('xgb_max_depth', 6),
            'learning_rate': self.best_params.get('xgb_lr', 0.01),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1,
            'scale_pos_weight': len(y[y == 0]) / len(y[y == 1])  # Handle imbalanced data
        }

        if self.use_gpu:
            params['gpu_id'] = 0

        model = xgb.XGBClassifier(**params)

        # Prepare validation set
        eval_set = [(X, y)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        # Train with early stopping
        model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )

        self.models['xgboost'] = model

        # Store feature importance
        self.feature_importance['xgboost'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    def _train_lightgbm(self, X, y, X_val=None, y_val=None):
        """Train LightGBM model with GPU support"""

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'num_leaves': self.best_params.get('lgb_leaves', 31),
            'learning_rate': self.best_params.get('lgb_lr', 0.01),
            'n_estimators': self.best_params.get('lgb_n_estimators', 500),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1,
            'class_weight': 'balanced'
        }

        if self.use_gpu:
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0

        model = lgb.LGBMClassifier(**params)

        # Prepare validation set
        eval_set = [(X, y)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        # Train with early stopping
        model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        self.models['lightgbm'] = model

        # Store feature importance
        self.feature_importance['lightgbm'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    def _train_catboost(self, X, y, X_val=None, y_val=None):
        """Train CatBoost model with GPU support"""

        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'task_type': 'GPU' if self.use_gpu else 'CPU',
            'learning_rate': 0.03,
            'iterations': 1000,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'class_weights': [1, 3]  # Handle imbalanced data
        }

        if self.use_gpu:
            params['gpu_ram_part'] = 0.5

        model = cb.CatBoostClassifier(**params)

        # Prepare validation pool
        eval_set = None
        if X_val is not None:
            eval_set = cb.Pool(X_val, y_val)

        # Train
        model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )

        self.models['catboost'] = model

        # Store feature importance
        self.feature_importance['catboost'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    def _prepare_lstm_data(self, features: pd.DataFrame, target: pd.Series = None) -> Tuple:
        """Prepare data for LSTM/Transformer (sequences)"""
        sequence_length = 60
        sequences = []
        targets = [] if target is not None else None

        for i in range(sequence_length, len(features)):
            seq = features.iloc[i - sequence_length:i].values
            sequences.append(seq)

            if target is not None:
                targets.append(target.iloc[i])

        sequences = np.array(sequences)

        if target is not None:
            targets = np.array(targets)
            return sequences, targets
        else:
            return sequences

    def predict_batch_gpu(self, symbols: List[str],
                          market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Batch prediction with GPU acceleration"""

        predictions = []

        # Prepare batch data
        batch_features = []
        valid_symbols = []

        for symbol in symbols:
            if symbol not in market_data:
                continue

            features = self.feature_engineer.create_all_features(
                market_data[symbol], symbol
            )

            if not features.empty and len(features) >= 60:
                batch_features.append(features)
                valid_symbols.append(symbol)

        if not batch_features:
            return []

        # GPU batch processing
        with torch.cuda.device(self.device) if self.use_gpu else torch.no_grad():
            # Scale features
            scaled_features = []
            for features in batch_features:
                scaled = self.feature_engineer.scale_features(features, fit=False)
                scaled_features.append(scaled)

            # Get predictions from all models
            for i, (symbol, features) in enumerate(zip(valid_symbols, scaled_features)):
                pred_result = self._get_ensemble_prediction(features, symbol)
                pred_result['symbol'] = symbol
                predictions.append(pred_result)

        return predictions

    def _get_ensemble_prediction(self, features: pd.DataFrame, symbol: str) -> Dict:
        """Get prediction from ensemble"""

        predictions = {}
        probabilities = {}

        # Neural network predictions
        if 'transformer' in self.models:
            trans_pred, trans_prob = self._predict_neural_model(
                self.models['transformer'], features
            )
            predictions['transformer'] = trans_pred
            probabilities['transformer'] = trans_prob

        if 'enhanced_lstm' in self.models:
            lstm_pred, lstm_prob = self._predict_neural_model(
                self.models['enhanced_lstm'], features
            )
            predictions['enhanced_lstm'] = lstm_pred
            probabilities['enhanced_lstm'] = lstm_prob

        # Tree model predictions
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            if model_name in self.models:
                pred, prob = self._predict_tree_model(
                    self.models[model_name], features
                )
                predictions[model_name] = pred
                probabilities[model_name] = prob

        # Weighted ensemble
        ensemble_prob = 0
        total_weight = 0

        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 0.1)
            ensemble_prob += prob * weight
            total_weight += weight

        ensemble_prob /= total_weight
        ensemble_pred = 1 if ensemble_prob > 0.6 else 0

        # Calculate confidence
        model_preds = list(predictions.values())
        agreement = sum(model_preds) / len(model_preds) if model_preds else 0
        confidence = ensemble_prob * (0.5 + 0.5 * abs(agreement - 0.5) * 2)

        # Get latest market data
        current_price = features.iloc[-1].get('close', 0)
        volatility = features.iloc[-1].get('volatility_20d', 0)
        volume_ratio = features.iloc[-1].get('volume_ratio_20', 1)

        return {
            'prediction': ensemble_pred,
            'probability': ensemble_prob,
            'confidence': confidence,
            'model_predictions': predictions,
            'model_probabilities': probabilities,
            'current_price': current_price,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'expected_return': self._calculate_expected_return(ensemble_prob, volatility),
            'timestamp': datetime.now()
        }

    def _predict_neural_model(self, model, features: pd.DataFrame) -> Tuple[int, float]:
        """Get prediction from neural network model"""
        model.eval()

        # Prepare sequence
        X_seq = self._prepare_lstm_data(features)

        if len(X_seq) == 0:
            return 0, 0.5

        # Take only the last sequence
        last_seq = X_seq[-1:]

        # Convert to tensor
        X_tensor = torch.FloatTensor(last_seq).to(self.device)

        # Predict
        with torch.no_grad():
            output = model(X_tensor)
            prob = output.cpu().numpy()[0, 0]

        pred = 1 if prob > 0.5 else 0

        return pred, float(prob)

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

        # Adjust for volatility
        vol_adjusted = prob_adjusted * (1 + min(volatility * 5, 0.5))

        return vol_adjusted

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate aggregated feature importance"""
        # Get importance from tree models
        importance_dfs = []

        for model_name in ['xgboost', 'lightgbm', 'catboost']:
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

    def explain_prediction(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Generate explainable AI insights for prediction"""

        # Get prediction
        features = self.feature_engineer.create_all_features(price_data, symbol)
        features_scaled = self.feature_engineer.scale_features(features, fit=False)

        prediction = self._get_ensemble_prediction(features_scaled, symbol)

        # SHAP-like analysis for tree models
        explanations = {}

        # For tree models, use built-in importance
        if 'xgboost' in self.models:
            # Get feature contributions
            model = self.models['xgboost']
            X = features_scaled.iloc[-1:].values

            # Get SHAP values (approximation)
            booster = model.get_booster()
            leaf_values = booster.predict(xgb.DMatrix(X), pred_leaf=True)

            # Calculate feature importance for this prediction
            feature_contribs = self._calculate_feature_contributions(
                model, X, features_scaled.columns
            )
            explanations['xgboost_contributions'] = feature_contribs

        # For neural networks, use gradient-based attribution
        if 'enhanced_lstm' in self.models:
            lstm_attribution = self._get_lstm_attribution(features_scaled)
            explanations['lstm_attribution'] = lstm_attribution

        # Aggregate explanations
        top_factors = self._aggregate_explanations(explanations, features_scaled)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(features_scaled)

        return {
            'prediction': prediction,
            'top_positive_factors': top_factors['positive'][:5],
            'top_negative_factors': top_factors['negative'][:5],
            'confidence_breakdown': self._get_confidence_breakdown(prediction),
            'risk_factors': risk_factors,
            'feature_values': self._get_key_feature_values(features_scaled)
        }

    def _calculate_feature_contributions(self, model, X, feature_names):
        """Calculate feature contributions for a single prediction"""
        # Simplified SHAP-like calculation
        base_pred = model.predict_proba(X)[0, 1]
        contributions = {}

        # Perturb each feature and measure impact
        for i, feature in enumerate(feature_names[:50]):  # Top 50 features only
            X_perturbed = X.copy()
            X_perturbed[0, i] = 0  # Set to mean (scaled data)

            perturbed_pred = model.predict_proba(X_perturbed)[0, 1]
            contribution = base_pred - perturbed_pred
            contributions[feature] = contribution

        return contributions

    def _get_lstm_attribution(self, features_scaled):
        """Get LSTM attribution using integrated gradients"""
        if 'enhanced_lstm' not in self.models:
            return {}

        model = self.models['enhanced_lstm']
        model.eval()

        # Prepare sequence
        X_seq = self._prepare_lstm_data(features_scaled)
        if len(X_seq) == 0:
            return {}

        last_seq = X_seq[-1:]
        X_tensor = torch.FloatTensor(last_seq).to(self.device).requires_grad_(True)

        # Get gradients
        output = model(X_tensor)
        output.backward()

        # Get gradient magnitudes
        gradients = X_tensor.grad.cpu().numpy()[0]

        # Average over sequence length
        feature_importance = np.abs(gradients).mean(axis=0)

        # Create attribution dict
        attribution = {}
        for i, importance in enumerate(feature_importance[:50]):
            if i < len(features_scaled.columns):
                attribution[features_scaled.columns[i]] = importance

        return attribution

    def _aggregate_explanations(self, explanations, features):
        """Aggregate explanations from different models"""
        all_contributions = {}

        # Combine all contributions
        for model_explanations in explanations.values():
            for feature, value in model_explanations.items():
                if feature not in all_contributions:
                    all_contributions[feature] = []
                all_contributions[feature].append(value)

        # Average contributions
        avg_contributions = {
            feature: np.mean(values)
            for feature, values in all_contributions.items()
        }

        # Sort by absolute contribution
        sorted_features = sorted(
            avg_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Separate positive and negative
        positive_factors = [
            {'name': feat, 'value': val, 'current': features.iloc[-1].get(feat, 0)}
            for feat, val in sorted_features if val > 0
        ]

        negative_factors = [
            {'name': feat, 'value': val, 'current': features.iloc[-1].get(feat, 0)}
            for feat, val in sorted_features if val < 0
        ]

        return {
            'positive': positive_factors,
            'negative': negative_factors
        }

    def _get_confidence_breakdown(self, prediction):
        """Break down confidence by model type"""
        model_probs = prediction['model_probabilities']

        # Group by model type
        neural_models = ['transformer', 'enhanced_lstm']
        tree_models = ['xgboost', 'lightgbm', 'catboost']

        neural_conf = np.mean([
            model_probs.get(m, 0.5) for m in neural_models if m in model_probs
        ])

        tree_conf = np.mean([
            model_probs.get(m, 0.5) for m in tree_models if m in model_probs
        ])

        return {
            'neural_networks': neural_conf,
            'tree_models': tree_conf,
            'ensemble': prediction['probability']
        }

    def _identify_risk_factors(self, features):
        """Identify potential risk factors in current market conditions"""
        risk_factors = []

        latest = features.iloc[-1]

        # Check volatility
        if latest.get('volatility_20d', 0) > 0.025:
            risk_factors.append({
                'factor': 'High Volatility',
                'value': latest['volatility_20d'],
                'threshold': 0.025,
                'severity': 'medium'
            })

        # Check RSI extremes
        if latest.get('rsi_14', 50) > 70:
            risk_factors.append({
                'factor': 'Overbought (RSI)',
                'value': latest['rsi_14'],
                'threshold': 70,
                'severity': 'medium'
            })
        elif latest.get('rsi_14', 50) < 30:
            risk_factors.append({
                'factor': 'Oversold (RSI)',
                'value': latest['rsi_14'],
                'threshold': 30,
                'severity': 'low'
            })

        # Check volume
        if latest.get('volume_ratio_20', 1) < 0.5:
            risk_factors.append({
                'factor': 'Low Volume',
                'value': latest['volume_ratio_20'],
                'threshold': 0.5,
                'severity': 'medium'
            })

        # Check trend
        if latest.get('price_to_sma_200', 1) < 0.95:
            risk_factors.append({
                'factor': 'Below Long-term Trend',
                'value': latest['price_to_sma_200'],
                'threshold': 0.95,
                'severity': 'high'
            })

        return risk_factors

    def _get_key_feature_values(self, features):
        """Get current values of key features"""
        latest = features.iloc[-1]

        key_features = {
            'Price Momentum': {
                'return_5d': latest.get('return_5d', 0),
                'return_20d': latest.get('return_20d', 0),
                'price_to_sma_20': latest.get('price_to_sma_20', 1),
                'price_to_sma_50': latest.get('price_to_sma_50', 1)
            },
            'Technical Indicators': {
                'rsi_14': latest.get('rsi_14', 50),
                'macd': latest.get('macd', 0),
                'bb_position_20': latest.get('bb_position_20', 0.5),
                'stoch_k': latest.get('stoch_k', 50)
            },
            'Volume Profile': {
                'volume_ratio_20': latest.get('volume_ratio_20', 1),
                'obv_signal': latest.get('obv_signal', 0),
                'money_flow_ratio': latest.get('money_flow_ratio', 1)
            },
            'Market Structure': {
                'volatility_20d': latest.get('volatility_20d', 0),
                'atr_pct_14': latest.get('atr_pct_14', 0),
                'trend_quality': latest.get('trend_quality', 0),
                'efficiency_ratio': latest.get('efficiency_ratio', 0)
            }
        }

        return key_features

    def save_models(self, path: str):
        """Save all models to disk"""
        import os
        os.makedirs(path, exist_ok=True)

        # Save neural network models
        for model_name in ['transformer', 'enhanced_lstm']:
            if model_name in self.models:
                torch.save(
                    self.models[model_name].state_dict(),
                    os.path.join(path, f'{model_name}_model.pth')
                )

        # Save tree models
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
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

        # Save model weights and parameters
        config = {
            'model_weights': self.model_weights,
            'best_params': self.best_params if hasattr(self, 'best_params') else {}
        }
        joblib.dump(config, os.path.join(path, 'model_config.pkl'))

        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load models from disk"""
        import os

        # Load configuration
        config_path = os.path.join(path, 'model_config.pkl')
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.model_weights = config['model_weights']
            self.best_params = config['best_params']

        # Load neural network models
        for model_name, model_class in [
            ('transformer', TransformerEncoder),
            ('enhanced_lstm', EnhancedLSTM)
        ]:
            model_path = os.path.join(path, f'{model_name}_model.pth')
            if os.path.exists(model_path):
                # Need to recreate model architecture
                if model_name == 'transformer':
                    model = TransformerEncoder(
                        input_dim=200,  # Will be set properly on first use
                        hidden_dim=self.best_params.get('transformer_hidden', 256),
                        num_heads=self.best_params.get('transformer_heads', 8),
                        num_layers=self.best_params.get('transformer_layers', 4)
                    ).to(self.device)
                else:
                    model = EnhancedLSTM(
                        input_dim=200,
                        hidden_dims=self.best_params.get('lstm_hidden_dims', [256, 128, 64]),
                        dropout=self.best_params.get('lstm_dropout', 0.2)
                    ).to(self.device)

                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                self.models[model_name] = model

        # Load tree models
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
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

        self.is_trained = True
        logger.info(f"Models loaded from {path}")