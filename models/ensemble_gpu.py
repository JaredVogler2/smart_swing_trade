# models/ensemble_gpu.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from datetime import datetime
import warnings

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, log_loss
)
import xgboost as xgb
import lightgbm as lgb

# Deep Learning with GPU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

# GPU acceleration
import cupy as cp
import cudf
from numba import cuda
import rapids_singlecell as rsc

# Custom modules
from models.features import FeatureEngineer
from config.settings import Config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AttentionLSTM(nn.Module):
    """Enhanced LSTM with multi-head attention and residual connections"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64],
                 num_heads=8, dropout=0.3):
        super(AttentionLSTM, self).__init__()

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Bidirectional LSTM layers with layer normalization
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0],
                             batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden_dims[0] * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_dims[0] * 2, hidden_dims[1],
                             batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(hidden_dims[1] * 2)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_dims[1] * 2, hidden_dims[2],
                             batch_first=True)
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout)

        # Multi-head attention with position encoding
        self.position_encoding = PositionalEncoding(hidden_dims[2])
        self.attention = nn.MultiheadAttention(
            hidden_dims[2], num_heads=num_heads, dropout=dropout
        )

        # Transformer encoder for deeper context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims[2], nhead=num_heads,
            dim_feedforward=hidden_dims[2] * 4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layers with skip connections
        self.fc1 = nn.Linear(hidden_dims[2], 128)
        self.ln4 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.ln5 = nn.LayerNorm(64)
        self.dropout5 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layers with residual connections
        out1, _ = self.lstm1(x)
        out1 = self.ln1(out1)
        out1 = self.dropout1(out1)

        out2, _ = self.lstm2(out1)
        out2 = self.ln2(out2)
        out2 = self.dropout2(out2)

        out3, _ = self.lstm3(out2)
        out3 = self.ln3(out3)
        out3 = self.dropout3(out3)

        # Add positional encoding
        out3 = self.position_encoding(out3)

        # Self-attention
        out3_t = out3.transpose(0, 1)
        attn_out, attn_weights = self.attention(out3_t, out3_t, out3_t)
        attn_out = attn_out.transpose(0, 1)

        # Transformer encoder
        trans_out = self.transformer(attn_out.transpose(0, 1))
        trans_out = trans_out.transpose(0, 1)

        # Take the last output with skip connection
        final_out = trans_out[:, -1, :] + out3[:, -1, :]

        # Output layers
        out = self.fc1(final_out)
        out = self.ln4(out)
        out = self.relu(out)
        out = self.dropout4(out)

        out = self.fc2(out)
        out = self.ln5(out)
        out = self.relu(out)
        out = self.dropout5(out)

        out = self.fc3(out)
        out = self.sigmoid(out)

        return out, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for attention mechanism"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class GRUModel(nn.Module):
    """GRU model for ensemble diversity"""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(GRUModel, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dims[0],
                           batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)

        self.gru2 = nn.GRU(hidden_dims[0] * 2, hidden_dims[1],
                           batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dims[1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)

        out, _ = self.gru2(out)
        out = self.dropout2(out)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out


class Conv1DModel(nn.Module):
    """1D CNN for pattern recognition"""

    def __init__(self, input_dim, sequence_length):
        super(Conv1DModel, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape for conv1d: (batch, features, sequence)
        x = x.transpose(1, 2)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class GPUEnsembleModel:
    """GPU-accelerated ensemble with advanced ML techniques"""

    def __init__(self, max_gpu_memory_mb=8192):
        self.max_gpu_memory = max_gpu_memory_mb
        self.models = {}
        self.feature_engineer = FeatureEngineer(enable_gpu=True)
        self.is_trained = False
        self.feature_importance = {}

        # Dynamic model weights based on performance
        self.model_weights = {
            'attention_lstm': 0.35,
            'gru': 0.10,
            'cnn': 0.10,
            'xgboost': 0.20,
            'lightgbm': 0.15,
            'catboost': 0.10
        }

        # Initialize GPU
        self._setup_gpu()

        # Model parameters
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.prediction_horizon = 3
        self.batch_size = 32

        # Training history for adaptive learning
        self.training_history = []

    def _setup_gpu(self):
        """Setup GPU with memory management"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(
                Config.GPU_MEMORY_FRACTION
            )

            # Enable TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # Initialize CuPy with memory pool
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            # Limit memory usage
            mempool.set_limit(size=self.max_gpu_memory * 1024 * 1024)

            logger.info(f"GPU initialized: {torch.cuda.get_device_name()}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        else:
            self.device = torch.device('cpu')
            logger.warning("GPU not available, using CPU")

    def _prepare_data_gpu(self, features: pd.DataFrame,
                          target: pd.Series = None) -> Tuple:
        """Prepare data on GPU using CuDF"""
        # Convert to CuDF DataFrame for GPU processing
        features_gpu = cudf.from_pandas(features)

        # Create sequences on GPU
        sequences = []
        targets = [] if target is not None else None

        for i in range(self.sequence_length, len(features_gpu)):
            seq = features_gpu.iloc[i - self.sequence_length:i].values
            sequences.append(seq)

            if target is not None:
                targets.append(target.iloc[i])

        # Convert to CuPy arrays
        sequences = cp.array(sequences)

        if target is not None:
            targets = cp.array(targets)
            return sequences, targets
        else:
            return sequences

    def train(self, train_data: Dict[str, pd.DataFrame],
              validation_data: Dict[str, pd.DataFrame] = None,
              use_multi_gpu: bool = False):
        """Train ensemble with GPU acceleration"""
        logger.info("Starting GPU-accelerated ensemble training...")

        # Prepare training data in parallel
        X_train_all = []
        y_train_all = []

        # Process symbols in batches for efficiency
        for symbol, df in train_data.items():
            if len(df) < self.sequence_length + 100:
                continue

            # Create features on GPU
            features = self.feature_engineer.create_features(df, symbol)

            if features.empty:
                continue

            # Create advanced targets
            target = self._create_advanced_target(df)

            # Remove last prediction_horizon rows
            features = features[:-self.prediction_horizon]
            target = target[:-self.prediction_horizon]

            # Remove NaN
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_idx]
            target = target[valid_idx]

            X_train_all.append(features)
            y_train_all.append(target)

        # Combine all data
        X_train = pd.concat(X_train_all)
        y_train = pd.concat(y_train_all)

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")

        # Scale features on GPU
        X_train_scaled = self._scale_features_gpu(X_train, fit=True)

        # Train each model with GPU acceleration
        self._train_deep_models(X_train_scaled, y_train, use_multi_gpu)
        self._train_tree_models_gpu(X_train_scaled, y_train)

        # Calculate feature importance on GPU
        self._calculate_feature_importance_gpu(X_train_scaled, y_train)

        # Optimize model weights based on validation
        if validation_data:
            self._optimize_weights(validation_data)

        self.is_trained = True
        logger.info("GPU ensemble training completed")

    def _create_advanced_target(self, df: pd.DataFrame) -> pd.Series:
        """Create sophisticated target variable"""
        # Multi-factor target considering:
        # 1. Price return
        # 2. Risk-adjusted return (Sharpe)
        # 3. Maximum adverse excursion

        close = df['close']
        high = df['high']
        low = df['low']

        # Forward returns
        forward_return = close.pct_change(self.prediction_horizon).shift(-self.prediction_horizon)

        # Forward volatility
        forward_vol = close.pct_change().rolling(self.prediction_horizon).std().shift(-self.prediction_horizon)

        # Risk-adjusted return
        sharpe = forward_return / (forward_vol + 1e-6)

        # Maximum adverse excursion (worst drawdown during holding period)
        mae = pd.Series(index=df.index, dtype=float)
        for i in range(len(df) - self.prediction_horizon):
            entry_price = close.iloc[i]
            period_low = low.iloc[i:i + self.prediction_horizon].min()
            mae.iloc[i] = (period_low - entry_price) / entry_price

        # Combined target: profitable trade with good risk/reward
        target = (
                (forward_return > 0.03) &  # 3% return
                (sharpe > 1.0) &  # Good risk-adjusted return
                (mae > -0.02)  # Max 2% drawdown
        ).astype(int)

        return target

    def _train_deep_models(self, X: pd.DataFrame, y: pd.Series,
                           use_multi_gpu: bool = False):
        """Train deep learning models with mixed precision"""
        logger.info("Training deep learning models on GPU...")

        # Prepare sequences
        X_seq, y_seq = self._prepare_lstm_data(X, y)

        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq)
        )

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, pin_memory=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=4
        )

        # Train Attention LSTM
        self._train_attention_lstm(train_loader, val_loader, X_seq.shape)

        # Train GRU
        self._train_gru(train_loader, val_loader, X_seq.shape)

        # Train CNN
        self._train_cnn(train_loader, val_loader, X_seq.shape)

    def _train_attention_lstm(self, train_loader, val_loader, input_shape):
        """Train attention LSTM with mixed precision"""
        model = AttentionLSTM(
            input_dim=input_shape[-1],
            hidden_dims=[256, 128, 64],
            num_heads=8,
            dropout=0.3
        ).to(self.device)

        # Use mixed precision training
        scaler = GradScaler()

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=50,
            steps_per_epoch=len(train_loader)
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs, _ = model(batch_X)
                    loss = criterion(outputs, batch_y)

                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)

                    with autocast():
                        outputs, _ = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.models['attention_lstm'] = model
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, "
                            f"Val Loss: {val_loss:.4f}")

    def _train_tree_models_gpu(self, X: pd.DataFrame, y: pd.Series):
        """Train tree models with GPU acceleration"""
        logger.info("Training tree models on GPU...")

        # Convert to GPU arrays for faster training
        X_gpu = cp.array(X.values)
        y_gpu = cp.array(y.values)

        # XGBoost with GPU
        xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': Config.GPU_DEVICE_ID,
            'max_depth': 8,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1,
            'scale_pos_weight': len(y[y == 0]) / len(y[y == 1])  # Handle imbalance
        }

        # Create DMatrix for efficiency
        dtrain = xgb.DMatrix(X_gpu, label=y_gpu)

        # Train with early stopping
        self.models['xgboost'] = xgb.train(
            xgb_params, dtrain,
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # LightGBM with GPU
        lgb_params = {
            'objective': 'binary',
            'device': 'gpu',
            'gpu_device_id': Config.GPU_DEVICE_ID,
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 0,  # Use all cores
            'force_col_wise': True,
            'histogram_pool_size': -1  # Unlimited
        }

        # Train LightGBM
        lgb_train = lgb.Dataset(X, label=y)
        self.models['lightgbm'] = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # CatBoost with GPU
        try:
            import catboost as cb

            cat_model = cb.CatBoostClassifier(
                iterations=1000,
                learning_rate=0.01,
                depth=8,
                task_type='GPU',
                devices=str(Config.GPU_DEVICE_ID),
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                early_stopping_rounds=50,
                verbose=False
            )

            cat_model.fit(X, y)
            self.models['catboost'] = cat_model

        except ImportError:
            logger.warning("CatBoost not available, skipping")

    def predict(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Generate prediction with GPU acceleration"""
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

        # Scale features on GPU
        features_scaled = self._scale_features_gpu(features, fit=False)

        # Get predictions from each model
        predictions = {}
        probabilities = {}
        attention_weights = None

        # Deep learning predictions
        if 'attention_lstm' in self.models:
            lstm_pred, lstm_prob, attn = self._predict_attention_lstm(features_scaled)
            predictions['attention_lstm'] = lstm_pred
            probabilities['attention_lstm'] = lstm_prob
            attention_weights = attn

        # Tree model predictions
        tree_preds = self._predict_tree_models_gpu(features_scaled)
        predictions.update(tree_preds['predictions'])
        probabilities.update(tree_preds['probabilities'])

        # Weighted ensemble prediction
        ensemble_prob = self._calculate_ensemble_prediction(probabilities)
        ensemble_pred = 1 if ensemble_prob > 0.65 else 0  # Higher threshold

        # Calculate advanced confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            predictions, probabilities, features_scaled
        )

        # Market context
        market_context = self._analyze_market_context(price_data)

        return {
            'symbol': symbol,
            'prediction': ensemble_pred,
            'probability': ensemble_prob,
            'confidence': confidence_metrics['overall_confidence'],
            'confidence_metrics': confidence_metrics,
            'model_predictions': predictions,
            'model_probabilities': probabilities,
            'attention_weights': attention_weights,
            'market_context': market_context,
            'expected_return': self._calculate_expected_return(
                ensemble_prob, market_context['volatility']
            ),
            'risk_score': self._calculate_risk_score(
                ensemble_prob, confidence_metrics, market_context
            ),
            'timestamp': datetime.now()
        }

    def _scale_features_gpu(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale features using GPU"""
        # Convert to CuDF for GPU processing
        features_gpu = cudf.from_pandas(features)

        if fit:
            # Fit scaler on GPU
            self.gpu_scaler_mean = features_gpu.mean()
            self.gpu_scaler_std = features_gpu.std()

        # Scale on GPU
        scaled_gpu = (features_gpu - self.gpu_scaler_mean) / (self.gpu_scaler_std + 1e-7)

        # Convert back to pandas
        return scaled_gpu.to_pandas()

    def _calculate_confidence_metrics(self, predictions: Dict,
                                      probabilities: Dict,
                                      features: pd.DataFrame) -> Dict:
        """Calculate advanced confidence metrics"""
        # Model agreement
        pred_values = list(predictions.values())
        agreement = np.std(pred_values)

        # Probability spread
        prob_values = list(probabilities.values())
        prob_spread = np.std(prob_values)

        # Feature quality score (based on missing values, outliers)
        feature_quality = 1.0 - (features.isna().sum().sum() / features.size)

        # Historical accuracy for similar patterns
        pattern_accuracy = self._get_pattern_accuracy(features)

        # Overall confidence
        overall_confidence = (
                0.3 * (1 - agreement) +  # Model agreement
                0.3 * (1 - prob_spread) +  # Probability consensus
                0.2 * feature_quality +  # Data quality
                0.2 * pattern_accuracy  # Historical accuracy
        )

        return {
            'overall_confidence': overall_confidence,
            'model_agreement': 1 - agreement,
            'probability_spread': prob_spread,
            'feature_quality': feature_quality,
            'pattern_accuracy': pattern_accuracy
        }

    def _analyze_market_context(self, price_data: pd.DataFrame) -> Dict:
        """Analyze current market context"""
        close = price_data['close']
        volume = price_data['volume']

        # Volatility regimes
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        vol_percentile = (volatility > close.pct_change().rolling(252).std()).sum() / 252

        # Trend strength
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        trend_strength = (close.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]

        # Volume profile
        volume_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

        # Market regime
        if trend_strength > 0.05 and volatility < 0.03:
            regime = 'strong_uptrend'
        elif trend_strength > 0.02:
            regime = 'uptrend'
        elif trend_strength < -0.05 and volatility < 0.03:
            regime = 'strong_downtrend'
        elif trend_strength < -0.02:
            regime = 'downtrend'
        else:
            regime = 'ranging'

        return {
            'volatility': volatility,
            'volatility_percentile': vol_percentile,
            'trend_strength': trend_strength,
            'volume_ratio': volume_ratio,
            'regime': regime,
            'current_price': close.iloc[-1],
            'distance_from_20ma': (close.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1],
            'distance_from_50ma': (close.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]
        }

    def _calculate_risk_score(self, probability: float,
                              confidence: Dict,
                              market_context: Dict) -> float:
        """Calculate comprehensive risk score"""
        # Base risk from probability
        base_risk = 1 - probability

        # Adjust for confidence
        confidence_adj = 1 - confidence['overall_confidence']

        # Market regime risk
        regime_risk = {
            'strong_uptrend': 0.2,
            'uptrend': 0.3,
            'ranging': 0.5,
            'downtrend': 0.7,
            'strong_downtrend': 0.8
        }.get(market_context['regime'], 0.5)

        # Volatility risk
        vol_risk = min(market_context['volatility'] * 10, 1.0)

        # Combined risk score
        risk_score = (
                0.4 * base_risk +
                0.2 * confidence_adj +
                0.2 * regime_risk +
                0.2 * vol_risk
        )

        return risk_score

    def save_models(self, path: str):
        """Save models with GPU state"""
        import os
        os.makedirs(path, exist_ok=True)

        # Save deep learning models
        for name in ['attention_lstm', 'gru', 'cnn']:
            if name in self.models:
                torch.save({
                    'model_state_dict': self.models[name].state_dict(),
                    'model_config': self.models[name].__dict__
                }, os.path.join(path, f'{name}_model.pth'))

        # Save tree models
        for name in ['xgboost', 'lightgbm', 'catboost']:
            if name in self.models:
                joblib.dump(
                    self.models[name],
                    os.path.join(path, f'{name}_model.pkl')
                )

        # Save scalers and configs
        joblib.dump({
            'gpu_scaler_mean': self.gpu_scaler_mean,
            'gpu_scaler_std': self.gpu_scaler_std,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }, os.path.join(path, 'model_config.pkl'))

        logger.info(f"Models saved to {path}")