# models/ensemble_gpu_windows.py
# GPU-accelerated ensemble model that works on Windows

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
    roc_auc_score, matthews_corrcoef, log_loss, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb

# Deep Learning with GPU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

# Custom modules
from models.enhanced_features import EnhancedFeatureEngineer as FeatureEngineer
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

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dims[2], num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Output layers with skip connections
        self.fc1 = nn.Linear(hidden_dims[2], 128)
        self.ln4 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.ln5 = nn.LayerNorm(64)
        self.dropout5 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 1)
        # No sigmoid - will use BCEWithLogitsLoss

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

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

        # Self-attention
        attn_out, attn_weights = self.attention(out3, out3, out3)

        # Take the last output with skip connection
        final_out = attn_out[:, -1, :] + out3[:, -1, :]

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
        # Return raw logits
        return out


class GPUEnsembleModel:
    """GPU-accelerated ensemble with advanced ML techniques"""

    def __init__(self, max_gpu_memory_mb=8192):
        self.max_gpu_memory = max_gpu_memory_mb
        self.models = {}
        self.feature_engineer = FeatureEngineer(use_gpu=False)  # Disable GPU for feature engineering
        self.is_trained = False
        self.feature_importance = {}
        self.optimal_threshold = 0.5  # Will be optimized during training

        # Dynamic model weights based on performance
        self.model_weights = {
            'attention_lstm': 0.35,
            'xgboost': 0.35,
            'lightgbm': 0.20,
            'rf': 0.10
        }

        # Initialize GPU
        self._setup_gpu()

        # Model parameters
        self.sequence_length = Config.SEQUENCE_LENGTH if hasattr(Config, 'SEQUENCE_LENGTH') else 20
        self.prediction_horizon = 3
        self.batch_size = 32

        # Training history for adaptive learning
        self.training_history = []

        # Scaler for features
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()

    def _setup_gpu(self):
        """Setup GPU with memory management"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

            # Set memory fraction
            if hasattr(Config, 'GPU_MEMORY_FRACTION'):
                torch.cuda.set_per_process_memory_fraction(
                    Config.GPU_MEMORY_FRACTION
                )

            # Enable TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            logger.info(f"GPU initialized: {torch.cuda.get_device_name()}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        else:
            self.device = torch.device('cpu')
            logger.warning("GPU not available, using CPU")

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

    def train(self, train_data: Dict[str, pd.DataFrame],
              validation_data: Dict[str, pd.DataFrame] = None,
              use_multi_gpu: bool = False):
        """Train ensemble with GPU acceleration"""
        logger.info("Starting GPU-accelerated ensemble training...")

        # Prepare training data
        X_train_all = []
        y_train_all = []

        # Process symbols
        print(f"DEBUG: Processing {len(train_data)} symbols for training")
        for symbol, df in train_data.items():
            print(f"DEBUG: Processing {symbol} with {len(df)} rows")

            if len(df) < self.sequence_length + 100:
                continue

            # Create features
            features = self.feature_engineer.create_all_features(df, symbol)

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

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        # Train each model with GPU acceleration
        self._train_deep_models(X_train_scaled, y_train)
        self._train_tree_models_gpu(X_train_scaled, y_train)

        # Calculate feature importance
        self._calculate_feature_importance(X_train_scaled, y_train)

        self.is_trained = True
        logger.info("GPU ensemble training completed")

    def _create_advanced_target(self, df: pd.DataFrame) -> pd.Series:
        """Create sophisticated target variable"""
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

    def _train_deep_models(self, X: pd.DataFrame, y: pd.Series):
        """Train deep learning models with mixed precision - keeping natural imbalance"""
        logger.info("Training deep learning models on GPU...")

        # Prepare sequences WITHOUT balancing - keep natural distribution
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

        # In _train_deep_models, before creating the DataLoader:
        print(f"X_seq shape: {X_seq.shape}")
        print(f"X_seq has NaN: {np.isnan(X_seq).any()}")
        print(f"X_seq has Inf: {np.isinf(X_seq).any()}")
        print(f"y_seq has NaN: {np.isnan(y_seq).any()}")

        # Check for extreme values
        print(f"X_seq stats - min: {X_seq.min()}, max: {X_seq.max()}, mean: {X_seq.mean()}")

        # Find which features have issues
        if np.isnan(X_seq).any():
            nan_features = np.where(np.isnan(X_seq).any(axis=(0, 1)))[0]
            print(f"Features with NaN: {nan_features}")

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, pin_memory=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=0
        )

        # Train Attention LSTM
        self._train_attention_lstm(train_loader, val_loader, X_seq.shape)

    def _train_attention_lstm(self, train_loader, val_loader, input_shape):
        """Train attention LSTM with proper handling of imbalanced data"""
        model = AttentionLSTM(
            input_dim=input_shape[-1],
            hidden_dims=[256, 128, 64],
            num_heads=8,
            dropout=0.3
        ).to(self.device)

        # Calculate class distribution for logging
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())

        pos_count = sum(all_labels)
        neg_count = len(all_labels) - pos_count
        pos_ratio = pos_count / len(all_labels)

        logger.info(f"Training class distribution - Negative: {neg_count} ({(1 - pos_ratio) * 100:.1f}%), "
                    f"Positive: {pos_count} ({pos_ratio * 100:.1f}%)")

        # Use class weights to handle imbalance
        # This makes the rare positive examples more important without changing the data distribution
        pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)

        logger.info(f"Using pos_weight: {pos_weight:.2f} to handle class imbalance")

        # Use mixed precision training
        scaler = GradScaler()

        # Loss function that handles imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Conservative learning rate and L2 regularization
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

        # Cosine annealing scheduler for stable training
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        # Training metrics
        best_val_f1 = 0  # Use F1 score instead of loss for imbalanced data
        patience = 15
        patience_counter = 0

        for epoch in range(100):  # More epochs with early stopping
            # Training
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                # Forward pass with autocast
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Add L2 regularization to prevent overfitting to majority class
                    l2_lambda = 0.01
                    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                    loss = loss + l2_lambda * l2_norm

                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at epoch {epoch}")
                    continue

                # Backward pass with gradient clipping
                scaler.scale(loss).backward()

                # Gradient clipping to prevent explosions
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # Collect predictions for metrics
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    train_preds.extend(probs.cpu().numpy())
                    train_labels.extend(batch_y.cpu().numpy())

            # Calculate training metrics
            train_preds = np.array(train_preds)
            train_labels = np.array(train_labels)

            # Use 0.5 threshold for now (can optimize later)
            train_preds_binary = (train_preds > 0.5).astype(int)

            # Calculate metrics that matter for imbalanced data
            train_precision = precision_score(train_labels, train_preds_binary, zero_division=0)
            train_recall = recall_score(train_labels, train_preds_binary, zero_division=0)
            train_f1 = f1_score(train_labels, train_preds_binary, zero_division=0)

            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)

                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    if not torch.isnan(loss):
                        val_loss += loss.item()

                        probs = torch.sigmoid(outputs)
                        val_preds.extend(probs.cpu().numpy())
                        val_labels.extend(batch_y.cpu().numpy())

            # Calculate validation metrics
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            val_preds_binary = (val_preds > 0.5).astype(int)

            val_precision = precision_score(val_labels, val_preds_binary, zero_division=0)
            val_recall = recall_score(val_labels, val_preds_binary, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

            # Also calculate AUC-ROC which is good for imbalanced data
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except:
                val_auc = 0.5

            # Update learning rate
            scheduler.step()

            # Save best model based on F1 score (balances precision and recall)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self.models['attention_lstm'] = model

                # Find optimal threshold
                precisions, recalls, thresholds = precision_recall_curve(val_labels, val_preds)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                best_threshold_idx = np.argmax(f1_scores)
                self.optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5

                logger.info(f"New best model! F1: {val_f1:.4f}, Optimal threshold: {self.optimal_threshold:.3f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Log progress every 5 epochs
            if epoch % 5 == 0:
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

                logger.info(f"Epoch {epoch}:")
                logger.info(f"  Train - Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, "
                            f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                logger.info(f"  Val   - Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, "
                            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")

        # Set default threshold if no model was saved
        if 'attention_lstm' not in self.models:
            logger.warning("No model met improvement criteria, using final model")
            self.models['attention_lstm'] = model
            self.optimal_threshold = 0.5

    def _train_tree_models_gpu(self, X: pd.DataFrame, y: pd.Series):
        """Train tree models with GPU acceleration"""
        logger.info("Training tree models on GPU...")

        # Split data for validation
        split_idx = int(0.9 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # XGBoost with GPU
        xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'max_depth': 8,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(
                y_train[y_train == 1]) > 0 else 1
        }

        self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        self.models['xgboost'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        # LightGBM with GPU
        lgb_params = {
            'objective': 'binary',
            'device': 'gpu',
            'gpu_device_id': 0,
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        self.models['lightgbm'] = lgb.LGBMClassifier(**lgb_params)
        self.models['lightgbm'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Random Forest (CPU only)
        from sklearn.ensemble import RandomForestClassifier
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        self.models['rf'].fit(X_train, y_train)

    def predict(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Generate prediction with GPU acceleration"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Create features
        features = self.feature_engineer.create_all_features(price_data, symbol)

        if features.empty or len(features) < self.sequence_length:
            return {
                'symbol': symbol,
                'prediction': 0,
                'confidence': 0,
                'error': 'Insufficient data'
            }

        # Scale features
        features_scaled = self.scaler.transform(features)
        features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

        # Get predictions from each model
        predictions = {}
        probabilities = {}

        # Deep learning predictions
        if 'attention_lstm' in self.models:
            lstm_pred, lstm_prob = self._predict_lstm(features_scaled)
            predictions['attention_lstm'] = lstm_pred
            probabilities['attention_lstm'] = lstm_prob

        # Tree model predictions
        for model_name in ['xgboost', 'lightgbm', 'rf']:
            if model_name in self.models:
                pred = self.models[model_name].predict(features_scaled.iloc[-1:])
                prob = self.models[model_name].predict_proba(features_scaled.iloc[-1:])[0, 1]
                predictions[model_name] = int(pred[0])
                probabilities[model_name] = float(prob)

        # Weighted ensemble prediction
        ensemble_prob = 0
        total_weight = 0
        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 0.1)
            ensemble_prob += prob * weight
            total_weight += weight

        ensemble_prob /= total_weight

        # Be more conservative with predictions for imbalanced data
        # Require higher confidence for positive predictions
        conservative_threshold = 0.7  # Instead of 0.5
        ensemble_pred = 1 if ensemble_prob > conservative_threshold else 0

        # Calculate confidence
        model_preds = list(predictions.values())
        agreement = sum(model_preds) / len(model_preds) if model_preds else 0
        confidence = ensemble_prob * (0.5 + 0.5 * abs(agreement - 0.5) * 2)

        # Market context
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
            'expected_return': self._calculate_expected_return(ensemble_prob, volatility),
            'timestamp': datetime.now()
        }

    def _predict_lstm(self, features: pd.DataFrame) -> Tuple[int, float]:
        """Get LSTM prediction with optimal threshold"""
        model = self.models['attention_lstm']
        model.eval()

        # Prepare sequence
        X_seq = self._prepare_lstm_data(features)

        # Take only the last sequence
        last_seq = X_seq[-1:] if len(X_seq) > 0 else X_seq

        # Convert to tensor
        X_tensor = torch.FloatTensor(last_seq).to(self.device)

        # Predict
        with torch.no_grad():
            with autocast():
                output = model(X_tensor)  # Raw logits
                # Apply sigmoid to get probability
                prob = torch.sigmoid(output).cpu().numpy()[0, 0]

        # Use optimal threshold if available, otherwise default to 0.5
        threshold = getattr(self, 'optimal_threshold', 0.5)
        pred = 1 if prob > threshold else 0

        return pred, prob

    def _calculate_expected_return(self, probability: float, volatility: float) -> float:
        """Calculate expected return based on probability and volatility"""
        base_return = 0.06  # 6% target
        prob_adjusted = base_return * (probability - 0.5) * 2
        vol_adjusted = prob_adjusted * (1 + volatility * 10)
        return vol_adjusted

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate feature importance from tree models"""
        importance_dict = {}

        # Get importance from tree models
        if 'xgboost' in self.models:
            importance_dict['xgboost'] = pd.Series(
                self.models['xgboost'].feature_importances_,
                index=X.columns
            )

        if 'lightgbm' in self.models:
            importance_dict['lightgbm'] = pd.Series(
                self.models['lightgbm'].feature_importances_,
                index=X.columns
            )

        if 'rf' in self.models:
            importance_dict['rf'] = pd.Series(
                self.models['rf'].feature_importances_,
                index=X.columns
            )

        # Average importance
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict)
            self.feature_importance = importance_df.mean(axis=1).sort_values(ascending=False)

            # Log top features
            logger.info("Top 20 features:")
            for feat, imp in self.feature_importance.head(20).items():
                logger.info(f"  {feat}: {imp:.4f}")

    def save_models(self, path: str):
        """Save models with GPU state"""
        import os
        os.makedirs(path, exist_ok=True)

        # Save deep learning models
        if 'attention_lstm' in self.models:
            torch.save({
                'model_state_dict': self.models['attention_lstm'].state_dict(),
                'input_dim': self.models['attention_lstm'].fc3.in_features,
                'optimal_threshold': self.optimal_threshold
            }, os.path.join(path, 'attention_lstm_model.pth'))

        # Save tree models
        for name in ['xgboost', 'lightgbm', 'rf']:
            if name in self.models:
                joblib.dump(
                    self.models[name],
                    os.path.join(path, f'{name}_model.pkl')
                )

        # Save scaler and configs
        joblib.dump({
            'scaler': self.scaler,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'optimal_threshold': self.optimal_threshold
        }, os.path.join(path, 'model_config.pkl'))

        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load models from disk"""
        import os

        # Load config
        config_path = os.path.join(path, 'model_config.pkl')
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.scaler = config['scaler']
            self.model_weights = config['model_weights']
            self.feature_importance = config.get('feature_importance', {})
            self.training_history = config.get('training_history', [])
            self.optimal_threshold = config.get('optimal_threshold', 0.5)

        # Load LSTM
        lstm_path = os.path.join(path, 'attention_lstm_model.pth')
        if os.path.exists(lstm_path):
            checkpoint = torch.load(lstm_path, map_location=self.device)

            # Get input dimension from saved model or use default
            input_dim = checkpoint.get('input_dim', 100)
            self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)

            # Create model
            model = AttentionLSTM(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                num_heads=8,
                dropout=0.3
            ).to(self.device)

            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models['attention_lstm'] = model

        # Load tree models
        for name in ['xgboost', 'lightgbm', 'rf']:
            model_path = os.path.join(path, f'{name}_model.pkl')
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)

        self.is_trained = True
        logger.info(f"Models loaded from {path}")