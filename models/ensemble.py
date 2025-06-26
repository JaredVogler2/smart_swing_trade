# models/ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from datetime import datetime
import warnings

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

# Custom modules
from models.features import FeatureEngineer
from config.settings import Config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """PyTorch LSTM model for GPU acceleration"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True,
                             bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_dims[0] * 2, hidden_dims[1], batch_first=True,
                             bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_dims[1] * 2, hidden_dims[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dims[2], num_heads=4)

        # Output layers
        self.fc1 = nn.Linear(hidden_dims[2], 16)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)
        out = self.dropout3(out)

        # Attention
        out = out.transpose(0, 1)  # Required for attention
        attn_out, _ = self.attention(out, out, out)
        out = attn_out.transpose(0, 1)

        # Take the last output
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout4(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


class EnsembleModel:
    """GPU-accelerated ensemble model combining LSTM, XGBoost, LightGBM, and RF"""

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and Config.USE_GPU
        self.models = {}
        self.feature_engineer = FeatureEngineer(enable_gpu=self.use_gpu)
        self.is_trained = False
        self.feature_importance = {}
        self.model_weights = {
            'lstm': 0.4,
            'xgboost': 0.3,
            'lightgbm': 0.2,
            'rf': 0.1
        }

        # Initialize device
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                self.use_gpu = False
                logger.warning("GPU not available, using CPU")
        else:
            self.device = torch.device('cpu')

        # Model parameters
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.prediction_horizon = 3  # 3 days for swing trading

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
            'gpu_id': Config.GPU_DEVICE_ID if self.use_gpu else None,
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

        return xgb.XGBClassifier(**params)

    def _create_lightgbm_model(self):
        """Create LightGBM model with GPU support"""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'gpu_device_id': Config.GPU_DEVICE_ID if self.use_gpu else None,
            'num_leaves': 31,
            'learning_rate': 0.01,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1
        }

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
              validation_data: Dict[str, pd.DataFrame] = None):
        """Train ensemble model on historical data"""
        logger.info("Starting ensemble model training...")

        # Prepare training data
        X_train_all = []
        y_train_all = []

        for symbol, df in train_data.items():
            if len(df) < self.sequence_length + 100:
                continue

            # Create features
            features = self.feature_engineer.create_features(df, symbol)

            if features.empty:
                continue

            # Create target (3-day forward return > 3% after costs)
            forward_return = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            target = (forward_return > 0.03).astype(int)  # 3% threshold for swing trades

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
        X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)

        # Train each model
        self._train_lstm(X_train_scaled, y_train)
        self._train_xgboost(X_train_scaled, y_train)
        self._train_lightgbm(X_train_scaled, y_train)
        self._train_random_forest(X_train_scaled, y_train)

        # Calculate feature importance
        self._calculate_feature_importance(X_train_scaled, y_train)

        self.is_trained = True
        logger.info("Ensemble model training completed")

    def _train_lstm(self, X: pd.DataFrame, y: pd.Series):
        """Train LSTM model"""
        logger.info("Training LSTM...")

        # Prepare sequences
        X_seq, y_seq = self._prepare_lstm_data(X, y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create model
        model = self._create_lstm_model(X_seq.shape)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop
        model.train()
        batch_size = 32
        n_epochs = 50

        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i + batch_size]
                batch_y = y_tensor[i:i + batch_size].unsqueeze(1)

                optimizer.zero_grad()
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

        self.models['lstm'] = model

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")

        model = self._create_xgboost_model()

        # Use early stopping
        X_train, X_val = X[:-1000], X[-1000:]
        y_train, y_val = y[:-1000], y[-1000:]

        model.fit(
            X_train, y_train,
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

    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")

        model = self._create_lightgbm_model()

        # Use early stopping
        X_train, X_val = X[:-1000], X[-1000:]
        y_train, y_val = y[:-1000], y[-1000:]

        model.fit(
            X_train, y_train,
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
            lstm_pred, lstm_prob = self._predict_lstm(features_scaled)
            predictions['lstm'] = lstm_pred
            probabilities['lstm'] = lstm_prob

        # XGBoost prediction
        if 'xgboost' in self.models:
            xgb_pred, xgb_prob = self._predict_tree_model(
                self.models['xgboost'], features_scaled
            )
            predictions['xgboost'] = xgb_pred
            probabilities['xgboost'] = xgb_prob

        # LightGBM prediction
        if 'lightgbm' in self.models:
            lgb_pred, lgb_prob = self._predict_tree_model(
                self.models['lightgbm'], features_scaled
            )
            predictions['lightgbm'] = lgb_pred
            probabilities['lightgbm'] = lgb_prob

        # Random Forest prediction
        if 'rf' in self.models:
            rf_pred, rf_prob = self._predict_tree_model(
                self.models['rf'], features_scaled
            )
            predictions['rf'] = rf_pred
            probabilities['rf'] = rf_prob

        # Weighted ensemble prediction
        ensemble_prob = 0
        total_weight = 0

        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 0.1)
            ensemble_prob += prob * weight
            total_weight += weight

        ensemble_prob /= total_weight
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