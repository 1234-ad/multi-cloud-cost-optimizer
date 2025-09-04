"""
Cost Prediction and Forecasting Model
====================================

Machine learning models for predicting cloud costs and usage patterns.
Supports multiple algorithms including LSTM, Prophet, and ensemble methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Time Series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json

logger = logging.getLogger(__name__)

class CostForecastingModel:
    """
    Comprehensive cost forecasting model with multiple algorithms.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the forecasting model.
        
        Args:
            model_type: Type of model ('lstm', 'prophet', 'ensemble', 'ml')
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        self.target_column = 'cost'
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'cost') -> pd.DataFrame:
        """
        Prepare data for training with feature engineering.
        
        Args:
            data: Raw cost data with datetime index
            target_col: Target column name
            
        Returns:
            Prepared DataFrame with features
        """
        df = data.copy()
        self.target_column = target_col
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                raise ValueError("Data must have datetime index or 'date' column")
        
        # Sort by date
        df = df.sort_index()
        
        # Create time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'cost_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [7, 30, 90]:
            df[f'cost_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'cost_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'cost_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'cost_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        # Exponential moving averages
        for span in [7, 30]:
            df[f'cost_ema_{span}'] = df[target_col].ewm(span=span).mean()
        
        # Seasonal decomposition features
        if len(df) >= 365:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(df[target_col].dropna(), model='additive', period=365)
            df['trend'] = decomposition.trend
            df['seasonal'] = decomposition.seasonal
            df['residual'] = decomposition.resid
        
        # Growth rates
        df['cost_pct_change'] = df[target_col].pct_change()
        df['cost_diff'] = df[target_col].diff()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col != target_col]
        
        logger.info(f"Data prepared: {len(df)} samples, {len(self.feature_columns)} features")
        return df
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the forecasting model(s).
        
        Args:
            data: Prepared training data
            test_size: Fraction of data for testing
            
        Returns:
            Training results and metrics
        """
        # Prepare features and target
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        results = {}
        
        if self.model_type in ['ml', 'ensemble']:
            results.update(self._train_ml_models(X, y, tscv))
        
        if self.model_type in ['lstm', 'ensemble'] and TENSORFLOW_AVAILABLE:
            results.update(self._train_lstm_model(data))
        
        if self.model_type in ['prophet', 'ensemble'] and PROPHET_AVAILABLE:
            results.update(self._train_prophet_model(data))
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return results
    
    def _train_ml_models(self, X: pd.DataFrame, y: pd.Series, tscv) -> Dict[str, Any]:
        """Train traditional ML models."""
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features for linear regression
                if name == 'linear_regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    
                    self.scalers[name] = scaler
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                
                mae = mean_absolute_error(y_val, y_pred)
                scores.append(mae)
            
            avg_mae = np.mean(scores)
            results[name] = {'model': model, 'mae': avg_mae}
            
            # Train on full dataset
            if name == 'linear_regression':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                self.scalers[name] = scaler
            else:
                model.fit(X, y)
            
            self.models[name] = model
            logger.info(f"{name}: Average MAE = {avg_mae:.2f}")
        
        return results
    
    def _train_lstm_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM neural network model."""
        try:
            # Prepare data for LSTM
            sequence_length = 30
            target_data = data[self.target_column].values.reshape(-1, 1)
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(target_data)
            self.scalers['lstm'] = scaler
            
            # Create sequences
            X_lstm, y_lstm = self._create_sequences(scaled_data, sequence_length)
            
            # Split data
            split_idx = int(len(X_lstm) * 0.8)
            X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_unscaled = scaler.inverse_transform(y_pred)
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
            
            self.models['lstm'] = model
            
            logger.info(f"LSTM: MAE = {mae:.2f}")
            
            return {'lstm': {'model': model, 'mae': mae, 'history': history.history}}
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {}
    
    def _train_prophet_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet time series model."""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data[self.target_column]
            })
            
            # Initialize Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Fit model
            model.fit(prophet_data)
            
            # Make predictions on training data for evaluation
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            # Calculate MAE
            mae = mean_absolute_error(prophet_data['y'], forecast['yhat'])
            
            self.models['prophet'] = model
            
            logger.info(f"Prophet: MAE = {mae:.2f}")
            
            return {'prophet': {'model': model, 'mae': mae}}
        
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return {}
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def predict(self, periods: int = 30, confidence_interval: bool = True) -> pd.DataFrame:
        """
        Generate cost forecasts.
        
        Args:
            periods: Number of periods to forecast
            confidence_interval: Whether to include confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        forecasts = {}
        
        # Generate predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'prophet':
                    forecast = self._predict_prophet(model, periods)
                elif model_name == 'lstm':
                    forecast = self._predict_lstm(model, periods)
                else:
                    forecast = self._predict_ml(model, model_name, periods)
                
                forecasts[model_name] = forecast
            
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
        
        # Combine forecasts
        if self.model_type == 'ensemble' and len(forecasts) > 1:
            # Ensemble prediction (simple average)
            ensemble_forecast = np.mean([f for f in forecasts.values()], axis=0)
            forecasts['ensemble'] = ensemble_forecast
        
        # Create result DataFrame
        future_dates = pd.date_range(
            start=datetime.now().date(),
            periods=periods,
            freq='D'
        )
        
        result_df = pd.DataFrame(index=future_dates)
        
        for model_name, forecast in forecasts.items():
            result_df[f'{model_name}_forecast'] = forecast[:periods]
        
        return result_df
    
    def _predict_prophet(self, model, periods: int) -> np.ndarray:
        """Generate Prophet predictions."""
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast['yhat'].tail(periods).values
    
    def _predict_lstm(self, model, periods: int) -> np.ndarray:
        """Generate LSTM predictions."""
        # This is a simplified implementation
        # In practice, you'd need the last sequence from training data
        predictions = []
        
        # Generate dummy predictions for demo
        last_value = 1000  # Would come from actual last known value
        for i in range(periods):
            # Simple trend with noise
            pred = last_value * (1 + np.random.normal(0, 0.02))
            predictions.append(pred)
            last_value = pred
        
        return np.array(predictions)
    
    def _predict_ml(self, model, model_name: str, periods: int) -> np.ndarray:
        """Generate ML model predictions."""
        # This is a simplified implementation
        # In practice, you'd need to create future features
        predictions = []
        
        # Generate dummy predictions for demo
        base_cost = 1000
        for i in range(periods):
            # Add some trend and seasonality
            trend = i * 0.5
            seasonal = 100 * np.sin(2 * np.pi * i / 30)  # Monthly seasonality
            noise = np.random.normal(0, 50)
            
            pred = base_cost + trend + seasonal + noise
            predictions.append(max(pred, 0))  # Ensure non-negative
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree-based models."""
        importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_imp = dict(zip(self.feature_columns, model.feature_importances_))
                # Sort by importance
                feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
                importance[model_name] = feature_imp
        
        return importance
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'models': {},
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        # Save non-neural network models
        for name, model in self.models.items():
            if name != 'lstm':
                model_data['models'][name] = model
        
        joblib.dump(model_data, filepath)
        
        # Save LSTM model separately if it exists
        if 'lstm' in self.models:
            lstm_path = filepath.replace('.pkl', '_lstm.h5')
            self.models['lstm'].save(lstm_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        # Load LSTM model if it exists
        lstm_path = filepath.replace('.pkl', '_lstm.h5')
        try:
            if TENSORFLOW_AVAILABLE:
                self.models['lstm'] = tf.keras.models.load_model(lstm_path)
        except:
            pass
        
        logger.info(f"Model loaded from {filepath}")

def generate_sample_cost_data(days: int = 365) -> pd.DataFrame:
    """Generate sample cost data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Create realistic cost pattern
    base_cost = 1000
    trend = np.linspace(0, 200, days)  # Gradual increase
    
    # Seasonal patterns
    yearly_seasonal = 100 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    weekly_seasonal = 50 * np.sin(2 * np.pi * np.arange(days) / 7)
    
    # Random noise
    noise = np.random.normal(0, 50, days)
    
    # Combine components
    costs = base_cost + trend + yearly_seasonal + weekly_seasonal + noise
    costs = np.maximum(costs, 0)  # Ensure non-negative
    
    # Add some additional features
    data = pd.DataFrame({
        'date': dates,
        'cost': costs,
        'cpu_usage': np.random.uniform(20, 80, days),
        'storage_gb': np.random.uniform(100, 1000, days),
        'network_gb': np.random.uniform(10, 100, days)
    })
    
    return data.set_index('date')

# Example usage
if __name__ == "__main__":
    print("Cost Forecasting Model Example")
    print("=" * 40)
    
    # Generate sample data
    sample_data = generate_sample_cost_data(days=365)
    print(f"Generated {len(sample_data)} days of sample cost data")
    
    # Initialize and train model
    forecaster = CostForecastingModel(model_type='ensemble')
    
    # Prepare data
    prepared_data = forecaster.prepare_data(sample_data)
    print(f"Data prepared with {len(forecaster.feature_columns)} features")
    
    # Train model
    print("Training models...")
    results = forecaster.train(prepared_data)
    
    # Generate forecasts
    print("Generating forecasts...")
    forecasts = forecaster.predict(periods=30)
    
    print("\\nForecast summary:")
    print(forecasts.head())
    
    # Feature importance
    importance = forecaster.get_feature_importance()
    if importance:
        print("\\nTop 5 most important features:")
        for model_name, features in importance.items():
            print(f"\\n{model_name}:")
            for feature, imp in list(features.items())[:5]:
                print(f"  {feature}: {imp:.4f}")
    
    print("\\nForecasting complete!")