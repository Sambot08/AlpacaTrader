import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class MLModel:
    """
    Machine Learning model for price prediction and trading signals
    """
    
    def __init__(self):
        """
        Initialize the ML model
        """
        self.models = {}  # Dictionary to store models for each symbol
        self.scalers = {}  # Dictionary to store scalers for each symbol
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist',
            'rsi',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_std',
            'atr',
            'daily_return', 'volatility'
        ]
        
    def prepare_features(self, df):
        """
        Prepare features for the ML model
        
        Args:
            df (DataFrame): DataFrame with OHLCV and technical indicators
            
        Returns:
            tuple: X (features) and y (target)
        """
        # Drop rows with NaN values
        df = df.dropna()
        
        # Create target variables (next day's return)
        df['target_return'] = df['close'].shift(-1) / df['close'] - 1
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        # Drop the last row since it doesn't have a target
        df = df[:-1]
        
        # Select features
        X = df[self.features]
        
        # Target for regression (predicting return)
        y_reg = df['target_return']
        
        # Target for classification (predicting direction)
        y_cls = df['target_direction']
        
        return X, y_reg, y_cls
        
    def train_model(self, symbol, historical_data, model_type="classification"):
        """
        Train the ML model for a symbol
        
        Args:
            symbol (str): Ticker symbol
            historical_data (DataFrame): Historical price data with indicators
            model_type (str): Type of model ('classification' or 'regression')
            
        Returns:
            float: Model accuracy or error
        """
        try:
            # Prepare features
            X, y_reg, y_cls = self.prepare_features(historical_data)
            
            if X.empty or len(X) < 20:
                logger.warning(f"Insufficient data for {symbol} to train model")
                return 0
            
            # Choose target based on model type
            y = y_cls if model_type == "classification" else y_reg
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            self.scalers[symbol] = scaler
            
            # Train model
            if model_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Model for {symbol} trained with accuracy: {accuracy:.4f}")
                
                # Save model
                self.models[symbol] = {
                    'model': model,
                    'type': 'classification',
                    'accuracy': accuracy,
                    'trained_at': datetime.now()
                }
                
                return accuracy
                
            else:  # Regression
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                logger.info(f"Model for {symbol} trained with RMSE: {rmse:.6f}")
                
                # Save model
                self.models[symbol] = {
                    'model': model,
                    'type': 'regression',
                    'rmse': rmse,
                    'trained_at': datetime.now()
                }
                
                return rmse
                
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return 0
    
    def predict(self, symbol, current_data):
        """
        Make predictions for a symbol
        
        Args:
            symbol (str): Ticker symbol
            current_data (DataFrame): Current market data with indicators
            
        Returns:
            dict: Prediction results
        """
        if symbol not in self.models:
            logger.warning(f"No model found for {symbol}")
            return {'prediction': None, 'confidence': 0, 'signal': 'HOLD'}
            
        try:
            # Get the latest data point
            latest_data = current_data.iloc[-1:][self.features]
            
            # Scale data
            scaled_data = self.scalers[symbol].transform(latest_data)
            
            # Make prediction
            model_info = self.models[symbol]
            model = model_info['model']
            
            if model_info['type'] == 'classification':
                # For classification model
                pred_proba = model.predict_proba(scaled_data)[0]
                pred_class = model.predict(scaled_data)[0]
                
                # Determine confidence
                confidence = pred_proba[pred_class]
                
                # Determine trading signal
                signal = "BUY" if pred_class == 1 else "SELL"
                
                # Adjust signal based on confidence
                if confidence < 0.6:
                    signal = "HOLD"
                
                return {
                    'prediction': bool(pred_class),
                    'confidence': float(confidence),
                    'signal': signal,
                    'predicted_direction': "UP" if pred_class == 1 else "DOWN"
                }
                
            else:
                # For regression model
                pred_return = model.predict(scaled_data)[0]
                
                # Determine trading signal based on predicted return
                signal = "HOLD"
                if pred_return > 0.01:  # 1% threshold for buy
                    signal = "BUY"
                elif pred_return < -0.01:  # -1% threshold for sell
                    signal = "SELL"
                
                return {
                    'prediction': float(pred_return),
                    'confidence': 0.5,  # No direct confidence for regression
                    'signal': signal,
                    'predicted_return_pct': float(pred_return * 100)
                }
                
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {str(e)}")
            return {'prediction': None, 'confidence': 0, 'signal': 'HOLD'}
            
    def needs_training(self, symbol):
        """
        Check if the model for a symbol needs training
        
        Args:
            symbol (str): Ticker symbol
            
        Returns:
            bool: True if model needs training, False otherwise
        """
        if symbol not in self.models:
            return True
            
        last_trained = self.models[symbol]['trained_at']
        
        # Retrain if model is more than 7 days old
        if datetime.now() - last_trained > timedelta(days=7):
            return True
            
        return False
        
    def save_models(self, directory="models"):
        """
        Save all models to disk
        
        Args:
            directory (str): Directory to save models
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for symbol, model_info in self.models.items():
            try:
                model_path = os.path.join(directory, f"{symbol}_model.joblib")
                scaler_path = os.path.join(directory, f"{symbol}_scaler.joblib")
                
                joblib.dump(model_info['model'], model_path)
                joblib.dump(self.scalers[symbol], scaler_path)
                
                logger.info(f"Model for {symbol} saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model for {symbol}: {str(e)}")
                
    def load_models(self, directory="models", symbols=None):
        """
        Load models from disk
        
        Args:
            directory (str): Directory to load models from
            symbols (list): List of symbols to load models for
            
        Returns:
            int: Number of models loaded
        """
        if not os.path.exists(directory):
            logger.warning(f"Model directory {directory} not found")
            return 0
            
        loaded_count = 0
        
        # If symbols not specified, try to load all models in directory
        if symbols is None:
            model_files = [f for f in os.listdir(directory) if f.endswith("_model.joblib")]
            symbols = [f.split("_model.joblib")[0] for f in model_files]
            
        for symbol in symbols:
            try:
                model_path = os.path.join(directory, f"{symbol}_model.joblib")
                scaler_path = os.path.join(directory, f"{symbol}_scaler.joblib")
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    logger.warning(f"Model or scaler for {symbol} not found")
                    continue
                    
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Determine model type
                if hasattr(model, 'predict_proba'):
                    model_type = 'classification'
                else:
                    model_type = 'regression'
                
                self.models[symbol] = {
                    'model': model,
                    'type': model_type,
                    'trained_at': datetime.fromtimestamp(os.path.getmtime(model_path))
                }
                
                self.scalers[symbol] = scaler
                loaded_count += 1
                
                logger.info(f"Model for {symbol} loaded from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {str(e)}")
                
        return loaded_count
