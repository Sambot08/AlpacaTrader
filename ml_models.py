import logging
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BaseMLModel:
    """Base class for ML models used in trading."""
    
    def __init__(self):
        self.model = None
        self.model_type = "base"
        self.is_trained = False
    
    def train(self, X, y):
        """Train the model on historical data."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if not self.is_trained:
            logger.warning(f"{self.model_type} model not trained, returning default prediction")
            return np.zeros(X.shape[0], dtype=int)
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making prediction with {self.model_type} model: {str(e)}")
            return np.zeros(X.shape[0], dtype=int)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            logger.warning(f"{self.model_type} model not trained, returning default probabilities")
            return np.array([[0.5, 0.5]] * X.shape[0])
        
        try:
            proba = self.model.predict_proba(X)
            # Return probability of positive class
            return proba[:, 1]
        except Exception as e:
            logger.error(f"Error getting prediction probabilities with {self.model_type} model: {str(e)}")
            return np.array([0.5] * X.shape[0])
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if not self.is_trained:
            logger.warning(f"Cannot save untrained {self.model_type} model")
            return False
        
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Saved {self.model_type} model to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.model_type} model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        try:
            if os.path.exists(filepath):
                self.model = joblib.load(filepath)
                self.is_trained = True
                logger.info(f"Loaded {self.model_type} model from {filepath}")
                return True
            else:
                logger.warning(f"Model file not found: {filepath}")
                return False
        except Exception as e:
            logger.error(f"Error loading {self.model_type} model: {str(e)}")
            return False

class RandomForestModel(BaseMLModel):
    """Random Forest classifier model for trading."""
    
    def __init__(self):
        super().__init__()
        self.model_type = "random_forest"
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            ))
        ])
    
    def train(self, X, y):
        """Train the Random Forest model."""
        try:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Random Forest model trained with metrics:")
            logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return False

class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting classifier model for trading."""
    
    def __init__(self):
        super().__init__()
        self.model_type = "gradient_boosting"
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
    
    def train(self, X, y):
        """Train the Gradient Boosting model."""
        try:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Gradient Boosting model trained with metrics:")
            logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {str(e)}")
            return False

class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression classifier model for trading."""
    
    def __init__(self):
        super().__init__()
        self.model_type = "logistic_regression"
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ])
    
    def train(self, X, y):
        """Train the Logistic Regression model."""
        try:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Logistic Regression model trained with metrics:")
            logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            logger.error(f"Error training Logistic Regression model: {str(e)}")
            return False

class SVMModel(BaseMLModel):
    """Support Vector Machine classifier model for trading."""
    
    def __init__(self):
        super().__init__()
        self.model_type = "svm"
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ))
        ])
    
    def train(self, X, y):
        """Train the SVM model."""
        try:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"SVM model trained with metrics:")
            logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            logger.error(f"Error training SVM model: {str(e)}")
            return False

class MLModelFactory:
    """Factory class to create ML models based on type."""
    
    def get_model(self, model_type):
        """Get an ML model instance based on type."""
        if model_type.lower() == 'random_forest':
            return RandomForestModel()
        elif model_type.lower() == 'gradient_boosting':
            return GradientBoostingModel()
        elif model_type.lower() == 'logistic_regression':
            return LogisticRegressionModel()
        elif model_type.lower() == 'svm':
            return SVMModel()
        else:
            # Default to Random Forest if type not recognized
            logger.warning(f"Unknown model type '{model_type}', defaulting to Random Forest")
            return RandomForestModel()
