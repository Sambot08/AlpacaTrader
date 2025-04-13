from app import db
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Date
from sqlalchemy.orm import relationship
from flask_login import UserMixin

class User(UserMixin, db.Model):
    """User model for authentication and profile management."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    api_key = db.Column(db.String(256))
    api_secret = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    strategies = relationship("TradingStrategy", back_populates="user")

class TradingStrategy(db.Model):
    """Model to store trading strategy configuration."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    description = db.Column(db.Text)
    symbols = db.Column(db.String(512), nullable=False)  # Comma-separated list of stock symbols
    ml_model = db.Column(db.String(64), nullable=False)  # Type of ML model to use
    risk_level = db.Column(db.Integer, default=3)  # 1-5 scale, 5 being highest risk
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, ForeignKey('user.id'))
    
    # Relationships
    user = relationship("User", back_populates="strategies")
    trade_records = relationship("TradeRecord", back_populates="strategy")

class TradeRecord(db.Model):
    """Model to store executed trade records."""
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # 'BUY' or 'SELL'
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='FILLED')  # 'FILLED', 'PENDING', 'FAILED'
    order_id = db.Column(db.String(64))  # Alpaca order ID
    strategy_id = db.Column(db.Integer, ForeignKey('trading_strategy.id'))
    
    # Relationships
    strategy = relationship("TradingStrategy", back_populates="trade_records")

class PerformanceMetric(db.Model):
    """Model to store daily performance metrics."""
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    portfolio_value = db.Column(db.Float, nullable=False)
    cash_balance = db.Column(db.Float, nullable=False)
    equity_value = db.Column(db.Float, nullable=False)
    daily_return = db.Column(db.Float)
    pnl = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    trade_count = db.Column(db.Integer, default=0)
