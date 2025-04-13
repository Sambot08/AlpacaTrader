from app import db
from datetime import datetime
from sqlalchemy import Index, String, Float, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.sql import func

class Trade(db.Model):
    """Model for storing trade information"""
    __tablename__ = 'trades'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    side = db.Column(db.String(10), nullable=False)  # buy or sell
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    order_id = db.Column(db.String(100), unique=True)
    profit_loss = db.Column(db.Float, nullable=True)
    strategy_used = db.Column(db.String(100), nullable=True)
    ml_confidence = db.Column(db.Float, nullable=True)  # Confidence of ML prediction
    status = db.Column(db.String(20), default='executed')  # executed, canceled, filled
    
    def __repr__(self):
        return f"<Trade {self.id} {self.side} {self.quantity} {self.symbol} @ {self.price}>"

class Position(db.Model):
    """Model for storing current positions"""
    __tablename__ = 'positions'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, unique=True, index=True)
    quantity = db.Column(db.Float, nullable=False)
    avg_entry_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float, nullable=True)
    market_value = db.Column(db.Float, nullable=True)
    profit_loss = db.Column(db.Float, nullable=True)
    profit_loss_pct = db.Column(db.Float, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    stop_loss_price = db.Column(db.Float, nullable=True)
    take_profit_price = db.Column(db.Float, nullable=True)
    entry_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Position {self.symbol} {self.quantity} @ {self.avg_entry_price}>"

class PerformanceMetric(db.Model):
    """Model for storing performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True, index=True)
    portfolio_value = db.Column(db.Float, nullable=False)
    cash_balance = db.Column(db.Float, nullable=False)
    equity_value = db.Column(db.Float, nullable=False, default=0.0)
    daily_return = db.Column(db.Float, nullable=True)
    daily_return_pct = db.Column(db.Float, nullable=True)
    total_trades = db.Column(db.Integer, nullable=True)
    winning_trades = db.Column(db.Integer, nullable=True)
    losing_trades = db.Column(db.Integer, nullable=True)
    sharpe_ratio = db.Column(db.Float, nullable=True)
    max_drawdown = db.Column(db.Float, nullable=True)
    max_drawdown_pct = db.Column(db.Float, nullable=True)
    volatility = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f"<PerformanceMetric {self.date} {self.portfolio_value}>"
