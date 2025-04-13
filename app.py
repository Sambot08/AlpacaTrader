import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///trading.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

# Import models and other modules after app initialization to avoid circular imports
with app.app_context():
    from models import User, TradingStrategy, TradeRecord, PerformanceMetric
    from trading_engine import TradingEngine
    from data_processor import DataProcessor
    
    # Create database tables
    db.create_all()
    
    # Initialize the trading engine
    data_processor = DataProcessor()
    trading_engine = TradingEngine(data_processor)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the trading dashboard."""
    # Get performance metrics
    with app.app_context():
        metrics = PerformanceMetric.query.order_by(PerformanceMetric.date.desc()).limit(30).all()
        trades = TradeRecord.query.order_by(TradeRecord.timestamp.desc()).limit(10).all()
        strategies = TradingStrategy.query.all()
    
    return render_template('dashboard.html', 
                          metrics=metrics, 
                          trades=trades,
                          strategies=strategies)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Render and process the settings page."""
    if request.method == 'POST':
        # Handle form submission for strategy settings
        strategy_name = request.form.get('strategy_name')
        stock_symbols = request.form.get('stock_symbols')
        ml_model_type = request.form.get('ml_model_type')
        risk_level = int(request.form.get('risk_level'))
        
        # Create or update strategy
        strategy = TradingStrategy.query.filter_by(name=strategy_name).first()
        if strategy:
            strategy.symbols = stock_symbols
            strategy.ml_model = ml_model_type
            strategy.risk_level = risk_level
        else:
            strategy = TradingStrategy(
                name=strategy_name,
                symbols=stock_symbols,
                ml_model=ml_model_type,
                risk_level=risk_level,
                is_active=True
            )
            db.session.add(strategy)
        
        db.session.commit()
        flash('Settings saved successfully!', 'success')
        return redirect(url_for('settings'))
    
    # Display current settings
    strategies = TradingStrategy.query.all()
    return render_template('settings.html', strategies=strategies)

@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    """API endpoint to start trading with a specific strategy."""
    try:
        strategy_id = request.json.get('strategy_id')
        if not strategy_id:
            return jsonify({"success": False, "message": "Strategy ID is required"}), 400
        
        strategy = TradingStrategy.query.get(strategy_id)
        if not strategy:
            return jsonify({"success": False, "message": "Strategy not found"}), 404
        
        # Activate the strategy
        strategy.is_active = True
        db.session.commit()
        
        # Start the trading engine with this strategy
        trading_engine.start_strategy(strategy)
        
        return jsonify({"success": True, "message": f"Trading started for strategy: {strategy.name}"})
    except Exception as e:
        logger.error(f"Error starting trading: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    """API endpoint to stop trading with a specific strategy."""
    try:
        strategy_id = request.json.get('strategy_id')
        if not strategy_id:
            return jsonify({"success": False, "message": "Strategy ID is required"}), 400
        
        strategy = TradingStrategy.query.get(strategy_id)
        if not strategy:
            return jsonify({"success": False, "message": "Strategy not found"}), 404
        
        # Deactivate the strategy
        strategy.is_active = False
        db.session.commit()
        
        # Stop the trading engine for this strategy
        trading_engine.stop_strategy(strategy)
        
        return jsonify({"success": True, "message": f"Trading stopped for strategy: {strategy.name}"})
    except Exception as e:
        logger.error(f"Error stopping trading: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/performance_data')
def performance_data():
    """API endpoint to get performance metrics for charting."""
    try:
        days = int(request.args.get('days', 30))
        metrics = PerformanceMetric.query.order_by(PerformanceMetric.date.desc()).limit(days).all()
        
        data = {
            "dates": [m.date.strftime('%Y-%m-%d') for m in reversed(metrics)],
            "portfolio_values": [float(m.portfolio_value) for m in reversed(metrics)],
            "returns": [float(m.daily_return) for m in reversed(metrics)]
        }
        
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Error fetching performance data: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/recent_trades')
def recent_trades():
    """API endpoint to get recent trades."""
    try:
        limit = int(request.args.get('limit', 10))
        trades = TradeRecord.query.order_by(TradeRecord.timestamp.desc()).limit(limit).all()
        
        trade_data = []
        for trade in trades:
            trade_data.append({
                "id": trade.id,
                "symbol": trade.symbol,
                "action": trade.action,
                "quantity": trade.quantity,
                "price": float(trade.price),
                "timestamp": trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "strategy_id": trade.strategy_id
            })
        
        return jsonify({"success": True, "trades": trade_data})
    except Exception as e:
        logger.error(f"Error fetching recent trades: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
