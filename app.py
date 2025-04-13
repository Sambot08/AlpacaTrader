import os
import logging
from datetime import datetime
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Local imports
from trading_engine.data_fetcher import AlpacaDataFetcher
from trading_engine.ml_model import MLModel
from trading_engine.risk_manager import RiskManager
from trading_engine.trade_executor import TradeExecutor
from trading_engine.strategy import TradingStrategy
from trading_engine.performance import PerformanceTracker

# Set up logging
logger = logging.getLogger(__name__)

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
# Make sure PostgreSQL URL starts with postgresql:// instead of postgres://
database_url = os.environ.get("DATABASE_URL", "sqlite:///trading_bot.db")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
db.init_app(app)

# Initialize trading components
def init_trading_engine():
    global data_fetcher, ml_model, risk_manager, trade_executor, trading_strategy, performance_tracker
    
    # Get API credentials from environment variables
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Default to paper trading
    data_source = os.environ.get("ALPACA_DATA_SOURCE", "iex")  # Default to IEX (free tier)
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found. Trading engine will be initialized but not active.")
        
    # Initialize components
    data_fetcher = AlpacaDataFetcher(api_key, api_secret, base_url, data_source=data_source)
    ml_model = MLModel()
    risk_manager = RiskManager()
    trade_executor = TradeExecutor(api_key, api_secret, base_url)
    performance_tracker = PerformanceTracker()
    
    # Strategy depends on other components
    trading_strategy = TradingStrategy(
        data_fetcher, 
        ml_model,
        risk_manager,
        trade_executor,
        performance_tracker
    )
    
    return trading_strategy

# Trading engine thread
def trading_thread():
    logger.info("Starting trading thread...")
    while trading_active:
        try:
            # Only run during market hours
            if data_fetcher.is_market_open():
                trading_strategy.execute_strategy()
                logger.info("Strategy executed successfully")
            else:
                logger.info("Market is closed. Waiting...")
            
            # Sleep for the configured interval
            time.sleep(trading_interval)
        except Exception as e:
            logger.error(f"Error in trading thread: {str(e)}")
            time.sleep(60)  # Wait a bit longer if there was an error

# Global variables
trading_active = False
trading_interval = 300  # 5 minutes by default
trading_thread_object = None
trading_strategy = None
data_fetcher = None
ml_model = None
risk_manager = None
trade_executor = None
performance_tracker = None

# Routes
@app.route('/')
def index():
    # Get account information if available
    account_info = None
    portfolio = None
    recent_trades = None
    market_status = False
    
    if data_fetcher:
        try:
            account_info = trade_executor.get_account()
            portfolio = trade_executor.get_positions()
            recent_trades = trade_executor.get_recent_orders(10)
            market_status = data_fetcher.is_market_open()
        except Exception as e:
            logger.error(f"Error fetching account data: {str(e)}")
            flash(f"Error fetching account data: {str(e)}", "danger")
    
    return render_template(
        'index.html',
        trading_active=trading_active,
        account_info=account_info,
        portfolio=portfolio,
        recent_trades=recent_trades,
        market_status=market_status,
        trading_interval=trading_interval
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global trading_interval, trading_strategy
    
    if request.method == 'POST':
        # Update trading parameters
        try:
            trading_interval = int(request.form.get('trading_interval', 300))
            
            # Update risk parameters
            max_position_size = float(request.form.get('max_position_size', 0.1))
            stop_loss_pct = float(request.form.get('stop_loss_pct', 0.02))
            take_profit_pct = float(request.form.get('take_profit_pct', 0.05))
            
            # Update API credentials
            api_key = request.form.get('api_key', '').strip()
            api_secret = request.form.get('api_secret', '').strip()
            api_base_url = request.form.get('api_base_url')
            
            # Save API credentials if provided
            if api_key and api_secret:
                # In a production app, use a secure method to store credentials
                # For now, we'll store them in environment variables
                os.environ['ALPACA_API_KEY'] = api_key
                os.environ['ALPACA_API_SECRET'] = api_secret
                os.environ['ALPACA_BASE_URL'] = api_base_url
                
                # Reinitialize the trading engine with new credentials
                try:
                    trading_strategy = init_trading_engine()
                    flash("API credentials updated and trading engine reinitialized!", "success")
                except Exception as e:
                    flash(f"Error reinitializing trading engine: {str(e)}", "danger")
            
            # Update risk parameters if engine is initialized
            if risk_manager:
                risk_manager.update_parameters(
                    max_position_size=max_position_size,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct
                )
                
                flash("Settings updated successfully!", "success")
            else:
                flash("Trading engine not initialized. Settings saved but not applied.", "warning")
                
        except ValueError as e:
            flash(f"Invalid input: {str(e)}", "danger")
            
        return redirect(url_for('settings'))
        
    # For GET requests, display current settings
    risk_params = {}
    if risk_manager:
        risk_params = risk_manager.get_parameters()
    
    # Get current API settings
    api_key = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    api_base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Get AI model settings
    model_type = "classification"  # Default
    confidence_threshold = 0.65
    lookback_days = 60
    
    if ml_model:
        model_params = ml_model.get_parameters() if hasattr(ml_model, 'get_parameters') else {}
        model_type = model_params.get('model_type', model_type)
        confidence_threshold = model_params.get('confidence_threshold', confidence_threshold)
        lookback_days = model_params.get('lookback_days', lookback_days)
    
    return render_template(
        'settings.html',
        trading_interval=trading_interval,
        risk_params=risk_params,
        api_key=api_key,
        api_secret=api_secret,
        api_base_url=api_base_url,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        lookback_days=lookback_days
    )

@app.route('/performance')
def performance():
    performance_data = None
    if performance_tracker:
        try:
            performance_data = performance_tracker.get_performance_stats()
        except Exception as e:
            logger.error(f"Error fetching performance data: {str(e)}")
            flash(f"Error fetching performance data: {str(e)}", "danger")
    
    return render_template('performance.html', performance_data=performance_data)

@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    global trading_active, trading_thread_object
    
    if not trading_active:
        try:
            trading_active = True
            trading_thread_object = threading.Thread(target=trading_thread, daemon=True)
            trading_thread_object.start()
            logger.info("Trading started")
            return jsonify({"success": True, "message": "Trading started"})
        except Exception as e:
            trading_active = False
            logger.error(f"Error starting trading: {str(e)}")
            return jsonify({"success": False, "message": f"Error: {str(e)}"})
    else:
        return jsonify({"success": False, "message": "Trading is already active"})

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    global trading_active
    
    if trading_active:
        trading_active = False
        logger.info("Trading stopped")
        return jsonify({"success": True, "message": "Trading stopped"})
    else:
        return jsonify({"success": False, "message": "Trading is not active"})

@app.route('/api/update_symbols', methods=['POST'])
def update_symbols():
    symbols = request.json.get('symbols', [])
    
    if not symbols:
        return jsonify({"success": False, "message": "No symbols provided"})
    
    try:
        if trading_strategy:
            trading_strategy.update_symbols(symbols)
            return jsonify({"success": True, "message": f"Updated symbols: {', '.join(symbols)}"})
        else:
            return jsonify({"success": False, "message": "Trading engine not initialized"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/api/account_info')
def account_info():
    if trade_executor:
        try:
            account = trade_executor.get_account()
            return jsonify({"success": True, "account": account})
        except Exception as e:
            logger.error(f"Error fetching account info: {str(e)}")
            return jsonify({"success": False, "message": f"Error: {str(e)}"})
    else:
        return jsonify({"success": False, "message": "Trading engine not initialized"})

@app.route('/api_config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        try:
            # Get API credentials
            api_key = request.form.get('api_key', '').strip()
            api_secret = request.form.get('api_secret', '').strip()
            api_base_url = request.form.get('api_base_url')
            data_source = request.form.get('data_source', 'iex')  # Default to IEX (free tier)
            
            # Save API credentials if provided
            if api_key and api_secret:
                # In a production app, use a secure method to store credentials
                # For now, we'll store them in environment variables
                os.environ['ALPACA_API_KEY'] = api_key
                os.environ['ALPACA_API_SECRET'] = api_secret
                os.environ['ALPACA_BASE_URL'] = api_base_url
                os.environ['ALPACA_DATA_SOURCE'] = data_source
                
                # Reinitialize the trading engine with new credentials
                try:
                    global trading_strategy, data_fetcher, ml_model, risk_manager, trade_executor, performance_tracker
                    trading_strategy = init_trading_engine()
                    flash("API credentials updated and trading engine reinitialized!", "success")
                except Exception as e:
                    flash(f"Error reinitializing trading engine: {str(e)}", "danger")
            else:
                flash("API credentials are required", "warning")
                
        except Exception as e:
            flash(f"Error updating API settings: {str(e)}", "danger")
            
        return redirect(url_for('api_config'))
    
    # For GET requests
    # Get current API settings
    api_key = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    api_base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    data_source = os.environ.get('ALPACA_DATA_SOURCE', 'iex')
    
    # Get account info if available
    account_info = None
    subscription_info = None
    
    if trade_executor:
        try:
            account_info = trade_executor.get_account()
            # Subscription info might be available depending on Alpaca API version
            subscription_info = {
                'plan': 'Free Tier',  # Default for paper trading
                'data_subscriptions': data_source.upper()
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {str(e)}")
    
    return render_template(
        'api_config.html',
        api_key=api_key,
        api_secret=api_secret,
        api_base_url=api_base_url,
        data_source=data_source,
        account_info=account_info,
        subscription_info=subscription_info
    )

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        api_secret = data.get('api_secret', '').strip()
        api_base_url = data.get('api_base_url', 'https://paper-api.alpaca.markets')
        
        if not api_key or not api_secret:
            return jsonify({"success": False, "message": "API key and secret are required"})
        
        # Create a temporary API connection to test credentials
        import alpaca_trade_api as tradeapi
        
        api = tradeapi.REST(
            api_key,
            api_secret,
            api_base_url,
            api_version='v2'
        )
        
        # Try to get account info to test connection
        account = api.get_account()
        
        # If we got here, the connection is successful
        return jsonify({
            "success": True,
            "message": "Connection successful",
            "account": {
                "account_number": account.account_number,
                "cash": account.cash,
                "portfolio_value": account.portfolio_value,
                "buying_power": account.buying_power,
                "equity": account.equity,
                "status": account.status
            }
        })
        
    except Exception as e:
        logger.error(f"Error testing API connection: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

# Initialize the database and trading engine
with app.app_context():
    from models import Trade, Position, PerformanceMetric
    db.create_all()
    
    # Initialize trading components
    try:
        trading_strategy = init_trading_engine()
        logger.info("Trading engine initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing trading engine: {str(e)}")
