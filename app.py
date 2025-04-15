import os
import logging
from datetime import datetime
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import dotenv
import praw
import requests

# Load environment variables from .env file
load_dotenv()
dotenv.load_dotenv()

# Local imports
from trading_engine.data_fetcher import AlpacaDataFetcher
from trading_engine.ml_model import MLModel
from trading_engine.risk_manager import RiskManager
from trading_engine.trade_executor import TradeExecutor
from trading_engine.strategy import TradingStrategy
from trading_engine.performance import PerformanceTracker
from extensions import db  # Import db from extensions

# Set up logging
logger = logging.getLogger(__name__)

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

# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent="your_user_agent",
    username="your_reddit_username",
    password="your_reddit_password"
)

# Initialize trading components
def init_trading_engine():
    global data_fetcher, ml_model, risk_manager, trade_executor, trading_strategy, performance_tracker
    
    # Get API credentials from environment variables
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Default to paper trading
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found. Trading engine will be initialized but not active.")
        
    # Ensure base_url does not end with a trailing slash
    base_url = base_url.rstrip('/')
    
    # Ensure base_url does not include '/v2'
    if base_url.endswith('/v2'):
        base_url = base_url[:-3]
    
    # Initialize components
    data_fetcher = AlpacaDataFetcher(api_key, api_secret, base_url)
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
            try:
                trading_thread_object = threading.Thread(target=trading_thread, daemon=True)
                trading_thread_object.start()
            finally:
                trading_thread_object.join()  # Ensure proper cleanup of the thread
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
            market_type = request.form.get('market_type', 'stock')  # Default to US stocks
            
            # Save API credentials if provided
            if api_key and api_secret:
                # In a production app, use a secure method to store credentials
                # For now, we'll store them in environment variables
                os.environ['ALPACA_API_KEY'] = api_key
                os.environ['ALPACA_API_SECRET'] = api_secret
                os.environ['ALPACA_BASE_URL'] = api_base_url
                os.environ['ALPACA_MARKET_TYPE'] = market_type
                
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
    market_type = os.environ.get('ALPACA_MARKET_TYPE', 'stock')
    
    # Get account info if available
    account_info = None
    subscription_info = None
    
    if trade_executor:
        try:
            account_info = trade_executor.get_account()
            # Show subscription info
            subscription_info = {
                'plan': 'Standard',  # Alpaca now provides real-time data
                'market_type': market_type.capitalize()
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {str(e)}")
    
    return render_template(
        'api_config.html',
        api_key=api_key,
        api_secret=api_secret,
        api_base_url=api_base_url,
        market_type=market_type,
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

# Enhanced Data API Endpoints
@app.route('/api/enhanced_data/news_sentiment')
def get_news_sentiment():
    """Get news sentiment data for trading symbols."""
    try:
        # Get the symbols from the active trading strategies
        symbols = []
        for strategy in TradingStrategy.query.filter_by(is_active=True).all():
            symbols.extend([s.strip() for s in strategy.symbols.split(',')])
        symbols = list(set(symbols))  # Remove duplicates
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']  # Default symbols
        
        # Initialize the news analyzer if it's not already initialized
        from trading_engine.news_analyzer import NewsAnalyzer
        news_analyzer = NewsAnalyzer()
        
        # Get news sentiment data
        sentiment_data = news_analyzer.get_news_sentiment_score(symbols)
        
        # Get news items for each symbol
        news_items = {}
        for symbol in symbols:
            news_data = news_analyzer.get_financial_news([symbol])
            if symbol in news_data:
                # Just get the headlines for display
                headlines = []
                for item in news_data[symbol][:3]:  # Get top 3 news items
                    if 'title' in item:
                        headlines.append({
                            'title': item['title'],
                            'sentiment': item.get('sentiment', 'neutral')
                        })
                news_items[symbol] = headlines
        
        # Format the response
        response = []
        for symbol in symbols:
            # Calculate the sentiment trend (positive, negative, neutral)
            sentiment_score = sentiment_data.get(symbol, 0)
            if sentiment_score > 0.2:
                trend = 'positive'
            elif sentiment_score < -0.2:
                trend = 'negative'
            else:
                trend = 'neutral'
                
            response.append({
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'trend': trend,
                'headlines': news_items.get(symbol, []),
                'last_updated': datetime.now().isoformat()
            })
            
        return jsonify({
            'success': True,
            'data': response
        })
        
    except Exception as e:
        logger.error(f"Error getting news sentiment data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error getting news sentiment data: {str(e)}"
        })

@app.route('/api/enhanced_data/social_sentiment')
def get_enhanced_social_sentiment():
    """Get social media sentiment data for trading symbols."""
    try:
        # Get the symbols from the active trading strategies
        symbols = []
        for strategy in TradingStrategy.query.filter_by(is_active=True).all():
            symbols.extend([s.strip() for s in strategy.symbols.split(',')])
        symbols = list(set(symbols))  # Remove duplicates
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']  # Default symbols
        
        # Initialize the social sentiment analyzer
        from trading_engine.social_sentiment import SocialSentimentAnalyzer
        social_analyzer = SocialSentimentAnalyzer()
        
        # Get sentiment data from different sources
        reddit_sentiment = social_analyzer.get_reddit_sentiment(symbols)
        stocktwits_sentiment = social_analyzer.get_stocktwits_sentiment(symbols)
        combined_sentiment = social_analyzer.get_combined_sentiment(symbols)
        
        # Format the response
        response = []
        for symbol in symbols:
            reddit_score = reddit_sentiment.get(symbol, 0)
            stocktwits_score = stocktwits_sentiment.get(symbol, 0)
            combined_score = combined_sentiment.get(symbol, 0)
            
            # Calculate trend
            if combined_score > 0.2:
                trend = 'positive'
            elif combined_score < -0.2:
                trend = 'negative'
            else:
                trend = 'neutral'
            
            # Generate a random number of mentions between 10 and 1000
            # In a real implementation, this would come from the API
            import random
            mentions = random.randint(10, 1000)
            
            response.append({
                'symbol': symbol,
                'reddit_sentiment': reddit_score,
                'stocktwits_sentiment': stocktwits_score,
                'combined_score': combined_score,
                'mentions': mentions,
                'trend': trend
            })
            
        return jsonify({
            'success': True,
            'data': response
        })
        
    except Exception as e:
        logger.error(f"Error getting social sentiment data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error getting social sentiment data: {str(e)}"
        })

@app.route('/api/enhanced_data/fundamental')
def get_fundamental_data():
    """Get fundamental data for trading symbols."""
    try:
        # Get the symbols from the active trading strategies
        symbols = []
        for strategy in TradingStrategy.query.filter_by(is_active=True).all():
            symbols.extend([s.strip() for s in strategy.symbols.split(',')])
        symbols = list(set(symbols))  # Remove duplicates
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']  # Default symbols
        
        # Initialize the fundamental analyzer
        from trading_engine.fundamental_analysis import FundamentalAnalyzer
        fundamental_analyzer = FundamentalAnalyzer()
        
        # Get fundamental data
        company_financials = fundamental_analyzer.get_company_financials(symbols)
        company_strength = fundamental_analyzer.evaluate_company_strength(symbols)
        sector_performance = fundamental_analyzer.get_sector_performance()
        
        # Format the response
        response = []
        for symbol in symbols:
            financials = company_financials.get(symbol, {})
            strength = company_strength.get(symbol, 50)
            
            response.append({
                'symbol': symbol,
                'company_strength': strength,
                'pe_ratio': financials.get('pe_ratio', None) or round(20 + (strength - 50) / 10, 1),
                'profit_margin': financials.get('profit_margin', None) or round(0.1 + (strength - 50) / 500, 3),
                'revenue_growth': financials.get('revenue_growth', None) or round(0.05 + (strength - 50) / 500, 3),
                'debt_to_equity': financials.get('debt_to_equity', None) or round(1.0 - (strength - 50) / 100, 2),
                'sector': 'Technology',  # This would come from real data
                'sector_performance': sector_performance.get('Technology', {}).get('ytd_return', 0.1) 
            })
            
        return jsonify({
            'success': True,
            'data': response
        })
        
    except Exception as e:
        logger.error(f"Error getting fundamental data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error getting fundamental data: {str(e)}"
        })

@app.route('/api/enhanced_data/ai_analysis')
def get_ai_analysis():
    """Get AI analysis and signal integration data."""
    try:
        # Get the symbols from the active trading strategies
        symbols = []
        for strategy in TradingStrategy.query.filter_by(is_active=True).all():
            symbols.extend([s.strip() for s in strategy.symbols.split(',')])
        symbols = list(set(symbols))  # Remove duplicates
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']  # Default symbols
        
        # Initialize the data integrator
        from trading_engine.data_integration import EnhancedDataIntegrator
        data_integrator = EnhancedDataIntegrator()
        
        # In a real implementation, this would use actual data
        # For demonstration, we'll create synthetic data that represents
        # what the actual implementation would return
        response = []
        
        for symbol in symbols:
            # Generate signal values for each data source
            import random
            
            # Technical signal (-1 to 1 scale)
            technical_signal = round(random.uniform(-0.8, 0.8), 2)
            technical_action = 'buy' if technical_signal > 0.2 else ('sell' if technical_signal < -0.2 else 'hold')
            
            # News signal (-1 to 1 scale)
            news_signal = round(random.uniform(-0.8, 0.8), 2)
            news_action = 'buy' if news_signal > 0.2 else ('sell' if news_signal < -0.2 else 'hold')
            
            # Social signal (-1 to 1 scale)
            social_signal = round(random.uniform(-0.8, 0.8), 2)
            social_action = 'buy' if social_signal > 0.2 else ('sell' if social_signal < -0.2 else 'hold')
            
            # Fundamental signal (0 to 100 scale, converted to -1 to 1)
            fundamental_strength = random.randint(20, 80)
            fundamental_signal = round((fundamental_strength - 50) / 50, 2)
            fundamental_action = 'buy' if fundamental_signal > 0.2 else ('sell' if fundamental_signal < -0.2 else 'hold')
            
            # Combined signal calculation (weighted average)
            weights = {
                'technical': 0.3,
                'news': 0.2,
                'social': 0.2,
                'fundamental': 0.3
            }
            
            combined_signal = (
                technical_signal * weights['technical'] +
                news_signal * weights['news'] +
                social_signal * weights['social'] +
                fundamental_signal * weights['fundamental']
            )
            
            combined_signal = round(combined_signal, 2)
            combined_action = 'buy' if combined_signal > 0.2 else ('sell' if combined_signal < -0.2 else 'hold')
            
            # Add to response
            response.append({
                'symbol': symbol,
                'technical': {
                    'signal': technical_signal,
                    'action': technical_action
                },
                'news': {
                    'signal': news_signal,
                    'action': news_action
                },
                'social': {
                    'signal': social_signal,
                    'action': social_action
                },
                'fundamental': {
                    'signal': fundamental_signal,
                    'action': fundamental_action
                },
                'combined': {
                    'signal': combined_signal,
                    'action': combined_action,
                    'confidence': abs(combined_signal)
                }
            })
        
        return jsonify({
            'success': True,
            'data': response
        })
        
    except Exception as e:
        logger.error(f"Error getting AI analysis data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error getting AI analysis data: {str(e)}"
        })

@app.route('/api/social_sentiment', methods=['GET'])
def get_social_sentiment():
    """API endpoint to fetch the latest social sentiment data."""
    try:
        symbol = request.args.get('symbol', None)
        if not symbol:
            return jsonify({"success": False, "message": "Symbol is required"}), 400

        # Query the latest sentiment data for the symbol
        sentiment_data = SocialSentiment.query.filter_by(symbol=symbol).order_by(SocialSentiment.timestamp.desc()).all()

        if not sentiment_data:
            return jsonify({"success": False, "message": "No sentiment data found for the symbol"}), 404

        # Format the response
        response = [
            {
                "source": data.source,
                "sentiment_score": data.sentiment_score,
                "timestamp": data.timestamp
            } for data in sentiment_data
        ]

        return jsonify({"success": True, "data": response})

    except Exception as e:
        logger.error(f"Error fetching social sentiment data: {str(e)}")
        return jsonify({"success": False, "message": "An error occurred while fetching sentiment data"}), 500

@app.route('/api/crypto_price', methods=['GET'])
def get_crypto_price():
    """Fetch live cryptocurrency price for Bitcoin (BTCUSD)."""
    try:
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        api_url = f"{base_url}/v1/crypto/BTCUSD/quote"
        response = requests.get(api_url, headers={"APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": os.environ.get("ALPACA_API_SECRET")})
        response.raise_for_status()
        data = response.json()
        return jsonify({
            "success": True,
            "price": data.get("ask_price", "N/A")
        })
    except Exception as e:
        logger.error(f"Error fetching Bitcoin price: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

# Initialize the database and trading engine
with app.app_context():
    from models import Trade, Position, PerformanceMetric, SocialSentiment
    db.create_all()
    
    # Initialize trading components
    try:
        trading_strategy = init_trading_engine()
        logger.info("Trading engine initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing trading engine: {str(e)}")
