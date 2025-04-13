import logging
import os
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from app import db
from models import TradeRecord, PerformanceMetric, TradingStrategy
from ml_models import MLModelFactory
from data_processor import DataProcessor
from risk_manager import RiskManager
from performance_tracker import PerformanceTracker

# Import new enhanced data source modules
from trading_engine.news_analyzer import NewsAnalyzer
from trading_engine.social_sentiment import SocialSentimentAnalyzer
from trading_engine.fundamental_analysis import FundamentalAnalyzer
from trading_engine.data_integration import EnhancedDataIntegrator

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine that orchestrates the data collection, analysis, 
    and trade execution processes with enhanced data sources including:
    - Technical analysis (price, volume, indicators)
    - News sentiment analysis
    - Social media sentiment analysis
    - Fundamental company data
    - Economic indicators
    """
    
    def __init__(self, data_processor):
        """Initialize the trading engine with Alpaca API credentials."""
        self.api_key = os.environ.get('ALPACA_API_KEY')
        self.api_secret = os.environ.get('ALPACA_API_SECRET')
        self.base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Initialize Alpaca API only if credentials are available
        self.api = None
        if self.api_key and self.api_secret:
            try:
                self.api = tradeapi.REST(
                    self.api_key, 
                    self.api_secret, 
                    self.base_url, 
                    api_version='v2'
                )
                logger.info("Trading Engine: Alpaca API initialized successfully")
            except Exception as e:
                logger.error(f"Trading Engine: Failed to initialize Alpaca API: {str(e)}")
        else:
            logger.warning("Trading Engine: Alpaca API credentials not found. Trading features will be unavailable.")
        
        self.data_processor = data_processor
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.ml_model_factory = MLModelFactory()
        
        # Initialize enhanced data sources
        try:
            self.data_integrator = EnhancedDataIntegrator()
            logger.info("Enhanced data integrator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing enhanced data integrator: {str(e)}")
            self.data_integrator = None
        
        # Dictionary to store active trading threads
        self.active_strategies = {}
        self.running = False
        
    def start_strategy(self, strategy):
        """Start trading with a particular strategy."""
        if strategy.id in self.active_strategies:
            logger.warning(f"Strategy {strategy.name} is already running")
            return
        
        # Check if API is available
        if self.api is None:
            logger.warning(f"Cannot start strategy {strategy.name}: Alpaca API not initialized")
            return
        
        # Create a new thread for this strategy
        strategy_thread = threading.Thread(
            target=self._run_strategy_loop,
            args=(strategy,),
            daemon=True
        )
        
        self.active_strategies[strategy.id] = {
            'thread': strategy_thread,
            'running': True
        }
        
        strategy_thread.start()
        logger.info(f"Started trading strategy: {strategy.name}")
    
    def stop_strategy(self, strategy):
        """Stop a running strategy."""
        if strategy.id not in self.active_strategies:
            logger.warning(f"Strategy {strategy.name} is not running")
            return
        
        self.active_strategies[strategy.id]['running'] = False
        logger.info(f"Stopping trading strategy: {strategy.name}")
        
        # Remove from active strategies
        del self.active_strategies[strategy.id]
    
    def _run_strategy_loop(self, strategy):
        """Main loop for running a trading strategy."""
        logger.info(f"Running strategy loop for {strategy.name}")
        
        # Check if API is available
        if self.api is None:
            logger.error(f"Cannot run strategy loop for {strategy.name}: Alpaca API not initialized")
            return
        
        # Parse symbols from comma-separated string
        symbols = [s.strip() for s in strategy.symbols.split(',')]
        
        # Initialize ML model based on strategy configuration
        ml_model = self.ml_model_factory.get_model(strategy.ml_model)
        
        # Continue running until the strategy is stopped
        while (strategy.id in self.active_strategies and 
               self.active_strategies[strategy.id]['running']):
            try:
                # Check if market is open
                try:
                    clock = self.api.get_clock()
                    is_market_open = clock.is_open
                except Exception as e:
                    logger.error(f"Error checking market status: {str(e)}")
                    # Assume market is open during weekdays, 9:30 AM - 4:00 PM ET
                    now = datetime.now()
                    is_weekday = now.weekday() < 5  # Monday to Friday
                    est_hour = (now.hour - 4) % 24  # Rough EST conversion (UTC-4)
                    is_market_hours = 9 <= est_hour < 16 or (est_hour == 9 and now.minute >= 30)
                    is_market_open = is_weekday and is_market_hours
                
                if not is_market_open:
                    # Market is closed, check again in 10 minutes
                    logger.info("Market is closed. Sleeping for 10 minutes...")
                    time.sleep(600)
                    continue
                
                # Process each symbol in the strategy
                for symbol in symbols:
                    self._process_symbol(symbol, strategy, ml_model)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for 5 minutes before checking again
                time.sleep(300)
            
            except Exception as e:
                logger.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)  # Sleep and retry on error
    
    def _process_symbol(self, symbol, strategy, ml_model):
        """Process a single symbol for trading decisions with enhanced data sources."""
        logger.info(f"Processing symbol {symbol} for strategy {strategy.name}")
        
        try:
            # Check if API is available
            if self.api is None:
                logger.error(f"Cannot process symbol {symbol}: Alpaca API not initialized")
                return
                
            # Get historical data
            historical_data = self.data_processor.get_historical_data(symbol, days=60)
            
            if historical_data is None or historical_data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return
            
            # Enrich the data with enhanced sources if available
            enhanced_data = {}
            try:
                if self.data_integrator:
                    logger.info(f"Enriching data for {symbol} with alternative data sources")
                    # Put the historical data in a dictionary with the symbol as key
                    price_data = {symbol: historical_data}
                    # Enrich with news, social, and fundamental data
                    enhanced_data = self.data_integrator.enrich_market_data([symbol], price_data)
                    
                    if symbol in enhanced_data:
                        logger.info(f"Successfully enriched data for {symbol}")
                        # Prepare features with enhanced data
                        enhanced_features = self.data_integrator.prepare_enhanced_features(enhanced_data)
                        
                        if symbol in enhanced_features:
                            # Use enhanced features for prediction
                            features = enhanced_features[symbol].values
                        else:
                            # Fall back to standard features if enhanced preparation failed
                            features = self.data_processor.prepare_features(historical_data)
                    else:
                        # Fall back to standard features
                        features = self.data_processor.prepare_features(historical_data)
                else:
                    # Fall back to standard features if data integrator is not available
                    features = self.data_processor.prepare_features(historical_data)
            except Exception as e:
                logger.error(f"Error enriching data for {symbol}: {str(e)}")
                # Fall back to standard features on error
                features = self.data_processor.prepare_features(historical_data)
            
            # Make prediction with ML model
            prediction = ml_model.predict(features)
            prediction_prob = ml_model.predict_proba(features)
            
            logger.info(f"Prediction for {symbol}: {prediction} with confidence {prediction_prob}")
            
            # Get trading signals from enhanced data if available
            enhanced_signals = None
            if self.data_integrator and symbol in enhanced_data:
                try:
                    # Create ML prediction dict for data integrator
                    ml_predictions = {
                        symbol: {
                            'action': 'buy' if prediction == 1 else 'sell',
                            'confidence': prediction_prob
                        }
                    }
                    
                    # Get trading signals incorporating all data sources
                    signals = self.data_integrator.create_trading_signals(enhanced_data, ml_predictions)
                    
                    if symbol in signals:
                        enhanced_signals = signals[symbol]
                        logger.info(f"Enhanced signals for {symbol}: {enhanced_signals['action']} with confidence {enhanced_signals['confidence']}")
                        
                        # Adjust the ML prediction based on enhanced signals
                        if enhanced_signals['action'] == 'buy':
                            prediction = 1
                            prediction_prob = enhanced_signals['confidence']
                        elif enhanced_signals['action'] == 'sell':
                            prediction = 0
                            prediction_prob = enhanced_signals['confidence']
                except Exception as e:
                    logger.error(f"Error getting enhanced signals for {symbol}: {str(e)}")
            
            # Get current position
            try:
                position = self.api.get_position(symbol)
                current_position_qty = int(position.qty)
                current_position_value = float(position.market_value)
            except Exception as e:
                logger.info(f"No position found for {symbol} or error: {str(e)}")
                # No position currently held
                current_position_qty = 0
                current_position_value = 0
            
            # Get account information
            try:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
            except Exception as e:
                logger.error(f"Error getting account information: {str(e)}")
                # Use a default value for demonstration
                buying_power = 10000.0
            
            # Determine action based on prediction and risk management
            action = self._determine_action(
                symbol, 
                prediction, 
                prediction_prob, 
                current_position_qty,
                current_position_value,
                buying_power,
                strategy
            )
            
            # Execute the action
            if action['type'] != 'HOLD':
                self._execute_trade(symbol, action, strategy)
        
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
    
    def _determine_action(self, symbol, prediction, prediction_prob, current_position_qty, 
                          current_position_value, buying_power, strategy):
        """
        Determine trading action based on prediction, current positions, 
        and risk management rules.
        """
        # Default action is to hold
        action = {'type': 'HOLD', 'qty': 0}
        
        # Check if API is available
        if self.api is None:
            logger.error(f"Cannot determine action for {symbol}: Alpaca API not initialized")
            return action
            
        # Get the current price from data processor (which will handle API unavailable case)
        try:
            prices = self.data_processor.get_latest_prices([symbol])
            if symbol in prices:
                current_price = prices[symbol]
            else:
                logger.error(f"No price available for {symbol}")
                return action
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return action
        
        # Apply risk management rules
        max_position_size = self.risk_manager.calculate_position_size(
            buying_power, 
            strategy.risk_level,
            prediction_prob
        )
        
        # Calculate the number of shares to buy or sell
        if prediction == 1:  # Buy signal
            if current_position_qty <= 0:
                # Calculate how many shares to buy
                qty_to_buy = int(max_position_size / current_price)
                
                if qty_to_buy > 0:
                    action = {'type': 'BUY', 'qty': qty_to_buy}
                    logger.info(f"Decision: BUY {qty_to_buy} shares of {symbol}")
            else:
                # Already have a long position, may add more based on confidence
                if prediction_prob > 0.75:  # High confidence
                    additional_qty = int((max_position_size - current_position_value) / current_price)
                    if additional_qty > 0:
                        action = {'type': 'BUY', 'qty': additional_qty}
                        logger.info(f"Decision: BUY additional {additional_qty} shares of {symbol}")
        
        elif prediction == 0:  # Sell signal
            if current_position_qty > 0:
                # Sell all current position
                action = {'type': 'SELL', 'qty': current_position_qty}
                logger.info(f"Decision: SELL {current_position_qty} shares of {symbol}")
            elif current_position_qty < 0:
                # May reduce short position based on confidence
                if prediction_prob < 0.6:  # Low confidence
                    action = {'type': 'BUY', 'qty': abs(current_position_qty)}
                    logger.info(f"Decision: BUY to cover {abs(current_position_qty)} shares of {symbol}")
        
        return action
    
    def _execute_trade(self, symbol, action, strategy):
        """Execute a trade based on the determined action."""
        try:
            # Check if API is available
            if self.api is None:
                logger.error(f"Cannot execute trade for {symbol}: Alpaca API not initialized")
                return
                
            order_id = None
            if action['type'] == 'BUY':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=action['qty'],
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                order_id = order.id
                logger.info(f"BUY order submitted for {action['qty']} shares of {symbol}")
            
            elif action['type'] == 'SELL':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=action['qty'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                order_id = order.id
                logger.info(f"SELL order submitted for {action['qty']} shares of {symbol}")
            
            # If no order was submitted, exit
            if not order_id:
                return
                
            # Get latest price for this symbol
            prices = self.data_processor.get_latest_prices([symbol])
            price = prices.get(symbol, 0.0)
                
            # Record the trade in the database
            trade = TradeRecord(
                symbol=symbol,
                action=action['type'],
                quantity=action['qty'],
                price=price,  # Use latest price as estimate
                total_amount=price * action['qty'],
                order_id=order_id,
                strategy_id=strategy.id
            )
            
            db.session.add(trade)
            db.session.commit()
            
            # Wait for order to fill and update the trade record
            self._wait_for_order_fill(order_id, trade.id)
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
    
    def _wait_for_order_fill(self, order_id, trade_id):
        """Wait for an order to fill and update the trade record."""
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Get the order status
                order = self.api.get_order(order_id)
                
                if order.status == 'filled':
                    # Update the trade record with fill details
                    trade = TradeRecord.query.get(trade_id)
                    if trade:
                        trade.price = float(order.filled_avg_price)
                        trade.total_amount = float(order.filled_avg_price) * float(order.filled_qty)
                        trade.status = 'FILLED'
                        db.session.commit()
                    
                    logger.info(f"Order {order_id} filled at price {order.filled_avg_price}")
                    return
                
                # If order is not filled, wait and check again
                time.sleep(5)
                attempts += 1
            
            except Exception as e:
                logger.error(f"Error checking order status: {str(e)}")
                time.sleep(5)
                attempts += 1
        
        logger.warning(f"Order {order_id} not filled after {max_attempts} attempts")
    
    def _update_performance_metrics(self):
        """Update performance metrics in the database."""
        try:
            # Get account information
            account = self.api.get_account()
            
            # Calculate daily return if we have previous metrics
            previous_metric = PerformanceMetric.query.order_by(
                PerformanceMetric.date.desc()
            ).first()
            
            daily_return = None
            if previous_metric:
                previous_value = previous_metric.portfolio_value
                current_value = float(account.portfolio_value)
                if previous_value > 0:
                    daily_return = (current_value - previous_value) / previous_value * 100
            
            # Create new performance metric
            metric = PerformanceMetric(
                date=datetime.utcnow().date(),
                portfolio_value=float(account.portfolio_value),
                cash_balance=float(account.cash),
                equity_value=float(account.equity),
                daily_return=daily_return,
                pnl=float(account.equity) - float(account.last_equity),
                # Other metrics would require more historical data
            )
            
            db.session.add(metric)
            db.session.commit()
            
            logger.info("Updated performance metrics")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
