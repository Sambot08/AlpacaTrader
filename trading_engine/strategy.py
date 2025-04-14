import logging
import time
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Implements the trading strategy using data fetcher, ML model, risk manager, and trade executor
    """
    
    def __init__(self, data_fetcher, ml_model, risk_manager, trade_executor, performance_tracker):
        """
        Initialize the trading strategy
        
        Args:
            data_fetcher: Data fetcher instance
            ml_model: ML model instance
            risk_manager: Risk manager instance
            trade_executor: Trade executor instance
            performance_tracker: Performance tracker instance
        """
        self.data_fetcher = data_fetcher
        self.ml_model = ml_model
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.performance_tracker = performance_tracker
        
        # Default symbols to trade
        self.symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META']
        self.historical_data = {}
        self.last_data_update = None
        
        # Strategy parameters
        self.lookback_days = 60  # 60 days of historical data
        self.confidence_threshold = 0.65  # Minimum confidence to enter a trade
        self.model_type = "classification"  # classification or regression
        self.update_interval = 3600  # Update historical data every hour
        
        # Initialize database
        self.initialize()
        
    def initialize(self):
        """
        Initialize the strategy - fetch initial data and train models
        """
        try:
            # Fetch initial historical data
            self.update_historical_data()
            
            # Train initial models
            for symbol in self.symbols:
                if symbol in self.historical_data:
                    self.ml_model.train_model(symbol, self.historical_data[symbol], self.model_type)
                    
            # Initial portfolio update
            account_info = self.trade_executor.get_account()
            if account_info:
                self.performance_tracker.update_portfolio_value(account_info['portfolio_value'])
                
            logger.info("Trading strategy initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading strategy: {str(e)}")
            
    def update_symbols(self, symbols):
        """
        Update the list of symbols to trade
        
        Args:
            symbols (list): List of ticker symbols
        """
        self.symbols = symbols
        logger.info(f"Updated symbols to trade: {', '.join(symbols)}")
        
        # Fetch data for new symbols
        self.update_historical_data()
        
        # Train models for new symbols
        for symbol in symbols:
            if symbol in self.historical_data and self.ml_model.needs_training(symbol):
                self.ml_model.train_model(symbol, self.historical_data[symbol], self.model_type)
                
    def update_historical_data(self, force=False):
        """
        Update historical data for all symbols
        
        Args:
            force (bool): Force update even if not due
        """
        # Check if update is needed
        if not force and self.last_data_update:
            time_since_update = datetime.now() - self.last_data_update
            if time_since_update.total_seconds() < self.update_interval:
                return
                
        try:
            # Calculate start date
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
            
            # Fetch data for all symbols
            data = self.data_fetcher.get_historical_data(
                self.symbols,
                timeframe='1D',
                start_date=start_date,
                end_date=end_date
            )
            
            # Check for insufficient data
            for symbol, data in data.items():
                if len(data) < self.lookback_days:
                    logger.warning(f"Insufficient data for {symbol} to train model")
                    continue

                # Update historical data for valid symbols
                self.historical_data[symbol] = data

            self.last_data_update = datetime.now()
            
            logger.info(f"Updated historical data for {len(data)} symbols")
        except Exception as e:
            logger.error(f"Error updating historical data: {str(e)}")
            
    def execute_strategy(self):
        """
        Execute the trading strategy
        """
        logger.info("Executing trading strategy...")
        
        try:
            # Update historical data if needed
            self.update_historical_data()
            
            # Check if models need training
            for symbol in self.symbols:
                if symbol in self.historical_data and self.ml_model.needs_training(symbol):
                    self.ml_model.train_model(symbol, self.historical_data[symbol], self.model_type)
                    
            # Get account info
            account_info = self.trade_executor.get_account()
            if not account_info:
                logger.error("Failed to get account information")
                return
                
            portfolio_value = account_info['portfolio_value']
            buying_power = account_info['buying_power']
            
            # Update portfolio value
            self.performance_tracker.update_portfolio_value(portfolio_value)
            
            # Check current positions
            current_positions = self.trade_executor.get_positions()
            position_symbols = [p['symbol'] for p in current_positions]
            
            # Get latest market data
            latest_data = {}
            for symbol in self.symbols:
                if symbol in self.historical_data:
                    latest_data[symbol] = self.historical_data[symbol]
                    
            # Make predictions and execute trades
            for symbol, data in latest_data.items():
                try:
                    # Make prediction
                    prediction = self.ml_model.predict(symbol, data)
                    signal = prediction.get('signal', 'HOLD')
                    confidence = prediction.get('confidence', 0)
                    
                    logger.info(f"Prediction for {symbol}: {signal} (confidence: {confidence:.4f})")
                    
                    # Check if we need to exit position
                    if symbol in position_symbols:
                        position = next((p for p in current_positions if p['symbol'] == symbol), None)
                        current_price = position['current_price']
                        
                        # Check exit signals from risk manager
                        exit_signal, exit_reason = self.risk_manager.check_exit_signals(
                            symbol, current_price, position
                        )
                        
                        # Exit position if needed
                        if exit_signal or signal == 'SELL':
                            # Close position
                            result = self.trade_executor.close_position(symbol)
                            if result:
                                logger.info(f"Closed position for {symbol} ({exit_reason})")
                                
                                # Update performance
                                pl = position['unrealized_pl']
                                pl_pct = position['unrealized_plpc']
                                self.performance_tracker.record_trade(
                                    symbol, 'SELL', position['qty'], current_price, pl, pl_pct
                                )
                                
                                # Update risk manager
                                self.risk_manager.update_position(symbol, current_price, position['qty'], 'sell')
                                
                        # Update trailing stop if needed
                        else:
                            self.risk_manager.update_trailing_stop(symbol, current_price)
                            
                    # Check if we should enter new position
                    elif signal == 'BUY' and confidence >= self.confidence_threshold:
                        latest_bar = self.data_fetcher.get_latest_bars([symbol])
                        if not latest_bar or symbol not in latest_bar:
                            logger.warning(f"No data for {symbol}, skipping")
                            continue
                            
                        current_price = latest_bar[symbol]['close']
                        
                        # Calculate position size
                        position_details = self.risk_manager.calculate_position_size(
                            symbol, current_price, portfolio_value, confidence
                        )
                        
                        shares = position_details['shares']
                        
                        # Submit order if shares > 0
                        if shares > 0:
                            # Check if we have enough buying power
                            cost = shares * current_price
                            if cost > buying_power:
                                logger.warning(f"Not enough buying power for {symbol}: need ${cost:.2f}, have ${buying_power:.2f}")
                                continue
                                
                            # Submit market order
                            order = self.trade_executor.submit_order(
                                symbol, shares, 'buy', 'market', 'day'
                            )
                            
                            if order:
                                logger.info(f"Bought {shares} shares of {symbol} @ {current_price:.2f}")
                                
                                # Update risk manager
                                self.risk_manager.update_position(symbol, current_price, shares, 'buy')
                                
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}")
                    
            # Update performance metrics
            positions = self.trade_executor.get_positions()
            self.performance_tracker.update_positions(positions)
            
            logger.info("Strategy execution completed")
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
