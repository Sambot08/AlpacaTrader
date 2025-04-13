import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our custom data modules
from trading_engine.news_analyzer import NewsAnalyzer
from trading_engine.social_sentiment import SocialSentimentAnalyzer
from trading_engine.fundamental_analysis import FundamentalAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedDataIntegrator:
    """
    Integrates multiple data sources for comprehensive market analysis
    and provides enhanced features for the machine learning models.
    """
    
    def __init__(self):
        """Initialize the data integrator and its data sources"""
        try:
            # Initialize the individual data modules
            self.news_analyzer = NewsAnalyzer()
            self.social_analyzer = SocialSentimentAnalyzer()
            self.fundamental_analyzer = FundamentalAnalyzer()
            
            logger.info("Enhanced data integrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing data integrator: {str(e)}")
            raise
    
    def enrich_market_data(self, symbols, price_data):
        """
        Enrich market data with alternative data sources
        
        Args:
            symbols (list): List of stock symbols
            price_data (dict): Historical price data for symbols
            
        Returns:
            dict: Dictionary of integrated data with additional features
        """
        integrated_data = {}
        
        try:
            # 1. Get news sentiment data
            logger.info("Fetching news sentiment data")
            news_sentiment = self.news_analyzer.get_news_sentiment_score(symbols)
            
            # 2. Get social media sentiment
            logger.info("Fetching social media sentiment data")
            social_sentiment = self.social_analyzer.get_combined_sentiment(symbols)
            
            # 3. Get company fundamentals
            logger.info("Fetching fundamental data")
            company_strength = self.fundamental_analyzer.evaluate_company_strength(symbols)
            
            # 4. Get economic indicators
            logger.info("Fetching economic indicators")
            economic_indicators = self.fundamental_analyzer.get_economic_indicators()
            
            # 5. Get sector performance
            logger.info("Fetching sector performance")
            sector_performance = self.fundamental_analyzer.get_sector_performance()
            
            # Integrate all data sources for each symbol
            for symbol in symbols:
                if symbol in price_data and not price_data[symbol].empty:
                    # Start with price data
                    symbol_data = price_data[symbol].copy()
                    
                    # Add sentiment features
                    symbol_data['news_sentiment'] = news_sentiment.get(symbol, 0)
                    symbol_data['social_sentiment'] = social_sentiment.get(symbol, 0)
                    
                    # Add fundamental features
                    symbol_data['company_strength'] = company_strength.get(symbol, 50)
                    
                    # Add the same economic indicators to all symbols
                    for indicator, value in economic_indicators.items():
                        # Clean indicator name for column name
                        col_name = f"econ_{indicator.lower().replace(' ', '_').replace('-', '_')}"
                        symbol_data[col_name] = value
                    
                    # Add combined sentiment score (weighted average of news and social)
                    symbol_data['combined_sentiment'] = (
                        news_sentiment.get(symbol, 0) * 0.6 + 
                        social_sentiment.get(symbol, 0) * 0.4
                    )
                    
                    integrated_data[symbol] = symbol_data
                else:
                    logger.warning(f"No price data available for {symbol}")
            
            logger.info(f"Successfully integrated data for {len(integrated_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error integrating data: {str(e)}")
        
        return integrated_data
    
    def prepare_enhanced_features(self, integrated_data):
        """
        Prepare enhanced feature set for ML models
        
        Args:
            integrated_data (dict): Dictionary of integrated data for each symbol
            
        Returns:
            dict: Dictionary of feature matrices for ML models
        """
        feature_matrices = {}
        
        for symbol, data in integrated_data.items():
            try:
                # Create feature matrix
                features = pd.DataFrame()
                
                # Technical indicators - already in the data
                tech_columns = [
                    'sma_10', 'sma_20', 'sma_50', 
                    'ema_12', 'ema_26',
                    'macd', 'macd_signal', 'macd_hist',
                    'rsi', 'bb_upper', 'bb_lower', 'bb_middle',
                    'atr', 'daily_return', 'volatility'
                ]
                
                # Alternative data columns
                alt_columns = [
                    'news_sentiment', 'social_sentiment', 
                    'company_strength', 'combined_sentiment'
                ]
                
                # Economic indicator columns (dynamically generated)
                econ_columns = [col for col in data.columns if col.startswith('econ_')]
                
                # Combine all features
                all_columns = tech_columns + alt_columns + econ_columns
                
                # Select only columns that exist in the data
                valid_columns = [col for col in all_columns if col in data.columns]
                
                features = data[valid_columns].copy()
                
                # Handle missing values
                features = features.fillna(method='ffill').fillna(0)
                
                # Add momentum and trend features
                for col in tech_columns:
                    if col in features.columns:
                        # Add change over 3 periods
                        features[f"{col}_change_3"] = features[col].pct_change(periods=3)
                        
                        # Add change over 5 periods
                        features[f"{col}_change_5"] = features[col].pct_change(periods=5)
                
                # Add sentiment momentum
                for col in alt_columns:
                    if col in features.columns:
                        # Rolling average of sentiment
                        features[f"{col}_avg_3"] = features[col].rolling(window=3).mean()
                
                # Fill NaN values created by pct_change and rolling calculations
                features = features.fillna(0)
                
                feature_matrices[symbol] = features
                
            except Exception as e:
                logger.error(f"Error preparing enhanced features for {symbol}: {str(e)}")
        
        return feature_matrices
    
    def create_trading_signals(self, integrated_data, ml_predictions=None):
        """
        Generate trading signals based on integrated data and ML predictions
        
        Args:
            integrated_data (dict): Dictionary of integrated data for each symbol
            ml_predictions (dict, optional): ML model predictions
            
        Returns:
            dict: Dictionary of trading signals for each symbol
        """
        signals = {}
        
        for symbol, data in integrated_data.items():
            try:
                # Start with neutral signal
                signal = {
                    'action': 'hold',
                    'confidence': 0.5,
                    'sources': {}
                }
                
                # Add technical signals
                tech_signal, tech_confidence = self._get_technical_signal(data)
                signal['sources']['technical'] = {
                    'signal': tech_signal,
                    'confidence': tech_confidence
                }
                
                # Add sentiment signals
                if 'combined_sentiment' in data.columns:
                    sentiment = data['combined_sentiment'].iloc[-1]
                    sent_signal = 'buy' if sentiment > 0.2 else ('sell' if sentiment < -0.2 else 'hold')
                    sent_confidence = min(abs(sentiment) * 2, 0.9)  # Scale to 0-0.9
                    
                    signal['sources']['sentiment'] = {
                        'signal': sent_signal,
                        'confidence': sent_confidence
                    }
                
                # Add fundamental signals
                if 'company_strength' in data.columns:
                    strength = data['company_strength'].iloc[-1]
                    fund_signal = 'buy' if strength > 65 else ('sell' if strength < 35 else 'hold')
                    fund_confidence = abs(strength - 50) / 50  # Scale to 0-1
                    
                    signal['sources']['fundamental'] = {
                        'signal': fund_signal,
                        'confidence': fund_confidence
                    }
                
                # Add ML predictions if available
                if ml_predictions and symbol in ml_predictions:
                    pred = ml_predictions[symbol]
                    signal['sources']['ml_model'] = {
                        'signal': pred['action'],
                        'confidence': pred['confidence']
                    }
                
                # Combine all signals with weights
                weights = {
                    'technical': 0.3,
                    'sentiment': 0.2,
                    'fundamental': 0.2,
                    'ml_model': 0.3
                }
                
                # Calculate weighted decision
                buy_score = 0
                sell_score = 0
                total_weight = 0
                
                for source, weight in weights.items():
                    if source in signal['sources']:
                        source_signal = signal['sources'][source]
                        if source_signal['signal'] == 'buy':
                            buy_score += source_signal['confidence'] * weight
                        elif source_signal['signal'] == 'sell':
                            sell_score += source_signal['confidence'] * weight
                        total_weight += weight
                
                # Normalize scores
                if total_weight > 0:
                    buy_score /= total_weight
                    sell_score /= total_weight
                
                # Determine final action
                if buy_score > sell_score and buy_score > 0.6:
                    signal['action'] = 'buy'
                    signal['confidence'] = buy_score
                elif sell_score > buy_score and sell_score > 0.6:
                    signal['action'] = 'sell'
                    signal['confidence'] = sell_score
                else:
                    signal['action'] = 'hold'
                    signal['confidence'] = 1 - (buy_score + sell_score) / 2
                
                signals[symbol] = signal
                
            except Exception as e:
                logger.error(f"Error creating trading signals for {symbol}: {str(e)}")
                signals[symbol] = {'action': 'hold', 'confidence': 0.5, 'sources': {}}
        
        return signals
    
    def _get_technical_signal(self, data):
        """
        Generate technical trading signal based on indicators
        
        Args:
            data (DataFrame): Price data with technical indicators
            
        Returns:
            tuple: (signal, confidence) where signal is 'buy', 'sell', or 'hold'
        """
        try:
            # Get the latest data point
            latest = data.iloc[-1]
            
            # Signal components
            signals = []
            
            # 1. Moving Average crossover
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                if latest['sma_20'] > latest['sma_50']:
                    signals.append(('buy', 0.6))  # Bullish MA crossover
                elif latest['sma_20'] < latest['sma_50']:
                    signals.append(('sell', 0.6))  # Bearish MA crossover
            
            # 2. RSI signals
            if 'rsi' in data.columns:
                if latest['rsi'] < 30:
                    signals.append(('buy', 0.7))  # Oversold
                elif latest['rsi'] > 70:
                    signals.append(('sell', 0.7))  # Overbought
            
            # 3. MACD signals
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                if latest['macd'] > latest['macd_signal']:
                    signals.append(('buy', 0.65))  # Bullish MACD crossover
                elif latest['macd'] < latest['macd_signal']:
                    signals.append(('sell', 0.65))  # Bearish MACD crossover
            
            # 4. Bollinger Band signals
            if 'close' in data.columns and 'bb_lower' in data.columns and 'bb_upper' in data.columns:
                if latest['close'] < latest['bb_lower']:
                    signals.append(('buy', 0.7))  # Price below lower band
                elif latest['close'] > latest['bb_upper']:
                    signals.append(('sell', 0.7))  # Price above upper band
            
            # Calculate overall signal
            if not signals:
                return 'hold', 0.5
                
            # Count buy/sell signals and their confidences
            buy_count = 0
            buy_conf_sum = 0
            sell_count = 0
            sell_conf_sum = 0
            
            for signal, conf in signals:
                if signal == 'buy':
                    buy_count += 1
                    buy_conf_sum += conf
                elif signal == 'sell':
                    sell_count += 1
                    sell_conf_sum += conf
            
            # Determine final signal
            if buy_count > sell_count:
                return 'buy', buy_conf_sum / len(signals)
            elif sell_count > buy_count:
                return 'sell', sell_conf_sum / len(signals)
            else:
                # If tied, use confidence to decide
                if buy_conf_sum > sell_conf_sum:
                    return 'buy', buy_conf_sum / len(signals)
                elif sell_conf_sum > buy_conf_sum:
                    return 'sell', sell_conf_sum / len(signals)
                else:
                    return 'hold', 0.5
                
        except Exception as e:
            logger.error(f"Error generating technical signal: {str(e)}")
            return 'hold', 0.5