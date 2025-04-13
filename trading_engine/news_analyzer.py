import os
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from trafilatura import fetch_url, extract

# Set up logging
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """
    Module for analyzing financial news and extracting sentiment data 
    to enhance trading decisions.
    """
    
    def __init__(self):
        """Initialize the news analyzer with required NLTK components"""
        
        try:
            # Download NLTK data for sentiment analysis
            nltk.download('punkt', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("News analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing news analyzer: {str(e)}")
            raise
    
    def get_financial_news(self, symbols, days_back=3):
        """
        Get financial news for a list of stock symbols
        
        Args:
            symbols (list): List of stock symbols
            days_back (int): Number of days of historical news to fetch
            
        Returns:
            dict: Dictionary mapping symbols to their news data with sentiment scores
        """
        news_data = {}
        
        for symbol in symbols:
            try:
                # Get news for this symbol
                symbol_news = self._fetch_symbol_news(symbol, days_back)
                
                if symbol_news:
                    # Analyze sentiment for each news item
                    for news in symbol_news:
                        if 'title' in news and 'summary' in news:
                            # Analyze title and summary sentiment
                            title_scores = self.sentiment_analyzer.polarity_scores(news['title'])
                            summary_scores = self.sentiment_analyzer.polarity_scores(news['summary'])
                            
                            # Combine scores with emphasis on title (0.4) and summary (0.6)
                            news['sentiment_score'] = (
                                title_scores['compound'] * 0.4 + 
                                summary_scores['compound'] * 0.6
                            )
                            
                            # Add sentiment classification
                            if news['sentiment_score'] >= 0.05:
                                news['sentiment'] = 'positive'
                            elif news['sentiment_score'] <= -0.05:
                                news['sentiment'] = 'negative'
                            else:
                                news['sentiment'] = 'neutral'
                    
                    news_data[symbol] = symbol_news
                else:
                    logger.warning(f"No news found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {str(e)}")
        
        return news_data
    
    def get_news_sentiment_score(self, symbols, days_back=3):
        """
        Get an aggregated sentiment score for each symbol based on recent news
        
        Args:
            symbols (list): List of stock symbols
            days_back (int): Number of days of historical news to analyze
            
        Returns:
            dict: Dictionary mapping symbols to their sentiment scores (-1 to 1)
        """
        sentiment_scores = {}
        
        # Get news with sentiment analysis
        news_data = self.get_financial_news(symbols, days_back)
        
        for symbol, news_items in news_data.items():
            if news_items:
                # Calculate weighted sentiment score based on recency
                # More recent news has higher weight
                total_weight = 0
                weighted_score = 0
                
                for i, news in enumerate(news_items):
                    # More recent news (lower index) gets higher weight
                    weight = 1.0 / (i + 1)  
                    if 'sentiment_score' in news:
                        weighted_score += news['sentiment_score'] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    sentiment_scores[symbol] = weighted_score / total_weight
                else:
                    sentiment_scores[symbol] = 0
            else:
                sentiment_scores[symbol] = 0
        
        # For symbols with no news, set a neutral score
        for symbol in symbols:
            if symbol not in sentiment_scores:
                sentiment_scores[symbol] = 0
        
        return sentiment_scores
    
    def _fetch_symbol_news(self, symbol, days_back=3):
        """
        Fetch recent news for a specific symbol
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days of historical news to fetch
            
        Returns:
            list: List of news items for the symbol
        """
        # For free sources, we can use Yahoo Finance or MarketWatch
        # This example uses a manual scraping approach
        
        news_items = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # Yahoo Finance URL for the symbol
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            
            # Fetch the webpage
            downloaded = fetch_url(url)
            
            # Extract text content
            content = extract(downloaded)
            
            if content:
                # Parse the content to extract news
                # This is a simplified approach - in a production system, 
                # you would want to use a proper HTML parser or API
                
                # For demonstration purposes, we'll create sample news items
                # based on the URL extraction
                news_items.append({
                    'title': f"Recent news for {symbol}",
                    'summary': content[:500] + "...",  # First 500 chars as summary
                    'url': url,
                    'published_date': datetime.now().isoformat()
                })
            
            # MarketWatch as a secondary source
            url2 = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
            downloaded2 = fetch_url(url2)
            content2 = extract(downloaded2)
            
            if content2:
                news_items.append({
                    'title': f"MarketWatch updates for {symbol}",
                    'summary': content2[:500] + "...", 
                    'url': url2,
                    'published_date': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
        
        return news_items
    
    def analyze_news_impact(self, symbols, historical_prices):
        """
        Analyze how news sentiment correlates with price movements
        
        Args:
            symbols (list): List of stock symbols
            historical_prices (dict): Dictionary of historical price data
            
        Returns:
            dict: Analysis of news impact on prices
        """
        results = {}
        
        for symbol in symbols:
            if symbol in historical_prices:
                # Get sentiment scores for news
                sentiment_data = self.get_news_sentiment_score([symbol])
                
                if symbol in sentiment_data:
                    sentiment_score = sentiment_data[symbol]
                    
                    # Get price data
                    price_data = historical_prices[symbol]
                    
                    # Calculate price change
                    if not price_data.empty and len(price_data) > 1:
                        latest_price = price_data['close'].iloc[-1]
                        prev_price = price_data['close'].iloc[-2]
                        price_change_pct = (latest_price - prev_price) / prev_price * 100
                        
                        results[symbol] = {
                            'sentiment_score': sentiment_score,
                            'price_change_pct': price_change_pct,
                            'correlation': 'positive' if (sentiment_score > 0 and price_change_pct > 0) or 
                                                      (sentiment_score < 0 and price_change_pct < 0) 
                                          else 'negative'
                        }
        
        return results