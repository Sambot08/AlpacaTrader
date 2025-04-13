import os
import re
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class SocialSentimentAnalyzer:
    """
    Analyzes social media and forum sentiment for trading decisions.
    Incorporates data from sources like Reddit (r/wallstreetbets, r/investing),
    Twitter/X, and StockTwits.
    """
    
    def __init__(self):
        """Initialize the social sentiment analyzer"""
        try:
            # Download NLTK data for sentiment analysis if not already downloaded
            nltk.download('vader_lexicon', quiet=True)
            
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Add finance-specific terms to the sentiment analyzer
            self._add_finance_lexicon()
            
            logger.info("Social sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing social sentiment analyzer: {str(e)}")
            raise
    
    def _add_finance_lexicon(self):
        """Add finance-specific terms to improve sentiment analysis"""
        # Bullish terms
        self.sentiment_analyzer.lexicon.update({
            'bullish': 3.0,
            'buy': 2.0,
            'long': 1.5,
            'calls': 1.5,
            'undervalued': 2.0,
            'moon': 3.0,
            'rocket': 3.0,
            'dip': 0.0,  # Neutral since it could be "buy the dip" or just "dip"
            'rally': 2.0,
            'breakout': 2.0,
        })
        
        # Bearish terms
        self.sentiment_analyzer.lexicon.update({
            'bearish': -3.0,
            'sell': -2.0,
            'short': -1.5,
            'puts': -1.5,
            'overvalued': -2.0,
            'crash': -3.0,
            'tank': -2.5,
            'drilling': -2.0,
            'bag': -1.5,  # As in "bag holder"
            'dump': -2.5,
        })
    
    def get_reddit_sentiment(self, symbols, subreddits=None, days_back=1):
        """
        Get sentiment from Reddit posts and comments
        
        Args:
            symbols (list): List of stock symbols
            subreddits (list): List of subreddits to search (default: wallstreetbets, investing)
            days_back (int): Number of days of historical data to analyze
            
        Returns:
            dict: Dictionary mapping symbols to their Reddit sentiment scores
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks']
            
        sentiment_scores = defaultdict(list)
        
        # For each symbol, analyze posts from the subreddits
        for symbol in symbols:
            try:
                # Search for posts containing the symbol
                for subreddit in subreddits:
                    posts = self._fetch_reddit_posts(symbol, subreddit, days_back)
                    
                    if posts:
                        for post in posts:
                            # Calculate sentiment for title and body
                            title_scores = self.sentiment_analyzer.polarity_scores(post['title'])
                            body_scores = self.sentiment_analyzer.polarity_scores(post['body'])
                            
                            # Combined sentiment (title is more important for sentiment)
                            combined_score = title_scores['compound'] * 0.6 + body_scores['compound'] * 0.4
                            
                            # Add to sentiment scores with multiplier based on votes
                            # Higher upvoted posts have more weight
                            vote_multiplier = min(1.0 + (post['upvotes'] / 100), 5.0)
                            
                            sentiment_scores[symbol].append(combined_score * vote_multiplier)
            
            except Exception as e:
                logger.error(f"Error analyzing Reddit sentiment for {symbol}: {str(e)}")
        
        # Calculate average sentiment score for each symbol
        avg_sentiment = {}
        for symbol, scores in sentiment_scores.items():
            if scores:
                avg_sentiment[symbol] = sum(scores) / len(scores)
            else:
                avg_sentiment[symbol] = 0.0
        
        # Ensure all symbols have a score
        for symbol in symbols:
            if symbol not in avg_sentiment:
                avg_sentiment[symbol] = 0.0
        
        return avg_sentiment
    
    def get_stocktwits_sentiment(self, symbols, days_back=1):
        """
        Get sentiment from StockTwits messages
        
        Args:
            symbols (list): List of stock symbols
            days_back (int): Number of days of historical data to analyze
            
        Returns:
            dict: Dictionary mapping symbols to their StockTwits sentiment scores
        """
        sentiment_scores = {}
        
        for symbol in symbols:
            try:
                # Fetch StockTwits messages
                messages = self._fetch_stocktwits_messages(symbol)
                
                if messages:
                    # Analyze sentiment of each message
                    scores = []
                    
                    for msg in messages:
                        # If StockTwits provides sentiment, use it
                        if 'sentiment' in msg and msg['sentiment'] in ['bullish', 'bearish']:
                            if msg['sentiment'] == 'bullish':
                                scores.append(0.75)  # Bullish sentiment
                            else:
                                scores.append(-0.75)  # Bearish sentiment
                        else:
                            # Otherwise, analyze the text
                            if 'body' in msg:
                                sentiment = self.sentiment_analyzer.polarity_scores(msg['body'])
                                scores.append(sentiment['compound'])
                    
                    # Calculate average sentiment
                    if scores:
                        sentiment_scores[symbol] = sum(scores) / len(scores)
                    else:
                        sentiment_scores[symbol] = 0.0
                else:
                    sentiment_scores[symbol] = 0.0
                    
            except Exception as e:
                logger.error(f"Error analyzing StockTwits sentiment for {symbol}: {str(e)}")
                sentiment_scores[symbol] = 0.0
        
        return sentiment_scores
    
    def get_combined_sentiment(self, symbols, days_back=1):
        """
        Get combined sentiment from all social sources
        
        Args:
            symbols (list): List of stock symbols
            days_back (int): Number of days of historical data to analyze
            
        Returns:
            dict: Dictionary mapping symbols to their overall social sentiment scores
        """
        # Weight for each source
        weights = {
            'reddit': 0.5,
            'stocktwits': 0.5
        }
        
        # Get sentiment from each source
        reddit_sentiment = self.get_reddit_sentiment(symbols, days_back=days_back)
        stocktwits_sentiment = self.get_stocktwits_sentiment(symbols, days_back=days_back)
        
        # Combine sentiments with weights
        combined_sentiment = {}
        
        for symbol in symbols:
            reddit_score = reddit_sentiment.get(symbol, 0.0)
            stocktwits_score = stocktwits_sentiment.get(symbol, 0.0)
            
            # Calculate weighted average
            combined_sentiment[symbol] = (
                reddit_score * weights['reddit'] +
                stocktwits_score * weights['stocktwits']
            )
        
        return combined_sentiment
    
    def _fetch_reddit_posts(self, symbol, subreddit, days_back=1):
        """
        Fetch posts from Reddit containing the symbol
        
        Args:
            symbol (str): Stock symbol
            subreddit (str): Subreddit to search
            days_back (int): Number of days back to search
            
        Returns:
            list: List of posts with title, body, and upvotes
        """
        # This is a simplified version - in production, you would use 
        # the Reddit API with proper authentication
        
        try:
            # For demonstration, we'll return a simple example
            # In a real implementation, you would make an API call to Reddit
            
            # Sample posts to simulate API response
            sample_posts = [
                {
                    'title': f"DD on {symbol} - Great potential for growth",
                    'body': f"I've analyzed {symbol} and found it to be undervalued. The company has solid fundamentals and upcoming catalysts.",
                    'upvotes': 25,
                    'created_utc': (datetime.now() - timedelta(hours=6)).timestamp()
                },
                {
                    'title': f"Is {symbol} a good buy right now?",
                    'body': f"Looking at the chart for {symbol}, it seems to be at a support level. What do you think?",
                    'upvotes': 5,
                    'created_utc': (datetime.now() - timedelta(hours=12)).timestamp()
                }
            ]
            
            # Filter by date
            min_timestamp = (datetime.now() - timedelta(days=days_back)).timestamp()
            filtered_posts = [post for post in sample_posts if post['created_utc'] >= min_timestamp]
            
            return filtered_posts
        
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {symbol} from r/{subreddit}: {str(e)}")
            return []
    
    def _fetch_stocktwits_messages(self, symbol):
        """
        Fetch messages from StockTwits for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            list: List of messages with body and sentiment
        """
        try:
            # For demonstration purposes - in production, you would use the StockTwits API
            # Sample messages to simulate API response
            sample_messages = [
                {
                    'body': f"$${symbol} looking strong today, I'm bullish on this one!",
                    'sentiment': 'bullish',
                    'created_at': (datetime.now() - timedelta(hours=2)).isoformat()
                },
                {
                    'body': f"$${symbol} breaking through resistance, could see new highs soon.",
                    'sentiment': 'bullish', 
                    'created_at': (datetime.now() - timedelta(hours=5)).isoformat()
                },
                {
                    'body': f"$${symbol} earnings weren't great, but the stock is holding up.",
                    'sentiment': None,  # No explicit sentiment provided
                    'created_at': (datetime.now() - timedelta(hours=8)).isoformat()
                }
            ]
            
            return sample_messages
            
        except Exception as e:
            logger.error(f"Error fetching StockTwits messages for {symbol}: {str(e)}")
            return []