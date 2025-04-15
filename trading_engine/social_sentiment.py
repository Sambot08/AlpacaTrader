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
from app import db
from models import SocialSentiment

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
    
    def store_sentiment_in_db(self, symbol, source, sentiment_score):
        """Store sentiment data in the database."""
        try:
            sentiment_entry = SocialSentiment(
                symbol=symbol,
                source=source,
                sentiment_score=sentiment_score,
                timestamp=datetime.utcnow()
            )
            db.session.add(sentiment_entry)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error storing sentiment data for {symbol} from {source}: {str(e)}")

    def refresh_sentiment_data(self, symbols):
        """Fetch and store sentiment data for all symbols."""
        try:
            for symbol in symbols:
                reddit_sentiment = self.get_reddit_sentiment([symbol])
                stocktwits_sentiment = self.get_stocktwits_sentiment([symbol])

                # Store in database
                self.store_sentiment_in_db(symbol, 'reddit', reddit_sentiment.get(symbol, 0.0))
                self.store_sentiment_in_db(symbol, 'stocktwits', stocktwits_sentiment.get(symbol, 0.0))

                combined_sentiment = self.get_combined_sentiment([symbol])
                self.store_sentiment_in_db(symbol, 'combined', combined_sentiment.get(symbol, 0.0))

            logger.info("Sentiment data refreshed successfully.")
        except Exception as e:
            logger.error(f"Error refreshing sentiment data: {str(e)}")
    
    def _fetch_reddit_posts(self, symbol, subreddit, days_back=1):
        """
        Fetch posts from Reddit containing the symbol using Reddit API
        """
        try:
            # Use the base URL from the environment variable
            base_url = os.environ.get("REDDIT_API_BASE_URL", "https://www.reddit.com")

            # Construct the API URL dynamically
            api_url = f"{base_url}/r/{subreddit}/search.json"

            headers = {
                'User-Agent': 'AlpacaTrader/1.0',
                'Authorization': f'Bearer {os.getenv("REDDIT_API_TOKEN")}'
            }
            params = {
                'q': symbol,
                'restrict_sr': True,
                'sort': 'new',
                'limit': 100
            }
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            posts = []
            for post in data['data']['children']:
                created_utc = post['data']['created_utc']
                if created_utc >= (datetime.now() - timedelta(days=days_back)).timestamp():
                    posts.append({
                        'title': post['data']['title'],
                        'body': post['data'].get('selftext', ''),
                        'upvotes': post['data']['ups'],
                        'created_utc': created_utc
                    })

            return posts
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {symbol} from r/{subreddit}: {str(e)}")
            return []

    def _fetch_stocktwits_messages(self, symbol):
        """
        Fetch messages from StockTwits for a symbol using StockTwits API
        """
        try:
            # Replace with actual StockTwits API integration
            base_url = f'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json'
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()

            messages = []
            for message in data['messages']:
                messages.append({
                    'body': message['body'],
                    'sentiment': message.get('entities', {}).get('sentiment', {}).get('basic'),
                    'created_at': message['created_at']
                })

            return messages
        except Exception as e:
            logger.error(f"Error fetching StockTwits messages for {symbol}: {str(e)}")
            return []