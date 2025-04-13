import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)

class AlpacaDataFetcher:
    """
    Handles fetching market data from Alpaca API
    """
    
    def __init__(self, api_key, api_secret, base_url):
        """
        Initialize the data fetcher with API credentials
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            base_url (str): Alpaca API base URL (paper or live)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        try:
            self.api = tradeapi.REST(
                api_key,
                api_secret,
                base_url,
                api_version='v2'
            )
            logger.info("Successfully connected to Alpaca API")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            raise
        
    def is_market_open(self):
        """
        Check if the market is currently open
        
        Returns:
            bool: True if the market is open, False otherwise
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
            
    def get_market_hours(self):
        """
        Get the current market hours
        
        Returns:
            dict: Dictionary with market open and close times
        """
        try:
            clock = self.api.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat(),
                'next_close': clock.next_close.isoformat(),
                'timestamp': clock.timestamp.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market hours: {str(e)}")
            return None
    
    def get_historical_data(self, symbols, timeframe='1D', start_date=None, end_date=None, limit=100):
        """
        Fetch historical price data for the given symbols
        
        Args:
            symbols (list): List of ticker symbols
            timeframe (str): Time interval (1Min, 5Min, 15Min, 1H, 1D)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            limit (int): Maximum number of data points
            
        Returns:
            dict: Dictionary of DataFrames with historical data for each symbol
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        result = {}
        
        for symbol in symbols:
            try:
                df = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start_date,
                    end=end_date,
                    limit=limit
                ).df
                
                if not df.empty:
                    # Add technical indicators
                    df = self._add_technical_indicators(df)
                    result[symbol] = df
                else:
                    logger.warning(f"No data found for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return result
    
    def get_latest_quotes(self, symbols):
        """
        Get the latest quotes for the given symbols
        
        Args:
            symbols (list): List of ticker symbols
            
        Returns:
            dict: Dictionary of latest quotes
        """
        result = {}
        
        for symbol in symbols:
            try:
                quote = self.api.get_latest_quote(symbol)
                result[symbol] = {
                    'bid': quote.bp,
                    'ask': quote.ap,
                    'bid_size': quote.bs,
                    'ask_size': quote.as_,
                    'timestamp': quote.t
                }
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {str(e)}")
                
        return result
    
    def get_latest_bars(self, symbols):
        """
        Get the latest bars for the given symbols
        
        Args:
            symbols (list): List of ticker symbols
            
        Returns:
            dict: Dictionary of latest bars
        """
        result = {}
        
        for symbol in symbols:
            try:
                bars = self.api.get_latest_bar(symbol)
                result[symbol] = {
                    'open': bars.o,
                    'high': bars.h,
                    'low': bars.l,
                    'close': bars.c,
                    'volume': bars.v,
                    'timestamp': bars.t
                }
            except Exception as e:
                logger.error(f"Error fetching bar for {symbol}: {str(e)}")
                
        return result
    
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with added technical indicators
        """
        # Calculate moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change() * 100
        
        # Calculate volatility (standard deviation of returns)
        df['volatility'] = df['daily_return'].rolling(window=21).std()
        
        return df
