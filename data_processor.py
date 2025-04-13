import logging
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import ta  # Technical Analysis library

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes stock market data for analysis and prediction.
    Fetches data from Alpaca API and computes technical indicators.
    """
    
    def __init__(self):
        """Initialize the DataProcessor with Alpaca API credentials."""
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
                logger.info("Alpaca API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca API: {str(e)}")
        else:
            logger.warning("Alpaca API credentials not found. API features will be unavailable.")
    
    def get_historical_data(self, symbol, days=60, timeframe='1D'):
        """
        Fetch historical data for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days of historical data to fetch
            timeframe (str): Time interval for data points (e.g., '1D', '1H')
            
        Returns:
            pandas.DataFrame: Historical data with OHLCV and computed indicators
        """
        try:
            # Check if API is available
            if self.api is None:
                logger.warning("Cannot fetch historical data: Alpaca API not initialized")
                # Return sample data for demonstration
                return self._get_sample_data(symbol, days)
            
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data from Alpaca
            df = self.api.get_bars(
                symbol,
                timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                return self._get_sample_data(symbol, days)
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            
            # Add technical indicators
            self._add_technical_indicators(df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Return sample data if we can't fetch real data
            return self._get_sample_data(symbol, days)
            
    def _get_sample_data(self, symbol, days=60):
        """
        Generate sample stock data for demonstration purposes when API is unavailable.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days of data to generate
            
        Returns:
            pandas.DataFrame: Sample historical data
        """
        logger.info(f"Generating sample data for {symbol}")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random stock prices with an upward trend
        np.random.seed(sum(ord(c) for c in symbol))  # Use symbol as seed for consistent results
        base_price = 100 + (ord(symbol[0]) % 100)  # Different starting price based on symbol
        
        # Create a somewhat realistic price series with some randomness
        price_changes = np.random.normal(0.0005, 0.01, size=len(date_range))
        price_changes = np.clip(price_changes, -0.05, 0.05)  # Limit daily changes
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + price_changes)
        prices = base_price * cum_returns
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices * (1 - np.random.uniform(0, 0.01, size=len(date_range))),
            'high': prices * (1 + np.random.uniform(0, 0.02, size=len(date_range))),
            'low': prices * (1 - np.random.uniform(0, 0.02, size=len(date_range))),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, size=len(date_range))
        }, index=date_range)
        
        # Add technical indicators
        self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Args:
            df (pandas.DataFrame): Dataframe with OHLCV data
            
        Returns:
            None: Modifies the dataframe in place
        """
        try:
            # Moving Averages
            df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # Exponential Moving Averages
            df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bollinger_upper'] = bollinger.bollinger_hband()
            df['bollinger_lower'] = bollinger.bollinger_lband()
            df['bollinger_pct'] = bollinger.bollinger_pband()
            
            # Average True Range (ATR)
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume Indicators
            df['volume_change'] = df['volume'].pct_change()
            df['volume_sma_5'] = ta.trend.sma_indicator(df['volume'], window=5)
            
            # Calculate returns
            df['daily_return'] = df['close'].pct_change()
            
            # Calculate price momentum
            df['momentum_1d'] = df['close'].pct_change(1)
            df['momentum_5d'] = df['close'].pct_change(5)
            
            # Create target variable (next day return > 0)
            df['target'] = df['close'].shift(-1) > df['close']
            df['target'] = df['target'].astype(int)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            logger.info(f"Added technical indicators to dataframe")
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for ML model from the dataframe.
        
        Args:
            df (pandas.DataFrame): Dataframe with technical indicators
            
        Returns:
            numpy.ndarray: Feature matrix for the ML model
        """
        try:
            # Select the latest data point for prediction
            latest_data = df.iloc[-1:].copy()
            
            # Define features to use
            features = [
                'SMA_5', 'SMA_20', 'SMA_50',
                'EMA_12', 'EMA_26',
                'MACD', 'MACD_signal', 'MACD_diff',
                'RSI',
                'bollinger_pct',
                'ATR',
                'volume_change',
                'momentum_1d', 'momentum_5d'
            ]
            
            # Extract features
            X = latest_data[features].values
            
            return X
        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            # Return a default feature array filled with zeros
            return np.zeros((1, 14))
    
    def get_market_calendar(self, start_date, end_date):
        """
        Get the market calendar between two dates.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of datetime objects representing market open days
        """
        try:
            # Check if API is available
            if self.api is None:
                logger.warning("Cannot fetch market calendar: Alpaca API not initialized")
                # Generate business days (Mon-Fri) for the date range
                all_days = pd.date_range(start=start_date, end=end_date, freq='D')
                market_days = [day for day in all_days if day.weekday() < 5]  # Monday-Friday
                return market_days
                
            calendar = self.api.get_calendar(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            market_days = [pd.Timestamp(day.date) for day in calendar]
            return market_days
        
        except Exception as e:
            logger.error(f"Error getting market calendar: {str(e)}")
            # Fallback to business days
            all_days = pd.date_range(start=start_date, end=end_date, freq='D')
            market_days = [day for day in all_days if day.weekday() < 5]  # Monday-Friday
            return market_days
    
    def get_latest_prices(self, symbols):
        """
        Get the latest prices for a list of symbols.
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Dictionary mapping symbols to their latest prices
        """
        try:
            # Check if API is available
            if self.api is None:
                logger.warning("Cannot fetch latest prices: Alpaca API not initialized")
                # Generate some realistic-looking prices based on symbol
                return self._get_sample_prices(symbols)
                
            prices = {}
            for symbol in symbols:
                # Get the latest trade
                trade = self.api.get_latest_trade(symbol)
                prices[symbol] = float(trade.price)
            
            return prices
        
        except Exception as e:
            logger.error(f"Error getting latest prices: {str(e)}")
            return self._get_sample_prices(symbols)
            
    def _get_sample_prices(self, symbols):
        """Generate sample prices for a list of symbols."""
        prices = {}
        for symbol in symbols:
            # Generate a consistent but different price for each symbol
            np.random.seed(sum(ord(c) for c in symbol))
            base_price = 100 + (ord(symbol[0]) % 100)
            variation = np.random.uniform(-0.5, 0.5)
            prices[symbol] = round(base_price + variation, 2)
        
        return prices
