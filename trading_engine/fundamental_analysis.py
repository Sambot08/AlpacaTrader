import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Set up logging
logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """
    Analyzes company fundamentals and economic indicators for enhancing
    trading decisions with additional data beyond technical analysis.
    """
    
    def __init__(self):
        """Initialize the fundamental analyzer"""
        try:
            # Initialize any API keys from environment variables
            self.fred_api_key = os.environ.get('FRED_API_KEY', '')  # For economic data
            
            logger.info("Fundamental analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing fundamental analyzer: {str(e)}")
            raise
    
    def get_company_financials(self, symbols):
        """
        Get key financial metrics for companies
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Dictionary mapping symbols to their financial metrics
        """
        financial_data = {}
        
        for symbol in symbols:
            try:
                # Fetch financial data
                financials = self._fetch_company_financials(symbol)
                
                if financials:
                    financial_data[symbol] = financials
                else:
                    logger.warning(f"No financial data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching financials for {symbol}: {str(e)}")
        
        return financial_data
    
    def get_economic_indicators(self):
        """
        Get current economic indicators that might affect the market
        
        Returns:
            dict: Dictionary of economic indicators and their values
        """
        indicators = {}
        
        try:
            # Attempt to fetch economic data from FRED or other sources
            if self.fred_api_key:
                # If we have a FRED API key, fetch actual data
                indicators = self._fetch_fred_indicators()
            else:
                # Otherwise, use latest estimates
                indicators = self._get_recent_economic_estimates()
                
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {str(e)}")
            
        return indicators
    
    def calculate_valuation_metrics(self, symbols, price_data):
        """
        Calculate valuation metrics for companies
        
        Args:
            symbols (list): List of stock symbols
            price_data (dict): Current price data for symbols
            
        Returns:
            dict: Dictionary mapping symbols to their valuation metrics
        """
        valuation_metrics = {}
        
        # Get financial data
        financials = self.get_company_financials(symbols)
        
        for symbol in symbols:
            if symbol in financials and symbol in price_data:
                try:
                    # Get current price
                    current_price = price_data[symbol]
                    
                    # Get financial metrics
                    eps = financials[symbol].get('eps', 0)
                    book_value = financials[symbol].get('book_value_per_share', 0)
                    revenue = financials[symbol].get('revenue', 0)
                    shares_outstanding = financials[symbol].get('shares_outstanding', 0)
                    
                    # Calculate valuation metrics
                    metrics = {}
                    
                    if eps and eps > 0:
                        metrics['pe_ratio'] = current_price / eps
                    else:
                        metrics['pe_ratio'] = None
                        
                    if book_value and book_value > 0:
                        metrics['price_to_book'] = current_price / book_value
                    else:
                        metrics['price_to_book'] = None
                        
                    if revenue and revenue > 0 and shares_outstanding and shares_outstanding > 0:
                        metrics['price_to_sales'] = (current_price * shares_outstanding) / revenue
                    else:
                        metrics['price_to_sales'] = None
                    
                    # Add more metrics as needed
                    
                    valuation_metrics[symbol] = metrics
                    
                except Exception as e:
                    logger.error(f"Error calculating valuation metrics for {symbol}: {str(e)}")
                    
        return valuation_metrics
    
    def get_sector_performance(self):
        """
        Get performance metrics for different market sectors
        
        Returns:
            dict: Dictionary mapping sectors to their performance metrics
        """
        sectors = [
            'Technology', 'Healthcare', 'Financial', 'Consumer Cyclical',
            'Consumer Defensive', 'Industrials', 'Basic Materials',
            'Energy', 'Utilities', 'Real Estate', 'Communication Services'
        ]
        
        sector_data = {}
        
        try:
            # In a production system, you would fetch this data from a sector ETF
            # or a market data API that provides sector performance
            
            # For demonstration, we'll create placeholder data
            # In a real implementation, fetch from sector ETF performance
            for sector in sectors:
                sector_data[sector] = self._fetch_sector_data(sector)
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            
        return sector_data
    
    def evaluate_company_strength(self, symbols):
        """
        Evaluate overall company strength based on fundamentals
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Dictionary mapping symbols to their strength scores (0-100)
        """
        strength_scores = {}
        
        # Get financial data
        financials = self.get_company_financials(symbols)
        
        for symbol in symbols:
            if symbol in financials:
                try:
                    # Calculate strength based on various metrics
                    financial_metrics = financials[symbol]
                    
                    # Start with a base score
                    score = 50  # Neutral starting point
                    
                    # Adjust based on profitability
                    if 'profit_margin' in financial_metrics:
                        profit_margin = financial_metrics['profit_margin']
                        if profit_margin > 0.20:  # Over 20% profit margin
                            score += 10
                        elif profit_margin > 0.10:  # Over 10% profit margin
                            score += 5
                        elif profit_margin < 0:  # Negative profit margin
                            score -= 10
                    
                    # Adjust based on debt levels
                    if 'debt_to_equity' in financial_metrics:
                        debt_to_equity = financial_metrics['debt_to_equity']
                        if debt_to_equity < 0.3:  # Low debt
                            score += 10
                        elif debt_to_equity > 1.5:  # High debt
                            score -= 10
                    
                    # Adjust based on growth
                    if 'revenue_growth' in financial_metrics:
                        revenue_growth = financial_metrics['revenue_growth']
                        if revenue_growth > 0.20:  # Over 20% growth
                            score += 10
                        elif revenue_growth > 0.10:  # Over 10% growth
                            score += 5
                        elif revenue_growth < 0:  # Negative growth
                            score -= 5
                    
                    # Cap the score at 0-100
                    score = max(0, min(100, score))
                    
                    strength_scores[symbol] = score
                    
                except Exception as e:
                    logger.error(f"Error evaluating company strength for {symbol}: {str(e)}")
                    strength_scores[symbol] = 50  # Default to neutral
            else:
                strength_scores[symbol] = 50  # Default to neutral
                
        return strength_scores
    
    def _fetch_company_financials(self, symbol):
        """
        Fetch financial data for a company
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Financial metrics for the company
        """
        try:
            # In a production system, you would make API calls to financial data providers
            # For example, using Alpha Vantage, Financial Modeling Prep, or IEX Cloud APIs
            
            # For demonstration, we'll create placeholder financial data
            # In a real implementation, this would call an actual API
            
            # Create sample financial metrics based on the symbol
            # These metrics would normally come from the API response
            financial_metrics = {
                'revenue': 1000000000,  # $1B revenue
                'net_income': 100000000,  # $100M net income
                'profit_margin': 0.10,  # 10% profit margin
                'eps': 2.50,  # $2.50 earnings per share
                'book_value_per_share': 15.0,  # $15 book value per share
                'debt_to_equity': 0.5,  # 0.5 debt to equity ratio
                'revenue_growth': 0.12,  # 12% revenue growth
                'shares_outstanding': 100000000,  # 100M shares outstanding
                'dividend_yield': 0.02,  # 2% dividend yield
                'return_on_equity': 0.15,  # 15% return on equity
            }
            
            return financial_metrics
            
        except Exception as e:
            logger.error(f"Error fetching company financials for {symbol}: {str(e)}")
            return {}
    
    def _fetch_fred_indicators(self):
        """
        Fetch economic indicators from FRED (Federal Reserve Economic Data)
        
        Returns:
            dict: Dictionary of economic indicators
        """
        indicators = {}
        
        # List of indicators to fetch (FRED series IDs)
        series_ids = {
            'GDP': 'GDP',
            'Unemployment Rate': 'UNRATE',
            'Inflation Rate': 'CPIAUCSL',
            'Federal Funds Rate': 'FEDFUNDS',
            '10-Year Treasury Yield': 'GS10',
            'Consumer Sentiment': 'UMCSENT'
        }
        
        try:
            if not self.fred_api_key:
                logger.warning("FRED API key not provided, cannot fetch economic indicators")
                return indicators
            
            # Base URL for FRED API
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            
            for name, series_id in series_ids.items():
                # Parameters for the API request
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1  # Get only the most recent observation
                }
                
                # Make the API request
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'observations' in data and len(data['observations']) > 0:
                        latest_value = data['observations'][0]['value']
                        indicators[name] = float(latest_value) if latest_value != '.' else None
                else:
                    logger.warning(f"Failed to fetch {name} (FRED series: {series_id})")
            
        except Exception as e:
            logger.error(f"Error fetching FRED indicators: {str(e)}")
            
        return indicators
    
    def _get_recent_economic_estimates(self):
        """
        Get recent estimates for economic indicators when API access is not available
        
        Returns:
            dict: Dictionary of estimated economic indicators
        """
        # These are placeholder values - in a production system,
        # you would fetch these from a reliable data source
        
        indicators = {
            'GDP Growth Rate': 2.1,  # Annual GDP growth rate (%)
            'Unemployment Rate': 3.7,  # Unemployment rate (%)
            'Inflation Rate': 3.2,  # Annual inflation rate (%)
            'Federal Funds Rate': 5.25,  # Federal funds rate (%)
            '10-Year Treasury Yield': 4.2,  # 10-year Treasury yield (%)
            'Consumer Sentiment': 63.8  # University of Michigan Consumer Sentiment
        }
        
        return indicators
    
    def _fetch_sector_data(self, sector):
        """
        Fetch performance data for a market sector
        
        Args:
            sector (str): Market sector name
            
        Returns:
            dict: Performance metrics for the sector
        """
        try:
            # In production, you would fetch actual data from sector ETFs or indices
            
            # Placeholder data - these would be API responses in a real implementation
            sector_data = {
                'ytd_return': 0.08,  # 8% year-to-date return
                'one_month_return': 0.02,  # 2% one-month return
                'three_month_return': 0.05,  # 5% three-month return
                'one_year_return': 0.12,  # 12% one-year return
                'pe_ratio': 22.5,  # Price-to-earnings ratio
                'dividend_yield': 0.018  # 1.8% dividend yield
            }
            
            # Add some variation based on the sector
            if sector == 'Technology':
                sector_data['ytd_return'] = 0.15
                sector_data['pe_ratio'] = 28.2
                sector_data['dividend_yield'] = 0.008
            elif sector == 'Energy':
                sector_data['ytd_return'] = 0.03
                sector_data['pe_ratio'] = 18.5
                sector_data['dividend_yield'] = 0.04
            elif sector == 'Healthcare':
                sector_data['ytd_return'] = 0.07
                sector_data['pe_ratio'] = 24.8
                sector_data['dividend_yield'] = 0.015
                
            return sector_data
            
        except Exception as e:
            logger.error(f"Error fetching sector data for {sector}: {str(e)}")
            return {}