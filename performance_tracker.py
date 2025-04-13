import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import os
from app import db
from models import PerformanceMetric, TradeRecord

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and analyzes trading performance metrics.
    """
    
    def __init__(self):
        """Initialize the PerformanceTracker with Alpaca API credentials."""
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
                logger.info("Performance Tracker: Alpaca API initialized successfully")
            except Exception as e:
                logger.error(f"Performance Tracker: Failed to initialize Alpaca API: {str(e)}")
        else:
            logger.warning("Performance Tracker: Alpaca API credentials not found. Performance tracking features will be limited.")
    
    def update_daily_metrics(self):
        """Update daily performance metrics based on account and trade data."""
        try:
            today = datetime.now().date()
            
            # Check if we already have metrics for today
            existing_metric = PerformanceMetric.query.filter_by(date=today).first()
            if existing_metric:
                logger.info(f"Performance metrics for {today} already exist")
                return
            
            # Get account information
            account = self.api.get_account()
            
            # Get previous day's metric for calculating daily return
            yesterday = today - timedelta(days=1)
            previous_metric = PerformanceMetric.query.filter_by(date=yesterday).first()
            
            # Calculate daily return
            daily_return = None
            if previous_metric:
                previous_value = previous_metric.portfolio_value
                current_value = float(account.portfolio_value)
                if previous_value > 0:
                    daily_return = (current_value - previous_value) / previous_value * 100
            
            # Get today's trades
            today_trades = TradeRecord.query.filter(
                TradeRecord.timestamp >= datetime.combine(today, datetime.min.time()),
                TradeRecord.timestamp <= datetime.combine(today, datetime.max.time())
            ).all()
            
            trade_count = len(today_trades)
            
            # Calculate win/loss metrics from completed trades
            win_count = 0
            for trade in today_trades:
                if trade.action == 'SELL':
                    # Find the corresponding BUY trade
                    buy_trade = TradeRecord.query.filter_by(
                        symbol=trade.symbol,
                        action='BUY'
                    ).order_by(TradeRecord.timestamp.desc()).first()
                    
                    if buy_trade and buy_trade.price < trade.price:
                        win_count += 1
            
            win_rate = win_count / trade_count if trade_count > 0 else None
            
            # Create new metric
            metric = PerformanceMetric(
                date=today,
                portfolio_value=float(account.portfolio_value),
                cash_balance=float(account.cash),
                equity_value=float(account.equity),
                daily_return=daily_return,
                pnl=float(account.equity) - float(account.last_equity),
                trade_count=trade_count,
                win_rate=win_rate
            )
            
            # Calculate other metrics if we have enough historical data
            self._calculate_advanced_metrics(metric)
            
            # Save to database
            db.session.add(metric)
            db.session.commit()
            
            logger.info(f"Updated performance metrics for {today}")
        
        except Exception as e:
            logger.error(f"Error updating daily metrics: {str(e)}")
    
    def _calculate_advanced_metrics(self, metric):
        """
        Calculate advanced performance metrics like Sharpe ratio and drawdown.
        
        Args:
            metric (PerformanceMetric): Current day's metric
        """
        try:
            # Get historical metrics for the past 30 days
            thirty_days_ago = metric.date - timedelta(days=30)
            historical_metrics = PerformanceMetric.query.filter(
                PerformanceMetric.date >= thirty_days_ago,
                PerformanceMetric.date < metric.date
            ).order_by(PerformanceMetric.date.asc()).all()
            
            if len(historical_metrics) < 5:
                logger.info(f"Not enough historical data to calculate advanced metrics")
                return
            
            # Get portfolio values
            portfolio_values = [m.portfolio_value for m in historical_metrics]
            portfolio_values.append(metric.portfolio_value)
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(daily_return)
            
            # Calculate Sharpe ratio (annualized)
            # Assuming risk-free rate of 0% for simplicity
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualize
                metric.sharpe_ratio = sharpe_ratio
            
            # Calculate maximum drawdown
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            metric.max_drawdown = max_drawdown * 100  # Convert to percentage
            
            logger.info(f"Calculated advanced metrics: Sharpe={metric.sharpe_ratio:.4f}, " +
                      f"Max Drawdown={metric.max_drawdown:.2f}%")
        
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
    
    def get_performance_summary(self, days=30):
        """
        Get a summary of performance over a specified time period.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Performance summary
        """
        try:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get metrics in date range
            metrics = PerformanceMetric.query.filter(
                PerformanceMetric.date >= start_date,
                PerformanceMetric.date <= end_date
            ).order_by(PerformanceMetric.date.asc()).all()
            
            if not metrics:
                logger.warning(f"No performance metrics found for the past {days} days")
                return {
                    "success": False,
                    "message": f"No data available for the past {days} days"
                }
            
            # Calculate summary statistics
            initial_value = metrics[0].portfolio_value
            final_value = metrics[-1].portfolio_value
            
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            # Calculate average daily return
            daily_returns = [m.daily_return for m in metrics if m.daily_return is not None]
            avg_daily_return = np.mean(daily_returns) if daily_returns else None
            
            # Get best and worst days
            best_day = max(metrics, key=lambda m: m.daily_return if m.daily_return is not None else -float('inf'))
            worst_day = min(metrics, key=lambda m: m.daily_return if m.daily_return is not None else float('inf'))
            
            # Get total trades and win rate
            total_trades = sum(m.trade_count for m in metrics if m.trade_count is not None)
            
            # Calculate average win rate
            win_rates = [m.win_rate for m in metrics if m.win_rate is not None]
            avg_win_rate = np.mean(win_rates) * 100 if win_rates else None
            
            # Get latest Sharpe ratio and max drawdown
            latest_sharpe = metrics[-1].sharpe_ratio
            latest_drawdown = metrics[-1].max_drawdown
            
            # Prepare summary
            summary = {
                "success": True,
                "time_period": f"{start_date} to {end_date}",
                "initial_value": float(initial_value),
                "final_value": float(final_value),
                "total_return": float(total_return),
                "avg_daily_return": float(avg_daily_return) if avg_daily_return is not None else None,
                "best_day": {
                    "date": best_day.date.strftime('%Y-%m-%d'),
                    "return": float(best_day.daily_return) if best_day.daily_return is not None else None
                },
                "worst_day": {
                    "date": worst_day.date.strftime('%Y-%m-%d'),
                    "return": float(worst_day.daily_return) if worst_day.daily_return is not None else None
                },
                "total_trades": total_trades,
                "avg_win_rate": float(avg_win_rate) if avg_win_rate is not None else None,
                "sharpe_ratio": float(latest_sharpe) if latest_sharpe is not None else None,
                "max_drawdown": float(latest_drawdown) if latest_drawdown is not None else None
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
