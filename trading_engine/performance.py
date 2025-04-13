import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and analyzes trading performance
    """
    
    def __init__(self):
        """
        Initialize the performance tracker
        """
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.positions = []
        
        # Create data directory if it doesn't exist
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Load existing data if available
        self.load_data()
        
    def load_data(self):
        """
        Load performance data from files
        """
        try:
            # Load portfolio history
            portfolio_file = os.path.join(self.data_dir, "portfolio_history.csv")
            if os.path.exists(portfolio_file):
                df = pd.read_csv(portfolio_file, parse_dates=['timestamp'])
                self.portfolio_history = df.to_dict('records')
                
            # Load trade history
            trade_file = os.path.join(self.data_dir, "trade_history.csv")
            if os.path.exists(trade_file):
                df = pd.read_csv(trade_file, parse_dates=['timestamp'])
                self.trade_history = df.to_dict('records')
                
            # Load daily returns
            returns_file = os.path.join(self.data_dir, "daily_returns.csv")
            if os.path.exists(returns_file):
                df = pd.read_csv(returns_file, parse_dates=['date'])
                self.daily_returns = df.to_dict('records')
                
            logger.info("Performance data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            
    def save_data(self):
        """
        Save performance data to files
        """
        try:
            # Save portfolio history
            if self.portfolio_history:
                df = pd.DataFrame(self.portfolio_history)
                df.to_csv(os.path.join(self.data_dir, "portfolio_history.csv"), index=False)
                
            # Save trade history
            if self.trade_history:
                df = pd.DataFrame(self.trade_history)
                df.to_csv(os.path.join(self.data_dir, "trade_history.csv"), index=False)
                
            # Save daily returns
            if self.daily_returns:
                df = pd.DataFrame(self.daily_returns)
                df.to_csv(os.path.join(self.data_dir, "daily_returns.csv"), index=False)
                
            logger.info("Performance data saved successfully")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
            
    def update_portfolio_value(self, portfolio_value, cash_balance=None):
        """
        Update portfolio value
        
        Args:
            portfolio_value (float): Current portfolio value
            cash_balance (float, optional): Current cash balance
        """
        timestamp = datetime.now()
        
        # If cash balance not provided, use last known cash balance
        if cash_balance is None and self.portfolio_history:
            cash_balance = self.portfolio_history[-1].get('cash_balance', 0)
            
        # Add to portfolio history
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash_balance': cash_balance
        })
        
        # Calculate daily return if we have previous values
        today = date.today()
        
        # Check if we already have a return for today
        today_return = next((r for r in self.daily_returns if r['date'].date() == today), None)
        
        if not today_return:
            # Find the last day's portfolio value
            yesterday = today - timedelta(days=1)
            yesterday_value = None
            
            for entry in reversed(self.portfolio_history[:-1]):
                if entry['timestamp'].date() <= yesterday:
                    yesterday_value = entry['portfolio_value']
                    break
                    
            if yesterday_value:
                daily_return = portfolio_value - yesterday_value
                daily_return_pct = (daily_return / yesterday_value) * 100
                
                self.daily_returns.append({
                    'date': timestamp,
                    'portfolio_value': portfolio_value,
                    'daily_return': daily_return,
                    'daily_return_pct': daily_return_pct
                })
        
        # Save data
        self.save_data()
        
    def record_trade(self, symbol, side, quantity, price, profit_loss=None, profit_loss_pct=None):
        """
        Record a trade
        
        Args:
            symbol (str): Ticker symbol
            side (str): Trade side ('BUY' or 'SELL')
            quantity (float): Trade quantity
            price (float): Trade price
            profit_loss (float, optional): Profit/loss amount
            profit_loss_pct (float, optional): Profit/loss percentage
        """
        timestamp = datetime.now()
        
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        })
        
        # Save data
        self.save_data()
        
    def update_positions(self, positions):
        """
        Update current positions
        
        Args:
            positions (list): List of position dictionaries
        """
        self.positions = positions
        
    def get_performance_stats(self):
        """
        Calculate performance statistics
        
        Returns:
            dict: Performance statistics
        """
        stats = {
            'total_trades': len(self.trade_history),
            'portfolio_value': 0,
            'cash_balance': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'portfolio_history': self.generate_portfolio_chart(),
            'returns_history': self.generate_returns_chart(),
            'daily_returns': []
        }
        
        # Get current portfolio value
        if self.portfolio_history:
            latest = self.portfolio_history[-1]
            stats['portfolio_value'] = latest['portfolio_value']
            stats['cash_balance'] = latest.get('cash_balance', 0)
            
        # Calculate total P&L
        if self.portfolio_history and len(self.portfolio_history) > 1:
            initial_value = self.portfolio_history[0]['portfolio_value']
            latest_value = self.portfolio_history[-1]['portfolio_value']
            stats['total_pnl'] = latest_value - initial_value
            stats['total_pnl_pct'] = (stats['total_pnl'] / initial_value) * 100
            
        # Calculate trade statistics
        winning_trades = [t for t in self.trade_history if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('profit_loss', 0) < 0]
        
        stats['winning_trades'] = len(winning_trades)
        stats['losing_trades'] = len(losing_trades)
        
        if self.trade_history:
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
            
        if winning_trades:
            stats['avg_win'] = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades)
            
        if losing_trades:
            stats['avg_loss'] = sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades)
            
        # Calculate profit factor
        if losing_trades and winning_trades:
            total_gains = sum(t.get('profit_loss', 0) for t in winning_trades)
            total_losses = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
            
            if total_losses > 0:
                stats['profit_factor'] = total_gains / total_losses
                
        # Calculate Sharpe ratio
        if self.daily_returns:
            returns = [r.get('daily_return_pct', 0) / 100 for r in self.daily_returns]
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    stats['sharpe_ratio'] = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
                    
        # Calculate maximum drawdown
        if self.portfolio_history:
            values = [entry['portfolio_value'] for entry in self.portfolio_history]
            max_dd, max_dd_pct = self._calculate_max_drawdown(values)
            
            stats['max_drawdown'] = max_dd
            stats['max_drawdown_pct'] = max_dd_pct
            
        # Get recent daily returns
        if self.daily_returns:
            stats['daily_returns'] = sorted(
                self.daily_returns,
                key=lambda x: x['date'],
                reverse=True
            )[:30]  # Last 30 days
            
        return stats
        
    def _calculate_max_drawdown(self, values):
        """
        Calculate maximum drawdown
        
        Args:
            values (list): List of portfolio values
            
        Returns:
            tuple: (max_drawdown, max_drawdown_percentage)
        """
        max_so_far = values[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for value in values:
            if value > max_so_far:
                max_so_far = value
            else:
                drawdown = max_so_far - value
                drawdown_pct = (drawdown / max_so_far) * 100
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct
                    
        return max_drawdown, max_drawdown_pct
        
    def generate_portfolio_chart(self):
        """
        Generate portfolio value chart
        
        Returns:
            str: Base64-encoded chart image
        """
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return None
            
        try:
            # Create DataFrame
            df = pd.DataFrame(self.portfolio_history)
            df['date'] = df['timestamp'].dt.date
            
            # Group by date and get the last value for each day
            daily_values = df.groupby('date')['portfolio_value'].last().reset_index()
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(daily_values['date'], daily_values['portfolio_value'], 'b-')
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        except Exception as e:
            logger.error(f"Error generating portfolio chart: {str(e)}")
            return None
            
    def generate_returns_chart(self):
        """
        Generate daily returns chart
        
        Returns:
            str: Base64-encoded chart image
        """
        if not self.daily_returns or len(self.daily_returns) < 2:
            return None
            
        try:
            # Create DataFrame
            df = pd.DataFrame(self.daily_returns)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(df['date'], df['daily_return_pct'], color=df['daily_return_pct'].apply(
                lambda x: 'green' if x > 0 else 'red'
            ))
            plt.title('Daily Returns (%)')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        except Exception as e:
            logger.error(f"Error generating returns chart: {str(e)}")
            return None
