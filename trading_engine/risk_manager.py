import logging
import numpy as np
from datetime import datetime, time, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages risk by determining position sizes and setting stop losses
    """
    
    def __init__(self, max_position_size=0.1, stop_loss_pct=0.02, take_profit_pct=0.05,
                max_trades_per_day=5, max_risk_per_trade=0.01, max_portfolio_risk=0.05):
        """
        Initialize the risk manager
        
        Args:
            max_position_size (float): Maximum position size as a fraction of portfolio
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            max_trades_per_day (int): Maximum number of trades per day
            max_risk_per_trade (float): Maximum risk per trade as a fraction of portfolio
            max_portfolio_risk (float): Maximum portfolio risk as a fraction of portfolio
        """
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades_per_day = max_trades_per_day
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        
        self.daily_trades = {}  # Dict to track daily trades by symbol
        self.positions = {}  # Dict to track current positions
        
    def update_parameters(self, **kwargs):
        """
        Update risk management parameters
        
        Args:
            **kwargs: Keyword arguments for parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated risk parameter {key} to {value}")
            else:
                logger.warning(f"Unknown risk parameter: {key}")
                
    def get_parameters(self):
        """
        Get all risk management parameters
        
        Returns:
            dict: Dictionary of risk parameters
        """
        return {
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_trades_per_day': self.max_trades_per_day,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk
        }
                
    def calculate_position_size(self, symbol, price, portfolio_value, confidence, volatility=None):
        """
        Calculate appropriate position size for a trade
        
        Args:
            symbol (str): Ticker symbol
            price (float): Current price
            portfolio_value (float): Total portfolio value
            confidence (float): Confidence score from ML model (0-1)
            volatility (float, optional): Stock volatility
            
        Returns:
            dict: Position size and risk parameters
        """
        # Check if we've reached max trades for the day
        today = datetime.now().date()
        if today not in self.daily_trades:
            self.daily_trades[today] = {}
            
        if symbol not in self.daily_trades[today]:
            self.daily_trades[today][symbol] = 0
            
        if sum(self.daily_trades[today].values()) >= self.max_trades_per_day:
            logger.info(f"Maximum daily trades reached ({self.max_trades_per_day})")
            return {'shares': 0, 'reason': 'MAX_TRADES_REACHED'}
            
        # Calculate max position value based on portfolio size
        max_position_value = portfolio_value * self.max_position_size
        
        # Adjust position size based on confidence
        position_value = max_position_value * confidence
        
        # Ensure minimum position size
        min_position_value = 100  # $100 minimum
        if position_value < min_position_value:
            position_value = min_position_value
            
        # Calculate number of shares
        shares = int(position_value / price)
        
        # Calculate stop loss and take profit
        stop_loss = price * (1 - self.stop_loss_pct)
        take_profit = price * (1 + self.take_profit_pct)
        
        # If risk is too high, reduce position size
        risk_per_share = price - stop_loss
        total_risk = risk_per_share * shares
        
        max_allowable_risk = portfolio_value * self.max_risk_per_trade
        if total_risk > max_allowable_risk:
            shares = int(max_allowable_risk / risk_per_share)
            logger.info(f"Reduced position size for {symbol} due to risk limits")
            
        return {
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_per_share': risk_per_share,
            'total_risk': risk_per_share * shares,
            'risk_pct': (risk_per_share * shares) / portfolio_value
        }
        
    def check_exit_signals(self, symbol, current_price, position_details):
        """
        Check if a position should be exited
        
        Args:
            symbol (str): Ticker symbol
            current_price (float): Current price
            position_details (dict): Position details
            
        Returns:
            tuple: (bool, str) - Exit signal and reason
        """
        if symbol not in self.positions:
            return False, "NO_POSITION"
            
        position = self.positions[symbol]
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        
        # Check stop loss
        if current_price <= stop_loss:
            return True, "STOP_LOSS"
            
        # Check take profit
        if current_price >= take_profit:
            return True, "TAKE_PROFIT"
            
        # Check trailing stop (if applicable)
        trailing_stop = position.get('trailing_stop', 0)
        if trailing_stop > 0 and current_price <= trailing_stop:
            return True, "TRAILING_STOP"
            
        # Time-based exit (close positions after certain time)
        entry_time = position.get('entry_time')
        if entry_time:
            # Convert string time to datetime if needed
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
                
            max_hold_time = timedelta(days=5)  # Maximum hold time of 5 days
            if datetime.now() - entry_time > max_hold_time:
                return True, "TIME_EXIT"
                
        return False, "HOLD"
        
    def update_position(self, symbol, price, shares, side, entry_time=None):
        """
        Update tracked positions
        
        Args:
            symbol (str): Ticker symbol
            price (float): Entry price
            shares (int): Number of shares
            side (str): 'buy' or 'sell'
            entry_time (datetime, optional): Entry time
        """
        if side.lower() == 'buy':
            # Calculate stop loss and take profit
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
            
            self.positions[symbol] = {
                'entry_price': price,
                'shares': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': 0,  # Initialize with no trailing stop
                'entry_time': entry_time if entry_time else datetime.now()
            }
            
            # Update daily trade count
            today = datetime.now().date()
            if today not in self.daily_trades:
                self.daily_trades[today] = {}
            if symbol not in self.daily_trades[today]:
                self.daily_trades[today][symbol] = 0
            self.daily_trades[today][symbol] += 1
            
            logger.info(f"Added position for {symbol}: {shares} shares @ {price}")
            
        elif side.lower() == 'sell' and symbol in self.positions:
            # Remove the position
            position = self.positions.pop(symbol)
            entry_price = position['entry_price']
            shares_held = position['shares']
            
            # Calculate profit/loss
            pl = (price - entry_price) * shares_held
            pl_pct = (price / entry_price - 1) * 100
            
            logger.info(f"Closed position for {symbol}: {shares_held} shares, P/L: ${pl:.2f} ({pl_pct:.2f}%)")
            
    def update_trailing_stop(self, symbol, current_price):
        """
        Update trailing stop for a position
        
        Args:
            symbol (str): Ticker symbol
            current_price (float): Current price
        """
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # Only update trailing stop if price has moved up
        if current_price > entry_price:
            # Calculate trailing stop distance (e.g., 50% of the gain)
            gain = current_price - entry_price
            trail_distance = gain * 0.5
            new_trailing_stop = current_price - trail_distance
            
            # Only update if new trailing stop is higher than current
            current_trailing_stop = position.get('trailing_stop', 0)
            if new_trailing_stop > current_trailing_stop:
                position['trailing_stop'] = new_trailing_stop
                logger.info(f"Updated trailing stop for {symbol} to {new_trailing_stop:.2f}")
