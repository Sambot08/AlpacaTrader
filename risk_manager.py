import logging
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages risk for trading strategies by implementing 
    position sizing and risk controls.
    """
    
    def __init__(self):
        """Initialize the RiskManager."""
        pass
    
    def calculate_position_size(self, account_value, risk_level, prediction_confidence):
        """
        Calculate the maximum position size based on account value, 
        risk level, and prediction confidence.
        
        Args:
            account_value (float): Current account value in dollars
            risk_level (int): Risk level from 1 (low risk) to 5 (high risk)
            prediction_confidence (float): Confidence in prediction (0-1)
            
        Returns:
            float: Recommended maximum position size in dollars
        """
        try:
            # Base percentage of account to risk based on risk level
            risk_percentages = {
                1: 0.01,  # 1% of account
                2: 0.02,  # 2% of account
                3: 0.05,  # 5% of account
                4: 0.10,  # 10% of account
                5: 0.15   # 15% of account
            }
            
            # Get base risk percentage
            base_risk = risk_percentages.get(risk_level, 0.05)
            
            # Adjust based on prediction confidence
            # Higher confidence allows for larger position size
            confidence_factor = np.clip(prediction_confidence, 0.5, 1.0)
            adjusted_risk = base_risk * ((confidence_factor - 0.5) * 2)
            
            # Calculate maximum position size
            max_position_size = account_value * adjusted_risk
            
            logger.info(f"Calculated position size: ${max_position_size:.2f} " +
                      f"(Risk level: {risk_level}, Confidence: {confidence_factor:.2f})")
            
            return max_position_size
        
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            # Return a conservative default
            return account_value * 0.01
    
    def check_risk_limits(self, current_positions, new_position, account_value):
        """
        Check if a new position would exceed risk limits.
        
        Args:
            current_positions (dict): Current positions {symbol: value}
            new_position (dict): Proposed new position {symbol: value}
            account_value (float): Current account value
            
        Returns:
            bool: True if position is within risk limits, False otherwise
        """
        try:
            # Calculate total position value
            total_position_value = sum(current_positions.values())
            
            # Add the new position
            new_symbol = list(new_position.keys())[0]
            new_value = new_position[new_symbol]
            
            total_with_new = total_position_value + new_value
            
            # Check if total exceeds 90% of account value (keeping some cash)
            if total_with_new > (account_value * 0.9):
                logger.warning(f"New position would exceed 90% of account value")
                return False
            
            # Check concentration risk (no single position > 20% of portfolio)
            for symbol, value in current_positions.items():
                if symbol == new_symbol:
                    value += new_value
                
                if value > (account_value * 0.2):
                    logger.warning(f"Position in {symbol} would exceed 20% of account value")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False
    
    def calculate_stop_loss(self, entry_price, risk_level, volatility=None):
        """
        Calculate a recommended stop loss price.
        
        Args:
            entry_price (float): Entry price of the position
            risk_level (int): Risk level from 1 (low risk) to 5 (high risk)
            volatility (float, optional): Stock volatility (e.g., ATR)
            
        Returns:
            float: Recommended stop loss price
        """
        try:
            # Default stop loss percentages based on risk level
            stop_loss_percentages = {
                1: 0.01,  # 1% below entry
                2: 0.02,  # 2% below entry
                3: 0.03,  # 3% below entry
                4: 0.05,  # 5% below entry
                5: 0.07   # 7% below entry
            }
            
            # Get base stop loss percentage
            base_percentage = stop_loss_percentages.get(risk_level, 0.03)
            
            # If volatility data is provided, adjust stop loss
            if volatility is not None:
                # Use ATR or other volatility measure to set stop
                # A common approach is 2-3x ATR
                volatility_multiplier = {
                    1: 1.0,
                    2: 1.5,
                    3: 2.0,
                    4: 2.5,
                    5: 3.0
                }
                
                multiplier = volatility_multiplier.get(risk_level, 2.0)
                atr_based_stop = entry_price - (volatility * multiplier)
                
                # Use the larger of percentage-based and ATR-based stop loss
                percentage_based_stop = entry_price * (1 - base_percentage)
                stop_loss = max(atr_based_stop, percentage_based_stop)
            else:
                # Use percentage-based stop loss if no volatility data
                stop_loss = entry_price * (1 - base_percentage)
            
            logger.info(f"Calculated stop loss: ${stop_loss:.2f} (Entry: ${entry_price:.2f})")
            
            return stop_loss
        
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            # Return a default stop loss (3% below entry)
            return entry_price * 0.97
