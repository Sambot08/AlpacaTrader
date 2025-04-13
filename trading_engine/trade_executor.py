import logging
import time
import alpaca_trade_api as tradeapi
from datetime import datetime

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Executes trades on Alpaca API
    """
    
    def __init__(self, api_key, api_secret, base_url):
        """
        Initialize the trade executor with API credentials
        
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
            logger.info("Successfully connected to Alpaca API for trading")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API for trading: {str(e)}")
            raise
            
    def get_account(self):
        """
        Get account information
        
        Returns:
            dict: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'daytrade_count': account.daytrade_count,
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error fetching account information: {str(e)}")
            return None
            
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            list: List of positions
        """
        try:
            positions = self.api.list_positions()
            result = []
            
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'current_price': float(position.current_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc) * 100,  # Convert to percentage
                    'side': position.side
                })
                
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            return []
            
    def get_recent_orders(self, limit=20):
        """
        Get recent orders
        
        Args:
            limit (int): Maximum number of orders to return
            
        Returns:
            list: List of recent orders
        """
        try:
            orders = self.api.list_orders(
                status='all',
                limit=limit,
                nested=True
            )
            
            result = []
            for order in orders:
                result.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'type': order.type,
                    'status': order.status,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'created_at': order.created_at.isoformat(),
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None
                })
                
            return result
        except Exception as e:
            logger.error(f"Error fetching recent orders: {str(e)}")
            return []
            
    def submit_order(self, symbol, qty, side, order_type='market', time_in_force='day',
                    limit_price=None, stop_price=None, client_order_id=None):
        """
        Submit an order
        
        Args:
            symbol (str): Ticker symbol
            qty (int/float): Order quantity
            side (str): Order side ('buy' or 'sell')
            order_type (str): Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force (str): Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
            limit_price (float, optional): Limit price for limit orders
            stop_price (float, optional): Stop price for stop orders
            client_order_id (str, optional): Client order ID
            
        Returns:
            dict: Order information
        """
        if not symbol or not qty or qty <= 0:
            logger.error(f"Invalid order parameters: symbol={symbol}, qty={qty}")
            return None
            
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            logger.info(f"Order submitted: {side} {qty} {symbol} @ {order_type}")
            
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {str(e)}")
            return None
            
    def cancel_order(self, order_id):
        """
        Cancel an order
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
            
    def cancel_all_orders(self):
        """
        Cancel all open orders
        
        Returns:
            int: Number of orders cancelled
        """
        try:
            cancelled = self.api.cancel_all_orders()
            logger.info(f"Cancelled {len(cancelled)} orders")
            return len(cancelled)
        except Exception as e:
            logger.error(f"Error cancelling orders: {str(e)}")
            return 0
            
    def close_position(self, symbol):
        """
        Close a position
        
        Args:
            symbol (str): Ticker symbol
            
        Returns:
            dict: Order information
        """
        try:
            order = self.api.close_position(symbol)
            logger.info(f"Position for {symbol} closed")
            
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return None
            
    def close_all_positions(self):
        """
        Close all positions
        
        Returns:
            bool: True if all positions closed successfully, False otherwise
        """
        try:
            self.api.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {str(e)}")
            return False
            
    def get_order_status(self, order_id):
        """
        Get order status
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order status
        """
        try:
            order = self.api.get_order(order_id)
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at.isoformat(),
                'filled_at': order.filled_at.isoformat() if order.filled_at else None
            }
        except Exception as e:
            logger.error(f"Error fetching order status for {order_id}: {str(e)}")
            return None
