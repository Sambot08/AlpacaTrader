import logging
from app import app, init_trading_engine
import os
from trading_engine.social_sentiment import SocialSentimentAnalyzer
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize the trading engine
    trading_engine = init_trading_engine()

    # Check if the trading engine's trade executor has a valid API connection
    if not trading_engine.trade_executor.api:
        logging.error("Trading engine failed to initialize. Check API credentials.")
    else:
        logging.info("Trading engine initialized successfully.")

    # Example stock symbol to analyze
    stock_symbol = "AAPL"

    # Initialize the social sentiment analyzer
    social_analyzer = SocialSentimentAnalyzer()

    # Fetch social sentiment for the stock
    try:
        reddit_sentiment = social_analyzer.get_reddit_sentiment([stock_symbol])
        stocktwits_sentiment = social_analyzer.get_stocktwits_sentiment([stock_symbol])
        combined_sentiment = social_analyzer.get_combined_sentiment([stock_symbol])

        logging.info(f"Social Sentiment for {stock_symbol}: ")
        logging.info(f"Reddit Sentiment: {reddit_sentiment.get(stock_symbol, 'N/A')}")
        logging.info(f"StockTwits Sentiment: {stocktwits_sentiment.get(stock_symbol, 'N/A')}")
        logging.info(f"Combined Sentiment: {combined_sentiment.get(stock_symbol, 'N/A')}")
    except Exception as e:
        logging.error(f"Error fetching social sentiment for {stock_symbol}: {str(e)}")

    # Function to fetch live stock price
    def fetch_live_stock_price(symbol):
        try:
            api_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quote"  # Updated to Alpaca's real API endpoint
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            return data.get("latestPrice", "N/A")
        except Exception as e:
            logging.error(f"Error fetching live stock price for {symbol}: {str(e)}")
            return "Error"

    # Fetch and log live stock price for the selected stock
    try:
        live_price = fetch_live_stock_price(stock_symbol)
        logging.info(f"Live Stock Price for {stock_symbol}: {live_price}")
    except Exception as e:
        logging.error(f"Error fetching live stock price: {str(e)}")

    # Initialize the scheduler
    scheduler = BackgroundScheduler()

    # Define the periodic task to refresh sentiment data
    def refresh_sentiment_task():
        symbols = ["AAPL", "TSLA", "AMZN"]  # Add more symbols as needed
        social_analyzer.refresh_sentiment_data(symbols)

    # Schedule the task to run every 5 minutes
    scheduler.add_job(refresh_sentiment_task, 'interval', minutes=5)

    # Start the scheduler
    scheduler.start()

    # Ensure the scheduler shuts down gracefully on app exit
    atexit.register(lambda: scheduler.shutdown())

    # Run the Flask application
    app.run(host="0.0.0.0", port=5000, debug=True)
