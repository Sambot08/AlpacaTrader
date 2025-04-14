# AlpacaTrader

AlpacaTrader is an algorithmic trading bot designed to automate trading strategies using the Alpaca API. It integrates machine learning, sentiment analysis, and fundamental data to make informed trading decisions.

## Features

- **Trading Engine**: Executes trades based on predefined strategies.
- **Machine Learning Models**: Predicts market trends using classification and regression models.
- **Sentiment Analysis**: Analyzes news and social media sentiment for trading signals.
- **Fundamental Analysis**: Evaluates company financials and sector performance.
- **Performance Tracking**: Tracks portfolio performance and key metrics.
- **Web Dashboard**: Provides a user-friendly interface to monitor and control trading activities.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sambot08/AlpacaTrader.git
   cd AlpacaTrader
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```env
     ALPACA_API_KEY=your_alpaca_api_key
     ALPACA_API_SECRET=your_alpaca_api_secret
     ALPACA_BASE_URL=https://paper-api.alpaca.markets
     SESSION_SECRET=your_session_secret
     DATABASE_URL=sqlite:///trading_bot.db
     ```

4. Initialize the database:
   ```bash
   flask db upgrade
   ```

5. Run the application:
   ```bash
   python main.py
   ```

## Usage

- Access the web dashboard at `http://localhost:5000`.
- Configure API settings, trading strategies, and risk parameters via the dashboard.
- Monitor portfolio performance and trading activity in real-time.

## Project Structure

```
AlpacaTrader/
├── app.py                 # Flask application entry point
├── trading_engine/        # Core trading engine modules
├── templates/             # HTML templates for the web dashboard
├── static/                # Static files (CSS, JS, images)
├── models.py              # Database models
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables (not included in the repo)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Alpaca API](https://alpaca.markets/) for providing trading infrastructure.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [Chart.js](https://www.chartjs.org/) for data visualization.