import logging
from app import app
import os

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask application
    app.run(host="0.0.0.0", port=5000, debug=True)
