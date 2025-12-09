import pickle
import logging
import sys
import os
import numpy as np
from thefuzz import process

#configurations
VECTOR_FILE = "show_vectors.pkl"
LOG_FILE = "app.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Prints to terminal
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    if not os.path.exists(VECTOR_FILE):
        logger.error(f"{VECTOR_FILE} not found! Run create_embeddings.py first.")
        sys.exit(1) #stop the program

    try:
        with open(VECTOR_FILE, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded {len(data)} shows from database")
        return data
    except Exception as e:
        logger.error(f"Failed to load pickle file: {e}")
        sys.exit(1)

def main():
    logger.info("Starting ShowSuggesterAI...")

    #load data
    show_data = load_data()

    #loadthe first show name 
    first_show = list(show_data.keys())[0]
    print(f"\ndata base loaded --> {first_show}")

if __name__ == "__main__":
    main()