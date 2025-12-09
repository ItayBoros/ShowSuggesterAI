import os
from dotenv import load_dotenv
import pickle
import pandas as pd
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

#configurations
CSV_FILE_PATH = 'imdb_tvshows.csv'
OUTPUT_FILE_PATH = 'show_vectors.pkl'
LOG_FILE = "setup.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_embedding(text):
    #remove new lines
    text = text.replace("\n", " ")

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def main():
    logger.info("Loading CSV file...")

    # Load CSV file
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"CSV file not found: {CSV_FILE_PATH}")
        return
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"Loaded {len(df)} shows from {CSV_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return
    
    embedding_dict = {}

    logger.info("Start embedding generation process...")

    for index, row in df.iterrows():
        try:
            title = row['Title']
            description = row['Description']

            vector = get_embedding(description)

            if vector:
                embedding_dict[title] = vector
                if index % 10 == 0:
                    logger.info(f"Processed {index} shows...")
        except KeyError as e:
            logger.error(f"Column missing in CSV row {index}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error at row {index}: {e}")
    if embedding_dict:
        try:
            with open(OUTPUT_FILE_PATH, 'wb') as f:
                pickle.dump(embedding_dict, f)
            logger.info(f"Embeddings saved to {OUTPUT_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    else:
        logger.warning("No embeddings were generated. Check CSV.")

    logger.info("Embedding generation process completed.")

if __name__ == "__main__":
    main()

