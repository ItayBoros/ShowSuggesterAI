import os
from dotenv import load_dotenv
import pickle
import pandas as pd
from usearch.index import Index
import numpy as np
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

#configurations
CSV_FILE_PATH = 'imdb_tvshows.csv'
DATA_FILE = 'show_vectors.pkl'
INDEX_FILE = 'show_vectors.usearch'
LOG_FILE = "setup.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

def get_embedding(text):
    try:
        #remove new lines
        text = text.replace("\n", " ")

        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding for text: {text[:30]}... Error: {e}")
        return None
    
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
    
    title_to_vector = {}
    ordered_titles = []
    vectors_list = []

    logger.info("Start embedding generation process...")

    for index, row in df.iterrows():
        try:
            title = row['Title']
            description = row['Description']
            # Add Genre to Description
            genre = row['Genres'] if 'Genres' in row and pd.notna(row['Genres']) else "Unknown Genre"
            actors = row['Actors'] if 'Actors' in row and pd.notna(row['Actors']) else "Unknown Cast"
            year = row['Years'] if 'Years' in row and pd.notna(row['Years']) else "Unknown Year"

            rating_text = ""
            if 'Rating' in row and pd.notna(row['Rating']):
                try:
                    r = float(row['Rating'])
                    if r >= 8.5:
                        rating_text = "Critically Acclaimed Masterpiece."
                    elif r >= 7.5:
                        rating_text = "Highly Rated."
                except:
                    pass
            
            duration_text = ""
            if 'EpisodeDuration(in Minutes)' in row and pd.notna(row['EpisodeDuration(in Minutes)']):
                duration_text = f"{row['EpisodeDuration(in Minutes)']} min episodes."

            text_to_embed = (
                f"Title: {title}. "
                f"Year: {year}. "
                f"Genre: {genre}. "
                f"{rating_text} "     
                f"Format: {duration_text} "
                f"Starring: {actors}. "
                f"Plot: {description}"
            )

            vector = get_embedding(text_to_embed)

            if vector:
                # Store data in three places
                title_to_vector[title] = vector
                ordered_titles.append(title)
                vectors_list.append(vector)
                if index % 10 == 0:
                    logger.info(f"Processed {index} shows...")

        except KeyError as e:
            logger.error(f"Column missing in CSV row {index}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error at row {index}: {e}")

    if vectors_list:
        try:
            #save Metadata
            data_package = {
                "dict": title_to_vector,
                "list": ordered_titles
            }
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(data_package, f)

            #build and save usearch index
            vectors_np = np.array(vectors_list, dtype=np.float32)
            dim = vectors_np.shape[1]
            
            index = Index(ndim=dim, metric='cos')
            index.add(np.arange(len(vectors_np)), vectors_np)
            index.save(INDEX_FILE)
            
            logger.info(f"Success! Saved {DATA_FILE} and {INDEX_FILE}")

        except Exception as e:
            logger.error(f"Error saving files: {e}")

    else:
        logger.warning("No embeddings were generated. Check CSV.")

    logger.info("Embedding generation process completed.")

if __name__ == "__main__":
    main()

