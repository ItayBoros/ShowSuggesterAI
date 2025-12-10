import pickle
import logging
import sys
import os
import numpy as np
from thefuzz import process
from usearch.index import Index

#configurations
VECTOR_FILE = "show_vectors.pkl"
LOG_FILE = "app.log"
TOP_N_RECOMMENDATIONS = 5

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

def get_valid_show_name(user_input, all_show_name):
    #split by comma and clean up
    raw_names = [name.strip() for name in user_input.split(',') if name.strip()]

    if len(raw_names) < 2:
        print("Please enter at least 2 shows.")
        return None
    
    matched_shows = []

    # find the best match for each show the user enter
    for raw in raw_names:
        best_match, score = process.extractOne(raw, all_show_name)
        matched_shows.append(best_match)

    #ask user to confirm
    formatted_names = ", ".join(matched_shows)
    print(f"\nMaking sure, do you mean {formatted_names}? (y/n)")

    confirmation = input().strip().lower()
    if confirmation == 'y':
        return matched_shows
    else:
        return None
    
def get_recommendations(user_shows, show_data):
    #get vectors for the user shows
    user_vectors = []
    for show in user_shows:
        user_vectors.append(show_data[show])
    
    #calculate the average vector == user profile
    average_vector = np.mean(user_vectors, axis=0)

    #preare the index
    all_titles = list(show_data.keys())
    #convent dict values to a matrix
    all_vectors = np.array(list(show_data.values()), dtype=np.float32)
    vector_dim =len(all_vectors[0])
    
    # use cosine similarity for better results
    index = Index(ndim=vector_dim, metric='cosine')
    index.add(np.arange(len(all_vectors)), all_vectors)

    #search for the closest shows
    search_count = TOP_N_RECOMMENDATIONS + len(user_shows)
    matches = index.search(average_vector, search_count)

    found_indices = matches.keys.flatten()
    found_distances = matches.distances.flatten()

    print("\nHere are the tv shows that i think you would love:")
    recommendations = []
    count = 0

    for i, idx in enumerate(found_indices):
        #make sure index is valid
        if idx >= len(all_titles):
            continue

        title = all_titles[idx]
        
        #skip shows the user already input
        if title in user_shows:
            continue
        
        #convert dist to similarity score
        dist = found_distances[i]
        score_percent = int((1 - dist) * 100)
        
        print(f"{title} ({score_percent}%)")
        recommendations.append(title)

        count += 1
        if count >= TOP_N_RECOMMENDATIONS:
            break
    
    return recommendations

def main():
    #load all show titles
    show_data = load_data()
    all_titles = list(show_data.keys())

    print("Welcome to ShowSuggesterAI!")   
    while True:
        print("\n\nWhich TV shows did you really like watching? Separate them by a comma.")
        print("Make sure to enter more than 1 show.")
        user_text = input("Input: ")

        #matching
        confirmed_shows= get_valid_show_name(user_text, all_titles)

        if confirmed_shows:
            print(f"\nGreat! genetaring recommendations now...")
            logger.info(f"user confirmed: {confirmed_shows}")

            recommendations = get_recommendations(confirmed_shows, show_data)
            logger.info(f"Recommendations generated: {recommendations}")
            break
        else:
            print("Sorry about that. Let's try again.")

if __name__ == "__main__":
    main()