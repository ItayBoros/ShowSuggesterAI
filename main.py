import pickle
import logging
import sys
import os
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from thefuzz import process
from usearch.index import Index
from openai import OpenAI
from dotenv import load_dotenv

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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        #divide by 2 to curve the results higher
        dist = found_distances[i]
        score_percent = int((1 - (dist / 2)) * 100)
        
        print(f"{title} ({score_percent}%)")
        recommendations.append(title)

        count += 1
        if count >= TOP_N_RECOMMENDATIONS:
            break
    
    return recommendations

def generate_creative_show(basis_shows):
    shows_str = ", ".join(basis_shows)
    prompt = f"""Invent a NEW TV show that combines the style/genre of these shows: {shows_str}.
    Provide exactly two lines:
    Line 1: The Title
    Line 2: A one-sentence plot summary.
    Do not add labels like "Title:" or "Plot:".
    """

    try:
        response  = client.chat.completions.create(
            model = "gpt-5-nano-2025-08-07",
            messages = [
                {"role": "user", "content": prompt}
            ],
        )
        content = response.choices[0].message.content.strip().split("\n")
        content = [line for line in content if line.strip()]

        if len(content) >= 2:
            return content[0], content[1]
        else :
            return "Mystery Show", content[0]
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        return "Unknown Show", "Could not generate plot."

def generate_poster(title, description):
    print(f"   [Generating poster for '{title}'...]")

    # sanitize the prompt for safety
    try:
        sanitizer_response = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{
                "role": "system", 
                "content": "You are an expert art director. Rewrite the following TV show plot into a visual description for a poster. It must be SAFE for work and strictly adhere to content policies (NO drugs, NO guns, NO blood, NO violence). Focus on the mood, lighting, atmosphere, and symbolic elements instead."
            }, {
                "role": "user", 
                "content": f"Title: {title}. Plot: {description}"
            }]
        )
        safe_prompt = sanitizer_response.choices[0].message.content + "dont generate any text in the image"
    except Exception as e:
        logger.error(f"Sanitization failed: {e}")
        safe_prompt = f"A cinematic poster for a show titled {title}, dramatic lighting, mystery atmosphere."
    
    #generate the image using the safe prompt
    print(f"   [Generating poster for '{title}'...]")
    try:
        response = client.images.generate(
            model="dall-e-3", 
            prompt=safe_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        
        #download and show image
        img_data = requests.get(image_url).content
        img = Image.open(BytesIO(img_data))
        img.show() # opens the image 
        logger.info(f"Poster generated for {title}")
        
    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        print("   (Could not generate image due to API error)")

def main():
    #load all show titles
    show_data = load_data()
    all_titles = list(show_data.keys())

    print("\n-----Welcome to ShowSuggesterAI!-----")   
    while True:
        print("\nWhich TV shows did you really like watching? Separate them by a comma.")
        print("Make sure to enter more than 1 show.")
        user_text = input("Input: ")

        #matching
        confirmed_shows= get_valid_show_name(user_text, all_titles)

        if confirmed_shows:
            print(f"\nGreat! genetaring recommendations now...")
            logger.info(f"user confirmed: {confirmed_shows}")

            recommendations = get_recommendations(confirmed_shows, show_data)
            logger.info(f"Recommendations generated: {recommendations}")

            #generate creative shows
            print("\nI have also created just for you two shows which I think you would love.")
            
            #show 1 
            title1, plot1 = generate_creative_show(confirmed_shows)
            print(f"\nShow #1 is based on the fact that you loved the input shows that you gave me.")
            print(f"Its name is {title1} and it is about {plot1}")
            generate_poster(title1, plot1)

            #show 2
            title2, plot2 = generate_creative_show(recommendations)
            print(f"\nShow #2 is based on the shows that I recommended for you.")
            print(f"Its name is {title2} and it is about {plot2}")
            generate_poster(title2, plot2)
            
            print("\nHere are also the 2 tv show ads. Hope you like them!")

        else:
            print("Sorry about that. Let's try again.")

if __name__ == "__main__":
    main()