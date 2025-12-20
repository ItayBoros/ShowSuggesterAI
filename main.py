import pickle
import logging
import sys
import os
from time import sleep
from usearch.index import Index
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from thefuzz import process
from usearch.index import Index
from openai import OpenAI
from dotenv import load_dotenv
import create_embeddings

#configurations
VECTOR_FILE = "show_vectors.pkl"
LOG_FILE = "app.log"
INDEX_FILE = 'show_vectors.usearch'
TOP_N_RECOMMENDATIONS = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_data():
    if not os.path.exists(VECTOR_FILE) or not os.path.exists(INDEX_FILE):
        logger.error(f"Missing {VECTOR_FILE} or {INDEX_FILE}. Run create_embeddings.py first.")
        print("\n--- FIRST TIME SETUP DETECTED ---")
        print("We need to analyze the show database. This will take 2-3 minutes.")
        print("Please wait...\n")
        create_embeddings.main()  #create the embeddings file
        print("\n--- Setup Complete! Starting App ---\n")

    try:
        with open(VECTOR_FILE, 'rb') as f:
            data_package = pickle.load(f)
        logger.info(f"Successfully loaded {len(data_package['list'])} shows from database")
        
        #load the O(1) index
        # We get dimensions from the first vector in the dict
        first_vec = next(iter(data_package['dict'].values()))
        dim = len(first_vec)

        index = Index(ndim=dim, metric='cos')
        index.load(INDEX_FILE) #fast load from disk
        return data_package, index
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
    
def get_recommendations(user_shows, data_package, index):
    show_dict = data_package['dict']
    show_list = data_package['list']
    #get vectors for the user shows
    user_vectors = []
    for show in user_shows:
        if show in show_dict:
            user_vectors.append(show_dict[show])
    
    if not user_vectors: return []

    #calculate the average vector == user profile
    average_vector = np.mean(np.array(user_vectors, dtype=np.float32), axis=0).astype(np.float32)

    search_count = TOP_N_RECOMMENDATIONS + len(user_shows) + 5
    matches = index.search(average_vector, search_count)

    found_indices = matches.keys.flatten()
    found_dists = matches.distances.flatten()

    print("\nHere are the tv shows that i think you would love:")
    recommendations = []
    count = 0

    for i, idx in enumerate(found_indices):
        idx = int(idx)
        #make sure index is valid
        if idx >= len(show_list):
            continue
       
        title = show_list[idx]
        
        #skip shows the user already input
        if title in user_shows:
            continue
        
        #convert dist to similarity score
        dist = float(found_dists[i])
        sigma = 4.0 
        sim_score = np.exp(-(dist ** 2) / sigma)
        score_percent = int(sim_score * 100)
        #cosmetic fix: reduce score a bit for each next recommendation
        score_percent = score_percent - count
        
        #final caps
        if score_percent > 99:
            score_percent = 99
        if score_percent < 1:
            score_percent = 1

        print(f"  {count+1}. {title}  ({score_percent}%)")
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
    data_package, index = load_data()
    all_titles = data_package['list']

    print("\n" + "=" * 50)
    print("                 ShowSuggesterAI  ")
    print("=" * 50 + "\n")   
    while True:
        print("\nWhich TV shows did you really like watching? Separate them by a comma.")
        print("Make sure to enter more than 1 show.")
        user_text = input("Input: ")

        #matching
        confirmed_shows= get_valid_show_name(user_text, all_titles)

        if confirmed_shows:
            print(f"\nGreat! generating recommendations now...")
            sleep(1.5) #simulate thinking
            logger.info(f"user confirmed: {confirmed_shows}")

            recommendations = get_recommendations(confirmed_shows, data_package,index)
            logger.info(f"Recommendations generated: {recommendations}")

            #generate creative shows
            print("\nI have also created just for you two shows which I think you would love.")
            
            #show 1 
            title1, plot1 = generate_creative_show(confirmed_shows)
            print(f"\nShow #1 is based on the fact that you loved the input shows that you gave me.")
            print(f"Show #1 – {title1}")
            print(f"  {plot1}\n")
            generate_poster(title1, plot1)

            #show 2
            title2, plot2 = generate_creative_show(recommendations)
            print(f"\nShow #2 is based on the shows that I recommended for you.")
            print(f"Show #2 – {title2}")
            print(f"  {plot2}\n")
            generate_poster(title2, plot2)
            print("\nHere are also the 2 tv show ads. Hope you like them!")

        else:
            print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly")

if __name__ == "__main__":
    main()