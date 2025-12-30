# ShowSuggesterAI

ShowSuggesterAI is an intelligent TV show recommendation engine powered by valid semantic search and generative AI. It goes beyond simple keyword matching by understanding the plot, genre, and "vibe" of shows you love to suggest hidden gems. Additionally, it leverages advanced generative models to dream up entirely new TV show concepts—complete with plots and posters—tailored specifically to your taste.

## Features

- **Smart Recommendations**: Utilizes OpenAI's `text-embedding-3-small` and vector search to understand deep semantic similarities between shows.
- **Fuzzy Matching**: Smartly handles typos in user input using `thefuzz` to ensure the system recognizes the shows you mean.
- **Creative AI Concepts**: Generates unique, never-before-seen TV show ideas (Title & Plot) based on your unique viewing profile.
- **Dynamic Poster Generation**: Creates stunning, cinematic posters for these invented shows using DALL-E 3.
- **High-Performance Search**: Built on `usearch` for lightning-fast vector retrieval.

## Tech Stack

- **Python 3.8+**
- **OpenAI API** (Embeddings, Chat completions, DALL-E 3)
- **USearch** (Vector Search Engine)
- **Pandas & NumPy** (Data Analysis)
- **TheFuzz** (String Matching)
- **Pillow** (Image Processing)

##  Prerequisites

- **Python**: Ensure you have Python installed.
- **OpenAI API Key**: You will need a valid API key from OpenAI.

##  Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ShowSuggesterAI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
   Add your OpenAI API key to the `.env` file:
   ```env
   OPENAI_API_KEY=your_sk_key_here
   ```

## Usage

Run the main application:

```bash
python main.py
```

### First Run Setup
The first time you launch the application, it will perform an initialization step:
- It reads the `imdb_tvshows.csv` dataset.
- Generates vector embeddings for all shows using the OpenAI API.
- Builds a fast search index.

**Note:** This one-time process may take **2-3 minutes**. Subsequent runs will launch instantly using the cached data.

### Interaction
1. When prompted, submit a list of a few TV shows you have enjoyed (separated by commas).
2. The AI will verify the titles.
3. Receive top personalized recommendations.
4. Enjoy two exclusive, AI-invented show concepts with custom-generated posters designed just for you.

##  Project Structure

- `main.py`: The entry point for the CLI application. Handles user interaction and orchestration.
- `create_embeddings.py`: Handles data loading, embedding generation, and index creation.
- `imdb_tvshows.csv`: Source dataset containing TV show metadata.
- `show_vectors.pkl`: Serialized storage for show metadata and vectors.
- `show_vectors.usearch`: High-performance vector search index.
- `requirements.txt`: Python dependency list.

## Credits
Built as part of the Advanced System Development using AI course.
