import os
import re
import gradio as gr
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Loads KEY=VALUE pairs from a .env file
def load_env_file(path='.env'):
    if not os.path.exists(path):
        print(f"Warning: .env file not found at {path}")
        return
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# --- 1. Configuration ---
load_env_file('.env')
TMDB_TOKEN = os.getenv("TMDB_TOKEN")   # Bearer token for TMDB API v3
TMDB_BASE_URL = "https://api.themoviedb.org/3"


# --- 2. Load Data and Models ---
# Loads MovieLens CSVs, merges them into a single DataFrame, and loads all 3 NCF models.
try:
    movies  = pd.read_csv('data/movies.csv')
    links   = pd.read_csv('data/links.csv')
    ratings = pd.read_csv('data/ratings.csv')

    # Merge ratings + links + movie titles into one working DataFrame
    movie_data = pd.merge(ratings, links, on='movieId')
    movie_data = pd.merge(movie_data, movies[['movieId', 'title']], on='movieId')

    # Load all 3 NCF levels: GMF (linear), MLP (non-linear), NeuMF (fusion)
    models = {
        "GMF":   tf.keras.models.load_model('models/ncf/aiomovie_ncf_gmf.keras'),
        "MLP":   tf.keras.models.load_model('models/ncf/aiomovie_ncf_mlp.keras'),
        "NeuMF": tf.keras.models.load_model('models/ncf/aiomovie_ncf_neumf.keras'),
    }

    # Each model uses a differently named embedding layer — map model name to layer name
    EMBEDDING_LAYER = {
        "GMF":   "movie_embedding_gmf",
        "MLP":   "movie_embedding_mlp",
        "NeuMF": "movie_embedding_neumf_mlp",  # NeuMF MLP branch used for cosine similarity
    }

    # Encode userId and tmdbId to contiguous integer indices (required by Embedding layers)
    user_ids = movie_data["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = sorted(movie_data["tmdbId"].unique().tolist())
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    # Build a tmdbId → title lookup map for fast title resolution after recommendations
    tmdb_id_to_title_map_df = pd.merge(movies[['movieId', 'title']], links[['movieId', 'tmdbId']], on='movieId', how='inner')
    tmdb_id_to_title_map = tmdb_id_to_title_map_df.drop_duplicates(subset=['tmdbId']).set_index('tmdbId')['title'].to_dict()

except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure 'data/' and 'models/ncf/' directories exist.")
    exit()


# --- 3. Core Functions ---

def get_movie_info_by_title(title_substring, movies_df, movie_data_df):
    """
    Searches for a movie by a substring of its title (case-insensitive).
    Returns (tmdbId, exact_title) of the first match, or (None, None) if not found.
    """
    matches = movies_df[movies_df['title'].str.contains(title_substring, case=False, na=False, regex=False)]
    if not matches.empty:
        matched_movie_id = matches.iloc[0]['movieId']
        exact_found_title = matches.iloc[0]['title']
        tmdb_id_row = movie_data_df[movie_data_df['movieId'] == matched_movie_id]
        if not tmdb_id_row.empty:
            return tmdb_id_row.iloc[0]['tmdbId'], exact_found_title
    return None, None


def recommend_movies_based_on_movie(input_movie_title, model, movie2movie_encoded, movies_df, tmdb_id_to_title_map, top_n=10, embedding_layer_name='movie_embedding_gmf'):
    """
    Generates top-N movie recommendations for a given title using cosine similarity
    on the trained movie embeddings from the specified NCF model.

    Strategy: extract the movie embedding vector, compute cosine similarity against
    all other movie embeddings, return the top-N most similar titles.
    """
    target_tmdb_id, found_title = get_movie_info_by_title(input_movie_title, movies_df, movie_data)
    if target_tmdb_id is None:
        return f"Movie '{input_movie_title}' not found in the dataset.", []

    target_movie_encoded = movie2movie_encoded.get(target_tmdb_id)
    if target_movie_encoded is None:
        return f"Encoded ID for '{found_title}' (TMDB ID: {target_tmdb_id}) not found in training data.", []

    # Extract the embedding matrix from the model (shape: num_movies × embedding_size)
    try:
        movie_embedding_layer = model.get_layer(embedding_layer_name)
    except ValueError:
        return f"Error: Embedding layer '{embedding_layer_name}' not found in the model.", []

    movie_embeddings = movie_embedding_layer.get_weights()[0]

    if target_movie_encoded >= len(movie_embeddings):
        return f"Error: Encoded ID {target_movie_encoded} for '{found_title}' is out of bounds.", []

    target_movie_embedding = movie_embeddings[target_movie_encoded]

    # Compute cosine similarity between the target movie and all other movies
    similarities = cosine_similarity(target_movie_embedding.reshape(1, -1), movie_embeddings).flatten()

    # Sort descending, skip index 0 (the movie itself), take top_n
    top_indices = similarities.argsort()[::-1]
    top_similar_encoded_indices = top_indices[1:top_n + 1]

    # Reverse mapping: encoded index → tmdbId → title
    movie_encoded_to_tmdb = {v: k for k, v in movie2movie_encoded.items()}
    recommended_titles = []
    for encoded_idx in top_similar_encoded_indices:
        recommended_tmdb_id = movie_encoded_to_tmdb.get(encoded_idx)
        if recommended_tmdb_id is not None:
            title = tmdb_id_to_title_map.get(recommended_tmdb_id)
            if title:
                recommended_titles.append(title)

    if recommended_titles:
        return f"Top {top_n} recommendations for '{found_title}':", recommended_titles
    return f"No recommendations could be generated for '{found_title}'.", []


def get_tmdb_movie_details(query):
    """
    Searches TMDB for a movie by title and returns (details_text, poster_url, movie_id).
    Returns (None, None, None) on failure or missing token.
    """
    if not TMDB_TOKEN:
        print("TMDB_TOKEN not set. Cannot fetch movie details.")
        return (None, None, None)

    headers = {"Authorization": f"Bearer {TMDB_TOKEN}"}
    params  = {"query": query, "include_adult": False, "language": "en-US", "page": 1}

    try:
        response = requests.get(f"{TMDB_BASE_URL}/search/movie", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return (None, None, None)

        movie       = results[0]
        movie_id    = movie.get("id")
        title       = movie.get("title") or "Unknown title"
        overview    = movie.get("overview") or "No overview available."
        release_date = movie.get("release_date") or "N/A"
        vote_average = movie.get("vote_average")
        popularity   = movie.get("popularity")
        poster_path  = movie.get("poster_path")
        poster_url   = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        details = (
            f"Title: {title}\n"
            f"Release date: {release_date}\n"
            f"Vote average: {vote_average if vote_average is not None else 'N/A'}\n"
            f"Popularity: {popularity if popularity is not None else 'N/A'}\n\n"
            f"Overview: {overview}"
        )
        return details, poster_url, movie_id
    except requests.RequestException as e:
        print(f"TMDB API request failed: {e}")
        return (None, None, None)


def get_watch_providers_html(movie_id):
    """
    Fetches streaming providers for a movie from TMDB.
    Prefers region 'BO' (Bolivia), falls back to first available region.
    Returns an HTML block with provider logos linking to the TMDB watch page.
    """
    if not TMDB_TOKEN or not movie_id:
        return ""
    headers = {"Authorization": f"Bearer {TMDB_TOKEN}"}
    try:
        response = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/watch/providers", headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", {})

        # Prefer BO region; fall back to first available region
        region   = results.get("BO") or next(iter(results.values()), None)
        if not region:
            return ""

        flatrate = region.get("flatrate", [])
        link     = region.get("link", "https://www.themoviedb.org")
        if not flatrate:
            return ""

        logos = "".join(
            f'<a href="{link}" target="_blank" rel="noopener noreferrer" title="{p.get("provider_name")}">'
            f'<img src="https://image.tmdb.org/t/p/w92{p.get("logo_path")}" style="height:40px;border-radius:8px;margin:4px;">'
            f'</a>'
            for p in flatrate if p.get("logo_path")
        )
        return f'<div style="margin-top:8px;"><strong>Watch Now</strong><br><div style="display:flex;flex-wrap:wrap;margin-top:6px;">{logos}</div></div>'
    except requests.RequestException as e:
        print(f"Watch providers API request failed: {e}")
        return ""


# --- 4. Gradio UI Builders ---

def build_movie_details_html(poster_url, details, watch_now_html):
    """
    Renders the selected movie's poster, metadata and watch providers
    as a single HTML block with poster on the left and details on the right.
    """
    poster_html = (
        f'<img src="{poster_url}" style="width:160px;min-width:160px;border-radius:10px;">'
        if poster_url else
        '<div style="width:160px;min-width:160px;height:240px;background:#333;border-radius:10px;"></div>'
    )
    # Convert plain-text details (newline-separated) into HTML paragraphs
    details_text = ""
    if details:
        for line in details.split("\n"):
            details_text += f'<p style="margin:2px 0;font-size:14px;">{line}</p>'

    return (
        f'<div style="display:flex;gap:16px;align-items:flex-start;">'
        f'{poster_html}'
        f'<div style="flex:1">{details_text}{watch_now_html}</div>'
        f'</div>'
    )


def build_recommendations_html(recommendation_titles):
    """
    Renders a numbered list of recommended movie titles as an HTML block.
    """
    if not recommendation_titles:
        return ""
    rows = []
    for i, title in enumerate(recommendation_titles, 1):
        rows.append(
            f'<div style="padding:8px 0;border-bottom:1px solid rgba(150,150,150,0.25);">'
            f'<span style="font-size:15px;">{i}. {title}</span>'
            f'</div>'
        )
    return f'<div style="padding:4px 0;">{"".join(rows)}</div>'


# --- 5. Main Gradio Handler ---

def gradio_recommendation_interface(movie_title, selected_models):
    """
    Main handler called when the user submits a movie title.
    1. Validates input.
    2. Runs recommendations for each selected NCF model.
    3. Fetches TMDB details and watch providers for the matched movie.
    4. Returns HTML for movie details and side-by-side recommendation columns.
    """
    if not movie_title or not movie_title.strip():
        return "", "", ""

    # Default to GMF if user deselects all models
    if not selected_models:
        selected_models = ["GMF"]

    # Use the first selected model to resolve the canonical matched title
    first_header, first_titles = recommend_movies_based_on_movie(
        movie_title, models[selected_models[0]], movie2movie_encoded, movies,
        tmdb_id_to_title_map, top_n=6, embedding_layer_name=EMBEDDING_LAYER[selected_models[0]]
    )

    if not isinstance(first_titles, list) or not first_titles:
        details, poster_url, _ = get_tmdb_movie_details(movie_title)
        details_html = build_movie_details_html(poster_url, details, "")
        return details_html, f"No recommendations found for '{movie_title}'. Try a different title.", ""

    # Extract matched title from header string, strip year suffix before TMDB query
    matched_title = first_header.split("'")[1] if "'" in first_header else movie_title
    clean_title   = re.sub(r'\s*\(\d{4}\)$', '', matched_title).strip()

    details, poster_url, movie_id = get_tmdb_movie_details(clean_title)
    watch_now_html = get_watch_providers_html(movie_id)
    details_html   = build_movie_details_html(poster_url, details, watch_now_html)

    # Build one recommendation column per selected model, displayed side by side
    columns = []
    for model_name in selected_models:
        _, titles = recommend_movies_based_on_movie(
            movie_title, models[model_name], movie2movie_encoded, movies,
            tmdb_id_to_title_map, top_n=6, embedding_layer_name=EMBEDDING_LAYER[model_name]
        )
        col_content = build_recommendations_html(titles if isinstance(titles, list) else [])
        columns.append(
            f'<div style="flex:1;min-width:220px;">'
            f'<h4 style="margin:0 0 8px 0;">{model_name}</h4>'
            f'{col_content}'
            f'</div>'
        )

    recommendations_html   = f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{"".join(columns)}</div>'
    recommendations_header = f"Top 6 recommendations for '{matched_title}':"

    return details_html, recommendations_header, recommendations_html


# --- 6. Gradio UI Layout ---

with gr.Blocks(title="AIOMovie - Movie Recommendation System") as demo:
    gr.HTML('<div style="text-align: center;">\n<h1>AIOMovie - Movie Recommendation System</h1>\n</div>')
    gr.HTML(
        '<div style="text-align:center;">'
        '<p style="margin:0;">Search a movie and get AI-powered recommendations using Neural Collaborative Filtering (NCF)</p>'
        '<p style="margin:0;">— a deep learning model that learns hidden user and movie preferences from millions of ratings, without needing to know anything about the movie\'s content.</p>'
        '</div>'
    )

    with gr.Row():
        movie_title = gr.Textbox(lines=1, placeholder="Enter a movie title (e.g., Toy Story), and Press Enter", label="Movie Title")
        # CheckboxGroup lets the user compare multiple NCF levels side by side; defaults to GMF
        model_selector = gr.CheckboxGroup(
            choices=["GMF", "MLP", "NeuMF"],
            value=["GMF"],
            label="NCF Model"
        )

    # Displays poster + metadata + watch providers in a single inline HTML block
    movie_details_html = gr.HTML(value="")

    with gr.Row():
        recommendations_header = gr.Markdown(value="")
    # Displays one column per selected model side by side
    recommendations_html = gr.HTML(value="")

    clear_btn = gr.Button("Clear", variant="secondary")

    def clear_all():
        # Resets all UI components to their default state
        return "", ["GMF"], "", "", ""

    clear_btn.click(
        fn=clear_all,
        outputs=[movie_title, model_selector, movie_details_html, recommendations_header, recommendations_html],
    )

    movie_title.submit(
        fn=gradio_recommendation_interface,
        inputs=[movie_title, model_selector],
        outputs=[movie_details_html, recommendations_header, recommendations_html],
    )

if __name__ == "__main__":
    demo.launch(share=False)
