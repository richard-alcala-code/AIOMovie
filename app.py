import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data and Model
try:
    movies = pd.read_csv('data/movies.csv')
    links = pd.read_csv('data/links.csv')
    ratings = pd.read_csv('data/ratings.csv')

    movie_data = pd.merge(ratings, links, on='movieId')
    movie_data = pd.merge(movie_data, movies[['movieId', 'title']], on='movieId')

    # Load the pre-trained Keras model
    model = tf.keras.models.load_model('aiomovie_model_v3.keras')

    # Recreate necessary mappings
    user_ids = movie_data["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = sorted(movie_data["tmdbId"].unique().tolist())
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    tmdb_id_to_title_map_df = pd.merge(movies[['movieId', 'title']], links[['movieId', 'tmdbId']], on='movieId', how='inner')
    tmdb_id_to_title_map = tmdb_id_to_title_map_df.drop_duplicates(subset=['tmdbId']).set_index('tmdbId')['title'].to_dict()

except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure 'movies.csv', 'links.csv', 'ratings.csv', and 'aiomovie_model_v3.keras' are in the correct directory.")


# 2. Define Recommendation Functions
def get_movie_info_by_title(title_substring, movies_df, movie_data_df):
    """
    Searches for a movie by a substring of its title (case-insensitive)
    and returns its tmdbId and the exact found title.
    Uses movies_df for initial title search and movie_data_df to link to tmdbId.
    """
    matching_movies_in_movies = movies_df[movies_df['title'].str.contains(title_substring, case=False, na=False, regex=False)]

    if not matching_movies_in_movies.empty:
        matched_movie_id = matching_movies_in_movies.iloc[0]['movieId']
        exact_found_title = matching_movies_in_movies.iloc[0]['title']

        tmdb_id_row = movie_data_df[movie_data_df['movieId'] == matched_movie_id]
        if not tmdb_id_row.empty:
            tmdb_id = tmdb_id_row.iloc[0]['tmdbId']
            return tmdb_id, exact_found_title
    return None, None

def recommend_movies_based_on_movie(input_movie_title, model, movie2movie_encoded, movies_df, tmdb_id_to_title_map, top_n=10):
    """
    Generates movie recommendations based on a given movie title using movie embeddings.
    """
    target_tmdb_id, found_title = get_movie_info_by_title(input_movie_title, movies_df, movie_data)
    if target_tmdb_id is None:
        return f"Movie '{input_movie_title}' not found in the dataset."

    target_movie_encoded = movie2movie_encoded.get(target_tmdb_id)
    if target_movie_encoded is None:
        return f"Encoded ID for '{found_title}' (TMDB ID: {target_tmdb_id}) not found. This movie might not be in the training data."

    movie_embedding_layer = model.get_layer('movie_embedding')
    movie_embeddings = movie_embedding_layer.get_weights()[0]

    target_movie_embedding = movie_embeddings[target_movie_encoded]

    similarities = cosine_similarity(target_movie_embedding.reshape(1, -1), movie_embeddings).flatten()

    top_similar_encoded_indices = similarities.argsort()[::-1][1:top_n+1]

    recommended_titles = []
    movie_encoded_to_tmdb = {v: k for k, v in movie2movie_encoded.items()}

    for encoded_idx in top_similar_encoded_indices:
        recommended_tmdb_id = movie_encoded_to_tmdb.get(encoded_idx)
        if recommended_tmdb_id is not None:
            title = tmdb_id_to_title_map.get(recommended_tmdb_id)
            if title:
                recommended_titles.append(title)

    if recommended_titles:
        return f"Top {top_n} recommendations for '{found_title}':\n" + "\n".join([f"{i+1}. {title}" for i, title in enumerate(recommended_titles)])
    else:
        return f"No recommendations could be generated for '{found_title}'."

# 3. Gradio Interface
def gradio_recommendation_interface(movie_title):
    # This function acts as a wrapper for Gradio
    return recommend_movies_based_on_movie(movie_title, model, movie2movie_encoded, movies, tmdb_id_to_title_map, top_n=10)

# Create the Gradio interface
demo = gr.Interface(
    fn=gradio_recommendation_interface,
    inputs=gr.Textbox(lines=1, placeholder="Enter a movie title (e.g., Toy Story)", label="Movie Title"),
    outputs=gr.Textbox(lines=10, label="Recommendations"),
    title="AIOMovie - Movie Recommendation System",
    description="Get movie recommendations based on a specific movie title using a deep learning model."
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
