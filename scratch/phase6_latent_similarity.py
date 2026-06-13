import os
import sys
# Add parent directory to path to import matrix_factorization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matrix_factorization import FunkSVD

def analyze_latent_similarity(movie_title="Star Wars (1977)", data_dir="data/ml-100k", top_n=5):
    # 1. Load data
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    items_path = os.path.join(data_dir, "u.item")
    items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies_df = pd.read_csv(items_path, sep='|', names=items_cols, 
                            usecols=['item_id', 'movie_title'], encoding='latin-1')
    
    # 2. Fit FunkSVD
    print("Training FunkSVD model...")
    model = FunkSVD(n_factors=15, lr=0.005, reg=0.02, epochs=20)
    model.fit(ratings_df)
    
    # 3. Get item index
    matching_movies = movies_df[movies_df['movie_title'] == movie_title]
    if len(matching_movies) == 0:
        print(f"Movie '{movie_title}' not found.")
        return
    item_id = matching_movies['item_id'].values[0]
    
    if item_id not in model.item_to_idx:
        print(f"Item ID {item_id} not seen in training.")
        return
    item_idx = model.item_to_idx[item_id]
    
    # 4. Extract latent vector for Star Wars
    star_wars_vector = model.Q[item_idx].reshape(1, -1)
    
    # 5. Compute cosine similarities with all other items in Q
    similarities = cosine_similarity(star_wars_vector, model.Q)[0]
    
    # Create DataFrame of results
    sim_df = pd.DataFrame({
        'item_idx': range(len(similarities)),
        'similarity': similarities
    })
    
    # Exclude self
    sim_df = sim_df[sim_df['item_idx'] != item_idx]
    
    # Get top N similar
    top_sim = sim_df.nlargest(top_n, 'similarity')
    
    print(f"\nTop {top_n} movies similar to '{movie_title}' in latent space (FunkSVD):")
    for _, row in top_sim.iterrows():
        nbr_idx = int(row['item_idx'])
        nbr_item_id = model.idx_to_item[nbr_idx]
        nbr_title = movies_df[movies_df['item_id'] == nbr_item_id]['movie_title'].values[0]
        print(f"  - {nbr_title} (Similarity: {row['similarity']:.4f})")

if __name__ == "__main__":
    analyze_latent_similarity()
