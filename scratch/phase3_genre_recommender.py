import os
import pandas as pd
import numpy as np

def recommend_by_genre(genre_name="Action", data_dir="data/ml-100k", n_recs=5, quantile_threshold=0.90):
    # 1. Load ratings
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    # 2. Load movies metadata with all genres
    items_path = os.path.join(data_dir, "u.item")
    genre_list = [
        "unknown", "Action", "Adventure", "Animation", "Children's", 
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy", 
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
        "Sci-Fi", "Thriller", "War", "Western"
    ]
    items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_list
    
    # Load all genre columns and title
    movies_df = pd.read_csv(items_path, sep='|', names=items_cols, encoding='latin-1')
    
    # Validate genre
    if genre_name not in genre_list:
        raise ValueError(f"Genre '{genre_name}' is not recognized. Choose from: {genre_list}")
        
    # 3. Filter movies belonging to target genre
    genre_movies = movies_df[movies_df[genre_name] == 1][['item_id', 'movie_title']]
    
    # 4. Merge ratings with target genre movies
    merged_df = pd.merge(ratings_df, genre_movies, on='item_id')
    
    if len(merged_df) == 0:
        print(f"No ratings found for genre: {genre_name}")
        return
        
    # 5. Compute stats per movie
    movie_stats = merged_df.groupby('movie_title').agg(
        vote_count=('rating', 'count'),
        vote_average=('rating', 'mean')
    ).reset_index()
    
    # Calculate C (genre-specific average rating) and m (genre-specific threshold)
    C = merged_df['rating'].mean()
    m = movie_stats['vote_count'].quantile(quantile_threshold)
    
    print(f"--- Genre Popularity Model Parameters ({genre_name}) ---")
    print(f"Genre Average Rating (C): {C:.4f}")
    print(f"Minimum Vote Cutoff (m, {quantile_threshold*100:.0f}th percentile): {m:.2f} ratings")
    
    # Apply Weighted Rating formula
    def weighted_rating(row):
        v = row['vote_count']
        R = row['vote_average']
        return (v / (v + m)) * R + (m / (v + m)) * C
        
    movie_stats['weighted_score'] = movie_stats.apply(weighted_rating, axis=1)
    
    # Sort and return recommendations
    recs = movie_stats.sort_values(by='weighted_score', ascending=False).head(n_recs)
    return recs[['movie_title', 'vote_count', 'vote_average', 'weighted_score']]

if __name__ == "__main__":
    # Test for "Action" genre
    recs_action = recommend_by_genre(genre_name="Action")
    print("\n--- Top 5 Action Recommendations (Weighted Score) ---")
    print(recs_action.to_string(index=False))
    
    # Test for "Comedy" genre
    recs_comedy = recommend_by_genre(genre_name="Comedy")
    print("\n--- Top 5 Comedy Recommendations (Weighted Score) ---")
    print(recs_comedy.to_string(index=False))
