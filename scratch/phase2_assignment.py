import os
import pandas as pd

def iterative_filtering(data_dir="data/ml-100k", min_user_ratings=20, min_movie_ratings=10):
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    initial_users = df['user_id'].nunique()
    initial_movies = df['item_id'].nunique()
    initial_ratings = len(df)
    initial_sparsity = (1 - (initial_ratings / (initial_users * initial_movies))) * 100
    
    print(f"--- Before Filtering ---")
    print(f"Users: {initial_users}, Movies: {initial_movies}, Ratings: {initial_ratings}")
    print(f"Sparsity: {initial_sparsity:.2f}%\n")
    
    iteration = 0
    while True:
        iteration += 1
        prev_ratings = len(df)
        
        # 1. Filter users
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['user_id'].isin(active_users)]
        
        # 2. Filter movies
        movie_counts = df['item_id'].value_counts()
        popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df['item_id'].isin(popular_movies)]
        
        # Check if dataset size stabilized
        current_ratings = len(df)
        if current_ratings == prev_ratings:
            print(f"Stabilized after {iteration} iterations.\n")
            break
            
    final_users = df['user_id'].nunique()
    final_movies = df['item_id'].nunique()
    final_ratings = len(df)
    final_sparsity = (1 - (final_ratings / (final_users * final_movies))) * 100
    
    print(f"--- After Filtering (Users >= {min_user_ratings}, Movies >= {min_movie_ratings}) ---")
    print(f"Users: {final_users}, Movies: {final_movies}, Ratings: {final_ratings}")
    print(f"Sparsity: {final_sparsity:.2f}%")
    return df

if __name__ == "__main__":
    filtered_df = iterative_filtering()
