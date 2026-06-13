import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_eda(data_dir="data/ml-100k"):
    # 1. Load the ratings data (u.data)
    # The ratings file is tab-separated: user id | item id | rating | timestamp
    ratings_path = os.path.join(data_dir, "u.data")
    if not os.path.exists(ratings_path):
        print(f"Error: {ratings_path} not found. Please run download_dataset.py first.")
        return
        
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    # 2. Load the items/movies data (u.item)
    # The items file contains movie metadata, separated by '|'
    items_path = os.path.join(data_dir, "u.item")
    items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    # We load with latin-1 encoding because of special characters in titles
    items_df = pd.read_csv(items_path, sep='|', names=items_cols, usecols=['item_id', 'movie_title'], encoding='latin-1')
    
    # 3. Merge ratings with movie titles
    df = pd.merge(ratings_df, items_df, on='item_id')
    
    print("--- MovieLens-100k Dataset Summary ---")
    print(df.head())
    
    # 4. Compute Basic Stats
    n_users = df['user_id'].nunique()
    n_movies = df['item_id'].nunique()
    n_ratings = len(df)
    avg_rating = df['rating'].mean()
    
    # Compute sparsity
    total_elements = n_users * n_movies
    sparsity = (1 - (n_ratings / total_elements)) * 100
    
    print("\n--- Basic Statistics ---")
    print(f"Number of unique users: {n_users}")
    print(f"Number of unique movies: {n_movies}")
    print(f"Total number of ratings: {n_ratings}")
    print(f"Average rating value: {avg_rating:.2f}")
    print(f"Matrix Sparsity: {sparsity:.2f}%")
    
    # Create directory for saving plots
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # 5. Plot 1: Distribution of Ratings
    plt.figure(figsize=(8, 5))
    rating_counts = df['rating'].value_counts().sort_index()
    plt.bar(rating_counts.index, rating_counts.values, color='#4A90E2', edgecolor='black', alpha=0.8)
    plt.title('Distribution of Movie Ratings in MovieLens-100k', fontsize=14, fontweight='bold')
    plt.xlabel('Rating (1-5)', fontsize=12)
    plt.ylabel('Count of Ratings', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plots_dir, 'ratings_distribution.png'), dpi=300)
    plt.close()
    print("Saved plot: plots/ratings_distribution.png")
    
    # 6. Plot 2: The Long-Tail Effect (Ratings per Movie)
    plt.figure(figsize=(10, 5))
    movie_rating_counts = df['item_id'].value_counts()
    plt.plot(range(len(movie_rating_counts)), movie_rating_counts.values, color='#D0021B', linewidth=2)
    plt.fill_between(range(len(movie_rating_counts)), movie_rating_counts.values, color='#D0021B', alpha=0.2)
    plt.title('Movie Popularity Distribution (The Long-Tail Effect)', fontsize=14, fontweight='bold')
    plt.xlabel('Movie Index (Sorted by popularity)', fontsize=12)
    plt.ylabel('Number of Ratings Received', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Draw vertical line to mark popular items (head) vs long tail (e.g. top 20% of movies)
    head_limit = int(0.2 * n_movies)
    plt.axvline(x=head_limit, color='black', linestyle='--', label=f'Top 20% Movies ({head_limit} items)')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(plots_dir, 'long_tail_effect.png'), dpi=300)
    plt.close()
    print("Saved plot: plots/long_tail_effect.png")

    # 7. Plot 3: User Activity distribution
    plt.figure(figsize=(8, 5))
    user_rating_counts = df['user_id'].value_counts()
    plt.hist(user_rating_counts.values, bins=50, color='#50E3C2', edgecolor='black', alpha=0.8)
    plt.title('User Activity (Number of Ratings per User)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plots_dir, 'user_activity_distribution.png'), dpi=300)
    plt.close()
    print("Saved plot: plots/user_activity_distribution.png")
    
    # Print movie stats: most rated and least rated
    print("\n--- Top 5 Most Rated Movies ---")
    most_rated = df.groupby('movie_title').size().sort_values(ascending=False).head(5)
    for title, count in most_rated.items():
        print(f" - {title}: {count} ratings")
        
    print("\n--- Bottom 5 Least Rated Movies (Cold Start Vulnerable) ---")
    least_rated = df.groupby('movie_title').size().sort_values(ascending=True).head(5)
    for title, count in least_rated.items():
        print(f" - {title}: {count} ratings")

if __name__ == "__main__":
    run_eda()
