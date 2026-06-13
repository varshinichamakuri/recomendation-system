import os
import pandas as pd
import numpy as np

class PopularityRecommender:
    def __init__(self, data_dir="data/ml-100k"):
        self.data_dir = data_dir
        self.ratings_df = None
        self.movies_df = None
        self.merged_df = None
        self.popular_movies_df = None
        self.global_mean_rating = 0.0
        
    def fit(self, quantile_threshold=0.90):
        """
        Fits the popularity model by calculating the IMDb Weighted Rating for all movies.
        """
        # Load ratings
        ratings_path = os.path.join(self.data_dir, "u.data")
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
        
        # Load movie metadata
        items_path = os.path.join(self.data_dir, "u.item")
        items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
        self.movies_df = pd.read_csv(items_path, sep='|', names=items_cols, 
                                     usecols=['item_id', 'movie_title'], encoding='latin-1')
        
        # Merge datasets
        self.merged_df = pd.merge(self.ratings_df, self.movies_df, on='item_id')
        
        # Calculate base statistics per movie: count (v) and mean rating (R)
        movie_stats = self.merged_df.groupby('movie_title').agg(
            vote_count=('rating', 'count'),
            vote_average=('rating', 'mean')
        ).reset_index()
        
        # Calculate C: Global mean rating across all movies
        self.global_mean_rating = self.merged_df['rating'].mean()
        C = self.global_mean_rating
        
        # Calculate m: Minimum votes required to be listed in the top charts.
        # We use the 90th percentile as the cutoff (meaning only top 10% of movies by vote count qualify)
        m = movie_stats['vote_count'].quantile(quantile_threshold)
        
        print(f"--- Popularity Model Parameters ---")
        print(f"Global Average Rating (C): {C:.4f}")
        print(f"Minimum Vote Cutoff (m, {quantile_threshold*100:.0f}th percentile): {m:.2f} ratings")
        
        # Apply the IMDb weighted rating formula:
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        def weighted_rating(row):
            v = row['vote_count']
            R = row['vote_average']
            return (v / (v + m)) * R + (m / (v + m)) * C
            
        movie_stats['weighted_score'] = movie_stats.apply(weighted_rating, axis=1)
        
        # Sort and store
        self.popular_movies_df = movie_stats.sort_values(by='weighted_score', ascending=False)
        
    def recommend(self, n=10):
        """
        Returns the top n popular recommendations.
        """
        if self.popular_movies_df is None:
            raise ValueError("Model has not been fitted yet. Run fit() first.")
        return self.popular_movies_df.head(n)[['movie_title', 'vote_count', 'vote_average', 'weighted_score']]

if __name__ == "__main__":
    recommender = PopularityRecommender()
    recommender.fit(quantile_threshold=0.90)
    
    print("\n--- Top 10 Popularity-Based Recommendations (IMDb Weighted Rating) ---")
    top_recs = recommender.recommend(10)
    print(top_recs.to_string(index=False))
