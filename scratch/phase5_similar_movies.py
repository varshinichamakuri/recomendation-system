import os
import sys
# Add parent directory to python path to import collaborative_filtering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from collaborative_filtering import ItemBasedCF

def test_similar_movies():
    print("Fitting ItemBasedCF model...")
    recommender = ItemBasedCF()
    recommender.fit()
    
    def get_similar_movies(movie_title, top_n=5):
        # Find the item_id for the given title
        matching_movies = recommender.movies_df[recommender.movies_df['movie_title'] == movie_title]
        if len(matching_movies) == 0:
            print(f"Movie '{movie_title}' not found in catalog.")
            return
            
        item_id = matching_movies['item_id'].values[0]
        
        if item_id not in recommender.item_similarity_df.index:
            print(f"Item ID {item_id} not in similarity matrix.")
            return
            
        # Get similarities and sort them
        similarities = recommender.item_similarity_df.loc[item_id]
        similarities = similarities.drop(item_id) # Drop self
        
        top_similar = similarities.nlargest(top_n)
        
        print(f"\nTop {top_n} movies similar to '{movie_title}':")
        for sim_item_id, score in top_similar.items():
            title = recommender.movies_df[recommender.movies_df['item_id'] == sim_item_id]['movie_title'].values[0]
            print(f"  - {title} (Similarity: {score:.4f})")

    # Test cases
    get_similar_movies("Star Wars (1977)")
    get_similar_movies("Silence of the Lambs, The (1991)")

if __name__ == "__main__":
    test_similar_movies()
