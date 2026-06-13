import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_user_overlaps(user_id=196, data_dir="data/ml-100k", top_n=5):
    # Load ratings
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    # Pivot to user-item matrix
    user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
    
    # Mean center ratings
    user_means = user_item_matrix.mean(axis=1)
    centered_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)
    
    # Compute similarity matrix
    user_similarity = cosine_similarity(centered_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.index
    )
    
    # Find top N similar users to user_id
    similarities = user_similarity_df.loc[user_id]
    # Drop self
    similarities = similarities.drop(user_id)
    top_neighbors = similarities.nlargest(top_n)
    
    print(f"=== Overlap Analysis for User {user_id} ===")
    user_rated_items = set(user_item_matrix.loc[user_id].dropna().index)
    print(f"User {user_id} has rated {len(user_rated_items)} items in total.\n")
    
    for neighbor_id, sim_score in top_neighbors.items():
        neighbor_rated_items = set(user_item_matrix.loc[neighbor_id].dropna().index)
        common_items = user_rated_items.intersection(neighbor_rated_items)
        
        print(f"Neighbor User {neighbor_id}:")
        print(f"  - Similarity Score: {sim_score:.4f}")
        print(f"  - Total Items Rated by Neighbor: {len(neighbor_rated_items)}")
        print(f"  - Common Items Rated (Overlap): {len(common_items)}")
        print(f"  - Overlap Items: {sorted(list(common_items)) if len(common_items) <= 5 else list(common_items)[:5]}")
        print()

if __name__ == "__main__":
    analyze_user_overlaps()
