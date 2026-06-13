import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    def __init__(self, data_dir="data/ml-100k"):
        self.data_dir = data_dir
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_means = None
        
    def fit(self):
        # 1. Load ratings
        ratings_path = os.path.join(self.data_dir, "u.data")
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
        
        # Load movies for mapping titles
        items_path = os.path.join(self.data_dir, "u.item")
        items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
        self.movies_df = pd.read_csv(items_path, sep='|', names=items_cols, 
                                     usecols=['item_id', 'movie_title'], encoding='latin-1')
        
        # Create user-item rating matrix
        self.user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
        
        # Calculate mean rating for each user (for mean centering)
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Mean-center the ratings: Subtract the user's mean rating from each of their ratings.
        # This solves the user bias problem (some users rate high, some low).
        # We fill NaNs with 0 in the centered matrix because:
        # Cosine similarity on mean-centered ratings filled with 0 is mathematically equivalent to Pearson Correlation.
        centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0)
        
        # Compute user-to-user cosine similarity matrix
        print("Calculating User-to-User similarity matrix...")
        self.user_similarity = cosine_similarity(centered_matrix)
        
        # Convert similarity matrix to a DataFrame for easier lookup
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
        print(f"User similarity shape: {self.user_similarity_df.shape}")
        
    def predict_rating(self, user_id, item_id, k=30):
        """
        Predicts the rating of a user for an item using the ratings of the top-k most similar users.
        Formula (Mean-Centered User-Based CF):
        pred(u, i) = mean(u) + sum(sim(u, v) * (r_v_i - mean(v))) / sum(abs(sim(u, v)))
        """
        # If the item or user is not in the training matrix, fallback to the user's mean or global average
        if user_id not in self.user_item_matrix.index:
            return 3.5
        if item_id not in self.user_item_matrix.columns:
            return self.user_means[user_id]
            
        # Get users who have rated the item
        users_who_rated = self.user_item_matrix[item_id].dropna().index
        
        # Filter out the target user themselves
        users_who_rated = users_who_rated[users_who_rated != user_id]
        
        if len(users_who_rated) == 0:
            return self.user_means[user_id]
            
        # Get similarity between target user and all users who rated the item
        similarities = self.user_similarity_df.loc[user_id, users_who_rated]
        
        # Select top-k most similar users
        top_k_users = similarities.nlargest(k)
        
        # Calculate rating deviations for these top-k users
        # deviation_v = r_v_i - mean(v)
        ratings_v = self.user_item_matrix.loc[top_k_users.index, item_id]
        means_v = self.user_means.loc[top_k_users.index]
        deviations = ratings_v - means_v
        
        # Calculate weighted average of deviations
        sim_sum = np.abs(top_k_users).sum()
        if sim_sum == 0:
            return self.user_means[user_id]
            
        weighted_deviation = np.dot(top_k_users, deviations) / sim_sum
        
        # Add target user's mean to the weighted deviation
        pred_rating = self.user_means[user_id] + weighted_deviation
        
        # Clip the output rating to the valid [1.0, 5.0] scale
        return np.clip(pred_rating, 1.0, 5.0)

    def recommend(self, user_id, n=10, k=30):
        """
        Recommends top n unrated movies for the user.
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User ID {user_id} not found in training data.")
            
        # Find items the user has already rated
        rated_items = self.user_item_matrix.loc[user_id].dropna().index
        
        # Candidates for recommendation: items NOT rated by the user
        all_items = self.user_item_matrix.columns
        candidate_items = [item for item in all_items if item not in rated_items]
        
        predictions = []
        for item_id in candidate_items:
            pred = self.predict_rating(user_id, item_id, k=k)
            predictions.append((item_id, pred))
            
        # Sort predictions in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n]
        
        # Map item IDs to movie titles
        rec_list = []
        for item_id, score in top_predictions:
            title = self.movies_df[self.movies_df['item_id'] == item_id]['movie_title'].values[0]
            rec_list.append({'item_id': item_id, 'movie_title': title, 'predicted_rating': score})
            
        return pd.DataFrame(rec_list)

class ItemBasedCF:
    def __init__(self, data_dir="data/ml-100k"):
        self.data_dir = data_dir
        self.user_item_matrix = None
        self.item_similarity = None
        
    def fit(self):
        # 1. Load ratings
        ratings_path = os.path.join(self.data_dir, "u.data")
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
        
        # Load movies
        items_path = os.path.join(self.data_dir, "u.item")
        items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
        self.movies_df = pd.read_csv(items_path, sep='|', names=items_cols, 
                                     usecols=['item_id', 'movie_title'], encoding='latin-1')
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
        
        # Mean-center item columns (Adjusted Cosine Similarity: subtract user average from rating)
        # In item-based CF, we center by user average across rows to remove user subjectivity.
        user_means = self.user_item_matrix.mean(axis=1)
        centered_matrix = self.user_item_matrix.sub(user_means, axis=0).fillna(0)
        
        # Calculate item-to-item cosine similarity matrix
        # Transposecentered matrix to compute cosine similarity between items (columns)
        print("Calculating Item-to-Item similarity matrix...")
        self.item_similarity = cosine_similarity(centered_matrix.T)
        
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity, 
            index=self.user_item_matrix.columns, 
            columns=self.user_item_matrix.columns
        )
        print(f"Item similarity shape: {self.item_similarity_df.shape}")
        
    def predict_rating(self, user_id, item_id, k=30):
        """
        Predicts the rating of a user for an item using the user's ratings for the top-k most similar items.
        Formula:
        pred(u, i) = sum(sim(i, j) * r_u_j) / sum(abs(sim(i, j)))
        """
        if user_id not in self.user_item_matrix.index:
            return 3.5
        if item_id not in self.user_item_matrix.columns:
            return 3.5
            
        # Get items rated by the target user
        items_rated_by_user = self.user_item_matrix.loc[user_id].dropna().index
        
        # Filter out the target item itself
        items_rated_by_user = items_rated_by_user[items_rated_by_user != item_id]
        
        if len(items_rated_by_user) == 0:
            return 3.5
            
        # Get similarity between target item and items rated by the user
        similarities = self.item_similarity_df.loc[item_id, items_rated_by_user]
        
        # Select top-k most similar items
        top_k_items = similarities.nlargest(k)
        
        # Calculate weighted average of the user's ratings for these similar items
        ratings_u = self.user_item_matrix.loc[user_id, top_k_items.index]
        
        # Use only items with positive similarities to prevent inverse preferences from distorting scores
        positive_similarities = top_k_items[top_k_items > 0]
        
        if len(positive_similarities) == 0:
            return self.user_item_matrix.loc[user_id].mean() # Fallback to user mean
            
        ratings_u = ratings_u[positive_similarities.index]
        
        sim_sum = positive_similarities.sum()
        pred_rating = np.dot(positive_similarities, ratings_u) / sim_sum
        
        return np.clip(pred_rating, 1.0, 5.0)

    def recommend(self, user_id, n=10, k=30):
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User ID {user_id} not found in training data.")
            
        rated_items = self.user_item_matrix.loc[user_id].dropna().index
        all_items = self.user_item_matrix.columns
        candidate_items = [item for item in all_items if item not in rated_items]
        
        predictions = []
        for item_id in candidate_items:
            pred = self.predict_rating(user_id, item_id, k=k)
            predictions.append((item_id, pred))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n]
        
        rec_list = []
        for item_id, score in top_predictions:
            title = self.movies_df[self.movies_df['item_id'] == item_id]['movie_title'].values[0]
            rec_list.append({'item_id': item_id, 'movie_title': title, 'predicted_rating': score})
            
        return pd.DataFrame(rec_list)

if __name__ == "__main__":
    test_user = 196  # Let's test with User ID 196
    
    print("\n=== USER-BASED COLLABORATIVE FILTERING ===")
    ub_recommender = UserBasedCF()
    ub_recommender.fit()
    ub_recs = ub_recommender.recommend(user_id=test_user, n=5, k=20)
    print(f"\nTop 5 Recommendations for User {test_user} (User-Based CF):")
    print(ub_recs.to_string(index=False))
    
    print("\n=== ITEM-BASED COLLABORATIVE FILTERING ===")
    ib_recommender = ItemBasedCF()
    ib_recommender.fit()
    ib_recs = ib_recommender.recommend(user_id=test_user, n=5, k=20)
    print(f"\nTop 5 Recommendations for User {test_user} (Item-Based CF):")
    print(ib_recs.to_string(index=False))
