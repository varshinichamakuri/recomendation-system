import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collaborative_filtering import UserBasedCF, ItemBasedCF
from matrix_factorization import FunkSVD

def calculate_precision_recall_at_k(recommendations, test_ratings, k=5, threshold=4.0):
    """
    Computes Precision@K and Recall@K for a single user.
    recommendations: List of item_ids recommended in order of preference.
    test_ratings: DataFrame of actual ratings for the user in the test set.
    """
    # Identify items the user liked in the test set (ratings >= threshold)
    liked_test_items = set(test_ratings[test_ratings['rating'] >= threshold]['item_id'])
    
    if len(liked_test_items) == 0:
        return None, None # Inactive user in test set or no highly rated items
        
    top_k_recs = recommendations[:k]
    
    # Calculate intersection
    hits = len(set(top_k_recs).intersection(liked_test_items))
    
    precision = hits / k
    recall = hits / len(liked_test_items)
    
    return precision, recall

def run_evaluation(data_dir="data/ml-100k", k_val=5, threshold=4.0):
    ratings_path = os.path.join(data_dir, "u.data")
    backup_path = os.path.join(data_dir, "u.data.bak")
    
    # 1. Load and filter data to remove severe cold-start noise
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    user_counts = df['user_id'].value_counts()
    movie_counts = df['item_id'].value_counts()
    filtered_df = df[df['user_id'].isin(user_counts[user_counts >= 50].index)]
    filtered_df = filtered_df[filtered_df['item_id'].isin(movie_counts[movie_counts >= 20].index)]
    
    # Train-test split (80/20)
    train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    
    # 2. Swap u.data to point to the training set only
    print("Backing up u.data and preparing train/test split...")
    shutil.copyfile(ratings_path, backup_path)
    train_df.to_csv(ratings_path, sep='\t', index=False, header=False)
    
    try:
        # 3. Fit Models
        print("\n--- Training User-Based CF ---")
        user_cf = UserBasedCF(data_dir=data_dir)
        user_cf.fit()
        
        print("\n--- Training Item-Based CF ---")
        item_cf = ItemBasedCF(data_dir=data_dir)
        item_cf.fit()
        
        print("\n--- Training FunkSVD ---")
        svd = FunkSVD(n_factors=15, lr=0.005, reg=0.02, epochs=20)
        svd.fit(train_df)
        
        # 4. Evaluate RMSE on the test set
        print("\n=== Evaluating RMSE ===")
        user_cf_errors = []
        item_cf_errors = []
        svd_errors = []
        
        for _, row in test_df.iterrows():
            u, i, r = row['user_id'], row['item_id'], row['rating']
            
            # Predict
            p_ucf = user_cf.predict_rating(u, i)
            p_icf = item_cf.predict_rating(u, i)
            p_svd = svd.predict(u, i)
            
            user_cf_errors.append((r - p_ucf) ** 2)
            item_cf_errors.append((r - p_icf) ** 2)
            svd_errors.append((r - p_svd) ** 2)
            
        rmse_ucf = np.sqrt(np.mean(user_cf_errors))
        rmse_icf = np.sqrt(np.mean(item_cf_errors))
        rmse_svd = np.sqrt(np.mean(svd_errors))
        
        print(f"User-Based CF RMSE: {rmse_ucf:.4f}")
        print(f"Item-Based CF RMSE: {rmse_icf:.4f}")
        print(f"FunkSVD RMSE:      {rmse_svd:.4f}")
        
        # 5. Evaluate Precision@K and Recall@K
        # We sample 100 users from the test set to save computation time
        print(f"\n=== Evaluating Precision@{k_val} & Recall@{k_val} ===")
        test_users = test_df['user_id'].unique()
        sampled_users = np.random.choice(test_users, size=min(10, len(test_users)), replace=False)
        
        metrics = {'User-CF': [], 'Item-CF': [], 'FunkSVD': []}
        
        for u in sampled_users:
            u_test = test_df[test_df['user_id'] == u]
            
            # 1) User CF Recs
            recs_ucf = user_cf.recommend(user_id=u, n=k_val, k=20)['item_id'].tolist()
            p, r = calculate_precision_recall_at_k(recs_ucf, u_test, k=k_val, threshold=threshold)
            if p is not None: metrics['User-CF'].append((p, r))
            
            # 2) Item CF Recs
            recs_icf = item_cf.recommend(user_id=u, n=k_val, k=20)['item_id'].tolist()
            p, r = calculate_precision_recall_at_k(recs_icf, u_test, k=k_val, threshold=threshold)
            if p is not None: metrics['Item-CF'].append((p, r))
            
            # 3) FunkSVD Recs
            # FunkSVD recommend method returns a DataFrame with 'item_id' column
            recs_svd = svd.recommend(user_id=u, ratings_df=train_df, movies_df=user_cf.movies_df, n=k_val)['item_id'].tolist()
            p, r = calculate_precision_recall_at_k(recs_svd, u_test, k=k_val, threshold=threshold)
            if p is not None: metrics['FunkSVD'].append((p, r))
            
        print("\nEvaluation Results Table:")
        print(f"{'Model':<15} | {'RMSE':<8} | {'Precision@' + str(k_val):<12} | {'Recall@' + str(k_val):<10}")
        print("-" * 55)
        for model_name, res_list in metrics.items():
            avg_p = np.mean([x[0] for x in res_list])
            avg_r = np.mean([x[1] for x in res_list])
            rmse = rmse_ucf if model_name == 'User-CF' else (rmse_icf if model_name == 'Item-CF' else rmse_svd)
            print(f"{model_name:<15} | {rmse:<8.4f} | {avg_p:<12.4f} | {avg_r:<10.4f}")
            
    finally:
        # Restore original dataset
        print("\nRestoring original u.data...")
        if os.path.exists(backup_path):
            shutil.move(backup_path, ratings_path)

if __name__ == "__main__":
    np.random.seed(42)
    run_evaluation()
