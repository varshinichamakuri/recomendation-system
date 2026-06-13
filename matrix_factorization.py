import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class FunkSVD:
    """
    Matrix Factorization using Stochastic Gradient Descent (SGD) on observed ratings.
    This is also known as FunkSVD (Simon Funk's algorithm popularized in the Netflix Prize).
    """
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, epochs=20):
        self.n_factors = n_factors # Number of latent dimensions (K)
        self.lr = lr # Learning rate (gamma)
        self.reg = reg # Regularization coefficient (lambda)
        self.epochs = epochs
        
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        self.P = None # User latent matrix (M x K)
        self.Q = None # Item latent matrix (N x K)
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None
        
    def fit(self, ratings_df):
        self.global_mean = ratings_df['rating'].mean()
        
        # Map raw user/item IDs to continuous matrix indices [0, M-1] / [0, N-1]
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Initialize latent factor matrices with small random normal values
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Initialize biases
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Prepare training samples: list of (u_idx, i_idx, rating)
        samples = []
        for _, row in ratings_df.iterrows():
            u_idx = self.user_to_idx[row['user_id']]
            i_idx = self.item_to_idx[row['item_id']]
            samples.append((u_idx, i_idx, row['rating']))
            
        print(f"Starting FunkSVD SGD training. Users: {n_users}, Items: {n_items}, Latent Factors: {self.n_factors}")
        
        # Optimization Loop (Stochastic Gradient Descent)
        for epoch in range(1, self.epochs + 1):
            # Shuffle samples to ensure random order in SGD updates
            np.random.shuffle(samples)
            
            sq_errors = 0
            for u, i, r in samples:
                # Prediction: global_mean + user_bias + item_bias + dot(P_u, Q_i)
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                sq_errors += err ** 2
                
                # Update Biases
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])
                
                # Update Latent Factors P and Q (vectorized updates for speed)
                p_temp = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_temp - self.reg * self.Q[i])
                
            epoch_rmse = np.sqrt(sq_errors / len(samples))
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - RMSE: {epoch_rmse:.4f}")
                
    def predict(self, user_id, item_id):
        # Handle unseen users or items during testing
        u = self.user_to_idx.get(user_id)
        i = self.item_to_idx.get(item_id)
        
        if u is None and i is None:
            return self.global_mean
        elif u is None:
            return self.global_mean + self.item_biases[i]
        elif i is None:
            return self.global_mean + self.user_biases[u]
            
        pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.P[u], self.Q[i])
        return np.clip(pred, 1.0, 5.0)

    def recommend(self, user_id, ratings_df, movies_df, n=5):
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not seen during training.")
            
        u = self.user_to_idx[user_id]
        P_u = self.P[u]
        
        # Vectorized prediction for all items: dot product self.Q (N_items x K) with P_u (K,)
        preds = self.global_mean + self.user_biases[u] + self.item_biases + np.dot(self.Q, P_u)
        preds = np.clip(preds, 1.0, 5.0)
        
        # Get movies already rated by the user
        rated_item_ids = ratings_df[ratings_df['user_id'] == user_id]['item_id'].unique()
        
        # Create a boolean mask to filter out already rated items
        keep_mask = np.ones(len(self.idx_to_item), dtype=bool)
        for iid in rated_item_ids:
            idx = self.item_to_idx.get(iid)
            if idx is not None:
                keep_mask[idx] = False
                
        # Get candidate item indices and predictions
        indices = np.arange(len(self.idx_to_item))
        candidate_indices = indices[keep_mask]
        candidate_preds = preds[keep_mask]
        
        # Get top n indices (argsort sorts ascending, so reverse with [::-1])
        top_k_indices = np.argsort(candidate_preds)[::-1][:n]
        top_item_indices = candidate_indices[top_k_indices]
        top_scores = candidate_preds[top_k_indices]
        
        rec_list = []
        for idx, score in zip(top_item_indices, top_scores):
            iid = self.idx_to_item[idx]
            title = movies_df[movies_df['item_id'] == iid]['movie_title'].values[0]
            rec_list.append({'item_id': iid, 'movie_title': title, 'predicted_rating': score})
            
        return pd.DataFrame(rec_list)

    def save_model(self, file_path):
        """
        Saves the trained model state to a file using pickle.
        """
        import pickle
        model_state = {
            'n_factors': self.n_factors,
            'lr': self.lr,
            'reg': self.reg,
            'epochs': self.epochs,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'P': self.P,
            'Q': self.Q,
            'global_mean': self.global_mean,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases
        }
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Model saved successfully to {file_path}")

    def load_model(self, file_path):
        """
        Loads the saved model state from a file.
        """
        import pickle
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
        with open(file_path, 'rb') as f:
            model_state = pickle.load(f)
            
        self.n_factors = model_state['n_factors']
        self.lr = model_state['lr']
        self.reg = model_state['reg']
        self.epochs = model_state['epochs']
        self.user_to_idx = model_state['user_to_idx']
        self.idx_to_user = model_state['idx_to_user']
        self.item_to_idx = model_state['item_to_idx']
        self.idx_to_item = model_state['idx_to_item']
        self.P = model_state['P']
        self.Q = model_state['Q']
        self.global_mean = model_state['global_mean']
        self.user_biases = model_state['user_biases']
        self.item_biases = model_state['item_biases']
        print(f"Model loaded successfully from {file_path}")


def sklearn_truncated_svd(ratings_df, movies_df, user_id, n=5, n_factors=20):
    """
    Alternative SVD implementation using Scikit-Learn's TruncatedSVD.
    This requires imputing missing values (we fill with user rating averages).
    """
    print("\n--- Running Scikit-Learn TruncatedSVD (Alternative) ---")
    user_item = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
    
    # Imputation: Fill missing values with user average
    user_means = user_item.mean(axis=1)
    user_item_imputed = user_item.T.fillna(user_means).T
    
    # Fit SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    latent_user_features = svd.fit_transform(user_item_imputed) # Shape: M x K
    latent_item_features = svd.components_.T # Shape: N x K
    
    # Reconstructed Matrix R_hat
    R_hat = np.dot(latent_user_features, latent_item_features.T)
    R_hat_df = pd.DataFrame(R_hat, index=user_item.index, columns=user_item.columns)
    
    # Make recommendations
    rated_items = user_item.loc[user_id].dropna().index
    candidate_items = [iid for iid in user_item.columns if iid not in rated_items]
    
    predictions = []
    for iid in candidate_items:
        pred = R_hat_df.loc[user_id, iid]
        predictions.append((iid, pred))
        
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:n]
    
    rec_list = []
    for iid, score in top_predictions:
        title = movies_df[movies_df['item_id'] == iid]['movie_title'].values[0]
        rec_list.append({'item_id': iid, 'movie_title': title, 'predicted_rating': score})
        
    return pd.DataFrame(rec_list)

if __name__ == "__main__":
    # Load dataset
    data_dir = "data/ml-100k"
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    items_path = os.path.join(data_dir, "u.item")
    items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies_df = pd.read_csv(items_path, sep='|', names=items_cols, 
                            usecols=['item_id', 'movie_title'], encoding='latin-1')
                            
    test_user = 196
    
    # 1. Custom FunkSVD
    recommender = FunkSVD(n_factors=15, lr=0.005, reg=0.02, epochs=20)
    recommender.fit(ratings_df)
    funk_recs = recommender.recommend(user_id=test_user, ratings_df=ratings_df, movies_df=movies_df, n=5)
    
    print(f"\nTop 5 Recommendations for User {test_user} (Custom FunkSVD):")
    print(funk_recs.to_string(index=False))
    
    # 2. Sklearn SVD
    sk_recs = sklearn_truncated_svd(ratings_df, movies_df, user_id=test_user, n=5, n_factors=15)
    print(f"\nTop 5 Recommendations for User {test_user} (Sklearn TruncatedSVD):")
    print(sk_recs.to_string(index=False))
