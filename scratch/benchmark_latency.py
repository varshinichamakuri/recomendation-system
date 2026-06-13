import os
import sys
import time
import pandas as pd
import numpy as np

# Add parent directory to path to import matrix_factorization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matrix_factorization import FunkSVD

def run_benchmark(data_dir="data/ml-100k", model_path="models/funksvd_model.pkl", n_samples=100):
    print("=== Recommender Inference Latency Benchmark ===")
    
    # 1. Load data
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)
    
    items_path = os.path.join(data_dir, "u.item")
    items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies_df = pd.read_csv(items_path, sep='|', names=items_cols, 
                            usecols=['item_id', 'movie_title'], encoding='latin-1')
    
    # 2. Check if trained model exists, if not, train and save
    if not os.path.exists(model_path):
        print("Pre-trained model not found. Training a new model for benchmarking...")
        model = FunkSVD(n_factors=15, lr=0.005, reg=0.02, epochs=20)
        model.fit(ratings_df)
        model.save_model(model_path)
    else:
        print(f"Pre-trained model found at {model_path}.")
        
    # 3. Benchmark model loading time
    print("\nBenchmarking Model Serialization...")
    start_load = time.perf_counter()
    model = FunkSVD()
    model.load_model(model_path)
    end_load = time.perf_counter()
    load_time_ms = (end_load - start_load) * 1000
    print(f"  - Model Load Time: {load_time_ms:.2f} ms")
    
    # 4. Benchmark Inference Latency
    print(f"\nBenchmarking inference latency over {n_samples} random user queries...")
    trained_users = list(model.user_to_idx.keys())
    np.random.seed(42)
    sampled_users = np.random.choice(trained_users, size=n_samples, replace=False)
    
    latencies = []
    
    for uid in sampled_users:
        start_time = time.perf_counter()
        # Recommend 5 movies for the user
        recs = model.recommend(user_id=uid, ratings_df=ratings_df, movies_df=movies_df, n=5)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
    # Stats
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    print("\n--- Latency Performance Metrics ---")
    print(f"Average Latency:  {avg_latency:.4f} ms")
    print(f"Std Deviation:    {std_latency:.4f} ms")
    print(f"Min Latency:      {min_latency:.4f} ms")
    print(f"Max Latency:      {max_latency:.4f} ms")
    print(f"95th Percentile:  {p95_latency:.4f} ms")
    print(f"99th Percentile:  {p99_latency:.4f} ms")
    print("-----------------------------------")
    print("Amazon Production SLA Check: " + ("PASS (avg < 20ms)" if avg_latency < 20.0 else "FAIL (avg >= 20ms)"))

if __name__ == "__main__":
    run_benchmark()
