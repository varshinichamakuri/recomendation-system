import argparse
import os
import pandas as pd

from download_dataset import download_and_extract_movielens
from eda import run_eda
from popularity_recommender import PopularityRecommender
from collaborative_filtering import UserBasedCF, ItemBasedCF
from matrix_factorization import FunkSVD

try:
    from scratch.phase7_eval import run_evaluation
except ImportError:
    run_evaluation = None


def print_recommendations(recs_df, title):
    print(f"\n--- {title} ---")
    print(recs_df.to_string(index=False))


def infer_default_user_id(data_dir):
    """Choose a valid user ID from the current dataset."""
    ratings_path = os.path.join(data_dir, "u.data")
    if not os.path.exists(ratings_path):
        return 1

    ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    if ratings_df.empty:
        return 1

    user_counts = ratings_df['user_id'].value_counts()
    if user_counts.empty:
        return 1

    return int(user_counts.idxmax())


def resolve_user_id(data_dir, requested_user_id=None):
    """Use the requested user id if it exists; otherwise fall back to a valid dataset user."""
    ratings_path = os.path.join(data_dir, "u.data")
    if not os.path.exists(ratings_path):
        return requested_user_id if requested_user_id is not None else 1

    ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    valid_user_ids = set(ratings_df['user_id'].astype(int).unique())

    if requested_user_id is not None and int(requested_user_id) in valid_user_ids:
        return int(requested_user_id)

    fallback_user_id = infer_default_user_id(data_dir)
    if requested_user_id is not None:
        print(f"User ID {requested_user_id} is not present in the current dataset; using user {fallback_user_id} instead.")
    return fallback_user_id


def main():
    parser = argparse.ArgumentParser(
        description="Run the MovieLens recommendation system components.")
    parser.add_argument("--download", action="store_true",
                        help="Download the MovieLens 100k dataset if it is missing.")
    parser.add_argument("--eda", action="store_true",
                        help="Run exploratory data analysis and save plots.")
    parser.add_argument("--popularity", action="store_true",
                        help="Run the popularity recommender and show the top movies.")
    parser.add_argument("--user-cf", action="store_true",
                        help="Run user-based collaborative filtering recommendations.")
    parser.add_argument("--item-cf", action="store_true",
                        help="Run item-based collaborative filtering recommendations.")
    parser.add_argument("--funk-svd", action="store_true",
                        help="Run matrix factorization recommendations using FunkSVD.")
    parser.add_argument("--eval", action="store_true",
                        help="Run the evaluation pipeline in scratch/phase7_eval.py.")
    parser.add_argument("--all", action="store_true",
                        help="Run all available components end to end.")
    parser.add_argument("--user-id", type=int, default=None,
                        help="User ID for personalized recommendations. If omitted, a valid dataset user is chosen automatically.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of recommendations to generate (default: 10).")
    parser.add_argument("--k", type=int, default=20,
                        help="Neighborhood size for collaborative filtering (default: 20).")
    parser.add_argument("--data-dir", default="data/ml-100k",
                        help="Path to the MovieLens 100k data directory.")

    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    root_dir = os.path.dirname(data_dir)
    resolved_user_id = resolve_user_id(data_dir, args.user_id)

    if args.download or not os.path.exists(os.path.join(data_dir, "u.data")):
        download_and_extract_movielens(dest_dir=root_dir)

    if args.eda or args.all:
        run_eda(data_dir=data_dir)

    if args.popularity or args.all:
        popularity = PopularityRecommender(data_dir=data_dir)
        popularity.fit(quantile_threshold=0.90)
        print_recommendations(popularity.recommend(args.top_n), "Top Popular Movies")

    if args.user_cf or args.all:
        user_cf = UserBasedCF(data_dir=data_dir)
        user_cf.fit()
        user_recs = user_cf.recommend(user_id=resolved_user_id, n=args.top_n, k=args.k)
        print_recommendations(user_recs, f"User-Based CF Recommendations for User {resolved_user_id}")

    if args.item_cf or args.all:
        item_cf = ItemBasedCF(data_dir=data_dir)
        item_cf.fit()
        item_recs = item_cf.recommend(user_id=resolved_user_id, n=args.top_n, k=args.k)
        print_recommendations(item_recs, f"Item-Based CF Recommendations for User {resolved_user_id}")

    if args.funk_svd or args.all:
        ratings_path = os.path.join(data_dir, "u.data")
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(ratings_path, sep='\t', names=ratings_cols)

        items_path = os.path.join(data_dir, "u.item")
        items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
        movies_df = pd.read_csv(items_path, sep='|', names=items_cols,
                                usecols=['item_id', 'movie_title'], encoding='latin-1')

        svd = FunkSVD(n_factors=15, lr=0.005, reg=0.02, epochs=20)
        svd.fit(ratings_df)
        svd_recs = svd.recommend(user_id=resolved_user_id, ratings_df=ratings_df, movies_df=movies_df, n=args.top_n)
        print_recommendations(svd_recs, f"FunkSVD Recommendations for User {resolved_user_id}")

    if args.eval or args.all:
        if run_evaluation is None:
            print("Evaluation module not found. Please ensure scratch/phase7_eval.py is available.")
        else:
            run_evaluation(data_dir=data_dir, k_val=args.top_n)

    if not any([args.download, args.eda, args.popularity, args.user_cf, args.item_cf, args.funk_svd, args.eval, args.all]):
        parser.print_help()


if __name__ == '__main__':
    import pandas as pd
    main()
