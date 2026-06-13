import os
import pandas as pd
from popularity_recommender import PopularityRecommender

DATA_DIR = os.path.abspath('data/ml-100k')
print('DATA_DIR', DATA_DIR)
for fname in ['u.data', 'u.item']:
    p = os.path.join(DATA_DIR, fname)
    print(fname, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else None)

rec = PopularityRecommender(data_dir=DATA_DIR)
rec.fit(quantile_threshold=0.90)
print('ratings_df rows', len(rec.ratings_df))
print('movies_df rows', len(rec.movies_df))
print('merged_df rows', len(rec.merged_df))
print(rec.popular_movies_df.head())
