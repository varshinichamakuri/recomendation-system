# MovieLens Recommendation System

This repository contains a simple recommendation system built on the MovieLens 100k dataset.
It includes:

- `download_dataset.py`: downloads the MovieLens 100k ratings and item metadata files.
- `eda.py`: exploratory data analysis and plot generation.
- `popularity_recommender.py`: popularity-based recommendations using an IMDb-style weighted score.
- `collaborative_filtering.py`: user-based and item-based collaborative filtering using cosine similarity.
- `matrix_factorization.py`: custom FunkSVD matrix factorization and an alternative TruncatedSVD implementation.
- `scratch/phase7_eval.py`: evaluation pipeline for RMSE and Precision/Recall metrics.
- `main.py`: unified entrypoint for running dataset download, EDA, recommenders, and evaluation.

## Setup

1. Create a Python environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Download the dataset (if it is not already present):

```bash
python download_dataset.py
```

## Usage

Run the main script to execute one or more components:

```bash
python main.py --download --eda --popularity --user-cf --item-cf --funk-svd --eval
```

Example with a single recommender:

```bash
python main.py --popularity --top-n 10
python main.py --user-cf --user-id 1 --top-n 5
python main.py --funk-svd --user-id 1 --top-n 5
```

## Notes

- The dataset should be available under `data/ml-100k/u.data` and `data/ml-100k/u.item`.
- If you omit `--user-id`, the CLI automatically selects a valid user from the current dataset to avoid runtime errors.
- Generated plots are saved to the `plots/` directory.
- `scratch/phase7_eval.py` performs train/test evaluation and calculates RMSE, Precision@K, and Recall@K.
