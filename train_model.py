#!/usr/bin/env python3
"""
train_model.py
--------------
This script trains a matrix factorization model on the MovieLens dataset.
It saves the learned user and item factor matrices (P and Q) and required mapping dictionaries
to files so that an inference script can load and use them.
"""

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# -----------------------------
# Data Loading & Mapping Creation
# -----------------------------
# Load MovieLens data (ensure these files are in the 'data/' folder)
movies_df = pd.read_csv('./data/movies.csv')         # Columns: movieId, title, genres
ratings_df = pd.read_csv('./data/ratings.csv')         # Columns: userId, movieId, rating, timestamp

# Create sorted lists of unique user and movie IDs
unique_user_ids = np.sort(ratings_df['userId'].unique())
unique_movie_ids = np.sort(ratings_df['movieId'].unique())

# Build mapping dictionaries
userId_to_index = {uid: idx for idx, uid in enumerate(unique_user_ids)}
movieId_to_index = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
index_to_movieId = {idx: mid for mid, idx in movieId_to_index.items()}

# Build the ratings list and a user_ratings dictionary
ratings = []          # Each entry: (user_index, movie_index, rating)
user_ratings = {}     # Dictionary mapping user_index to {movie_index: rating}
for _, row in ratings_df.iterrows():
    uid = row['userId']
    mid = row['movieId']
    r = float(row['rating'])
    if uid in userId_to_index and mid in movieId_to_index:
        u_idx = userId_to_index[uid]
        m_idx = movieId_to_index[mid]
        ratings.append((u_idx, m_idx, r))
        user_ratings.setdefault(u_idx, {})[m_idx] = r

ratings = np.array(ratings)
num_users = len(unique_user_ids)
num_movies = len(unique_movie_ids)

# -----------------------------
# Matrix Factorization Training
# -----------------------------
# Hyperparameters
num_factors = 20   # Number of latent factors
epochs = 20        # Number of training iterations
alpha = 0.005      # Learning rate
reg = 0.02         # Regularization strength

# Initialize latent factor matrices with small random values
P = np.random.normal(scale=0.1, size=(num_users, num_factors))
Q = np.random.normal(scale=0.1, size=(num_movies, num_factors))

print("Starting training for {} epochs...".format(epochs))
for epoch in range(1, epochs + 1):
    np.random.shuffle(ratings)  # Shuffle the training data each epoch
    # Wrap the ratings iterable with tqdm for a progress bar
    for (u_idx, m_idx, r) in tqdm(ratings, desc=f"Epoch {epoch}/{epochs}", unit="rating"):
        u = int(u_idx)
        m = int(m_idx)
        r = float(r)
        # Compute prediction and error
        pred = np.dot(P[u], Q[m])
        error = r - pred
        P_old = P[u].copy()
        # Update latent factors for user u and item m
        P[u] += alpha * (error * Q[m] - reg * P[u])
        Q[m] += alpha * (error * P_old - reg * Q[m])
    
    # Compute RMSE on the training data (optional; for monitoring training progress)
    errors = []
    for (u_idx, m_idx, r) in ratings:
        u = int(u_idx)
        m = int(m_idx)
        pred = np.dot(P[u], Q[m])
        errors.append((r - pred) ** 2)
    train_rmse = np.sqrt(np.mean(errors))
    print("Epoch {}: Training RMSE = {:.4f}".format(epoch, train_rmse))

# -----------------------------
# Persisting the Trained Model
# -----------------------------
# Save the latent factor matrices
np.save('./model/model_P.npy', P)
np.save('./model/model_Q.npy', Q)

# Save mapping dictionaries and the user_ratings dictionary for inference.
# We use pickle here for convenience.
model_data = {
    'userId_to_index': userId_to_index,
    'movieId_to_index': movieId_to_index,
    'index_to_movieId': index_to_movieId,
    'user_ratings': user_ratings
}

with open('./model/model_mappings.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Training complete! Model saved as 'model_P.npy', 'model_Q.npy', and 'model_mappings.pkl'.")
