#!/usr/bin/env python3
"""
infer.py
--------
This script loads the pre-trained model parameters and mapping data,
computes recommendations for a given user, and prints the result in JSON format.

Usage:
    python infer.py <userId>
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# Load the Pre-trained Model & Mappings
# -----------------------------
P = np.load('./model/model_P.npy')
Q = np.load('./model/model_Q.npy')

with open('./model/model_mappings.pkl', 'rb') as f:
    model_data = pickle.load(f)

userId_to_index = model_data['userId_to_index']
movieId_to_index = model_data['movieId_to_index']
index_to_movieId = model_data['index_to_movieId']
user_ratings = model_data['user_ratings']

# Load movies data to retrieve movie titles
movies_df = pd.read_csv('./data/movies.csv')
# Build a simple mapping from movieId to title
movieId_to_title = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_for_user(user_index, topN=10):
    """
    For a given user index, compute predicted scores for all movies not yet rated
    using the pre-trained MF model and return the top-N recommendations.
    """
    num_movies = Q.shape[0]
    scores = {}
    for m_idx in range(num_movies):
        # Skip movies that the user has already rated
        if m_idx in user_ratings.get(user_index, {}):
            continue
        score = float(np.dot(P[user_index], Q[m_idx]))
        scores[m_idx] = score

    # Sort the movies based on predicted score (highest first)
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topN]
    recommendations = []
    for m_idx, score in top_items:
        orig_movie_id = int(index_to_movieId[m_idx])
        recommendations.append({
            "movieId": orig_movie_id,
            "title": movieId_to_title.get(orig_movie_id, "(Unknown Title)"),
            "score": round(score, 3)
        })
    return recommendations

# -----------------------------
# Main: Process Command-Line Argument and Output Recommendations as JSON
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <userId>")
        sys.exit(0)
    try:
        user_id = int(sys.argv[1])
    except ValueError:
        print(json.dumps([]))
        sys.exit(0)
    
    if user_id not in userId_to_index:
        # If the user ID is not found, return an empty list
        print(json.dumps([]))
    else:
        user_index = userId_to_index[user_id]
        recs = recommend_for_user(user_index, topN=10)
        print(json.dumps(recs))
