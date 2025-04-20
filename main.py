import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional, Set, Tuple
import os
from pydantic import BaseModel, Field 
from huggingface_hub import hf_hub_download

REPO_ID_P = "aditya-sr-iitj/netflix-challenge-model-model_P.npy"
MODEL_P_FILENAME = "model_P.npy"
REPO_ID_Q = "aditya-sr-iitj/netflix-challenge-model-model_Q.npy"
MODEL_Q_FILENAME = "model_Q.npy"
REPO_ID_MAPPINGS = "aditya-sr-iitj/netflix-challenge-model-model_mappings.pkl"
MAPPINGS_FILENAME = "model_mappings.pkl"

MOVIES_CSV_PATH = "./data/movies.csv"

# --- Global Variables for Loaded Data ---
P: Optional[np.ndarray] = None
Q: Optional[np.ndarray] = None
userId_to_index: Optional[Dict[int, int]] = None
movieId_to_index: Optional[Dict[int, int]] = None
index_to_movieId: Optional[Dict[int, int]] = None
user_ratings: Optional[Dict[int, Dict[int, float]]] = None
movieId_to_title: Optional[Dict[int, str]] = None
index_to_userId: Optional[Dict[int, int]] = None
movieId_to_tmdbId: Optional[Dict[int, Optional[int]]] = None
app_is_ready = False # Flag to indicate if loading was successful

# Loading Function
def load_model_and_data():
    """
    Loads all necessary models and mappings from Hugging Face Hub,
    and movie data from a local CSV file.
    """
    global P, Q, userId_to_index, movieId_to_index, index_to_movieId, \
           user_ratings, movieId_to_title, movieId_to_tmdbId, index_to_userId, app_is_ready

    try:
        print(f"INFO:     Downloading/Loading models and mappings from separate Hugging Face repos...")

        # Download/Load P, Q from their specific Hugging Face Repos
        print(f"INFO:     Accessing P matrix from repo: {REPO_ID_P}")
        local_p_path = hf_hub_download(repo_id=REPO_ID_P, filename=MODEL_P_FILENAME)
        print(f"INFO:     Accessing Q matrix from repo: {REPO_ID_Q}")
        local_q_path = hf_hub_download(repo_id=REPO_ID_Q, filename=MODEL_Q_FILENAME)
        P = np.load(local_p_path)
        Q = np.load(local_q_path)
        print(f"INFO:     Loaded P matrix from {local_p_path}")
        print(f"INFO:     Loaded Q matrix from {local_q_path}")

        # Download/Load Mappings from its specific Hugging Face Repo
        print(f"INFO:     Accessing mappings from repo: {REPO_ID_MAPPINGS}")
        local_mappings_path = hf_hub_download(repo_id=REPO_ID_MAPPINGS, filename=MAPPINGS_FILENAME)
        print(f"INFO:     Loading mappings and ratings data from {local_mappings_path}...")
        with open(local_mappings_path, 'rb') as f:
            model_data = pickle.load(f)
        print("INFO:     Mappings and ratings data loaded.")

        # Assign global variables
        userId_to_index = model_data['userId_to_index']
        movieId_to_index = model_data['movieId_to_index']
        index_to_movieId = model_data['index_to_movieId']
        user_ratings = model_data['user_ratings']
        index_to_userId = model_data['index_to_userId']

        # Load movie titles from local CSV
        if not os.path.exists(MOVIES_CSV_PATH):
            raise FileNotFoundError(f"Movies CSV file not found: {MOVIES_CSV_PATH}")
        print(f"INFO:     Loading movie titles from {MOVIES_CSV_PATH}...")
        print(f"INFO:     Loading movie titles and TMDB IDs from {MOVIES_CSV_PATH}...")

        movies_df = pd.read_csv(MOVIES_CSV_PATH)
        movieId_to_title = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}
        movieId_to_tmdbId = {row['movieId']: row['tmdbId'] for _, row in movies_df.iterrows()}
        print("INFO:     Movie titles and TMDB IDs loaded.")

        # Final check
        required_vars = {
            "P matrix": P is not None,
            "Q matrix": Q is not None,
            "userId_to_index": bool(userId_to_index),
            "movieId_to_index": bool(movieId_to_index),
            "index_to_movieId": bool(index_to_movieId),
            "user_ratings": user_ratings is not None,
            "movieId_to_title": bool(movieId_to_title),
            "index_to_userId": bool(index_to_userId),
            "movieId_to_tmdbId": bool(movieId_to_tmdbId)
        }
        missing = [name for name, present in required_vars.items() if not present]
        if not missing:
            app_is_ready = True
            print("INFO:     Model, mappings, ratings, and title/TMDB ID data loaded successfully.")
        else:
             raise ValueError(f"One or more essential data components missing after loading: {', '.join(missing)}")

    except FileNotFoundError as e: # Handles missing CSV or potentially errors during download if file doesn't exist on Hub (though hf_hub_download usually raises specific errors)
        print(f"ERROR:    {e}")
        app_is_ready = False
    except KeyError as e:
        print(f"ERROR:    Missing key {e} in mappings file downloaded from {REPO_ID_MAPPINGS}/{MAPPINGS_FILENAME}.")
        app_is_ready = False
    except Exception as e: # Catch potential download errors from huggingface_hub or other issues
        print(f"ERROR:    Failed to load model or data during startup: {e}")
        app_is_ready = False


# Recommendation Function (remains the same)
def recommend_for_user(user_index: int, topN: int = 10) -> List[Dict[str, Any]]:
    """
    For a given user index, compute predicted scores for all movies not yet rated
    using the pre-trained MF model and return the top-N recommendations.
    (Excludes already rated movies)
    """
    if not app_is_ready or P is None or Q is None or user_ratings is None or index_to_movieId is None or movieId_to_title is None or movieId_to_tmdbId is None:
         print("ERROR:    Recommendation function called but data not ready.")
         return []

    num_movies = Q.shape[0]
    scores = {}
    # Ensure user_index exists in user_ratings before accessing potentially large dict
    rated_movie_indices = user_ratings.get(user_index, {})

    for m_idx in range(num_movies):
        if m_idx in rated_movie_indices:
            continue
        # Ensure P and Q dimensions match user_index and m_idx before dot product
        if user_index < P.shape[0] and m_idx < Q.shape[0]:
             score = float(np.dot(P[user_index], Q[m_idx]))
             scores[m_idx] = score
        else:
             print(f"WARN:     Index out of bounds skipped: user_index={user_index} (P shape {P.shape}), m_idx={m_idx} (Q shape {Q.shape})")


    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topN]
    recommendations = []
    for m_idx, score in top_items:
        # Check if m_idx is valid before accessing index_to_movieId
        if m_idx in index_to_movieId:
            orig_movie_id = int(index_to_movieId[m_idx])
            recommendations.append({
                "movieId": orig_movie_id,
                "title": movieId_to_title.get(orig_movie_id, "(Unknown Title)"),
                "score": round(score, 3),
                "tmdbId":movieId_to_tmdbId.get(orig_movie_id, None)
            })
        else:
             print(f"WARN:     m_idx {m_idx} not found in index_to_movieId mapping.")

    return recommendations


# User Matching Logic Function (remains the same)
def _find_best_matching_user_index(target_movie_indices: Set[int]) -> Tuple[Optional[int], int, float]:
    """
    Internal helper to find the user_index that best matches a set of movie_indices.

    Args:
        target_movie_indices: A set of movie indices to match against.

    Returns:
        A tuple containing:
        - The best matching user_index (int) or None if no match.
        - The maximum number of matches found (int).
        - The average rating of the matched movies for the best user (float).
    """
    if not user_ratings: # Check if data is loaded
        return None, -1, -1.0

    best_match_user_index: Optional[int] = None
    max_match_count = -1
    best_avg_rating = -1.0

    for user_index, rated_movies in user_ratings.items(): # rated_movies is {movie_idx: rating}
        user_rated_indices = set(rated_movies.keys())
        matched_indices = user_rated_indices.intersection(target_movie_indices)
        match_count = len(matched_indices)

        if match_count > 0:
            sum_ratings = sum(rated_movies[m_idx] for m_idx in matched_indices)
            avg_rating = sum_ratings / match_count

            if match_count > max_match_count:
                max_match_count = match_count
                best_avg_rating = avg_rating
                best_match_user_index = user_index
            elif match_count == max_match_count:
                if avg_rating > best_avg_rating:
                    best_avg_rating = avg_rating
                    best_match_user_index = user_index

    # If max_match_count is still -1, it means no user matched any movie
    if max_match_count == -1:
         max_match_count = 0 # Set count to 0 if no matches

    return best_match_user_index, max_match_count, best_avg_rating


# Pydantic Models (remain the same)
class RecommendationItem(BaseModel):
    movieId: int
    title: str
    score: float
    tmdbId: int

class FindUserRequest(BaseModel):
    movie_ids: List[int] = Field(..., min_items=1, max_items=20, description="List of 1 to 20 movie IDs")

class FindUserResponse(BaseModel):
    matched_user_id: Optional[int] = Field(description="The ID of the user who best matched the input movies, or null if no suitable match found.")
    match_count: Optional[int] = Field(description="Number of movies from the input list rated by the matched user.")
    average_rating: Optional[float] = Field(description="Average rating given by the matched user to the matched movies.")


# FastAPI App Initialization (remains the same)
app = FastAPI(
    title="Movie Recommender API",
    description="Provides movie recommendations based on user ID and finds users/recommendations matching movie lists.",
    version="1.3.0" # Incremented version for HF integration
)


# Event Handler for Startup (remains the same)
@app.on_event("startup")
async def startup_event():
    """Load data when the application starts."""
    load_model_and_data()


# API Endpoints (remain the same)
@app.get("/", tags=["Health Check"])
async def read_root():
    """Basic health check endpoint."""
    if app_is_ready:
        return {"status": "OK", "message": "Recommender API is running and models are loaded."}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: Models or data failed to load. Check server logs.")

@app.get(
    "/recommendations/{user_id}",
    response_model=List[RecommendationItem],
    tags=["Recommendations"],
    summary="Get movie recommendations for a specific user",
)
async def get_recommendations(user_id: int):
    """
    Takes a `user_id` and returns a list of top 10 movie recommendations.
    - **Excludes movies the user has already rated.**
    - Returns an empty list `[]` if the user ID is not found.
    """
    if not app_is_ready: raise HTTPException(status_code=503, detail="Service Unavailable: Models or data not loaded.")
    if not userId_to_index: raise HTTPException(status_code=500, detail="Internal Server Error: User mapping not loaded.")

    user_index = userId_to_index.get(user_id)
    if user_index is None:
        # User not found, return empty list
        return []

    recs = recommend_for_user(user_index, topN=10)
    return recs

@app.post(
    "/find_matching_user",
    response_model=FindUserResponse,
    tags=["User Matching"],
    summary="Find user who rated the most movies from a given list",
)
async def find_matching_user(request: FindUserRequest):
    """
    Accepts a list of 1 to 5 movie IDs. Finds the user who has rated the
    highest number of movies from this list (using avg rating as tie-breaker).
    Returns the `userId` of the best matching user, or `null`.
    """
    if not app_is_ready or not movieId_to_index or not index_to_userId:
        raise HTTPException(status_code=503, detail="Service Unavailable: Data required for matching is not loaded.")

    input_movie_ids = request.movie_ids
    target_movie_indices: Set[int] = set()
    invalid_ids = []

    for movie_id in input_movie_ids:
        movie_index = movieId_to_index.get(movie_id)
        if movie_index is not None:
            target_movie_indices.add(movie_index)
        else:
            invalid_ids.append(movie_id)
            print(f"Warning: Input movieId {movie_id} not found in mapping, skipping.")

    if not target_movie_indices:
         # If *all* input IDs were invalid after checking
         raise HTTPException(status_code=400, detail=f"None of the provided movie IDs were found in the dataset. Invalid IDs: {invalid_ids}")

    # Call the helper function
    best_user_index, match_count, avg_rating = _find_best_matching_user_index(target_movie_indices)

    # Format the response
    if best_user_index is not None:
        matched_user_id = index_to_userId.get(best_user_index)
        if matched_user_id is None:
             print(f"ERROR: Could not find original userId for user_index {best_user_index}")
             # Return null if mapping fails unexpectedly
             return FindUserResponse(matched_user_id=None, match_count=None, average_rating=None)

        return FindUserResponse(
            matched_user_id=matched_user_id,
            match_count=match_count,
            average_rating=round(avg_rating, 3) if avg_rating >= 0 else None
            )
    else:
        # No user rated any of the valid input movies
        return FindUserResponse(matched_user_id=None, match_count=0, average_rating=None)


# Endpoint combining find_matching_user and recommend_for_user (remains the same)
@app.post(
    "/recommendations_from_selection",
    response_model=List[RecommendationItem],
    tags=["Recommendations"],
    summary="Get recommendations based on a list of liked movies",
    responses={
        200: {"description": "Successful recommendations response (or empty list if no matching user found)"},
        400: {"description": "Invalid input (e.g., all movie IDs invalid)"},
        503: {"description": "Service unavailable (models not loaded)"}
    }
)
async def get_recommendations_from_selection(request: FindUserRequest):
    """
    Accepts a list of 1 to 5 movie IDs.
    1. Finds the user profile that best matches this list (most rated movies, highest avg rating tie-breaker).
    2. Generates movie recommendations based on that matched user's profile.
    3. Returns a list of recommended movies (excluding movies the matched user has already rated).
    Returns an empty list `[]` if no suitable matching user profile could be found based on the input movies.
    """
    if not app_is_ready or not movieId_to_index or not index_to_userId:
        raise HTTPException(status_code=503, detail="Service Unavailable: Data required for matching is not loaded.")

    input_movie_ids = request.movie_ids
    target_movie_indices: Set[int] = set()
    invalid_ids = []

    # Step 1: Convert input IDs to indices
    for movie_id in input_movie_ids:
        movie_index = movieId_to_index.get(movie_id)
        if movie_index is not None:
            target_movie_indices.add(movie_index)
        else:
            invalid_ids.append(movie_id)
            print(f"Warning: Input movieId {movie_id} not found in mapping, skipping.")

    if not target_movie_indices:
         raise HTTPException(status_code=400, detail=f"None of the provided movie IDs were found in the dataset. Invalid IDs: {invalid_ids}")

    # Step 2: Find the best matching user using the helper
    best_user_index, _, _ = _find_best_matching_user_index(target_movie_indices) # We only need the index here

    # Step 3: Generate recommendations if a user was found
    if best_user_index is not None:
        print(f"INFO: Found matching user_index {best_user_index}. Generating recommendations.")
        recommendations = recommend_for_user(user_index=best_user_index, topN=10)
        return recommendations
    else:
        # No suitable matching user found for the input movies
        print("INFO: No matching user found for the provided movie list. Returning empty recommendations.")
        return [] 


# Optional: Run with Uvicorn (remains the same)
if __name__ == "__main__":
    import uvicorn
    print("--- Starting Uvicorn Development Server ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)