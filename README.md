# Movie Recommendation System API using Matrix Factorization

## Overview

This project implements a movie recommendation system using Matrix Factorization (MF), a popular collaborative filtering technique. It exposes a FastAPI application that serves recommendations based on user preferences. The system is trained on the MovieLens dataset.

A key feature of this API allows users (even new ones without a rating history) to get personalized movie recommendations by simply providing a list of 1 to 20 movies they like. The system identifies an existing user profile in the dataset that best matches the provided movie list and generates recommendations based on that profile's learned preferences.

The pre-trained model components (User latent factors P, Item latent factors Q, and data mappings) are hosted on Hugging Face Hub and are automatically downloaded when the API server starts.

## Features

* **Matrix Factorization Model:** Uses MF to learn latent user and item representations from the MovieLens dataset.
* **FastAPI Backend:** Provides a robust and fast API service.
* **Hugging Face Hub Integration:** Automatically downloads and loads pre-trained model components.
* **User-Specific Recommendations:** Endpoint to get top N recommendations for a known `user_id`.
* **Recommendations from Selection:** Endpoint to get top N recommendations based on a list of 1-20 favorite `movie_ids` provided by the user (finds a proxy user profile).
* **User Profile Matching:** Endpoint to find the existing user ID whose ratings best match a given list of movies.
* **Automatic API Documentation:** Interactive API documentation provided by FastAPI (Swagger UI/ReDoc).
* **Data Preprocessing & Training Scripts:** Includes scripts to preprocess the raw data and retrain the MF model from scratch.

## Technology Stack

* **Python 3.x**
* **FastAPI:** For building the web API.
* **Uvicorn:** ASGI server to run FastAPI.
* **NumPy:** For numerical operations and matrix handling.
* **Pandas:** For data loading and manipulation during training/preprocessing.
* **Hugging Face Hub (`huggingface_hub`):** For downloading pre-trained model files.
* **Pickle:** For saving/loading Python objects (mappings, user ratings).
* **python-dotenv:** For managing environment variables (optional, if needed for configuration).

## Project Structure
```bash
├── .env              # Optional: Environment variables (not strictly needed for model paths anymore)
├── .gitignore        # Git ignore file
├── data/
│   ├── movies.csv    # Movie metadata (Required for running the API)
│   └── ratings.csv   # User ratings (Required for training/preprocessing only)
├── main.py           # FastAPI application logic, API endpoints, model loading
├── preprocess_ratings.py # Script to preprocess ratings and create user_ratings mapping
├── train_model.py    # Script to train the Matrix Factorization model
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Model Details

* **Algorithm:** Matrix Factorization trained using Stochastic Gradient Descent (SGD).
* **Dataset:** MovieLens dataset (specifically `movies.csv` and `ratings.csv`).
* **Pre-trained Components:**
    * The User latent factor matrix (P), Item latent factor matrix (Q), and essential data mappings (`userId_to_index`, `movieId_to_index`, `index_to_movieId`, `index_to_userId`, `user_ratings`) are required to run the API.
    * These components are automatically downloaded by `main.py` at startup from the following Hugging Face Hub repositories:
        * **P Matrix:** `aditya-sr-iitj/netflix-challenge-model-model_P.npy`
        * **Q Matrix:** `aditya-sr-iitj/netflix-challenge-model-model_Q.npy`
        * **Mappings:** `aditya-sr-iitj/netflix-challenge-model-model_mappings.pkl`

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.7+
    * `pip` package manager

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Data Files:**
    * Ensure the `movies.csv` file is present in the `data/` directory. This file is **required** by `main.py` to map `movieId`s to titles. You can typically download this from the MovieLens dataset sources.
    * The `ratings.csv` file is **only required** if you intend to run `preprocess_ratings.py` or `train_model.py`. It is *not* needed just to run the API with the pre-trained models from Hugging Face Hub.

## Running the Application

1.  **Start the API Server:**
    Use Uvicorn to run the FastAPI application defined in `main.py`:
    ```bash
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
    ```
    * `--reload`: Enables auto-reloading during development (server restarts on code changes). Remove for production.
    * `--host`: Specifies the host address.
    * `--port`: Specifies the port number.

2.  **Access the API:**
    * The API will be available at `http://127.0.0.1:8000`.
    * On the first run (or if the model files aren't cached), you will see logs indicating the download progress from Hugging Face Hub.
    * Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.
    * Alternative API documentation (ReDoc) is available at `http://127.0.0.1:8000/redoc`.

## API Endpoints

The following endpoints are available:

### `GET /`

* **Description:** Health check endpoint. Returns the status of the API.
* **Response (Success - 200):**
    ```json
    {
      "status": "OK",
      "message": "Recommender API is running and models are loaded."
    }
    ```
* **Response (Error - 503):** If models failed to load.

### `GET /recommendations/{user_id}`

* **Description:** Get the top 10 (can be changed to n in topN in recommend_for_user function) movie recommendations for a specific `user_id` present in the dataset. Excludes movies the user has already rated.
* **Path Parameter:**
    * `user_id` (int): The original ID of the user.
* **Response (Success - 200):**
    ```json
    [
      {
        "movieId": 123,
        "title": "Some Movie Title (Year)",
        "score": 4.875
      },
      // ... 9 more recommendations
    ]
    ```
    * Returns an empty list `[]` if the `user_id` is not found.
* **Response (Error - 503):** If models are not loaded.

### `POST /find_matching_user`

* **Description:** Finds the user profile in the dataset that best matches a provided list of movie IDs. The match is based on the highest number of movies rated from the input list, with the average rating used as a tie-breaker.
* **Request Body:**
    ```json
    {
      "movie_ids": [1, 296, 593, 1196, 2571] // List of 1 to 20 movie IDs
    }
    ```
* **Response (Success - 200):**
    ```json
    {
      "matched_user_id": 42, // The ID of the best matching user, or null
      "match_count": 4,      // Number of movies matched, or null/0
      "average_rating": 4.75 // Average rating by the matched user for matched movies, or null
    }
    ```
* **Response (Error - 400):** If none of the provided `movie_ids` are found in the dataset.
* **Response (Error - 503):** If models/data are not loaded.

### `POST /recommendations_from_selection`

* **Description:** The primary endpoint for generating recommendations based on user input. It takes a list of 1-20 movie IDs, finds the best matching user profile (using the same logic as `/find_matching_user`), and returns the top 10 (can be changed to n in topN in recommend_for_user function) movie recommendations for that *matched* user profile. Excludes movies the matched user has already rated.
* **Request Body:**
    ```json
    {
      "movie_ids": [1, 296, 593, 1196, 2571] // List of 1 to 20 movie IDs user likes
    }
    ```
* **Response (Success - 200):**
    ```json
    [
      {
        "movieId": 456,
        "title": "Another Movie (Year)",
        "score": 4.912
      },
      // ... 9 more recommendations based on the matched user profile
    ]
    ```
    * Returns an empty list `[]` if no suitable matching user profile is found for the input movies.
* **Response (Error - 400):** If none of the provided `movie_ids` are found in the dataset.
* **Response (Error - 503):** If models/data are not loaded.

## Retraining the Model (Optional)

If you want to retrain the Matrix Factorization model using the provided `ratings.csv`:

1.  **Ensure Data:** Make sure `data/movies.csv` and `data/ratings.csv` are present.
2.  **Run the Training Script:**
    ```bash
    python train_model.py
    ```
3.  **Output:** The script will train the model and save the following files locally in the `model/` directory (it will create the directory if it doesn't exist):
    * `model/model_P.npy`: User latent factor matrix.
    * `model/model_Q.npy`: Item latent factor matrix.
    * `model/model_mappings.pkl`: Mappings (`userId_to_index`, `movieId_to_index`, `index_to_movieId`) and the basic `user_ratings` structure based on the training data.
    * **Note:** These locally trained files are *not* automatically used by `main.py`, which defaults to downloading from Hugging Face Hub. You would need to modify `main.py` to load these local files if desired.

## Data Preprocessing (Optional)

The `preprocess_ratings.py` script serves a specific purpose: it takes an existing `model_mappings.pkl` (like the one generated by `train_model.py` or potentially an older version) and the full `ratings.csv` file to generate a more comprehensive `user_ratings` dictionary (mapping `user_index` to `{movie_index: rating}`) and an `index_to_userId` mapping. It saves these back into the `model_mappings.pkl` file.

This is useful if you need to update or generate the `user_ratings` structure needed by the `recommend_for_user` function and the matching logic in `main.py` without retraining the P and Q matrices.

1.  **Ensure Data:** Requires `data/ratings.csv` and an *existing* `model/model_mappings.pkl` file (containing at least `userId_to_index`, `movieId_to_index`, `index_to_movieId`).
2.  **Run the Script:**
    ```bash
    python preprocess_ratings.py
    ```
3.  **Output:** Updates the `model/model_mappings.pkl` file with the `user_ratings` and `index_to_userId` keys populated based on `ratings.csv`.

## Contirubution
Aditya Signh Rathore
