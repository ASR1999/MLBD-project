# preprocess_ratings.py
import pandas as pd
import pickle
import os
from collections import defaultdict
import dotenv

dotenv.load_dotenv('/home/uwcuser/datasets/MedQA/__MACOSX/data_clean/textbooks/en/test/MLBD-netflix-challenge/.env')

print("--- Starting Pre-processing Script ---")

RATINGS_CSV_PATH = os.getenv('RATINGS_CSV_PATH')
MAPPINGS_PATH = os.getenv('MAPPINGS_PATH')

# --- Load Existing Mappings ---
print(f"Loading existing mappings from: {MAPPINGS_PATH}")
try:
    with open(MAPPINGS_PATH, 'rb') as f:
        # Load only the mappings needed, assume P/Q are handled separately
        existing_data = pickle.load(f)
        userId_to_index = existing_data['userId_to_index']
        movieId_to_index = existing_data['movieId_to_index']
        index_to_movieId = existing_data['index_to_movieId']
        # Keep other data if it was already there (like user_ratings if partially populated?)
        # Or start fresh for the required ones. Let's assume we rebuild user_ratings
        print("Successfully loaded existing userId_to_index, movieId_to_index, index_to_movieId.")
except FileNotFoundError:
    print(f"ERROR: Mappings file not found at {MAPPINGS_PATH}. Cannot proceed.")
    exit(1)
except KeyError as e:
    print(f"ERROR: Missing key {e} in mappings file. Cannot proceed.")
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to load mappings: {e}")
    exit(1)


# --- Create index_to_userId mapping ---
print("Creating index_to_userId mapping...")
index_to_userId = {v: k for k, v in userId_to_index.items()}
print(f"Created index_to_userId mapping for {len(index_to_userId)} users.")

# --- Process ratings.csv ---
print(f"Loading and processing ratings data from: {RATINGS_CSV_PATH}")
user_ratings_by_index = defaultdict(dict) # Use defaultdict for convenience: {user_idx: {movie_idx: rating}}
processed_count = 0
skipped_count = 0

try:
    # Consider using chunking if 890MB still causes memory issues even for this script
    # chunksize = 1_000_000
    # for chunk in pd.read_csv(RATINGS_CSV_PATH, chunksize=chunksize):
    #     for _, row in chunk.iterrows(): ...
    
    ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    print(f"Loaded {len(ratings_df)} ratings.")

    for _, row in ratings_df.iterrows():
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = float(row['rating'])

        # Get internal indices
        user_index = userId_to_index.get(user_id)
        movie_index = movieId_to_index.get(movie_id)

        if user_index is not None and movie_index is not None:
            user_ratings_by_index[user_index][movie_index] = rating
            processed_count += 1
        else:
            skipped_count += 1 # Rating for a user/movie not in our model's mappings

        if processed_count % 5_000_000 == 0 and processed_count > 0: # Print progress periodically
             print(f"Processed {processed_count} ratings...")


    print(f"Finished processing ratings. Processed: {processed_count}, Skipped (user/movie not in map): {skipped_count}")
    print(f"Built user_ratings_by_index structure for {len(user_ratings_by_index)} users.")

except FileNotFoundError:
    print(f"ERROR: ratings.csv not found at {RATINGS_CSV_PATH}. Cannot proceed.")
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to process ratings.csv: {e}")
    exit(1)

# --- Prepare final data structure for saving ---
# Include original mappings and the new structures
final_model_data = {
    'userId_to_index': userId_to_index,
    'movieId_to_index': movieId_to_index,
    'index_to_movieId': index_to_movieId,
    'index_to_userId': index_to_userId,            # Added
    'user_ratings': dict(user_ratings_by_index) # Convert defaultdict back to dict for saving
}

# --- Save Updated Mappings ---
print(f"Saving updated mappings (including user_ratings) back to: {MAPPINGS_PATH}")
try:
    with open(MAPPINGS_PATH, 'wb') as f:
        pickle.dump(final_model_data, f)
    print("Successfully saved updated mappings.")
except Exception as e:
    print(f"ERROR: Failed to save updated mappings: {e}")
    exit(1)

print("--- Pre-processing Script Finished ---")