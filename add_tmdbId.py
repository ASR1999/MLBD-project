import pandas as pd
import os

MOVIES_CSV_PATH = "./data/movies.csv"
LINKS_CSV_PATH = "/home/uwcuser/datasets/MedQA/__MACOSX/data_clean/textbooks/en/test/ml-latest/links.csv"

print(f"Attempting to update {MOVIES_CSV_PATH} by adding 'tmdbId' from {LINKS_CSV_PATH}")

if not os.path.exists(MOVIES_CSV_PATH):
    print(f"Error: Movies file not found at '{MOVIES_CSV_PATH}'")
elif not os.path.exists(LINKS_CSV_PATH):
    print(f"Error: Links file not found at '{LINKS_CSV_PATH}'")
else:
    try:
        # Read the CSV files
        print(f"Reading {MOVIES_CSV_PATH}...")
        movies_df = pd.read_csv(MOVIES_CSV_PATH)
        print(f"Reading {LINKS_CSV_PATH}...")
        links_df = pd.read_csv(LINKS_CSV_PATH)

        # 1. Select only necessary columns from links_df
        links_subset_df = links_df[['movieId', 'tmdbId']].copy()

        # 2. Check for duplicate movieIds in links - keep the first one if found
        if links_subset_df['movieId'].duplicated().any():
            print("Warning: Duplicate movieIds found in links.csv. Keeping the first occurrence.")
            links_subset_df = links_subset_df.drop_duplicates(subset='movieId', keep='first')

        # 3. Handle potential non-numeric tmdbId values before merge (optional but safer)
        # Convert tmdbId to numeric, setting errors='coerce' will turn non-numeric into NaN
        links_subset_df['tmdbId'] = pd.to_numeric(links_subset_df['tmdbId'], errors='coerce')
        # Convert to nullable Integer type (allows NaN)
        links_subset_df['tmdbId'] = links_subset_df['tmdbId'].astype('Int64')

        # --- Merging ---
        print("Merging dataframes on 'movieId'...")
        # Use 'left' merge to keep all movies from movies.csv
        # If a movie doesn't have a corresponding entry in links_subset_df, its tmdbId will be <NA>
        merged_df = pd.merge(movies_df, links_subset_df, on='movieId', how='left')

        # --- Final Output Preparation ---
        # Ensure the desired columns are present and in order
        # If 'tmdbId' was already somehow in movies_df, the merge adds '_y', so we handle it
        if 'tmdbId_y' in merged_df.columns:
             merged_df = merged_df.rename(columns={'tmdbId_y': 'tmdbId'})
             # Decide what to do with potential 'tmdbId_x' if it existed
             if 'tmdbId_x' in merged_df.columns:
                 print("Warning: 'tmdbId' column already existed in movies.csv. Overwriting with value from links.csv.")
                 merged_df = merged_df.drop(columns=['tmdbId_x'])

        # Select and order the final columns
        # Include 'genres' which was in the original movies.csv
        final_columns = ['movieId', 'title', 'genres', 'tmdbId']
        # Ensure all expected columns are present before selecting
        final_df = merged_df[[col for col in final_columns if col in merged_df.columns]]


        # --- Saving ---
        print(f"Saving updated data back to {MOVIES_CSV_PATH}...")
        # index=False prevents pandas from writing the DataFrame index as a column
        # na_rep='' writes missing (<NA>) values as empty strings in the CSV
        final_df.to_csv(MOVIES_CSV_PATH, index=False, na_rep='')

        print("-" * 30)
        print(f"Successfully updated '{MOVIES_CSV_PATH}'.")
        print("Preview of the first few rows of the updated file:")
        print(final_df.head())
        print(f"\nShape of updated dataframe: {final_df.shape}")
        print(f"\nData types:\n{final_df.dtypes}")
        print("-" * 30)

    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the input files is empty or improperly formatted. {e}")
    except KeyError as e:
        print(f"Error: A required column is missing from one of the CSV files. Missing column: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")