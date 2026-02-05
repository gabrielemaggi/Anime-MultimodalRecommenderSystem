import re

import pandas as pd

from Encoders import *
from Libs.indexing_db import *


class GoalParsing:
    def __init__(self, csv_path="./Dataset/AnimeList.csv"):
        """
        Initializes the parser by loading valid genres and studios from the CSV.
        """
        self.valid_genres = set()
        self.valid_studios = set()
        self._load_knowledge_base(csv_path)

    def _load_knowledge_base(self, path):
        """
        Reads the CSV and extracts unique values for genres and studios.
        Assumes columns are named 'genre' and 'studio' or similar.
        """
        try:
            df = pd.read_csv(path)

            # Helper to split comma-separated strings and add to set
            def extract_unique(column_name, target_set):
                if column_name in df.columns:
                    # Drop NAs, convert to string, split by comma, strip whitespace
                    for item in df[column_name].dropna().astype(str):
                        parts = [p.strip() for p in item.split(",")]
                        target_set.update(parts)

            # Adjust column names here if your CSV is different
            # Common names checked: 'genre', 'Genres', 'studio', 'Studios'
            genre_col = next((c for c in df.columns if "genre" in c.lower()), None)
            studio_col = next((c for c in df.columns if "studio" in c.lower()), None)

            if genre_col:
                extract_unique(genre_col, self.valid_genres)
            if studio_col:
                extract_unique(studio_col, self.valid_studios)

            print(
                f"Loaded {len(self.valid_genres)} genres and {len(self.valid_studios)} studios."
            )

        except FileNotFoundError:
            print(f"Error: {path} not found. Ensure the file exists.")
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def extract_entities(self, text):
        """
        Scans the text for valid genres and studios using Regex.
        Returns lists of found matches.
        """
        found_genres = []
        found_studios = []

        # Search genres
        for genre in self.valid_genres:
            # \b matches word boundaries, re.escape escapes special chars like '+' in 'Hentai+'
            pattern = r"\b" + re.escape(genre) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                found_genres.append(genre)

        # Search studios
        for studio in self.valid_studios:
            pattern = r"\b" + re.escape(studio) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                found_studios.append(studio)

        return found_genres, found_studios

    def process_request(self, text, user_instance, indexer_instance):
        """
        Parses text, gets embeddings, and applies filters to the user instance.
        """
        print(f"Processing request: '{text}'")
        genres, studios = self.extract_entities(text)

        if not genres and not studios:
            print("No matching genres or studios found in text.")
            return

        print(f"Found Genres: {genres} | Found Studios: {studios}")

        # 1. Get Embeddings
        results = indexer_instance.encode_tabular_genre_studio(
            genres=genres, studios=studios
        )
        # print(results)
        # 2. Iterate through 'genres' results
        if "genres" in results and results["genres"]:
            for genre_name, embedding in results["genres"].items():
                print(f"Applying filter for Genre: {genre_name}")
                query = indexer_instance.align_embedding(embedding, modality="tab")
                user_instance.add_filtering(query, "append")

        # 3. Iterate through 'studios' results
        if "studios" in results and results["studios"]:
            for studio_name, embedding in results["studios"].items():
                print(f"Applying filter for Studio: {studio_name}")
                query = indexer_instance.align_embedding(embedding, modality="tab")
                user_instance.add_filtering(query, "append")

    def process_sypnopsis(self, text, user_instance, indexer_instance):
        embedding = indexer_instance.encode_sypnopsis(text)
        print(f"Applying filter for Synopsis: {text}")
        query = indexer_instance.align_embedding(embedding, modality="syn")
        user_instance.add_filtering(query, "append")

    def process_image(self, image, user_instance, indexer_instance):
        embedding = indexer_instance.encode_image(image)
        print(f"Applying filter for Image: {image}")
        query = indexer_instance.align_embedding(embedding, modality="vis")
        user_instance.add_filtering(query, "append")
