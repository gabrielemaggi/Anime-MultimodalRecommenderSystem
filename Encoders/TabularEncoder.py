import os

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from node2vec import Node2Vec

from .Encoder import *


class TabularEncoder:
    def __init__(self, embedding_dim=384):

        self.embedding_dim = embedding_dim
        # Internal state
        self.df = None
        self.G = None
        self.model_path = "./Embeddings/anime_tabular_model.model"
        self.vectors_path = "./Embeddings/anime_tabular_embedding.vec"
        self.model = None
        # self.embeddings_loaded = False

    def encode(self, cvs_path):
        self.__load(cvs_path)
        self.__build_graph()
        self.__train_model()
        self.save_model(
            "./Embeddings/anime_tabular_model.model",
            "./Embeddings/anime_tabular_embedding.vec",
        )
        return self.return_embeddings()

    def __load(self, filepath):
        """Loads and cleans the DataFrame."""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        if "genre" in self.df.columns:
            self.df["genre"] = self.df["genre"].fillna("")
        if "studio" in self.df.columns:
            self.df["studio"] = self.df["studio"].fillna("")
        print(f"Data loaded. Rows: {len(self.df)}")

    def __build_graph(self):
        """Builds the weighted NetworkX graph."""
        if self.df is None:
            self.load_data()

        self.G = nx.Graph()
        print("Building graph with weighted edges...")

        BASE_GENRE_WEIGHT = 1.0
        BASE_STUDIO_WEIGHT = 1.0

        for index, row in self.df.iterrows():
            anime_node = f"Anime_{row['title']}"
            self.G.add_node(anime_node, type="anime", title=row["title"])

            try:
                score = float(row.get("score", 5.0))
                if np.isnan(score):
                    score = 5.0
                score_by = float(row.get("scored_by", 1.0))
                if np.isnan(score_by):
                    score_by = 1.0
            except (ValueError, TypeError):
                score = 5.0
                score_by = 1.0

            score_factor = (score / 10.0) + 0.1
            popularity_factor = np.log10(score_by + 10.0)
            quality_multiplier = score_factor * popularity_factor

            if row["genre"]:
                genres_list = [g.strip() for g in row["genre"].split(",")]
                for genre in genres_list:
                    if genre:
                        genre_node = f"Genre_{genre}"
                        final_weight = BASE_GENRE_WEIGHT * quality_multiplier
                        self.G.add_edge(anime_node, genre_node, weight=final_weight)

            studio = str(row["studio"]).strip()
            if studio:
                studio_node = f"Studio_{studio}"
                final_weight = BASE_STUDIO_WEIGHT * quality_multiplier
                self.G.add_edge(anime_node, studio_node, weight=final_weight)

        print(f"Graph Construction Complete! Nodes: {self.G.number_of_nodes()}")

    def __train_model(self):
        """Trains Node2Vec if the graph is built."""
        if self.G is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        print("Training Node2Vec model...")
        node2vec = Node2Vec(
            self.G,
            dimensions=self.embedding_dim,
            walk_length=30,  # 30
            num_walks=100,  # 100 production
            workers=12,
            weight_key="weight",
            temp_folder="./Embeddings/",
        )
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

    def save_model(self, model_path, vector_path):
        """Saves the model in .model, .vec, and .pt formats."""
        if not self.model:
            print("No model to save.")
            return

        # 1. Save Gensim Model (.model)
        print(f"Saving Gensim model to {model_path}...")
        self.model.save(model_path)

        # 2. Save KeyedVectors format (.vec)
        print(f"Saving text vectors to {vector_path}...")
        self.model.wv.save_word2vec_format(vector_path)

    # copia in indexing
    def load_embeddings(self):
        """
        Attempts to load existing models in order: .model -> .pt -> .vec
        """
        # Priority 1: Gensim Model
        if os.path.exists(self.model_path):
            print(f"[LOAD] Found {self.model_path}. Loading Gensim model...")
            self.model = Word2Vec.load(self.model_path)
            self.embeddings_loaded = True
            return True

        # Priority 3: Vector File
        elif os.path.exists(self.vectors_path):
            print(f"[LOAD] Found {self.vectors_path}. Loading text vectors...")
            self.model = KeyedVectors.load_word2vec_format(
                self.vectors_path, binary=False
            )
            self.embeddings_loaded = True
            return True

        return False

    def fit_or_load(self):
        """Main pipeline: Load if possible, otherwise Train and Save."""
        if self.load_embeddings():
            print("Model successfully loaded from storage.")
        else:
            print("No checkpoints found. Starting training pipeline...")
            self.build_data_and_train()

    def return_embeddings(self):
        """
        Returns a list of dictionaries mapping IDs to Embeddings.
        Format: [{"id": row_id, "embedding": np_array}, ...]
        """
        # 1. Handle Model Type (Word2Vec wrapper vs KeyedVectors)
        if self.model is None:
            return []
        vectors = self.model.wv if hasattr(self.model, "wv") else self.model
        results = []
        # 2. If DataFrame is loaded, map strictly to Data Rows (Filtering out Genre/Studio nodes)
        if self.df is not None:
            for index, row in self.df.iterrows():
                # Reconstruct the specific node name used in __build_graph
                node_key = f"Anime_{row['title']}"

                # print(vectors)
                # print(vectors[node_key])
                # print(node_key)

                if node_key in vectors:
                    # Use 'id' column if it exists, otherwise fallback to title
                    row_id = row["id"] if "id" in row else row["title"]

                    results.append({row_id: vectors[node_key].tolist()})
        else:
            for node_key in vectors.index_to_key:
                results.append({node_key: vectors[node_key].tolist()})
        return results

    def recommend(self, anime_title, top_k=5):
        """Finds similar entities."""
        if not self.embeddings_loaded or self.model is None:
            print("Error: Model not loaded.")
            return

        query_node = f"Anime_{anime_title}"

        # Handle difference between Word2Vec (has .wv) and KeyedVectors (is .wv)
        vectors = self.model.wv if isinstance(self.model, Word2Vec) else self.model

        if query_node not in vectors:
            print(f"Anime '{anime_title}' not found in vocabulary.")
            return []

        print(f"\nRecommendations for '{anime_title}':")
        similar = vectors.most_similar(query_node, topn=top_k)

        results = []
        for node_name, score in similar:
            clean_name = (
                node_name.replace("Anime_", "")
                .replace("Genre_", "[Genre] ")
                .replace("Studio_", "[Studio] ")
            )
            results.append((clean_name, score))
            print(f"- {clean_name} (Similarity: {score:.4f})")

        return results

    def run_model(self, anime_title):

        self.load_embeddings()

        vectors = self.model.wv if hasattr(self.model, "wv") else self.model

        node_key = f"Anime_{anime_title}"

        if node_key in vectors:
            return vectors[node_key]
        else:
            print(f"Anime '{anime_title}' not found in model vocabulary.")
            # Optional: Suggest similar titles if available
            if hasattr(vectors, "key_to_index"):
                similar_titles = [
                    k.replace("Anime_", "")
                    for k in vectors.key_to_index.keys()
                    if k.startswith("Anime_") and anime_title.lower() in k.lower()
                ]
                if similar_titles:
                    print(f"Did you mean one of these? {similar_titles[:3]}")
            return None

    def test_genre_studio_embeddings(self):
        """
        Test function to retrieve embeddings for genre and studio nodes.
        Returns embeddings for specified genres and studios.
        """
        if self.model is None:
            print(
                "Error: Model not loaded. Call fit_or_load() or load_embeddings() first."
            )
            return None

        # Handle difference between Word2Vec and KeyedVectors
        vectors = self.model.wv if hasattr(self.model, "wv") else self.model

        results = {"genres": {}, "studios": {}}

        # Get all available genre and studio nodes
        available_genres = []
        available_studios = []

        for node_key in vectors.index_to_key:
            if node_key.startswith("Genre_"):
                available_genres.append(node_key.replace("Genre_", ""))
            elif node_key.startswith("Studio_"):
                available_studios.append(node_key.replace("Studio_", ""))

        print(f"\n--- Available Genres ({len(available_genres)}) ---")
        print(available_genres[:10], "..." if len(available_genres) > 10 else "")

        print(f"\n--- Available Studios ({len(available_studios)}) ---")
        print(available_studios[:10], "..." if len(available_studios) > 10 else "")

        # Get embeddings for genres
        for genre in available_genres:
            node_key = f"Genre_{genre}"
            if node_key in vectors:
                results["genres"][genre] = vectors[node_key]

        # Get embeddings for studios
        for studio in available_studios:
            node_key = f"Studio_{studio}"
            if node_key in vectors:
                results["studios"][studio] = vectors[node_key]

        print(f"\n--- Embedding Retrieval Complete ---")
        print(f"Retrieved {len(results['genres'])} genre embeddings")
        print(f"Retrieved {len(results['studios'])} studio embeddings")
        print(f"Embedding dimension: {self.embedding_dim}")

        return results

    def get_specific_embeddings(self, genres=None, studios=None):
        """
        Get embeddings for specific genres and/or studios.

        Args:
            genres: List of genre names (e.g., ['Action', 'Comedy'])
            studios: List of studio names (e.g., ['Toei Animation', 'Madhouse'])

        Returns:
            Dictionary with genre and studio embeddings
        """

        if self.model is None:
            self.load_embeddings()

        vectors = self.model.wv if hasattr(self.model, "wv") else self.model

        results = {"genres": {}, "studios": {}}

        # Get genre embeddings
        if genres:
            for genre in genres:
                node_key = f"Genre_{genre}"
                if node_key in vectors:
                    results["genres"][genre] = vectors[node_key]
                    print(f"✓ Found embedding for genre: {genre}")
                else:
                    print(f"✗ Genre not found: {genre}")

        # Get studio embeddings
        if studios:
            for studio in studios:
                node_key = f"Studio_{studio}"
                if node_key in vectors:
                    results["studios"][studio] = vectors[node_key]
                    print(f"✓ Found embedding for studio: {studio}")
                else:
                    print(f"✗ Studio not found: {studio}")

        return results


# --- MAIN TEST SCRIPT ---
if __name__ == "__main__":
    csv_file = "../Dataset/AnimeList.csv"

    recommender = TabularEncoder(embedding_dim=384)

    print("--- 1. STARTING PIPELINE ---")
    recommender.fit_or_load()

    print("\n--- 2. TESTING GENRE AND STUDIO EMBEDDINGS ---")
    all_embeddings = recommender.test_genre_studio_embeddings()

    # Example: Get specific embeddings
    print("\n--- 3. GETTING SPECIFIC EMBEDDINGS ---")
    specific = recommender.get_specific_embeddings(
        genres=["Action", "Comedy", "Drama"],
        studios=["Toei Animation", "Madhouse", "Bones"],
    )

    # Display sample embedding
    if specific["genres"]:
        sample_genre = list(specific["genres"].keys())[0]
        print(f"\nSample embedding for '{sample_genre}':")
        print(f"Shape: {specific['genres'][sample_genre].shape}")
        print(f"First 5 values: {specific['genres'][sample_genre][:5]}")
