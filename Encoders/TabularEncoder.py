import os
import pickle

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

    def _old(self, anime_title):

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

    def add_anime_to_model(self, anime_data, retrain=True):
        """
        Add a new anime to the graph and optionally retrain the model.

        Args:
            anime_data: Dictionary with anime info (title, genre, studio, score, scored_by)
            retrain: Whether to retrain the model after adding (default: True)

        Returns:
            The embedding for the new anime
        """
        if self.G is None:
            raise ValueError("Graph not built. Load the graph first.")

        if self.model is None:
            self.load_embeddings()

        # Extract data
        anime_title = anime_data.get("title")
        genres = anime_data.get("genre", "")
        studios = anime_data.get("studios", "") or anime_data.get("studio", "")
        score = anime_data.get("score", 5.0)
        scored_by = anime_data.get("scored_by", 1.0)

        if not anime_title:
            raise ValueError("anime_data must contain 'title'")

        # Create anime node
        anime_node = f"Anime_{anime_title}"

        # Check if already exists
        if anime_node in self.G:
            print(f"Anime '{anime_title}' already exists in graph.")
            vectors = self.model.wv if hasattr(self.model, "wv") else self.model
            if anime_node in vectors:
                return vectors[anime_node]

        print(f"Adding '{anime_title}' to graph...")

        # Add node to graph
        self.G.add_node(anime_node, type="anime", title=anime_title)

        # Calculate quality multiplier (same as __build_graph)
        try:
            score = float(score) if score else 5.0
            scored_by = float(scored_by) if scored_by else 1.0
        except (ValueError, TypeError):
            score = 5.0
            scored_by = 1.0

        score_factor = (score / 10.0) + 0.1
        popularity_factor = np.log10(scored_by + 10.0)
        quality_multiplier = score_factor * popularity_factor

        BASE_GENRE_WEIGHT = 1.0
        BASE_STUDIO_WEIGHT = 1.0

        # Add genre edges
        if genres:
            if isinstance(genres, str):
                genres_list = [g.strip() for g in genres.split(",")]
            else:
                genres_list = genres

            for genre in genres_list:
                if genre:
                    genre_node = f"Genre_{genre}"
                    # Add genre node if it doesn't exist
                    if genre_node not in self.G:
                        self.G.add_node(genre_node, type="genre")
                        print(f"  + New genre node: {genre}")

                    final_weight = BASE_GENRE_WEIGHT * quality_multiplier
                    self.G.add_edge(anime_node, genre_node, weight=final_weight)
                    print(
                        f"  ✓ Connected to genre: {genre} (weight: {final_weight:.2f})"
                    )

        # Add studio edges
        if studios:
            if isinstance(studios, str):
                studios_list = [s.strip() for s in studios.split(",")]
            else:
                studios_list = studios

            for studio in studios_list:
                if studio:
                    studio_node = f"Studio_{studio}"
                    # Add studio node if it doesn't exist
                    if studio_node not in self.G:
                        self.G.add_node(studio_node, type="studio")
                        print(f"  + New studio node: {studio}")

                    final_weight = BASE_STUDIO_WEIGHT * quality_multiplier
                    self.G.add_edge(anime_node, studio_node, weight=final_weight)
                    print(
                        f"  ✓ Connected to studio: {studio} (weight: {final_weight:.2f})"
                    )

        # Retrain if requested
        if retrain:
            print("Generating random walks for new node...")
            embedding = self._incremental_train(anime_node)

            # Save updated model
            self.save_model(self.model_path, self.vectors_path)
            print(f"✅ Model updated and saved")

            return embedding
        else:
            print("⚠️  Node added to graph but model not retrained")
            return None

    def _incremental_train(self, new_node=None, num_walks=100, walk_length=30):
        """
        Incrementally train the model on new nodes using random walks.

        Args:
            new_node: Specific node to focus walks on (optional)
            num_walks: Number of random walks per node
            walk_length: Length of each walk

        Returns:
            Embedding for the new node (if specified)
        """
        if self.G is None:
            raise ValueError("Graph not available")

        if self.model is None:
            raise ValueError("Model not loaded")

        # Generate walks
        print(f"Generating random walks...")

        if new_node:
            # Focus walks on the new node and its neighbors
            nodes_to_walk = [new_node]
            # Add neighbors for context
            neighbors = list(self.G.neighbors(new_node))
            nodes_to_walk.extend(neighbors)
            print(f"Focusing on new node and {len(neighbors)} neighbors")
            # More walks for the new node itself
            node_walk_counts = {new_node: num_walks}
            for neighbor in neighbors:
                node_walk_counts[neighbor] = max(
                    10, num_walks // 10
                )  # Fewer walks for neighbors
        else:
            # Walk all nodes
            nodes_to_walk = list(self.G.nodes())
            node_walk_counts = {node: num_walks for node in nodes_to_walk}

        # Generate walks
        walks = []
        for node in nodes_to_walk:
            walk_count = node_walk_counts.get(node, num_walks)
            for _ in range(walk_count):
                walk = self._random_walk(node, walk_length)
                walks.append(walk)

        print(f"Generated {len(walks)} walks")

        # Build vocabulary for new nodes
        vectors = self.model.wv if hasattr(self.model, "wv") else self.model

        new_words = []
        for walk in walks:
            for node in walk:
                if node not in vectors:
                    new_words.append(node)

        if new_words:
            unique_new = list(set(new_words))
            print(f"Adding {len(unique_new)} new words to vocabulary...")
            self.model.build_vocab(walks, update=True)

        # Train on new walks
        print("Training model on new walks...")
        self.model.train(walks, total_examples=len(walks), epochs=self.model.epochs)

        print("✅ Incremental training complete")

        # Return embedding for new node
        if new_node:
            vectors = self.model.wv if hasattr(self.model, "wv") else self.model
            if new_node in vectors:
                return vectors[new_node]
            else:
                print(f"⚠️  Warning: {new_node} not in vectors after training")

        return None

    def _random_walk(self, start_node, walk_length):
        """
        Perform a weighted random walk from start_node.

        Args:
            start_node: Starting node
            walk_length: Length of the walk

        Returns:
            List of node names representing the walk
        """
        walk = [start_node]

        for _ in range(walk_length - 1):
            current = walk[-1]
            neighbors = list(self.G.neighbors(current))

            if not neighbors:
                break

            # Get edge weights
            weights = []
            for neighbor in neighbors:
                weight = self.G[current][neighbor].get("weight", 1.0)
                weights.append(weight)

            # Normalize weights to probabilities
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
            else:
                probabilities = [1.0 / len(neighbors)] * len(neighbors)

            # Choose next node based on weights
            next_node = np.random.choice(neighbors, p=probabilities)
            walk.append(next_node)

        return walk

    def run_model(
        self,
        anime_title=None,
        genres=None,
        studios=None,
        score=None,
        scored_by=None,
        auto_add=False,
    ):
        """
        Get embedding for an anime. Optionally add to model if not found.

        Args:
            anime_title: Title of the anime
            genres: List or comma-separated string of genres
            studios: List or comma-separated string of studios
            score: MAL score (0-10)
            scored_by: Number of users who scored it
            auto_add: If True, automatically add missing anime to model

        Returns:
            numpy array embedding
        """
        self.load_embeddings()

        # Also load graph if not loaded
        if self.G is None:
            print("Loading graph structure...")
            self._load_graph()

        vectors = self.model.wv if hasattr(self.model, "wv") else self.model

        # Try to get existing embedding
        if anime_title:
            node_key = f"Anime_{anime_title}"
            if node_key in vectors:
                return vectors[node_key]

        # If auto_add is enabled and we have enough info
        if auto_add and anime_title and (genres or studios):
            print(f"Auto-adding '{anime_title}' to model...")
            anime_data = {
                "title": anime_title,
                "genre": genres,
                "studios": studios,
                "score": score,
                "scored_by": scored_by,
            }
            return self.add_anime_to_model(anime_data, retrain=True)

        # Otherwise, fall back to averaging
        print(f"Anime '{anime_title}' not found. Constructing from metadata...")
        return self._construct_from_metadata(genres, studios, score, scored_by)

    def _construct_from_metadata(self, genres, studios, score=None, scored_by=None):
        """Fallback: construct embedding from genre/studio averaging"""
        vectors = self.model.wv if hasattr(self.model, "wv") else self.model

        embeddings_to_average = []

        # Process genres
        if genres:
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(",")]

            for genre in genres:
                if genre:
                    genre_key = f"Genre_{genre}"
                    if genre_key in vectors:
                        embeddings_to_average.append(vectors[genre_key])
                        print(f"  ✓ Genre: {genre}")

        # Process studios
        if studios:
            if isinstance(studios, str):
                studios = [s.strip() for s in studios.split(",")]

            for studio in studios:
                if studio:
                    studio_key = f"Studio_{studio}"
                    if studio_key in vectors:
                        embeddings_to_average.append(vectors[studio_key])
                        print(f"  ✓ Studio: {studio}")

        if embeddings_to_average:
            avg_embedding = np.mean(embeddings_to_average, axis=0)
            print(f"  → Averaged {len(embeddings_to_average)} embeddings")
            return avg_embedding
        else:
            print(f"  ✗ No valid metadata. Returning zero vector.")
            return np.zeros(self.embedding_dim)

    def _load_graph(self):
        """Load graph structure from saved file or rebuild from CSV"""
        graph_path = "./Embeddings/anime_graph.pkl"

        if os.path.exists(graph_path):
            try:
                import pickle

                print(f"Loading graph from {graph_path}...")
                with open(graph_path, "rb") as f:
                    self.G = pickle.load(f)
                print(f"Graph loaded: {self.G.number_of_nodes()} nodes")
                return
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"⚠️  Failed to load graph: {e}")
                print("Removing corrupted graph file and rebuilding...")
                try:
                    os.remove(graph_path)
                except:
                    pass

        # If we get here, we need to rebuild
        print("Rebuilding graph from CSV...")
        if self.df is None:
            # Try to load the default CSV
            csv_path = "./Dataset/AnimeList.csv"
            if os.path.exists(csv_path):
                self.__load(csv_path)
            else:
                raise ValueError(
                    "Cannot load graph: no saved graph and no CSV found at ./Dataset/AnimeList.csv"
                )

        # Build the graph
        self.__build_graph()

        # Save for next time
        try:
            import pickle

            with open(graph_path, "wb") as f:
                pickle.dump(self.G, f)
            print(f"✅ Graph saved to {graph_path}")
        except Exception as e:
            print(f"⚠️  Failed to save graph: {e}")

    def save_model(self, model_path, vector_path):
        """Save model and graph with error handling"""
        if not self.model:
            print("No model to save.")
            return

        try:
            # Save Gensim Model
            print(f"Saving model to {model_path}...")
            self.model.save(model_path)
            print("✅ Model saved")
        except Exception as e:
            print(f"⚠️  Failed to save model: {e}")

        try:
            # Save vectors
            print(f"Saving vectors to {vector_path}...")
            self.model.wv.save_word2vec_format(vector_path)
            print("✅ Vectors saved")
        except Exception as e:
            print(f"⚠️  Failed to save vectors: {e}")

        # Save graph structure
        if self.G:
            try:
                import pickle

                graph_path = "./Embeddings/anime_graph.pkl"
                print(f"Saving graph to {graph_path}...")
                with open(graph_path, "wb") as f:
                    pickle.dump(self.G, f)
                print("✅ Graph saved")
            except Exception as e:
                print(f"⚠️  Failed to save graph: {e}")


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
