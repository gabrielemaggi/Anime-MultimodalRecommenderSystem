import pandas as pd
import networkx as nx
import numpy as np
import os
import torch  # Required for .pt files
from node2vec import Node2Vec
from gensim.models import KeyedVectors, Word2Vec


class TabularEmbedder:
    def __init__(self,
                 csv_path,
                 model_path="anime_node2vec_weighted.model",
                 vectors_path="anime_embeddings_node2vec_weighted.vec",
                 embedding_dim=384):

        self.csv_path = csv_path
        self.model_path = model_path
        self.vectors_path = vectors_path
        self.embedding_dim = embedding_dim

        # Internal state
        self.df = None
        self.G = None
        self.model = None  # Can hold Word2Vec (Gensim) or KeyedVectors
        self.embeddings_loaded = False

    def load_data(self):
        """Loads and cleans the DataFrame."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)

        if 'genre' in self.df.columns:
            self.df['genre'] = self.df['genre'].fillna('')
        if 'studio' in self.df.columns:
            self.df['studio'] = self.df['studio'].fillna('')

        print(f"Data loaded. Rows: {len(self.df)}")

    def build_graph(self):
        """Builds the weighted NetworkX graph."""
        if self.df is None:
            self.load_data()

        self.G = nx.Graph()
        print("Building graph with weighted edges...")

        BASE_GENRE_WEIGHT = 1.0
        BASE_STUDIO_WEIGHT = 2.0

        for index, row in self.df.iterrows():
            anime_node = f"Anime_{row['title']}"
            self.G.add_node(anime_node, type='anime', title=row['title'])

            try:
                score = float(row.get('score', 5.0))
                if np.isnan(score): score = 5.0
                score_by = float(row.get('scored_by', 1.0))
                if np.isnan(score_by): score_by = 1.0
            except (ValueError, TypeError):
                score = 5.0
                score_by = 1.0

            score_factor = (score / 10.0) + 0.1
            popularity_factor = np.log10(score_by + 10.0)
            quality_multiplier = score_factor * popularity_factor

            if row['genre']:
                genres_list = [g.strip() for g in row['genre'].split(',')]
                for genre in genres_list:
                    if genre:
                        genre_node = f"Genre_{genre}"
                        final_weight = BASE_GENRE_WEIGHT * quality_multiplier
                        self.G.add_edge(anime_node, genre_node, weight=final_weight)

            studio = str(row['studio']).strip()
            if studio:
                studio_node = f"Studio_{studio}"
                final_weight = BASE_STUDIO_WEIGHT * quality_multiplier
                self.G.add_edge(anime_node, studio_node, weight=final_weight)

        print(f"Graph Construction Complete! Nodes: {self.G.number_of_nodes()}")

    def train_model(self):
        """Trains Node2Vec if the graph is built."""
        if self.G is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        print("Training Node2Vec model...")

        node2vec = Node2Vec(
            self.G,
            dimensions=self.embedding_dim,
            walk_length=30,
            num_walks=100,
            workers=12,
            weight_key='weight'
        )

        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        print("Training complete.")
        self.embeddings_loaded = True

    def save_model(self):
        """Saves the model in .model, .vec, and .pt formats."""
        if not self.model:
            print("No model to save.")
            return

        # 1. Save Gensim Model (.model)
        print(f"Saving Gensim model to {self.model_path}...")
        self.model.save(self.model_path)

        # 2. Save KeyedVectors format (.vec)
        print(f"Saving text vectors to {self.vectors_path}...")
        self.model.wv.save_word2vec_format(self.vectors_path)

        # Get vectors and vocab
        vectors = self.model.wv.vectors  # numpy array
        vocab_list = self.model.wv.index_to_key  # list of words

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
            self.model = KeyedVectors.load_word2vec_format(self.vectors_path, binary=False)
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

    def build_data_and_train(self):
        """Forces full training cycle."""
        self.load_data()
        self.build_graph()
        self.train_model()
        self.save_model()

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
            clean_name = node_name.replace("Anime_", "").replace("Genre_", "[Genre] ").replace("Studio_", "[Studio] ")
            results.append((clean_name, score))
            print(f"- {clean_name} (Similarity: {score:.4f})")

        return results



# --- MAIN TEST SCRIPT ---
if __name__ == "__main__":

    # 1. Configuration with requested filenames
    csv_file = 'AnimeList.csv'

    # Define the specific filenames you requested
    recommender = TabularEmbedder(
        csv_path=csv_file,
        model_path='anime_node2vec_weighted.model',  # Gensim format
        vectors_path='anime_embeddings.vec',  # Text format
        embedding_dim=384  # Kept small for faster testing, use 384 for production
    )

    print("--- 1. STARTING PIPELINE ---")
    # This will train ONLY if files don't exist.
    # Otherwise it loads 'anime_node2vec_weighted.model' or 'anime_node2vec.pt'
    recommender.fit_or_load()

    print("\n--- 2. TESTING RECOMMENDATIONS ---")
    # Replace 'Naruto' with an actual title from your CSV
    recommender.recommend("Naruto", top_k=5)
