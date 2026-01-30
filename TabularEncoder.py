import pandas as pd
import networkx as nx
import numpy as np
import os
import torch  # Required for .pt files
from node2vec import Node2Vec
from gensim.models import KeyedVectors, Word2Vec
from Encoder import *

class TabularEncoder():
    def __init__(self, embedding_dim=384):

        self.embedding_dim = embedding_dim
        # Internal state
        self.df = None
        self.G = None
        self.model = None  # Can hold Word2Vec (Gensim) or KeyedVectors
        # self.embeddings_loaded = False

    def encode(self, cvs_path):
        self.__load(cvs_path)
        self.__build_graph()
        self.__train_model()
        return self.return_embeddings()
        #self.save_model("Embeddings/anime_tabular_model.model", "Embeddings/anime_tabular_embedding.vec")

    def __load(self, filepath):
        """Loads and cleans the DataFrame."""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        if 'genre' in self.df.columns:
            self.df['genre'] = self.df['genre'].fillna('')
        if 'studio' in self.df.columns:
            self.df['studio'] = self.df['studio'].fillna('')
        print(f"Data loaded. Rows: {len(self.df)}")

    def __build_graph(self):
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

    def __train_model(self):
        """Trains Node2Vec if the graph is built."""
        if self.G is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        print("Training Node2Vec model...")
        node2vec = Node2Vec(
            self.G,
            dimensions=self.embedding_dim,
            walk_length=3,  # 30
            num_walks=5,   # 100 production
            workers=12,
            weight_key='weight',
            temp_folder='./Embeddings/'
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
        print(f"Saving text vectors to {vectors_path}...")
        self.model.wv.save_word2vec_format(vectors_path)

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

    def return_embeddings(self):
            """
            Returns a list of dictionaries mapping IDs to Embeddings.
            Format: [{"id": row_id, "embedding": np_array}, ...]
            """
            # 1. Handle Model Type (Word2Vec wrapper vs KeyedVectors)
            if self.model is None:
                return []
            vectors = self.model.wv if hasattr(self.model, 'wv') else self.model
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
                        row_id = row['id'] if 'id' in row else row['title']

                        results.append({
                            row_id: vectors[node_key].tolist()
                        })
            else:
                for node_key in vectors.index_to_key:
                    results.append({
                        node_key: vectors[node_key].tolist()
                    })
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
