import numpy as np
import os
import json
from pathlib import Path
import pandas as pd
from typing import Optional, List, Tuple
import pickle
from sklearn.metrics.pairwise import cosine_similarity


from VisualEncoder import *
from SynopsisEncoder import *
from TabularEncoder import *
from VectorDatabase import *
from Fusion import *
# Import your existing classes
# from tabular_embedder import TabularEmbedder
# from visual_encoder import VisualEncoder
# from fusion import Fusion


class Indexing:
    def __init__(self):
        self.synopsis_encoder = None
        self.visual_encoder = None
        self.tabular_encoder = None

        self.synopsis_embeddings = None;
        self.visual_embeddings = None;
        self.tabular_embeddings = None;
        self.fused_embeddings = None;

        self.anime_embeddings = None;

        self.synopsis_path = "./Embeddings/anime_syno_embeddings.json"
        self.visual_path = "./Embeddings/anime_poster_embeddings.json"
        self.tabular_path = "./Embeddings/anime_tabular_embeddings.json"

        self.tabular_model_path = "./Embeddings/anime_tabular_model.model"
        self.tabular_vector_path = "./Embeddings/anime_tabular_embedding.vec"

        self.image_dir = Path("./dataset/images/")
        self.dataset = Path("./AnimeList.csv")

        # metadata are columns of the DB to be stored in the vector db
        self.AM = ['id', 'title', 'title_english', 'title_japanese', 'genre', 'sypnopsis']
        self.vector_db = None
        self.anime_metadata = None


    def calculate_synopsis_embedding(self):
        if not Path(self.synopsis_path).exists():
            print("Extracting synopsis features...")
            self.synopsis_encoder = SynopsisEncoder()
            self.synopsis_embeddings = self.synopsis_encoder.encode(self.dataset)
            self.save(self.synopsis_embeddings, self.synopsis_path)
        else:
            print("Loading existing synopsis embeddings...")
            self.load(self.synopsis_path, type='syn')
        pass

    def calculate_visual_embedding(self):
        if not Path(self.visual_path).exists():
            image_paths = [str(f) for f in self.image_dir.glob("*") if f.suffix.lower() in ('.jpg', '.png', '.webp')]
            if not image_paths:
                print("No images found.")
            else:
                print("Extracting visual features...")
                self.visual_encoder = VisualEncoder(model_size='small')
                self.visual_embeddings = self.visual_encoder.encode(image_paths)
                self.save(self.visual_embeddings, self.visual_path)
        else:
            print("Loading existing visual embeddings...")
            self.load(self.visual_path, type='vis')
        pass

    def calculate_tabular_embedding(self):
        if not Path(self.tabular_path).exists():
            print("Extracting tabular features...")
            self.tabular_encoder = TabularEncoder()
            self.tabular_embeddings = self.tabular_encoder.encode(self.dataset)
            self.save(self.tabular_embeddings, self.tabular_path)
        else:
            print("Loading existing tabular embeddings...")
            self.load(self.tabular_path, type='tab')



    def calculate_embeddings(self):
        self.calculate_synopsis_embedding()
        self.calculate_visual_embedding()
        self.calculate_tabular_embedding()

        self.joint_embeddings()

        # Fused?
        self.fuse(method="mean")
        self.build_vector_database()


    def joint_embeddings(self) -> np.ndarray:
        """
        Joins all three embeddings on ID and returns a single matrix.
        Format: [ID, syn_v1...vN, vis_v1...vN, tab_v1...vN]
        """
        # 1. Find the intersection of IDs (only items present in all 3)
        syn_ids = set(self.synopsis_embeddings.keys())
        vis_ids = set(self.visual_embeddings.keys())
        tab_ids = set(self.tabular_embeddings.keys())

        common_ids = sorted(list(syn_ids & vis_ids & tab_ids))

        if not common_ids:
            raise ValueError("No common IDs found across all embedding sets.")

        joined_data = []

        for item_id in common_ids:
            # Extract the vectors
            syn_vec = self.synopsis_embeddings[item_id] # (D1,)
            vis_vec = self.visual_embeddings[item_id]   # (D2,)
            tab_vec = self.tabular_embeddings[item_id]   # (D3,)

            # Combine: [id] + syn + vis + tab
            # np.concatenate requires all arrays to be the same dimension
            # id: [[syn], [vis]. [tab]]
            combined_row = {item_id: [[syn_vec], [vis_vec], [tab_vec]]}
            joined_data.append(combined_row)

        self.anime_embeddings = np.array(joined_data)

    def fuse(self, method='weighted', weights: Optional[List[float]] = None):

        fusion_engine = Fusion(
            (self.anime_embeddings)
        )

        print(f"Esecuzione fusione con metodo: {method}...")

        if method == 'mean':
            self.fused_embeddings = fusion_engine.mean_fusion()

        elif method == 'concatenate':
            self.fused_embeddings = fusion_engine.concatenate()

        elif method == 'weighted':
            # If you don't weight use a balanced weight
            if weights is None:
                weights = [0.4, 0.4, 0.2]
            self.fused_embeddings = fusion_engine.weighted_average_fusion(weights=weights)
        else:
            raise ValueError(f"Metodo di fusione {method} non supportato.")
        print(f"Fusione completata. Shape finale: {self.fused_embeddings.shape}")
        return self.fused_embeddings


    def save(self, embeddings, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        self.save_vector_db()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=4)

        print(f"Embeddings successfully saved to {path}")

    def load(self, path, type):
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Assign data to the correct attribute based on type
            if type == "syn":
                self.synopsis_embeddings = data
                print(f"Loaded 'syn' embeddings from {path}")
            elif type == "vis":
                self.visual_embeddings = data
                print(f"Loaded 'vis' embeddings from {path}")
            elif type == "tab":
                self.tabular_embeddings = data
                print(f"Loaded 'tab' embeddings from {path}")
            else:
                print(f"Error: Unknown type '{type}'")

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {path}")
        except Exception as e:
            print(f"Error loading file: {e}")

    def retrieve_top_k(self, query_embedding: np.ndarray, k: int = 10) -> List[int]:
        """
        Performs cosine similarity retrieval on embeddings where the first column is the ID.

        Args:
            query_embedding: The vector to search for (WITHOUT the ID column).
            k: Number of nearest neighbors to return.
        """
        if self.fused_embeddings is None:
            raise ValueError("Fused embeddings not found. Please run fuse() first.")

        # 1. Separate IDs from the features
        # Assuming fused_embeddings shape is (N, 1 + D)
        item_ids = self.fused_embeddings[:, 0]
        feature_matrix = self.fused_embeddings[:, 1:]

        # 2. Prepare query (ensure it doesn't include an ID and is 2D)
        if query_embedding.ndim == 1:
            query_vec = query_embedding.reshape(1, -1)
        else:
            query_vec = query_embedding

        # 3. Calculate Cosine Similarity
        # This ignores the ID column and only compares feature vectors
        similarities = cosine_similarity(query_vec, feature_matrix).flatten()

        # 4. Get the indices of the top K scores
        top_k_indices = similarities.argsort()[-k:][::-1]

        # 5. Map indices back to the actual IDs from the first column
        top_k_ids = item_ids[top_k_indices].astype(int).tolist()

        print(f"Top {k} retrieval complete.")
        return top_k_ids

    # Vector Database Functions
    def build_vector_database(self):
        """Build vector database from embeddings"""

        # Load anime metadata
        df = pd.read_csv(self.dataset)
        self.anime_metadata = df[self.AM].to_dict('records')

        # Choose which embeddings to use
        if self.fused_embeddings is not None:
            embeddings = np.array(self.fused_embeddings)
        else:
            raise ValueError("No fused embeddings available")
        # Initialize vector database
        dimension = embeddings.shape[1]
        self.vector_db = VectorDatabase(dimension)
        # Add vectors
        print(f"Adding {len(embeddings)} vectors to database...")
        self.vector_db.add_vectors(embeddings, self.anime_metadata)
        print("Vector database built successfully!")

    def search_similar_anime(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for similar anime"""
        if self.vector_db is None:
            raise ValueError("Vector database not built. Call build_vector_database() first")
        results = self.vector_db.search(query_embedding, k=top_k)
        return results

    def save_vector_db(self, index_path: str = "./Embeddings/faiss.index",
                       metadata_path: str = "./Embeddings/metadata.pkl"):
        """Save vector database"""
        if self.vector_db is not None:
            self.vector_db.save(index_path, metadata_path)
            print(f"Vector database saved to {index_path}")

    def load_vector_db(self, index_path: str = "./Embeddings/faiss.index",
                       metadata_path: str = "./Embeddings/metadata.pkl"):
        """Load vector database"""
        dimension = 384
        self.vector_db = VectorDatabase(dimension)
        self.vector_db.load(index_path, metadata_path)
        print("Vector database loaded successfully!")
