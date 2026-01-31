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
        self.anime_ids = [];

        self.synopsis_path = "./Embeddings/anime_syno_embeddings.json"
        self.visual_path = "./Embeddings/anime_poster_embeddings.json"
        self.tabular_path = "./Embeddings/anime_tabular_embeddings.json"

        self.tabular_model_path = "./Embeddings/anime_tabular_model.model"
        self.tabular_vector_path = "./Embeddings/anime_tabular_embedding.vec"

        self.image_dir = Path("./dataset/images/")
        self.dataset = Path("./AnimeList.csv")

        # metadata are columns of the DB to be stored in the vector db
        self.AM = ['title', 'title_english', 'title_japanese', 'genre', 'sypnopsis']
        self.vector_db = None
        self.anime_metadata = None

        self.anime_db = './Embeddings/AnimeVecDb'


    def calculate_synopsis_embedding(self):
        if not Path(self.synopsis_path).exists():
            print("Extracting synopsis features...")
            self.synopsis_encoder = SynopsisEncoder()
            self.synopsis_embeddings = self.synopsis_encoder.encode(self.dataset)
            self.save(self.synopsis_embeddings, self.synopsis_path)
        else:
            print("Loading existing synopsis embeddings...")
            self.load(self.synopsis_path, type='syn')


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
        # if not os.path.exists(self.anime_db + '.index'):
        self.calculate_synopsis_embedding()
        self.calculate_visual_embedding()
        self.calculate_tabular_embedding()

        self.joint_embeddings()

        # Fused?
        self.fuse(method="mean")
        self.build_vector_database()
            #else:
            #self.load_vector_db(self.anime_db + '.index', self.anime_db + '.pkl')


    def joint_embeddings(self) -> dict:
        # Flatten each list-of-dicts into a single ID->embedding mapping
        syn_dict = {k: v for d in self.synopsis_embeddings for k, v in d.items()}
        vis_dict = {k: v for d in self.visual_embeddings for k, v in d.items()}
        tab_dict = {k: v for d in self.tabular_embeddings for k, v in d.items()}

        # Find IDs present in all three embedding sets
        common_ids = sorted(set(syn_dict.keys()) & set(vis_dict.keys()) & set(tab_dict.keys()))
        if not common_ids:
            raise ValueError("No common IDs found across all embedding sets.")

        # Build joint dictionary: {id: [synopsis_embedding, visual_embedding, tabular_embedding]}
        joint_dict = {
            item_id: [
                syn_dict[item_id],  # Synopsis embedding (preserved as-is)
                vis_dict[item_id],  # Visual embedding (preserved as-is)
                tab_dict[item_id]   # Tabular embedding (preserved as-is)
            ]
            for item_id in common_ids
        }

        # Store results as instance attributes
        self.anime_ids = common_ids
        self.anime_embeddings = joint_dict

        return joint_dict


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
        return self.fused_embeddings


    def save(self, embeddings, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

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

    def build_vector_database(self):
        """
        Builds the vector database by aligning fused embeddings with metadata.
        """
        # 1. Validate prerequisites
        if self.fused_embeddings is None:
            raise ValueError("No fused embeddings available. Run fusion step first.")
        if not self.anime_ids:
            raise ValueError("anime_ids not found. Ensure fusion step populated this attribute.")
        if not self.AM:
            raise ValueError("Metadata columns (self.AM) not defined.")

        # 2. Load dataset
        try:
            df = pd.read_csv(self.dataset)
        except Exception as e:
            raise IOError(f"Failed to load dataset: {e}")

        # 3. Validate ID column
        id_col = 'id' if 'id' in df.columns else ('anime_id' if 'anime_id' in df.columns else None)
        if id_col is None:
            raise ValueError(f"Dataset must contain 'id' or 'anime_id'. Found: {list(df.columns)}")

        # 4. Build metadata lookup
        df[id_col] = df[id_col].astype(str)
        available_cols = [col for col in self.AM if col in df.columns]
        if not available_cols:
            raise ValueError(f"None of {self.AM} found in dataset. Available: {list(df.columns)}")

        metadata_lookup = df.set_index(id_col)[available_cols].to_dict('index')
        print(f"Loaded metadata for {len(metadata_lookup)} items")

        # 5. Align embeddings + metadata IN CONSISTENT ORDER
        embedding_list = []
        metadata_list = []
        missing_metadata = []

        for anime_id in self.anime_ids:
            str_id = str(anime_id)

            # Get embedding (handle dict/list formats)
            if isinstance(self.fused_embeddings, dict):
                emb = self.fused_embeddings.get(str_id)
            else:  # list of dicts
                emb = next((item.get(str_id) for item in self.fused_embeddings if str_id in item), None)

            if emb is None:
                raise ValueError(f"Embedding missing for ID {str_id}")

            # Get metadata
            meta = metadata_lookup.get(str_id)
            if meta is None:
                missing_metadata.append(str_id)
                meta = {"id": str_id, "title": "Unknown", "_warning": "Metadata missing"}
            else:
                meta = meta.copy()
                meta["id"] = str_id

            # Normalize embedding format
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb).flatten()

            embedding_list.append(emb)
            metadata_list.append(meta)

        # 6. Warn about missing metadata
        if missing_metadata:
            print(f"⚠️ Warning: {len(missing_metadata)} items missing metadata (e.g., {missing_metadata[:3]})")

        # 7. CRITICAL FIX: Use aligned lists + validate lengths
        if len(embedding_list) != len(metadata_list):
            raise ValueError(f"Embedding count ({len(embedding_list)}) != metadata count ({len(metadata_list)})")

        embeddings_matrix = np.array(embedding_list, dtype='float32')
        dimension = embeddings_matrix.shape[1]

        # ✅ FIX: Assign metadata BEFORE adding vectors
        self.anime_metadata = metadata_list  # <-- THIS WAS MISSING

        # Initialize and populate DB
        self.vector_db = VectorDatabase(dimension, distance="cosine")
        self.vector_db.add_vectors(embeddings_matrix, self.anime_metadata)  # Now uses correct metadata

        print(f"✅ Vector DB built with {len(self.vector_db.metadata)} items")
        print(f"   - Metadata fields: {list(self.anime_metadata[0].keys())}")
        print(f"   - Embedding dimension: {dimension}")

        # Save
        self.vector_db.save(
            index_path=self.anime_db + '.index',
            metadata_path=self.anime_db + '.pkl'
        )

    def search_similar_anime(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for similar anime"""
        if self.vector_db is None:
            raise ValueError("Vector database not built. Call build_vector_database() first")
        results = self.vector_db.search(query_embedding, k=top_k)
        return results

    def load_vector_db(self, index_path: str = "./Embeddings/faiss.index",
                       metadata_path: str = "./Embeddings/metadata.pkl"):
        """Load vector database"""
        dimension = 384
        self.vector_db = VectorDatabase(dimension)
        self.vector_db.load(index_path, metadata_path)
        print("Vector database loaded successfully!")
