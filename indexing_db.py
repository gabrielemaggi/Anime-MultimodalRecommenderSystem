import numpy as np
import os
import json
from pathlib import Path
import pandas as pd
from typing import Optional, List
import pickle

from VisualEncoder import *
from SynopsisEncoder import *
from TabularEncoder import *
from VectorDatabase import *
from Fusion import *


class IndexingDB:
    def __init__(self):
        # Encoders
        self.synopsis_encoder = None
        self.visual_encoder = None
        self.tabular_encoder = None

        # Paths
        self.synopsis_path = "./Embeddings/anime_syno_embeddings.json"
        self.visual_path = "./Embeddings/anime_poster_embeddings.json"
        self.tabular_path = "./Embeddings/anime_tabular_embeddings.json"
        self.image_dir = Path("./dataset/images/")
        self.dataset = Path("./AnimeList.csv")
        self.anime_db_index = './Embeddings/AnimeVecDb.index'
        self.anime_db_metadata = './Embeddings/AnimeVecDb.pkl'

        # Metadata columns to store
        self.AM = ['title', 'title_english', 'title_japanese', 'genre', 'sypnopsis']

        # Vector database
        self.vector_db = None


    def _load_or_create_embeddings(self, path: str, encoder_fn, embed_type: str):
        """Generic method to load existing embeddings or create new ones"""
        if Path(path).exists():
            print(f"Loading existing {embed_type} embeddings...")
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Extracting {embed_type} features...")
            embeddings = encoder_fn()
            self._save_embeddings(embeddings, path)
            return embeddings


    def _save_embeddings(self, embeddings, path):
        """Save embeddings to JSON file"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=4)
        print(f"Embeddings successfully saved to {path}")


    def build_vector_database(self, fusion_method: str = 'mean',
                             fusion_weights: Optional[List[float]] = None):
        """
        Main method: builds the complete vector database from scratch or loads existing one

        Args:
            fusion_method: 'mean', 'concatenate', or 'weighted'
            fusion_weights: weights for weighted fusion (defaults to [0.4, 0.4, 0.2])
        """
        # Check if database already exists
        if Path(self.anime_db_index).exists() and Path(self.anime_db_metadata).exists():
            print("Loading existing vector database...")
            self.load_vector_database()
            return

        print("Building vector database from scratch...")

        # 1. Load or create embeddings
        synopsis_embeddings = self._load_or_create_embeddings(
            self.synopsis_path,
            lambda: SynopsisEncoder().encode(self.dataset),
            "synopsis"
        )

        visual_embeddings = self._load_or_create_embeddings(
            self.visual_path,
            lambda: self._create_visual_embeddings(),
            "visual"
        )

        tabular_embeddings = self._load_or_create_embeddings(
            self.tabular_path,
            lambda: TabularEncoder().encode(self.dataset),
            "tabular"
        )

        # 2. Align embeddings by common IDs
        print("Aligning embeddings...")
        aligned_data = self._align_embeddings(
            synopsis_embeddings,
            visual_embeddings,
            tabular_embeddings
        )

        # 3. Fuse embeddings
        print(f"Fusing embeddings using {fusion_method} method...")
        fused_embeddings = self._fuse_embeddings(
            aligned_data,
            fusion_method,
            fusion_weights
        )

        # 4. Load metadata and build vector database
        print("Building vector database...")
        self._create_vector_database(fused_embeddings, aligned_data['ids'])

        print("✅ Vector database built successfully!")


    def _create_visual_embeddings(self):
        """Create visual embeddings from image directory"""
        image_paths = [
            str(f) for f in self.image_dir.glob("*")
            if f.suffix.lower() in ('.jpg', '.png', '.webp')
        ]

        if not image_paths:
            raise ValueError("No images found in image directory")

        self.visual_encoder = VisualEncoder(model_size='small')
        return self.visual_encoder.encode(image_paths)


    def _align_embeddings(self, synopsis_emb, visual_emb, tabular_emb) -> dict:
        """
        Align embeddings from different sources by common IDs

        Returns:
            dict with keys: 'ids', 'synopsis', 'visual', 'tabular'
        """
        # Flatten list-of-dicts to single dict
        syn_dict = {k: v for d in synopsis_emb for k, v in d.items()}
        vis_dict = {k: v for d in visual_emb for k, v in d.items()}
        tab_dict = {k: v for d in tabular_emb for k, v in d.items()}

        # Find common IDs
        common_ids = sorted(set(syn_dict.keys()) & set(vis_dict.keys()) & set(tab_dict.keys()))

        if not common_ids:
            raise ValueError("No common IDs found across all embedding sets")

        print(f"Found {len(common_ids)} items with all three embedding types")

        # Align by common IDs
        return {
            'ids': common_ids,
            'synopsis': [syn_dict[id] for id in common_ids],
            'visual': [vis_dict[id] for id in common_ids],
            'tabular': [tab_dict[id] for id in common_ids]
        }


    def _fuse_embeddings(self, aligned_data: dict, method: str,
                        weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Fuse aligned embeddings using specified method

        Returns:
            numpy array of shape (n_items, embedding_dim)
        """
        # Prepare data for Fusion class (dict format: {id: [syn_emb, vis_emb, tab_emb]})
        joint_dict = {
            item_id: [
                aligned_data['synopsis'][i],
                aligned_data['visual'][i],
                aligned_data['tabular'][i]
            ]
            for i, item_id in enumerate(aligned_data['ids'])
        }

        fusion_engine = Fusion(joint_dict)

        if method == 'mean':
            fused = fusion_engine.mean_fusion()
        elif method == 'concatenate':
            fused = fusion_engine.concatenate()
        elif method == 'weighted':
            if weights is None:
                weights = [0.4, 0.4, 0.2]
            fused = fusion_engine.weighted_average_fusion(weights=weights)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

        return fused


    def _create_vector_database(self, fused_embeddings: np.ndarray, anime_ids: List[str]):
        """
        Create vector database with embeddings and metadata
        """
        # Load dataset
        try:
            df = pd.read_csv(self.dataset)
        except Exception as e:
            raise IOError(f"Failed to load dataset: {e}")

        # Identify ID column
        id_col = 'id' if 'id' in df.columns else ('anime_id' if 'anime_id' in df.columns else None)
        if id_col is None:
            raise ValueError(f"Dataset must contain 'id' or 'anime_id'. Found: {list(df.columns)}")

        df[id_col] = df[id_col].astype(str)

        # Get available metadata columns
        available_cols = [col for col in self.AM if col in df.columns]
        if not available_cols:
            raise ValueError(f"None of {self.AM} found in dataset. Available: {list(df.columns)}")

        metadata_lookup = df.set_index(id_col)[available_cols].to_dict('index')

        # Build metadata list aligned with embeddings
        metadata_list = []
        valid_embeddings = []

        for i, anime_id in enumerate(anime_ids):
            str_id = str(anime_id)

            # Get metadata
            meta = metadata_lookup.get(str_id)
            if meta is None:
                print(f"⚠️ Warning: Metadata missing for ID {str_id}, skipping...")
                continue

            # Add ID to metadata
            meta = meta.copy()
            meta['id'] = str_id

            metadata_list.append(meta)
            valid_embeddings.append(fused_embeddings[i] if isinstance(fused_embeddings, list)
                                  else fused_embeddings[str_id])

        # Convert to numpy array
        embeddings_matrix = np.array(valid_embeddings, dtype='float32')
        dimension = embeddings_matrix.shape[1]

        # Initialize and populate vector database
        self.vector_db = VectorDatabase(dimension, distance="cosine")
        self.vector_db.add_vectors(embeddings_matrix, metadata_list)

        print(f"✅ Vector DB created with {len(metadata_list)} items")
        print(f"   - Metadata fields: {list(metadata_list[0].keys())}")
        print(f"   - Embedding dimension: {dimension}")

        # Save database
        self.vector_db.save(self.anime_db_index, self.anime_db_metadata)


    def load_vector_database(self):
        """Load existing vector database"""
        if not Path(self.anime_db_index).exists() or not Path(self.anime_db_metadata).exists():
            raise FileNotFoundError("Vector database files not found. Build database first.")

        # Load metadata to get dimension
        with open(self.anime_db_metadata, 'rb') as f:
            metadata = pickle.load(f)

        # Infer dimension from saved index
        import faiss
        temp_index = faiss.read_index(self.anime_db_index)
        dimension = temp_index.d

        self.vector_db = VectorDatabase(dimension, distance="cosine")
        self.vector_db.load(self.anime_db_index, self.anime_db_metadata)

        print(f"✅ Vector database loaded: {len(self.vector_db.metadata)} items, dim={dimension}")


    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Search for similar anime using query embedding

        Args:
            query_embedding: Query vector (should match database dimension)
            top_k: Number of results to return

        Returns:
            List of dictionaries with metadata and similarity scores
        """
        if self.vector_db is None:
            raise ValueError("Vector database not loaded. Call build_vector_database() or load_vector_database() first")

        return self.vector_db.search(query_embedding, k=top_k)


    def get_database_info(self) -> dict:
        """Get information about the current vector database"""
        if self.vector_db is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "total_items": len(self.vector_db.metadata),
            "dimension": self.vector_db.dimension,
            "metadata_fields": list(self.vector_db.metadata[0].keys()) if self.vector_db.metadata else []
        }
