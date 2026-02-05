import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch.nn.functional as F
from .Fusion import *
from Encoders import *
from .trainableFusion import *
from .VectorDatabase import *
import torch

class Indexing:
    def __init__(self):
        # Encoders
        self.synopsis_encoder = None
        self.visual_encoder = None
        self.tabular_encoder = None

        # Paths
        self.synopsis_path = "../Embeddings/anime_syno_embeddings.json"
        self.visual_path = "../Embeddings/anime_poster_embeddings.json"
        self.tabular_path = "../Embeddings/anime_tabular_embeddings.json"

        self.image_dir = Path("../Dataset/images/")
        self.dataset = Path("../Dataset/AnimeList.csv")
        self.anime_db_index = "../Embeddings/AnimeVecDb.index"
        self.anime_db_metadata = "../Embeddings/AnimeVecDb.pkl"

        # Metadata columns to store
        self.AM = ["title", "title_english", "title_japanese", "genre", "sypnopsis"]

        # Vector database
        self.vector_db = None

        # Cached dataset
        self._dataset_df = None

        # Fusion settings (store for encoding queries)
        self.fusion_method = "trainable"
        self.fusion_weights = [0.7, 0.1, 0.2]
        self.fusion_engine = FusionTrainer(load_model=True)

    def _load_dataset(self) -> pd.DataFrame:
        """Load and cache the dataset"""
        if self._dataset_df is None:
            self._dataset_df = pd.read_csv(self.dataset)
        return self._dataset_df

    def _ensure_encoders_loaded(self):
        """Ensure all encoders are initialized"""
        if self.synopsis_encoder is None:
            self.synopsis_encoder = SynopsisEncoder()
        if self.visual_encoder is None:
            self.visual_encoder = VisualEncoder(model_size="small")
        if self.tabular_encoder is None:
            self.tabular_encoder = TabularEncoder()

    def _load_or_create_embeddings(self, path: str, encoder_fn, embed_type: str):
        """Generic method to load existing embeddings or create new ones"""
        if Path(path).exists():
            print(f"Loading existing {embed_type} embeddings...")
            with open(path, "r", encoding="utf-8") as f:
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

        with open(path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, indent=4)
        print(f"Embeddings successfully saved to {path}")

    def _normalize_embeddings(
        self, embedding_list: List[Dict[str, List[float]]]
    ) -> List[Dict[str, List[float]]]:
        """
        Normalizes a list of dictionary embeddings (L2 Norm).
        Structure: [{'id': [vector]}, ...] -> Tensor -> Normalize -> [{'id': [normalized_vector]}, ...]
        """

        # 1. Flatten the list of dicts into parallel lists of IDs and Vectors
        # This assumes each dict in the list has exactly one key-value pair, based on your _align logic
        ids = []
        vectors = []
        for item in embedding_list:
            for k, v in item.items():
                ids.append(k)
                vectors.append(v)
        # 2. Convert to PyTorch Tensor (Batch processing is much faster)
        # shape: (num_items, embedding_dim)
        tensor_vectors = torch.tensor(vectors, dtype=torch.float)
        # 3. Apply L2 Normalization
        # dim=1 applies it across the embedding dimension (the rows)
        normalized_tensor = F.normalize(tensor_vectors, p=2, dim=1)
        # 4. Convert back to Python List for compatibility with the rest of your pipeline
        normalized_vectors = normalized_tensor.tolist()
        # 5. Reconstruct the original list-of-dicts structure
        return [{id_val: vec} for id_val, vec in zip(ids, normalized_vectors)]

    def build_vector_database(
        self,
    ):  # (self, fusion_method: str = 'mean',fusion_weights: Optional[List[float]] = None):
        """
        Main method: builds the complete vector database from scratch or loads existing one

        Args:
            fusion_method: 'mean', 'concatenate', or 'weighted'
            fusion_weights: weights for weighted fusion (defaults to [0.4, 0.4, 0.2])
        """
        # Store fusion settings
        # self.fusion_method = fusion_method
        # self.fusion_weights = fusion_weights

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
            "synopsis",
        )

        visual_embeddings = self._load_or_create_embeddings(
            self.visual_path, lambda: self._create_visual_embeddings(), "visual"
        )

        tabular_embeddings = self._load_or_create_embeddings(
            self.tabular_path, lambda: TabularEncoder().encode(self.dataset), "tabular"
        )

        print("Normalizing embeddings...")
        synopsis_embeddings = self._normalize_embeddings(synopsis_embeddings)
        visual_embeddings = self._normalize_embeddings(visual_embeddings)
        tabular_embeddings = self._normalize_embeddings(tabular_embeddings)

        # 2. Align embeddings by common IDs
        print("Aligning embeddings...")
        aligned_data = self._align_embeddings(
            synopsis_embeddings, visual_embeddings, tabular_embeddings
        )

        # 3. Fuse embeddings
        print(f"Fusing embeddings using {self.fusion_method} method...")
        fused_embeddings = self._fuse_embeddings(
            aligned_data, self.fusion_method, self.fusion_weights
        )

        # 4. Load metadata and build vector database
        print("Building vector database...")
        self._create_vector_database(fused_embeddings, aligned_data["ids"])

        print("✅ Vector database built successfully!")

    def _create_visual_embeddings(self):
        """Create visual embeddings from image directory"""
        image_paths = [
            str(f)
            for f in self.image_dir.glob("*")
            if f.suffix.lower() in (".jpg", ".png", ".webp")
        ]

        if not image_paths:
            raise ValueError("No images found in image directory")

        self.visual_encoder = VisualEncoder(model_size="small")
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
        common_ids = sorted(
            set(syn_dict.keys()) & set(vis_dict.keys()) & set(tab_dict.keys())
        )

        if not common_ids:
            raise ValueError("No common IDs found across all embedding sets")

        print(f"Found {len(common_ids)} items with all three embedding types")

        # Align by common IDs
        return {
            "ids": common_ids,
            "synopsis": [syn_dict[id] for id in common_ids],
            "visual": [vis_dict[id] for id in common_ids],
            "tabular": [tab_dict[id] for id in common_ids],
        }

    def _fuse_embeddings(
        self, aligned_data: dict, method: str, weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Fuse aligned embeddings using specified method

        Returns:
            numpy array of shape (n_items, embedding_dim)
        """
        # Prepare data for Fusion class (dict format: {id: [syn_emb, vis_emb, tab_emb]})
        joint_dict = {
            item_id: [
                aligned_data["synopsis"][i],
                aligned_data["visual"][i],
                aligned_data["tabular"][i],
            ]
            for i, item_id in enumerate(aligned_data["ids"])
        }

        if method == "trainable":
            syn_embeddings = np.array(aligned_data["synopsis"])
            vis_embeddings = np.array(aligned_data["visual"])
            tab_embeddings = np.array(aligned_data["tabular"])

            fusion_engine = FusionTrainer(
                np.array(aligned_data["ids"]),
                syn_embeddings,
                vis_embeddings,
                tab_embeddings,
                384,
            )
            fusion_engine.train()
            fused = fusion_engine.transform()
        else:
            fusion_engine = Fusion(joint_dict)
            if method == "mean":
                fused = fusion_engine.mean_fusion()
            elif method == "concatenate":
                fused = fusion_engine.concatenate()
            elif method == "weighted":
                if weights is None:
                    weights = [0.4, 0.4, 0.2]
                fused = fusion_engine.weighted_average_fusion(weights=weights)
            else:
                raise ValueError(f"Unsupported fusion method: {method}")

        return fused

    def _fuse_single_embeddings(
        self,
        anime_id,
        synopsis_emb: np.ndarray,
        visual_emb: np.ndarray,
        tabular_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse a single set of embeddings using the same method as the database

        Returns:
            Fused embedding vector
        """
        method = self.fusion_method
        weights = self.fusion_weights

        if method == "trainable":
            fusion_engine = FusionTrainer(
                anime_id, synopsis_emb, visual_emb, tabular_emb, 384, load_model=True
            )
            fusion_engine.train()
            fused = fusion_engine.transform()

        else:
            joint_dict = {anime_id: [synopsis_emb, visual_emb, tabular_emb]}
            fusion_engine = Fusion(joint_dict)
            if method == "mean":
                fused = fusion_engine.mean_fusion()
            elif method == "concatenate":
                fused = fusion_engine.concatenate()
            elif method == "weighted":
                if weights is None:
                    weights = [0.4, 0.4, 0.2]
                fused = fusion_engine.weighted_average_fusion(weights=weights)
            else:
                raise ValueError(f"Unsupported fusion method: {method}")

        return fused

    def _create_vector_database(
        self, fused_embeddings: np.ndarray, anime_ids: List[str]
    ):
        """
        Create vector database with embeddings and metadata
        """
        # Load dataset
        try:
            df = pd.read_csv(self.dataset)
        except Exception as e:
            raise IOError(f"Failed to load dataset: {e}")

        # Identify ID column
        id_col = (
            "id"
            if "id" in df.columns
            else ("anime_id" if "anime_id" in df.columns else None)
        )
        if id_col is None:
            raise ValueError(
                f"Dataset must contain 'id' or 'anime_id'. Found: {list(df.columns)}"
            )

        df[id_col] = df[id_col].astype(str)

        # Get available metadata columns
        available_cols = [col for col in self.AM if col in df.columns]
        if not available_cols:
            raise ValueError(
                f"None of {self.AM} found in dataset. Available: {list(df.columns)}"
            )

        metadata_lookup = df.set_index(id_col)[available_cols].to_dict("index")

        # Build metadata list aligned with embeddings
        metadata_list = []
        valid_embeddings = []

        for i, anime_id in enumerate(anime_ids):
            str_id = str(anime_id)

            # Get metadata
            meta = metadata_lookup.get(str_id)
            if meta is None:
                print(f"Warning: Metadata missing for ID {str_id}, skipping...")
                continue

            # Add ID to metadata
            meta = meta.copy()
            meta["id"] = str_id

            metadata_list.append(meta)
            valid_embeddings.append(
                fused_embeddings[i]
                if isinstance(fused_embeddings, list)
                else fused_embeddings[str_id]
            )

        # Convert to numpy array
        embeddings_matrix = np.array(valid_embeddings, dtype="float32")
        dimension = embeddings_matrix.shape[1]

        # Initialize and populate vector database
        self.vector_db = VectorDatabase(dimension)
        self.vector_db.add_vectors(embeddings_matrix, metadata_list)

        print(f"Vector DB created with {len(metadata_list)} items")
        print(f"   - Metadata fields: {list(metadata_list[0].keys())}")
        print(f"   - Embedding dimension: {dimension}")

        # Save database
        self.vector_db.save(self.anime_db_index, self.anime_db_metadata)

    def load_vector_database(self):
        """Load existing vector database"""
        if (
            not Path(self.anime_db_index).exists()
            or not Path(self.anime_db_metadata).exists()
        ):
            raise FileNotFoundError(
                "Vector database files not found. Build database first."
            )

        # Load metadata to get dimension
        with open(self.anime_db_metadata, "rb") as f:
            metadata = pickle.load(f)

        # Infer dimension from saved index
        import faiss

        temp_index = faiss.read_index(self.anime_db_index)
        dimension = temp_index.d

        self.vector_db = VectorDatabase(dimension)
        self.vector_db.load(self.anime_db_index, self.anime_db_metadata)

        print(
            f"Vector database loaded: {len(self.vector_db.metadata)} items, dim={dimension}"
        )

    def encode_by_id(self, anime_id: Union[str, int]) -> np.ndarray:
        """
        Encode an anime by its ID from the dataset

        Args:
            anime_id: The ID of the anime to encode

        Returns:
            Fused embedding vector
        """
        # Load dataset
        df = self._load_dataset()
        # Identify ID column
        id_col = (
            "id"
            if "id" in df.columns
            else ("anime_id" if "anime_id" in df.columns else None)
        )
        if id_col is None:
            raise ValueError(f"Dataset must contain 'id' or 'anime_id'")
        # Find the row
        row = df[df[id_col] == anime_id]
        if row.empty:
            raise ValueError(f"Anime with ID {anime_id} not found in dataset")
        row_data = row.iloc[0].to_dict()
        # Encode using the row data
        return self.encode_from_data(row_data, anime_id=anime_id)

    def encode_from_data(
        self,
        data: Dict,
        anime_id: Optional[Union[str, int]] = None,
        image_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode anime data into a fused embedding vector

        Args:
            data: Dictionary containing anime information (must include fields for synopsis and tabular encoding)
            anime_id: Optional anime ID (if not provided, will look for 'id' or 'anime_id' in data)
            image_path: Optional path to anime poster image (if not provided, will try to find in image_dir)

        Returns:
            Fused embedding vector that can be used for search

        Example:
            >>> data = {
            ...     'title': 'New Anime',
            ...     'genre': 'Action, Fantasy',
            ...     'sypnopsis': 'An epic adventure...',
            ...     'episodes': 24,
            ...     'rating': 8.5
            ... }
            >>> embedding = indexer.encode_from_data(data, image_path='path/to/poster.jpg')
            >>> results = indexer.search(embedding, top_k=5)
        """
        self._ensure_encoders_loaded()

        # Determine anime ID
        if anime_id is None:
            anime_id = data.get("id") or data.get("anime_id")
            if anime_id is None:
                anime_id = "temp"  # Use temporary ID if none provided

        # 1. Encode synopsis
        synopsis = data.get("sypnopsis") or data.get("synopsis") or ""
        if not synopsis:
            raise ValueError("Data must contain 'sypnopsis' or 'synopsis' field")

        synopsis_embedding = self.synopsis_encoder.run_model(synopsis)

        # 2. Encode visual (poster image)
        if image_path is None:
            # Try to find image in image directory
            possible_extensions = [".jpg", ".png", ".webp"]
            for ext in possible_extensions:
                potential_path = self.image_dir / f"{anime_id}{ext}"
                if potential_path.exists():
                    image_path = str(potential_path)
                    break

        if image_path is None or not Path(image_path).exists():
            raise ValueError(
                f"Image not found. Provide image_path or place image at {self.image_dir}/{anime_id}.[jpg|png|webp]"
            )

        visual_embedding = self.visual_encoder.run_model(image_path)

        # 3. Encode tabular data
        # Convert data to DataFrame row for tabular encoder
        # df_row = pd.DataFrame([data])
        print(data.get("title"))
        tabular_embedding = self.tabular_encoder.run_model(data.get("title"))

        # 4. Fuse embeddings
        fused_embedding = self._fuse_single_embeddings(
            anime_id, synopsis_embedding, visual_embedding, tabular_embedding
        )

        # self.vector_db.add if save
        return fused_embedding.get("embedding")

    def encode_image(self, image):
        self._ensure_encoders_loaded()
        return self.visual_encoder.run_model(image)

    def encode_sypnopsis(self, text):
        self._ensure_encoders_loaded()
        return self.synopsis_encoder.run_model(text)

    def encode_tabular(self, anime_title):
        self._ensure_encoders_loaded()
        return self.synopsis_encoder.run_model(anime_title)

    def align_embedding(self, embedding, modality):
        # print(embedding)
        aligned = self.fusion_engine.encode_single_modality(embedding, modality)
        return aligned

    def encode_tabular_genre_studio(self, genres=None, studios=None):
        self._ensure_encoders_loaded()
        # Return a dict of embeddings of genres and/or studio inserted
        embeddings = self.tabular_encoder.get_specific_embeddings(
            genres=genres, studios=studios
        )

        return embeddings

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
            raise ValueError(
                "Vector database not loaded. Call build_vector_database() or load_vector_database() first"
            )

        return self.vector_db.search(query_embedding, k=top_k)

    def search_by_id(self, anime_id: Union[str, int], top_k: int = 5) -> List[dict]:
        """
        Find similar anime by encoding an anime ID from the dataset

        Args:
            anime_id: The ID of the anime to use as query
            top_k: Number of similar anime to return

        Returns:
            List of similar anime with metadata and similarity scores
        """
        query_embedding = self.encode_by_id(anime_id)
        return self.search(query_embedding, top_k=top_k)

    def search_by_data(
        self, data: Dict, top_k: int = 5, image_path: Optional[str] = None
    ) -> List[dict]:
        """
        Find similar anime by encoding provided anime data

        Args:
            data: Dictionary containing anime information
            top_k: Number of similar anime to return
            image_path: Optional path to anime poster image

        Returns:
            List of similar anime with metadata and similarity scores
        """
        query_embedding = self.encode_from_data(data, image_path=image_path)
        return self.search(query_embedding, top_k=top_k)

    def get_db_embedding_by_id(self, id):
        return self.vector_db.get_embedding_by_id(id)

    def get_database_info(self) -> dict:
        """Get information about the current vector database"""
        if self.vector_db is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "total_items": len(self.vector_db.metadata),
            "dimension": self.vector_db.dimension,
            "fusion_method": self.fusion_method,
            "fusion_weights": self.fusion_weights,
            "metadata_fields": list(self.vector_db.metadata[0].keys())
            if self.vector_db.metadata
            else [],
        }

    def add_new_anime_to_db(self, data: Dict, image_path: Optional[str] = None):
        """
        Encodes a new anime and adds it permanently to the FAISS index and metadata.
        """
        if self.vector_db is None:
            self.load_vector_database()

        # 1. Generate the fused embedding
        # encode_from_data returns the vector (np.ndarray)
        fused_vector = self.encode_from_data(data, image_path=image_path)

        # print(f"Encoded vector: {fused_vector}")

        # 2. Prepare for FAISS (needs to be 2D array: 1 x dimension)
        embeddings_matrix = np.array([fused_vector], dtype="float32")

        # 3. Prepare metadata
        # Ensure ID is in the metadata for future lookups
        meta = data.copy()
        if "id" not in meta:
            meta["id"] = str(data.get("anime_id", "unknown"))

        # 4. Add to VectorDatabase
        self.vector_db.add_vectors(embeddings_matrix, [meta])

        # 5. Save changes to disk
        self.vector_db.save(self.anime_db_index, self.anime_db_metadata)
        print(
            f"✅ Successfully added '{meta.get('title')}' to the vector database and saved."
        )
