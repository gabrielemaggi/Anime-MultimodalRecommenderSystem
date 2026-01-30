import faiss
import numpy as np
import pickle

class VectorDatabase:
    def __init__(self, dimension: int, distance="cosine"):
        self.dimension = dimension
        if distance == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.metadata = []  # NOW stores full metadata dicts (not just IDs)

    def add_vectors(self, embeddings: np.ndarray, metadata: list = None):
        """
        Add vectors to the database (MINIMAL CHANGE: accepts np array + metadata list)

        Args:
            embeddings: numpy array of shape (n, dimension)
            metadata: list of dicts with length n (optional; if None, uses indices as IDs)
        """
        embeddings = embeddings.astype('float32')

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings)

        # Store metadata if provided, else store simple IDs
        if metadata is not None:
            if len(metadata) != embeddings.shape[0]:
                raise ValueError("Metadata length must match embeddings count")
            self.metadata.extend(metadata)
        else:
            # Backward compatibility: store indices as IDs
            start_idx = len(self.metadata)
            self.metadata.extend([{"id": i} for i in range(start_idx, start_idx + embeddings.shape[0])])

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Search for k nearest neighbors (returns full metadata)"""
        query_vector = query_vector.astype('float32').reshape(1, -1)

        # Normalize for cosine similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()  # Return full metadata dict
                result['distance'] = float(dist)
                result['similarity'] = float(1 - dist/2) if isinstance(self.index, faiss.IndexFlatIP) else None
                results.append(result)
        return results

    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
