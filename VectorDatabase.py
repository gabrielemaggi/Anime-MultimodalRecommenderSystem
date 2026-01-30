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
        # Or use: faiss.IndexFlatIP(dimension) for cosine similarity

        self.metadata = []  # Store anime IDs/names

    def add_vectors(self, embeddings: np.ndarray, metadata: list):
        """Add vectors to the database"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Search for k nearest neighbors"""
        query_vector = query_vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'metadata': self.metadata[idx],
                    'distance': float(dist)
                })
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
