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

        if isinstance(self.index, faiss.IndexFlatIP):
                faiss.normalize_L2(embeddings)

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
        print(query_vector)
        query_vector = query_vector.astype('float32').reshape(1, -1)

        # Normalize for cosine similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()
                # Se l'indice è Inner Product e hai normalizzato L2:
                # dist è già la Cosine Similarity (da -1 a 1)
                if isinstance(self.index, faiss.IndexFlatIP):
                    result['similarity'] = float(dist)
                    # La distanza coseno è definita come 1 - similarity
                    result['distance'] = 1.0 - float(dist)
                else:
                    # Se usassi IndexFlatL2, dist sarebbe la distanza euclidea al quadrato
                    result['distance'] = float(dist)
                    result['similarity'] = None # O formula di conversione L2->Sim
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

    def get_embedding_by_id(self, item_id):
        """
        Get embedding vector by ID from metadata
        Args:
            item_id: The ID to search for in metadata
        Returns:
            numpy array of the embedding vector, or None if not found
        """
        # Find the index in metadata list where id matches

        for idx, meta in enumerate(self.metadata):
            if int(meta.get('id')) == item_id:
                # print("Found!")
                # Reconstruct the vector from the FAISS index
                vector = self.index.reconstruct(int(idx))
                return vector
        return None
