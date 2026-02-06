import numpy as np

from Libs.indexing_db import *


def log_embeddings_info(embeddings, sample_key=None, top_n=5):
    """
    Logs metadata and a snippet of the embeddings dictionary.
    """
    print("\n--- [Embedding Inspection Log] ---")

    if not isinstance(embeddings, dict):
        print(f"Error: Expected dict, got {type(embeddings)}")
        return

    total_keys = len(embeddings)
    print(f"Total Entries: {total_keys}")

    if total_keys == 0:
        print("Status: Dictionary is empty.")
        return

    # Grab a sample key if none provided
    if sample_key is None or sample_key not in embeddings:
        sample_key = list(embeddings.keys())[0]

    sample_val = embeddings[sample_key]

    # Handle both numpy arrays and lists
    shape = getattr(sample_val, "shape", len(sample_val))
    dtype = getattr(sample_val, "dtype", type(sample_val[0]))

    print(f"Sample ID:    {sample_key}")
    print(f"Vector Shape: {shape}")
    print(f"Data Type:    {dtype}")

    # Show a snippet of the actual values
    snippet = sample_val[:top_n]
    print(f"Vector Preview (first {top_n}): {snippet}")
    print("----------------------------------\n")


if __name__ == "__main__":
    indexer = Indexing()
    indexer.build_vector_database()
    results = indexer.search_by_id(26055, top_k=5)

    info = indexer.get_database_info()
    print("-" * 50)
    print("results: ", results)
    print("-" * 50)
    print("info: ", info)

    # Test code to add to your script
    # indexer = Indexing()
    indexer.load_vector_database()

    # Get an anime ID that exists in your database
    anime_id = 30  # Use an actual ID from your dataset

    # 1. Get the embedding stored in the database
    db_embedding = indexer.get_db_embedding_by_id(anime_id)
    print(f"DB embedding shape: {db_embedding.shape}")
    print(f"DB embedding (first 5): {db_embedding[:5]}")

    # 2. Encode the same anime fresh
    fresh_embedding = indexer.encode_by_id(anime_id)
    print(f"\nFresh embedding shape: {fresh_embedding.shape}")
    print(f"Fresh embedding (first 5): {fresh_embedding[:5]}")

    # 3. Compare them
    import torch
    import torch.nn.functional as F

    db_tensor = torch.tensor(db_embedding).unsqueeze(0)
    fresh_tensor = torch.tensor(fresh_embedding).unsqueeze(0)
    similarity = F.cosine_similarity(db_tensor, fresh_tensor, dim=1)
    print(f"\nSimilarity: {similarity.item()}")

    # 4. Self-similarity check (should be 1.0)
    self_sim = F.cosine_similarity(db_tensor, db_tensor, dim=1)
    print(f"Self-similarity (should be 1.0): {self_sim.item()}")
