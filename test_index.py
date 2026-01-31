from indexing import *
import numpy as np

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
    shape = getattr(sample_val, 'shape', len(sample_val))
    dtype = getattr(sample_val, 'dtype', type(sample_val[0]))

    print(f"Sample ID:    {sample_key}")
    print(f"Vector Shape: {shape}")
    print(f"Data Type:    {dtype}")

    # Show a snippet of the actual values
    snippet = sample_val[:top_n]
    print(f"Vector Preview (first {top_n}): {snippet}")
    print("----------------------------------\n")


if __name__ == "__main__":
    indexer = Indexing()

    print("--- Starting Embedding Pipeline ---")

    indexer.calculate_embeddings()
    print(f"The type of fused_embeddings is: {type(indexer.fused_embeddings)}")
    log_embeddings_info(indexer.fused_embeddings, sample_key='32785')

    similars = indexer.search_similar_anime(indexer.fused_embeddings.get('32785'))

    print(similars)
