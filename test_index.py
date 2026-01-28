if __name__ == "__main__":
    # 1. Initialize the Indexing Engine
    indexer = Indexing()

    print("--- Starting Embedding Pipeline ---")

    # 2. Calculate individual embeddings (Synopsis, Visual, and Tabular)
    # This will check for existing .json files before running heavy computations
    indexer.calculate_embeddings()
