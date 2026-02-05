import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec


def get_anime_vector(model, genres, studio=None, title=None):
    """
    Creates a single vector by averaging the vectors of
    Genres + Studio (No Author).
    """
    vectors = []

    # 1. Collect Genre Vectors
    for genre in genres:
        node_name = f"Genre_{genre}"
        if node_name in model.wv:
            vectors.append(model.wv[node_name])
        else:
            print(f"Warning: Genre '{genre}' not found in graph.")

    # 2. Collect Studio Vector
    if studio:
        node_name = f"Studio_{studio}"
        if node_name in model.wv:
            vectors.append(model.wv[node_name])
        else:
            print(f"Warning: Studio '{studio}' not found in graph.")

    if title:
        node_name = f"Anime_{title}"
        if node_name in model.wv:
            vectors.append(model.wv[node_name])
        else:
            print(f"Warning: Title '{title}' not found in graph.")

    # 3. Handle Edge Case
    if not vectors:
        return np.zeros(model.vector_size)

    # 4. Average them (Mean Pooling)
    final_embedding = np.mean(vectors, axis=0)
    return final_embedding

# ---------------------------------------------------------
# 7. USAGE EXAMPLE
# ---------------------------------------------------------
model = Word2Vec.load("./Embeddings/anime_node2vec_weighted.model")

# Let's search for a mix: "Action" + "Fantasy" made by "Kyoto Animation"
search_genres = ["Romance", "Fantasy", "Action"]
search_studio = "MAPPA"
search_title = "JoJo no Kimyou na Bouken: Stardust Crusaders 2nd Season"

# Get the vector representation of this hypothetical anime
query_vector = get_anime_vector(model, search_genres, search_studio, search_title)

# Find closest existing nodes
similar_nodes = model.wv.most_similar(positive=[query_vector], topn=10)

print(f"\nRecommendations for {search_studio} ({search_genres}):")
for node_name, score in similar_nodes:
    # Filter to ensure we only recommend Anime (not other studios/genres)
    if node_name.startswith("Anime_"):
        clean_name = node_name.replace("Anime_", "")
        print(f"- {clean_name} (Score: {score:.3f})")
