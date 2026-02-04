import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec
from indexing_db import *
from User import *

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

if __name__ == "__main__":
    indexer = Indexing()
    indexer.build_vector_database()

    results = indexer.encode_tabular_genre_studio(genres=['Drama', 'Romance'], studios=['MAPPA'])
    # print(results)

    embedding = results.get('genres').get('Romance')
    # print("aaaaa", embedding)
    query = indexer.align_embedding(embedding, modality='tab')

    indexer.load_vector_database()
    u = User("MrPeanut02")
    u.debug_plot_watchlist()
    u.findCentersOfClusters()
    u.add_filtering(query, "move")

    print("---"*40)
    print(u.get_nearest_anime_from_clusters(indexer, 10))

    # similars = indexer.search(query)
    # print(similars)
