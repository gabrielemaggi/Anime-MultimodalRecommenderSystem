import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import sys
import pandas as pd
import numpy as np
from SynopsisEncoder import *
from visual_embedding import *


def find_similar(query_embedding, database_embeddings, k=5):
    """
    Finds the top k most similar images from the database using Cosine Similarity.

    Args:
        query_embedding (np.array): Shape (1, dim) - The image you want recommendations for.
        database_embeddings (np.array): Shape (N, dim) - The catalog of all item embeddings.
        k (int): Number of recommendations to return.

    Returns:
        indices (np.array): Indices of the top k items in the database.
        scores (np.array): Similarity scores (0 to 1).
    """
    # Compute Cosine Similarity
    # Result shape: (1, N_database_items)
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), database_embeddings)

    # Get top K indices (sorted descending)
    top_k_indices = similarities[0].argsort()[-k:][::-1]
    top_k_scores = similarities[0][top_k_indices]

    return top_k_indices, top_k_scores


def stats_of_embedding(embeddings):
    print("Stats of first 10 vectors:")
    total_sum = 0
    size = list(embeddings)[1]
    for idx, e in enumerate(embeddings):
        sum = e.sum()
        if(idx < 10):
            print(f"vector {idx}: \n")
            print(f"\t shape of vector: {e.size}")
            print(f"\t vector zeros dimension:{e.size() - e.nonzero()} ")
            print(f"\t vector maximum value: {e.max()}")
            print(f"\t vector minimum value: {e.min()}")
            print(f"\t sum of all the dimension of the vector: {sum}")
        total_sum += sum
    print(f"average sum of dimension of all the vectors : {total_sum/ size}")


if __name__ == "__main__":
    data = pd.read_csv("AnimeList.csv", sep= ',')
    # data_sample = data.head(5)

    output_file = "anime_syno_embeddings.npy"
    encoder = BertEncoder()

    if not(os.path.exists(output_file)):
        column_name = 'sypnopsis'
        data[column_name] = data[column_name].fillna("No description available").astype(str)
        sentences = data[column_name].tolist()
        embeddings = encoder.run_model_batch(sentences, batch_size=64)
    else:
        embeddings = np.load(output_file)

    query_sentence = "Joutarou Kuujou and his allies have finally made it to Egypt, where the immortal Dio awaits. Upon their arrival, the group gains a new comrade: Iggy, a mutt who wields the Stand \"The Fool.\" It's not all good news however, as standing in their path is a new group of Stand users who serve Dio, each with a Stand representative of an ancient Egyptian god. As their final battle approaches, it is a race against time to break Joutarou's mother free from her curse and end Dio's reign of terror over the Joestar family once and for all."

    print(query_sentence)
    query = encoder.run_model(query_sentence)

    indices, scores = find_similar(query, embeddings, k=5)

    print("\nTop 5 Recommendations:")
    for idx, score in zip(indices, scores):
        print(f"Item: {data.iloc[idx]['sypnopsis']} | Similarity: {score:.4f}")

    print(f"Data Shape: {data.shape}")
    print(f"Embeddings Shape: {embeddings.shape}")

    np.save(output_file, embeddings)
