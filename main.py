import os
from pathlib import Path

import numpy as np
import pandas as pd

# Environment setup
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from SynopsisEncoder import *
from visual_embedding import *


# --- FUNCTIONS ---
def get_synopsis_embeddings(csv_path, output_file):
    """Loads existing BERT embeddings or generates them from CSV."""
    if os.path.exists(output_file):
        print(f"Found existing synopsis embeddings at {output_file}. Loading...")
        return np.load(output_file)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Cannot generate embeddings.")
        return None

    print(f"Generating new synopsis embeddings from {csv_path}...")
    data = pd.read_csv(csv_path)
    encoder = BertEncoder()

    sentences = (
        data["sypnopsis"].fillna("No description available").astype(str).tolist()
    )
    embeddings = encoder.run_model_batch(sentences, batch_size=64)

    np.save(output_file, embeddings)
    return embeddings

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

#checking cosine simularite between the geners in order to check if bert is a good option to embed ganers
def check_ganer_by_sBert():
    ganers = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Slice of Life", "Romance", "Sci-Fi", "Shounen",
              "Seinen", "Shoujo", "Josei", "Isekai", "Mecha", "Psychological", "Horror", "Mystery", "Sports",
              "Supernatural", "Historical", "Music", "Thriller", "School", "Ecchi", "Harem", "Military", "Space",
              "Parody", "Samurai", "Magic", "Demons", "Vampire", "Police", "Martial Arts", "Game"]
    encoder = BertEncoder()
    embeddings = encoder.run_model_batch(ganers, batch_size=64)
    similarities = encoder.model.similarity(embeddings, embeddings)
    similarity_df = pd.DataFrame(similarities, columns=ganers, index=ganers)
    similarity_df.to_csv("similarity_matrix.csv")

def get_poster_embeddings(image_dir, output_file):
    """Loads existing DINO embeddings or generates them from image folder."""
    if os.path.exists(output_file):
        print(f"Found existing poster embeddings at {output_file}. Loading...")
        return np.load(output_file)

    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} not found. Cannot generate embeddings.")
        return None

    print(f"Generating new poster embeddings from {image_dir}...")
    dino = DinoRecommender(model_size="small")
    folder_path = Path(image_dir)
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    catalog_images = [
        str(f) for f in folder_path.iterdir() if f.suffix.lower() in valid_extensions
    ]

    if not catalog_images:
        print("No valid images found.")
        return None

    embeddings = dino.extract_features(catalog_images)
    np.save(output_file, embeddings)
    return embeddings


def get_recommendations(query_embedding, database_embeddings, data, top_k=5):
    """
    Computes similarity and prints the most similar items from the dataframe.
    """
    # Reshape query to (1, dim) if it's a flat vector
    query_vec = query_embedding.reshape(1, -1)

    # Calculate Cosine Similarity
    # Result: (1, N) where N is number of items in database
    scores = cosine_similarity(query_vec, database_embeddings)[0]

    # Get indices of top_k highest scores
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append(
            {
                "id": idx,
                "score": scores[idx],
                "sypnopsis": data.iloc[idx]["sypnopsis"],
                "title": data.iloc[idx].get(
                    "title", f"Item {idx}"
                ),  # Fallback if title column exists
            }
        )

    return results


# --- MAIN ---

if __name__ == "__main__":
    csv_file = "AnimeList.csv"
    data = pd.read_csv(csv_file)

    # Skip generation if the .npy files exist
    syno_embeddings = get_synopsis_embeddings(
        csv_path="AnimeList.csv", output_file="anime_syno_embeddings.npy"
    )

    poster_embeddings = get_poster_embeddings(
        image_dir="./dataset/images/", output_file="anime_poster_embeddings.npy"
    )

    print("\nProcessing complete.")
    if syno_embeddings is not None:
        query_vec = syno_embeddings[0]

        recommendations = get_recommendations(query_vec, syno_embeddings, data, top_k=5)


        print("\n--- Results ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [ID: {rec['id']}].")
            print(f"   Score: {rec['score']:.4f}].")
            print(f"   Synopsis: {rec['sypnopsis']}.\n")
            print(f"   Title: {rec['title']}.\n")
