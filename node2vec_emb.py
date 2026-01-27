import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec


# 1. Load the CSV
# Make sure your CSV has columns: 'title', 'genre', 'studio'
df = pd.read_csv('AnimeList.csv')

# 2. Preprocessing
# Fill NaNs to prevent errors.
# NOTE: Check if your CSV column is named 'genre' or 'genres'
df['genre'] = df['genre'].fillna('')
df['studio'] = df['studio'].fillna('')

# 3. Initialize the Graph
G = nx.Graph()

print("Building graph...")
"""
# 4. Iterate through the CSV rows
for index, row in df.iterrows():
    # --- A. Create the Anime Node ---
    # We use the Title as the unique identifier
    anime_node = f"Anime_{row['title']}"
    G.add_node(anime_node, type='anime', title=row['title'])

    # --- B. Process Genres ---
    # Assuming the column in CSV is named 'genre'
    # "Action, Shounen" -> ["Action", "Shounen"]
    if row['genre']:
        genres_list = [g.strip() for g in row['genre'].split(',')]

        for genre in genres_list:
            if genre:
                genre_node = f"Genre_{genre}"
                # Link Anime to Genre
                G.add_edge(anime_node, genre_node)

    # --- C. Process Studio ---
    studio = row['studio'].strip()
    if studio:
        studio_node = f"Studio_{studio}"
        # Link Anime to Studio
        G.add_edge(anime_node, studio_node)
"""
# Define Base Weights
BASE_GENRE_WEIGHT = 1.0
BASE_STUDIO_WEIGHT = 2.0

for index, row in df.iterrows():

    anime_node = f"Anime_{row['title']}"
    G.add_node(anime_node, type='anime', title=row['title'])

    # 1. Get the Score (Handle missing data)
    try:
        # Fill NaN with 5.0 (neutral score)
        score = float(row.get('score', 5.0))
        score_by = float(row.get('scored_by', 1.0))

    except:
        score = 5.0
        score_by = 1.0

    # 2. Calculate a "Quality Multiplier"
    # quality_multiplier = (score / 10.0) + 0.1

    # A. Normalize Score (0.1 to 1.1)
    score_factor = (score / 10.0) + 0.1
    popularity_factor = np.log10(score_by + 10.0)
    quality_multiplier = score_factor * popularity_factor

    # --- Process Genres ---
    if row['genre']:
        genres_list = [g.strip() for g in row['genre'].split(',')]
        for genre in genres_list:
            if genre:
                genre_node = f"Genre_{genre}"

                # Apply the multiplier
                final_weight = BASE_GENRE_WEIGHT * quality_multiplier

                G.add_edge(anime_node, genre_node, weight=final_weight)

    # --- Process Studio ---
    studio = row['studio'].strip()
    if studio:
        studio_node = f"Studio_{studio}"

        # Apply the multiplier
        final_weight = BASE_STUDIO_WEIGHT * quality_multiplier

        G.add_edge(anime_node, studio_node, weight=final_weight)

print(f"Graph Construction Complete!")
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")

# ---------------------------------------------------------
# 5. TRAIN THE MODEL (This step was missing in your snippet)
# ---------------------------------------------------------
print("Training Node2Vec model... (This might take a moment)")

# dimensions: size of the vector (64 or 128 is usually good for this size)
# walk_length: how deep the random walk goes
# num_walks: how many times it walks from each node
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=20, workers=12, weight_key='weight')

# Fit the model (train it)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

print("Model Trained!")
model.save("anime_node2vec_weighted.model")
model.wv.save_word2vec_format("anime_embeddings_node2vec_weighted.vec")
# model.save("anime_node2vec.pt")
# ---------------------------------------------------------
# 6. Recommendation Function (Updated: No Author)
# ---------------------------------------------------------
