import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from Encoder import *

class SynopsisEncoder(Encoder):

    def __init__(self, sbert_size="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(sbert_size)

    def encode(self, csv_path, batch_size=64):
        data = self.__load(csv_path)
        sentences = (
            data["sypnopsis"].fillna("No description available").astype(str).tolist()
        )
        embeddings = self.__run_model_batch(sentences, batch_size=batch_size)

        return [
            {"id": row_id, "embedding": emb.tolist() if hasattr(emb, "tolist") else emb}
            for row_id, emb in zip(data["id"], embeddings)
        ]

    def __load(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Cannot generate embeddings.")
            return None
        print(f"Generating new synopsis embeddings from {csv_path}...")
        data = pd.read_csv(csv_path)
        return data

    def getModel(self):
        return self.model

    def run_model(self, sentences):
        embedding = self.model.encode(sentences)
        return embedding

    def __run_model_batch(self, sentences, batch_size=32):
        """
        Extracts embeddings for a list of sentences in batches.
        """
        features_list = []
        # Process in batches to manage memory
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            # Filter out None or empty values if necessary to mimic the 'valid_indices' logic
            # This step ensures we don't crash on bad input
            valid_batch = [s for s in batch_sentences if s and isinstance(s, str) and s.strip()]

            if not valid_batch:
                continue

            # SentenceTransformer .encode() handles tokenization, device movement,
            # and no_grad internally. It returns a numpy array by default.
            batch_embeddings = self.model.encode(
                valid_batch,
                batch_size=len(valid_batch),
                show_progress_bar=False,
                convert_to_numpy=True
            )

            features_list.append(batch_embeddings)

            print(f"Processed {min(i + batch_size, len(sentences))}/{len(sentences)} sentences...")

        if not features_list:
            return np.array([])
        return np.vstack(features_list)
