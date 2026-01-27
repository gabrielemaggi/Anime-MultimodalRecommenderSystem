import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class BertEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def getModel(self):
        return self.model


    def run_model_batch(self, sentences, batch_size=32):
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

        # Stack the list of arrays into a single large matrix
        return np.vstack(features_list)

    def run_model(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
