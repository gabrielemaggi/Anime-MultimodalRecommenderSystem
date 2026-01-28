import numpy as np
import os
import json
from pathlib import Path
import pandas as pd
from typing import Optional, List, Tuple
import pickle

from VisualEncoder import *
from SynopsisEncoder import *
from TabularEncoder import *


# Import your existing classes
# from tabular_embedder import TabularEmbedder
# from visual_encoder import VisualEncoder
# from fusion import Fusion

class Indexing:
    def __init__(self):
        self.synopsis_encoder = None
        self.visual_encoder = None
        self.tabular_encoder = None

        self.synopsis_embeddings = None;
        self.visual_embeddings = None;
        self.tabular_embeddings = None;
        self.fused_embeddings = None;

        self.synopsis_path = "./Embeddings/anime_syno_embeddings.json"
        self.visual_path = "./Embeddings/anime_poster_embeddings.json"
        self.tabular_path = "./Embeddings/anime_tabular_embeddings.json"

        self.tabular_model_path = "./Embeddings/anime_tabular_model.model"
        self.tabular_vector_path = "./Embeddings/anime_tabular_embedding.vec"

        self.image_dir = Path("./dataset/images/")
        self.dataset = Path("./AnimeList.csv")


    def calculate_synopsis_embedding(self):
        if not Path(self.synopsis_path).exists():
            print("Extracting synopsis features...")
            self.synopsis_encoder = SynopsisEncoder()
            self.synopsis_embeddings = self.synopsis_encoder.encode(self.dataset)
            self.save(self.synopsis_embeddings, self.synopsis_path)
        else:
            print("Loading existing synopsis embeddings...")
            self.load(self.synopsis_path, type='syn')
        pass

    def calculate_visual_embedding(self):
        if not Path(self.visual_path).exists():
            image_paths = [str(f) for f in self.image_dir.glob("*") if f.suffix.lower() in ('.jpg', '.png', '.webp')]
            if not image_paths:
                print("No images found.")
            else:
                print("Extracting visual features...")
                self.visual_encoder = VisualEncoder(model_size='small')
                self.visual_embeddings = self.visual_encoder.encode(image_paths)
                self.save(self.visual_embeddings, self.visual_path)
        else:
            print("Loading existing visual embeddings...")
            self.embeddings =  self.load(self.visual_path, type='vis')
        pass

    def calculate_tabular_embedding(self):
        if not Path(self.tabular_path).exists():
            print("Extracting tabular features...")
            self.tabular_encoder = TabularEncoder()
            self.tabular_embeddings = self.tabular_encoder.encode(self.dataset)
            self.save(self.tabular_embeddingsh, self.tabular_path)
        else:
            print("Loading existing tabular embeddings...")
            self.tabular_embeddings = self.load(self.tabular_path, type='tab')
        pass


    def calculate_embeddings(self):
        self.calculate_synopsis_embedding()
        self.calculate_visual_embedding()
        self.calculate_tabular_embedding()

    def fuse(self, method='weighted', weights: Optional[List[float]] = None):
        fusion_engine = Fusion(
            self.synopsis_embeddings,
            self.visual_embeddings,
            self.tabular_embeddings
        )

        print(f"Esecuzione fusione con metodo: {method}...")

        if method == 'mean':
            self.fused_embeddings = fusion_engine.mean_fusion()

        elif method == 'concatenate':
            self.fused_embeddings = fusion_engine.concatenate()

        elif method == 'weighted':
            # Se non passi pesi, usa un default bilanciato
            if weights is None:
                weights = [0.4, 0.4, 0.2]  # Esempio: dai meno peso ai dati tabulari
            self.fused_embeddings = fusion_engine.weighted_average_fusion(weights=weights)

        else:
            raise ValueError(f"Metodo di fusione {method} non supportato.")

        print(f"Fusione completata. Shape finale: {self.fused_embeddings.shape}")

        return self.fused_embeddings



    def save(self, embeddings, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=4)

        print(f"Embeddings successfully saved to {path}")

    def load(self, path, type):
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Assign data to the correct attribute based on type
            if type == "syn":
                self.synopsis_embeddings = data
                print(f"Loaded 'syn' embeddings from {path}")
            elif type == "vis":
                self.visual_embeddings = data
                print(f"Loaded 'vis' embeddings from {path}")
            elif type == "tab":
                self.tabular_embeddings = data
                print(f"Loaded 'tab' embeddings from {path}")
            else:
                print(f"Error: Unknown type '{type}'")

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {path}")
        except Exception as e:
            print(f"Error loading file: {e}")
