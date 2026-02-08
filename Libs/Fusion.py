from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class Fusion:
    def __init__(self, anime_embeddings: List[Dict]):
        """
        Initialize with joint embeddings.

        Expected format (flexible):
          Option A (current): [{id1: [[syn], [vis], [tab]]}, {id2: [[syn], [vis], [tab]]}, ...]
          Option B (recommended): {id1: [syn, vis, tab], id2: [syn, vis, tab], ...}
        """
        if not anime_embeddings:
            raise ValueError("anime_embeddings cannot be empty")

        # Normalize input to consistent internal format
        self.item_ids: List[str] = []
        self.synopsis_embeddings: List[np.ndarray] = []
        self.visual_embeddings: List[np.ndarray] = []
        self.tabular_embeddings: List[np.ndarray] = []

        # Handle both list-of-dicts and flat dict formats
        if isinstance(anime_embeddings, dict):
            items = anime_embeddings.items()
        else:
            items = []
            for item_dict in anime_embeddings:
                if not isinstance(item_dict, dict):
                    raise TypeError(f"Expected dict, got {type(item_dict)}")
                if len(item_dict) != 1:
                    raise ValueError(
                        f"Each dict must have exactly 1 key, got {len(item_dict)}"
                    )
                item_id = next(iter(item_dict.keys()))
                items.append((item_id, item_dict[item_id]))

        for item_id, embeddings in items:
            if len(embeddings) != 3:
                raise ValueError(
                    f"Expected 3 embeddings per item, got {len(embeddings)} for ID {item_id}"
                )

            # Handle double-nested format [[vec]] vs [vec]
            syn = np.array(
                embeddings[0][0]
                if isinstance(embeddings[0], list) and len(embeddings[0]) == 1
                else embeddings[0]
            )
            vis = np.array(
                embeddings[1][0]
                if isinstance(embeddings[1], list) and len(embeddings[1]) == 1
                else embeddings[1]
            )
            tab = np.array(
                embeddings[2][0]
                if isinstance(embeddings[2], list) and len(embeddings[2]) == 1
                else embeddings[2]
            )

            self.item_ids.append(str(item_id))
            self.synopsis_embeddings.append(syn)
            self.visual_embeddings.append(vis)
            self.tabular_embeddings.append(tab)

        # Convert to arrays
        self.synopsis_embeddings = np.array(self.synopsis_embeddings)
        self.visual_embeddings = np.array(self.visual_embeddings)
        self.tabular_embeddings = np.array(self.tabular_embeddings)

        # Build O(1) ID lookup
        self._id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}

        self.n_samples = len(self.item_ids)
        print(f"Fusion initialized with {self.n_samples} samples")
        print(f"  Synopsis shape: {self.synopsis_embeddings.shape}")
        print(f"  Visual shape: {self.visual_embeddings.shape}")
        print(f"  Tabular shape: {self.tabular_embeddings.shape}")

    def _get_embedding_dim(self, emb: np.ndarray) -> int:
        """Safely get feature dimension for 1D or 2D arrays."""
        return emb.shape[1] if emb.ndim == 2 else emb.shape[0]

    def concatenate(
        self, as_list: bool = False
    ) -> Union[Dict[str, np.ndarray], List[Dict]]:
        result = {}
        for i, item_id in enumerate(self.item_ids):
            result[item_id] = np.concatenate(
                [
                    self.synopsis_embeddings[i],
                    self.visual_embeddings[i],
                    self.tabular_embeddings[i],
                ]
            )

        return (
            [{"id": k, "embedding": v} for k, v in result.items()]
            if as_list
            else result
        )

    def mean_fusion(
        self, as_list: bool = False
    ) -> Union[Dict[str, np.ndarray], List[Dict]]:
        dims = [
            self._get_embedding_dim(self.synopsis_embeddings),
            self._get_embedding_dim(self.visual_embeddings),
            self._get_embedding_dim(self.tabular_embeddings),
        ]
        if len(set(dims)) != 1:
            raise ValueError(
                f"Mean fusion requires equal dimensions. Got: synopsis={dims[0]}, "
                f"visual={dims[1]}, tabular={dims[2]}"
            )

        result = {}
        for i, item_id in enumerate(self.item_ids):
            stacked = np.stack(
                [
                    self.synopsis_embeddings[i],
                    self.visual_embeddings[i],
                    self.tabular_embeddings[i],
                ]
            )
            result[item_id] = np.mean(stacked, axis=0)

        return (
            [{"id": k, "embedding": v} for k, v in result.items()]
            if as_list
            else result
        )

    def weighted_average_fusion(
        self, weights: Optional[List[float]] = None, as_list: bool = False
    ) -> Union[Dict[str, np.ndarray], List[Dict]]:
        dims = [
            self._get_embedding_dim(self.synopsis_embeddings),
            self._get_embedding_dim(self.visual_embeddings),
            self._get_embedding_dim(self.tabular_embeddings),
        ]
        if len(set(dims)) != 1:
            raise ValueError(
                f"Weighted fusion requires equal dimensions. Got: synopsis={dims[0]}, "
                f"visual={dims[1]}, tabular={dims[2]}"
            )

        if weights is None:
            weights = [1 / 3, 1 / 3, 1 / 3]
        else:
            if len(weights) != 3:
                raise ValueError(f"Expected 3 weights, got {len(weights)}")
            total = sum(weights)
            if total == 0:
                raise ValueError("Weights sum to zero")
            weights = [w / total for w in weights]

        print(
            f"Using weights: synopsis={weights[0]:.2f}, visual={weights[1]:.2f}, tabular={weights[2]:.2f}"
        )

        result = {}
        for i, item_id in enumerate(self.item_ids):
            result[item_id] = (
                weights[0] * self.synopsis_embeddings[i]
                + weights[1] * self.visual_embeddings[i]
                + weights[2] * self.tabular_embeddings[i]
            )

        return (
            [{"id": k, "embedding": v} for k, v in result.items()]
            if as_list
            else result
        )

    def get_embedding_by_id(self, item_id: str, modality: str = "all") -> np.ndarray:
        if item_id not in self._id_to_idx:
            raise ValueError(
                f"Item ID '{item_id}' not found. Available IDs: {list(self._id_to_idx.keys())[:5]}..."
            )

        idx = self._id_to_idx[item_id]
        if modality == "synopsis":
            return self.synopsis_embeddings[idx]
        elif modality == "visual":
            return self.visual_embeddings[idx]
        elif modality == "tabular":
            return self.tabular_embeddings[idx]
        elif modality == "all":
            return np.concatenate(
                [
                    self.synopsis_embeddings[idx],
                    self.visual_embeddings[idx],
                    self.tabular_embeddings[idx],
                ]
            )
        else:
            raise ValueError(
                f"Unknown modality '{modality}'. Choose from: 'synopsis', 'visual', 'tabular', 'all'"
            )
