import numpy as np
from typing import Dict, List, Optional

class Fusion:
    def __init__(self, anime_embeddings: List[Dict]):
        """
        Initialize the Fusion class with joint embeddings.

        :param anime_embeddings: List of dicts with structure:
                                 [{id: [[syn_vec], [vis_vec], [tab_vec]]}, ...]
        """
        if anime_embeddings is None or len(anime_embeddings) == 0:
            raise ValueError("anime_embeddings cannot be None or empty")

        self.anime_embeddings = anime_embeddings
        self.n_samples = len(anime_embeddings)

        # Get dimensions for validation
        first_item = list(anime_embeddings[0].values())[0]
        self.syn_dim = len(first_item[0][0])
        self.vis_dim = len(first_item[1][0])
        self.tab_dim = len(first_item[2][0])

        print(f"Fusion initialized with {self.n_samples} samples")
        print(f"  Synopsis dim: {self.syn_dim}")
        print(f"  Visual dim: {self.vis_dim}")
        print(f"  Tabular dim: {self.tab_dim}")

    def concatenate(self) -> Dict[str, np.ndarray]:
        """
        Concatenates all embeddings along feature axis.
        For each ID: concatenate [syn_vec] + [vis_vec] + [tab_vec]

        :return: Dict mapping item_id -> concatenated embedding vector
        """
        result = {}

        for item_dict in self.anime_embeddings:
            # Get the ID and embeddings
            item_id = list(item_dict.keys())[0]
            embeddings = item_dict[item_id]  # [[syn], [vis], [tab]]

            # Extract vectors and concatenate
            syn_vec = np.array(embeddings[0][0])  # From [[syn_vec]]
            vis_vec = np.array(embeddings[1][0])  # From [[vis_vec]]
            tab_vec = np.array(embeddings[2][0])  # From [[tab_vec]]

            # Concatenate all three
            concatenated = np.concatenate([syn_vec, vis_vec, tab_vec])
            result[item_id] = concatenated

        output_dim = self.syn_dim + self.vis_dim + self.tab_dim
        print(f"Concatenation complete. Output dim: {output_dim}")
        return result

    def mean_fusion(self) -> Dict[str, np.ndarray]:
        """
        Performs mean fusion by averaging all embeddings.
        Note: Only works if all embeddings have the same dimension.

        :return: Dict mapping item_id -> mean fused embedding vector
        """
        # Check if dimensions are compatible
        if not (self.syn_dim == self.vis_dim == self.tab_dim):
            raise ValueError(
                f"Mean fusion requires all embeddings to have same dimension. "
                f"Got: Synopsis={self.syn_dim}, Visual={self.vis_dim}, Tabular={self.tab_dim}. "
                f"Use weighted_average_fusion or concatenate instead."
            )

        result = {}

        for item_dict in self.anime_embeddings:
            # Get the ID and embeddings
            item_id = list(item_dict.keys())[0]
            embeddings = item_dict[item_id]  # [[syn], [vis], [tab]]

            # Extract vectors
            syn_vec = np.array(embeddings[0][0])
            vis_vec = np.array(embeddings[1][0])
            tab_vec = np.array(embeddings[2][0])

            # Stack and compute mean
            stacked = np.stack([syn_vec, vis_vec, tab_vec])
            result[item_id] = np.mean(stacked, axis=0)

        print(f"Mean fusion complete. Output dim: {self.syn_dim}")
        return result

    def weighted_average_fusion(self, weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Performs weighted average fusion of all embeddings.
        Note: Only works if all embeddings have the same dimension.

        :param weights: List of weights [syn_weight, vis_weight, tab_weight].
                       If None, equal weights [1/3, 1/3, 1/3] are used.
                       Will be normalized to sum to 1.
        :return: Dict mapping item_id -> weighted average embedding vector
        """
        # Check if dimensions are compatible
        if not (self.syn_dim == self.vis_dim == self.tab_dim):
            raise ValueError(
                f"Weighted fusion requires all embeddings to have same dimension. "
                f"Got: Synopsis={self.syn_dim}, Visual={self.vis_dim}, Tabular={self.tab_dim}. "
                f"Use concatenate instead."
            )

        # Set default equal weights if not provided
        if weights is None:
            weights = np.array([1/3, 1/3, 1/3])
        else:
            weights = np.asarray(weights)
            if len(weights) != 3:
                raise ValueError(f"Expected 3 weights, but got {len(weights)}.")
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)

        print(f"Using weights: Synopsis={weights[0]:.3f}, Visual={weights[1]:.3f}, Tabular={weights[2]:.3f}")

        result = {}

        for item_dict in self.anime_embeddings:
            # Get the ID and embeddings
            item_id = list(item_dict.keys())[0]
            embeddings = item_dict[item_id]  # [[syn], [vis], [tab]]

            # Extract vectors
            syn_vec = np.array(embeddings[0][0])
            vis_vec = np.array(embeddings[1][0])
            tab_vec = np.array(embeddings[2][0])

            # Weighted sum
            weighted_sum = (
                weights[0] * syn_vec +
                weights[1] * vis_vec +
                weights[2] * tab_vec
            )
            result[item_id] = weighted_sum

        print(f"Weighted fusion complete. Output dim: {self.syn_dim}")
        return result

    def get_embedding_by_id(self, item_id: str, modality: str = 'all') -> np.ndarray:
        """
        Retrieve embedding for a specific item directly from anime_embeddings.

        :param item_id: The ID of the item
        :param modality: 'synopsis', 'visual', 'tabular', or 'all' (concatenated)
        :return: The requested embedding vector
        """
        # Find the item in anime_embeddings
        item_embeddings = None
        for item_dict in self.anime_embeddings:
            if item_id in item_dict:
                item_embeddings = item_dict[item_id]
                break

        if item_embeddings is None:
            raise ValueError(f"Item ID {item_id} not found in embeddings")

        # Extract based on modality
        if modality == 'synopsis':
            return np.array(item_embeddings[0][0])
        elif modality == 'visual':
            return np.array(item_embeddings[1][0])
        elif modality == 'tabular':
            return np.array(item_embeddings[2][0])
        elif modality == 'all':
            syn_vec = np.array(item_embeddings[0][0])
            vis_vec = np.array(item_embeddings[1][0])
            tab_vec = np.array(item_embeddings[2][0])
            return np.concatenate([syn_vec, vis_vec, tab_vec])
        else:
            raise ValueError(f"Unknown modality: {modality}. Use 'synopsis', 'visual', 'tabular', or 'all'")

    def get_all_ids(self) -> List[str]:
        """
        Get list of all item IDs in the embeddings.

        :return: List of item IDs
        """
        return [list(item_dict.keys())[0] for item_dict in self.anime_embeddings]
