import numpy as np

class Fusion:
    def __init__(self, synopsis_embeddings, poster_embeddings, tabular_embeddings, *args):
        """
        Initialize the Fusion class with multiple embedding modalities.

        :param synopsis_embeddings: embedding of synopses.
        :param poster_embeddings: embedding of posters.
        :param tabular_embeddings: embedding of tabular features.
        :param *args: other useful embeddings.
        """
        if synopsis_embeddings is None or poster_embeddings is None or tabular_embeddings is None:
            raise ValueError("synopsis or poster or tabular embeddings cannot be None")

        self.embedding_store = [
            np.asarray(synopsis_embeddings),
            np.asarray(poster_embeddings),
            np.asarray(tabular_embeddings)
        ]

        # samples in the embeddings
        n_samples = self.embedding_store[0].shape[0]

        # saving other optional embeddings
        for index, embedding in enumerate(args):
            arr = np.asarray(embedding)
            # check sample consistency
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"Dimension mismatch in optional embedding #{index + 1}: "
                    f"Expected {n_samples} samples, but got {arr.shape[0]}."
                )
            self.embedding_store.append(arr)

    def concatenate(self):
        """
        Concatenates all stored embeddings along feature axes.

        :return: Concatenated embeddings of shape (n_samples, sum_of_all_features)
        """
        return np.concatenate(self.embedding_store, axis=1)

    def mean_fusion(self):
        """
        Performs mean fusion by averaging all embeddings across the feature axis.

        :return: Mean fused embeddings of shape (n_samples, max_feature_dim)

        Note: This method assumes all embeddings have the same feature dimension.
        If they don't, consider using weighted_average_fusion with appropriate weights.
        """
        # Stack embeddings and compute mean across embedding modalities (axis=0)
        stacked = np.stack(self.embedding_store, axis=0)
        return np.mean(stacked, axis=0)

    def weighted_average_fusion(self, weights=None):
        """
        Performs weighted average fusion of all embeddings.

        :param weights: Array of weights for each embedding modality.
                       If None, equal weights (1/n_embeddings) are used.
                       Must sum to 1 or will be normalized.
        :return: Weighted average fused embeddings of shape (n_samples, max_feature_dim)

        Example:
            fusion = Fusion(synop_emb, poster_emb, tabular_emb)
            # Equal weights
            result = fusion.weighted_average_fusion()
            # Custom weights
            result = fusion.weighted_average_fusion(weights=[0.5, 0.3, 0.2])
        """
        n_embeddings = len(self.embedding_store)

        # Set default equal weights if not provided
        if weights is None:
            weights = np.ones(n_embeddings) / n_embeddings
        else:
            weights = np.asarray(weights)

            # Validate weights length
            if len(weights) != n_embeddings:
                raise ValueError(
                    f"Expected {n_embeddings} weights, but got {len(weights)}."
                )

            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)

        # Compute weighted average
        weighted_sum = np.zeros_like(self.embedding_store[0])
        for embedding, weight in zip(self.embedding_store, weights):
            weighted_sum += weight * embedding

        return weighted_sum
