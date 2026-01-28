import numpy as np


class Fusion:
    def __init__(self, synopsis_embeddings, poster_embeddings, *args):
        """
        :param synopsis_embeddings: embedding of synopses.
        :param poster_embeddings: embedding of posters.
        :param *args: other useful embeddings.
        """
        if synopsis_embeddings is None or poster_embeddings is None:
            raise ValueError("synopsis or poster embeddings cannot be None")

        self.embedding_store = [np.asarray(synopsis_embeddings), np.asarray(poster_embeddings)]

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
        """
        return np.concatenate(self.embedding_store, axis=1)
