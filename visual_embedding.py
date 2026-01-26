import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class DinoRecommender:
    def __init__(self, model_size='small', device=None):

        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading DINOv2 ({model_size}) on {self.device}...")

        # Map simple names to official DINOv2 repository names
        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }

        if model_size not in model_map:
            raise ValueError(
                f"Invalid model_size. Choose from: {list(model_map.keys())}")

        # Load model from PyTorch Hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_map[model_size])
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (essential for deterministic output)

        # Define the standardized DINOv2 preprocessing pipeline
        # DINOv2 expects images to be multiples of the patch size (14), usually 224x224 or 518x518
        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image_path):
        """
        Loads and transforms an image for the model.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(
                0)  # Add batch dimension
            return img_tensor
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_features(self, image_paths, batch_size=32):
        """
        Extracts embeddings for a list of image paths.

        Returns:
            np.ndarray: Matrix of shape (n_images, embedding_dim)
        """
        features_list = []

        # Process in batches to manage memory
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            batch_tensors = []
            valid_indices = []

            for idx, path in enumerate(batch_paths):
                tensor = self.preprocess_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    valid_indices.append(idx)

            if not batch_tensors:
                continue

            # Stack tensors: (Batch_Size, 3, 224, 224)
            batch_input = torch.cat(batch_tensors).to(self.device)

            with torch.no_grad():
                # DINOv2 forward pass
                # .cpu() moves data back to RAM, .numpy() converts to array
                batch_features = self.model(batch_input).cpu().numpy()

            features_list.append(batch_features)

            print(
                f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")

        if not features_list:
            return np.array([])

        return np.vstack(features_list)


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


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Recommender
    recommender = DinoRecommender(model_size='small')

    # 2. Database of n_images
    folder_path = Path("./dataset/images/")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    catalog_images = [
        str(f) for f in folder_path.iterdir()
        if f.suffix.lower() in valid_extensions
    ]

    if not catalog_images:
        print(f"Nessuna immagine trovata in {folder_path}")
    else:
        print(f"Trovate {len(catalog_images)} immagini.")

        print("\n--- Extracting Catalog Features ---")
        catalog_embeddings = recommender.extract_features(catalog_images)

        if len(catalog_embeddings) > 0:
            # (5, 384) for small model
            print(f"Extracted features shape: {catalog_embeddings.shape}")
            print(catalog_embeddings)
            # 4. Run a Recommendation (Online Step)
            query_image = "./dataset/images/26055.jpg"
            print(f"\n--- Processing Query: {query_image} ---")

            query_tensor = recommender.preprocess_image(query_image)
            if query_tensor is not None:
                # Extract single feature
                with torch.no_grad():
                    query_embedding = recommender.model(
                        query_tensor.to(recommender.device)).cpu().numpy()

                # Find recommendations
                indices, scores = find_similar(
                    query_embedding, catalog_embeddings, k=3)

                print("\nTop 3 Recommendations:")
                for idx, score in zip(indices, scores):
                    print(
                        f"Item: {catalog_images[idx]} | Similarity: {score:.4f}")
