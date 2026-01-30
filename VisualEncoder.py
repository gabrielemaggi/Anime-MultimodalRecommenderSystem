import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from Encoder import *

class VisualEncoder(Encoder):
    def __init__(self, model_size='small', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = self._get_model_name(model_size)

        print(f"Loading {self.model_name} on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_model_name(self, size):
        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        if size not in model_map:
            raise ValueError(f"Invalid size. Choose from: {list(model_map.keys())}")
        return model_map[size]

    def __load(self, image_path):
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

    def encode(self, image_paths, batch_size=32):
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
                    tensor = self.__load(path)
                    if tensor is not None:
                        batch_tensors.append(tensor)
                        valid_indices.append(idx)

                if not batch_tensors:
                    continue
                # Stack tensors: (Batch_Size, 3, 224, 224)
                batch_input = torch.cat(batch_tensors).to(self.device)

                with torch.no_grad():
                    # DINOv2 forward pass
                    batch_features = self.model(batch_input).cpu().numpy()

                features_list.append(batch_features)

                print(
                    f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")

            if not features_list:
                return np.array([])

            full_embeddings = np.vstack(features_list)

            return [
                {os.path.splitext(os.path.basename(path))[0]: emb.tolist()}
                for path, emb in zip(image_paths, full_embeddings)
            ]

            return np.vstack(features_list)


    def run_model(self, images):
        embedding = self.model.encode(images)
        return embedding
