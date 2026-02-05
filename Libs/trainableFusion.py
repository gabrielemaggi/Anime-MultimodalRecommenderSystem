import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CoMMFusion(nn.Module):
    def __init__(self, syn_dim, vis_dim, tab_dim, output_dim=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- Encoders ---
        self.enc_syn = nn.Sequential(
            nn.Linear(syn_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)
        )
        self.enc_vis = nn.Sequential(
            nn.Linear(vis_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)
        )
        self.enc_tab = nn.Sequential(
            nn.Linear(tab_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)
        )
        # --- Learnable Modality Importance Weights ---
        # We initialize these to 0.0 so that exp(0) = 1.0
        # This means all modalities start as equally important.
        self.weight_syn = nn.Parameter(torch.zeros([]))
        self.weight_vis = nn.Parameter(torch.zeros([]))
        self.weight_tab = nn.Parameter(torch.zeros([]))
        # --- Learnable Logit Scale (Temperature) ---
        # We initialize to log(1/0.07) approx 2.65
        # This acts as the inverse temperature for the Loss function.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.output_dim = output_dim

    def forward(self, syn, vis, tab, return_centroid=False):
        # 1. Project and normalize individual embeddings
        z_s = F.normalize(self.enc_syn(syn), p=2, dim=-1)
        z_v = F.normalize(self.enc_vis(vis), p=2, dim=-1)
        z_t = F.normalize(self.enc_tab(tab), p=2, dim=-1)

        # 2. Calculate Importance Weights
        # exp() ensures weights are always positive
        w_s = torch.exp(self.weight_syn)
        w_v = torch.exp(self.weight_vis)
        w_t = torch.exp(self.weight_tab)

        # 3. Create Weighted Centroid
        # If the model learns that 'vis' is noisy, w_v will decrease.
        weighted_sum = (w_s * z_s) + (w_v * z_v) + (w_t * z_t)

        # 4. Normalize the Centroid
        centroid = F.normalize(weighted_sum, p=2, dim=-1)

        if return_centroid:
            return z_s, z_v, z_t, centroid

        return centroid

    def comm_loss(self, z_s, z_v, z_t, centroid):
        """
        CoMM Alignment Loss using Learnable Temperature.
        """
        batch_size = z_s.size(0)
        labels = torch.arange(batch_size, device=z_s.device)

        # 1. Recover the dynamic temperature
        # exp(logit_scale) is equivalent to (1 / temperature)
        # We clamp it to max 100 to prevent numerical instability (standard in CLIP)
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # 2. Calculate Similarities scaled by temperature
        # Math: similarity * (1/0.07)
        logits_s = torch.matmul(z_s, centroid.T) * logit_scale
        logits_v = torch.matmul(z_v, centroid.T) * logit_scale
        logits_t = torch.matmul(z_t, centroid.T) * logit_scale

        # 3. Multi-modal Cross Entropy
        loss_s = F.cross_entropy(logits_s, labels)
        loss_v = F.cross_entropy(logits_v, labels)
        loss_t = F.cross_entropy(logits_t, labels)

        return (loss_s + loss_v + loss_t) / 3


class FusionTrainer:
    def __init__(
        self,
        item_ids=None,
        syn_embs=None,
        vis_embs=None,
        tab_embs=None,
        output_dim=384,
        load_model=False,
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = "./Embeddings/fusion_model.pt"

        # Handle load_model case (model-only initialization)
        if load_model:
            self.model = CoMMFusion(384, 384, 384, 384)
            self.model = self.load(self.save_path, self.device)
            # Initialize embeddings as None - they can be set later if needed
            self.syn_embs = syn_embs.astype("float32") if syn_embs is not None else None
            self.vis_embs = vis_embs.astype("float32") if vis_embs is not None else None
            self.tab_embs = tab_embs.astype("float32") if tab_embs is not None else None
            self.item_ids = item_ids

        else:
            # Training mode - require embeddings
            if syn_embs is None or vis_embs is None or tab_embs is None:
                raise ValueError(
                    "When load_model=False, syn_embs, vis_embs, and tab_embs are required for training"
                )

            self.syn_embs = syn_embs.astype("float32")
            self.vis_embs = vis_embs.astype("float32")
            self.tab_embs = tab_embs.astype("float32")
            self.item_ids = item_ids

            self.model = CoMMFusion(
                syn_embs.shape[1], vis_embs.shape[1], tab_embs.shape[1], output_dim
            ).to(self.device)

    def train_2(self, epochs=50, batch_size=128):
        if os.path.exists(self.save_path):
            self.load(self.save_path)
            return self.model

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )
        loader = DataLoader(
            TensorDataset(self.syn_embs, self.vis_embs, self.tab_embs),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for syn, vis, tab in loader:
                syn, vis, tab = (
                    syn.to(self.device),
                    vis.to(self.device),
                    tab.to(self.device),
                )

                # Get individual views and the shared centroid
                z_s, z_v, z_t, centroid = self.model(
                    syn, vis, tab, return_centroid=True
                )

                # Calculate CoMM loss
                loss = self.model.comm_loss(z_s, z_v, z_t, centroid)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1} | CoMM Loss: {total_loss / len(loader):.4f}")

        self.save(self.save_path)

    def train(self, epochs=500, batch_size=2048, patience=15, min_delta=1e-4):
        if os.path.exists(self.save_path):
            self.load(self.save_path)
            return self.model

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )

        # --- Early Stopping Setup ---
        best_loss = float("inf")
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # Note: Ideally, use a separate validation loader here.
        # For this example, we will monitor the training loss trend.
        loader = DataLoader(
            TensorDataset(self.syn_embs, self.vis_embs, self.tab_embs),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for syn, vis, tab in loader:
                syn, vis, tab = (
                    syn.to(self.device),
                    vis.to(self.device),
                    tab.to(self.device),
                )

                z_s, z_v, z_t, centroid = self.model(
                    syn, vis, tab, return_centroid=True
                )
                loss = self.model.comm_loss(z_s, z_v, z_t, centroid)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1} | CoMM Loss: {avg_loss:.4f}")

            # --- Early Stopping Logic ---
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    # Load the best weights before exiting
                    self.model.load_state_dict(best_model_wts)
                    break

        self.save(self.save_path)

    def transform(
        self, syn_embs=None, vis_embs=None, tab_embs=None, as_list: bool = False
    ):
        """
        Fuse embeddings using trained model.
        Returns SAME STRUCTURE as Fusion class:
          - as_list=False: {id1: embedding1, id2: embedding2, ...}
          - as_list=True:  [{'id': id1, 'embedding': embedding1}, ...]
        """
        if syn_embs is None:  # Transform all training data
            syn_embs, vis_embs, tab_embs = self.syn_embs, self.vis_embs, self.tab_embs
            ids = self.item_ids  # Use stored IDs
        else:
            ids = ["query_temp"]

        self.model.eval()
        with torch.no_grad():
            syn = torch.from_numpy(syn_embs.astype("float32")).to(self.device)
            vis = torch.from_numpy(vis_embs.astype("float32")).to(self.device)
            tab = torch.from_numpy(tab_embs.astype("float32")).to(self.device)
            fused = self.model(syn, vis, tab).cpu().numpy()

        # RETURN STRUCTURED OUTPUT (MATCHES Fusion CLASS)
        if isinstance(ids, int):
            print({ids: fused})
            return {"id": ids, "embedding": fused}
        else:
            result = {id_: fused[i] for i, id_ in enumerate(ids)}
            out = (
                [{"id": k, "embedding": v} for k, v in result.items()]
                if as_list
                else result
            )
            print(out)
            return out

    def save(self, path: str = "fusion_model.pt"):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "dims": {
                    "syn_dim": self.syn_embs.shape[1],
                    "vis_dim": self.vis_embs.shape[1],
                    "tab_dim": self.tab_embs.shape[1],
                    "output_dim": self.model.output_dim,
                },
                "item_ids": self.item_ids,  # ← SAVE IDs FOR LATER USE
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)

        self.model.eval()
        return self.model

    def encode_single_modality(self, embedding, modality_type="syn"):
        """
        Encode a single modality embedding through its respective encoder.
        Returns the normalized, aligned embedding in the shared space (NOT the centroid).

        Args:
            embedding: numpy array or torch tensor of shape (embedding_dim,) or (batch_size, embedding_dim)
            modality_type: str, one of ['syn', 'vis', 'tab']

        Returns:
            numpy array of the encoded embedding in the shared space
        """
        self.model.eval()

        # Convert to tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding.astype("float32"))

        # Add batch dimension if needed
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.device)

        with torch.no_grad():
            # Select the appropriate encoder
            if modality_type == "syn":
                encoded = F.normalize(self.model.enc_syn(embedding), p=2, dim=-1)
            elif modality_type == "vis":
                encoded = F.normalize(self.model.enc_vis(embedding), p=2, dim=-1)
            elif modality_type == "tab":
                encoded = F.normalize(self.model.enc_tab(embedding), p=2, dim=-1)
            else:
                raise ValueError(
                    f"Invalid modality_type: {modality_type}. Must be 'syn', 'vis', or 'tab'"
                )

        return encoded.cpu().numpy().squeeze()


class TensorDataset(Dataset):
    def __init__(self, syn, vis, tab):
        self.syn, self.vis, self.tab = syn, vis, tab

    def __len__(self):
        return len(self.syn)

    def __getitem__(self, i):
        return self.syn[i], self.vis[i], self.tab[i]
