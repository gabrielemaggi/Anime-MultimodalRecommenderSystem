import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import copy

class SimpleFusion(nn.Module):
    """Minimal trainable fusion with InfoNCE loss"""
    def __init__(self, syn_dim: int, vis_dim: int, tab_dim: int, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(syn_dim + vis_dim + tab_dim, self.output_dim)

    def forward(self, syn, vis, tab):
        # 1. L2-normalize each modality (critical!)
        syn = F.normalize(syn, p=2, dim=-1)
        vis = F.normalize(vis, p=2, dim=-1)
        tab = F.normalize(tab, p=2, dim=-1)
        # 2. Concatenate and project
        x = torch.cat([syn, vis, tab], dim=-1)
        x = self.proj(x)
        # 3. L2-normalize final embedding (for cosine similarity)
        return F.normalize(x, p=2, dim=-1)


class CoMMFusion(nn.Module):
    def __init__(self, syn_dim, vis_dim, tab_dim, output_dim=256):
        super().__init__()
        # Independent encoders to map different input sizes to the same shared space
        self.enc_syn = nn.Sequential(nn.Linear(syn_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.enc_vis = nn.Sequential(nn.Linear(vis_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.enc_tab = nn.Sequential(nn.Linear(tab_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.output_dim = output_dim

    def forward(self, syn, vis, tab, return_centroid=False):

        # Project and normalize
        z_s = F.normalize(self.enc_syn(syn), p=2, dim=-1)
        z_v = F.normalize(self.enc_vis(vis), p=2, dim=-1)
        z_t = F.normalize(self.enc_tab(tab), p=2, dim=-1)

        # The "Centroid" is the CoMM representation
        centroid = (z_s + z_v + z_t) # / 3
        centroid = F.normalize(centroid, p=2, dim=-1)

        if return_centroid:
            return z_s, z_v, z_t, centroid
        return centroid



class FusionTrainer:
    def __init__(self, item_ids, syn_embs, vis_embs, tab_embs, output_dim=256, load_model=False):
        self.syn_embs = syn_embs.astype('float32')
        self.vis_embs = vis_embs.astype('float32')
        self.tab_embs = tab_embs.astype('float32')

        self.item_ids = item_ids
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_path = "my_fusion_v1.pt"

        self.model = CoMMFusion(384, 384, 384, 384)

        if load_model:
             self.model = self.load(self.save_path, self.device)
        else:
            self.model = CoMMFusion(
                syn_embs.shape[1],
                vis_embs.shape[1],
                tab_embs.shape[1],
                output_dim
            ).to(self.device)

    def comm_loss(self, z_s, z_v, z_t, centroid, temperature=0.07):
        """
        CoMM Alignment: Each modality is pulled towards the shared centroid.
        """
        batch_size = z_s.size(0)
        labels = torch.arange(batch_size, device=self.device)

        # 1. Similarity of each modality to the shared Centroid
        # We use the logit_scale (temperature) to sharpen the distribution
        logits_s = torch.matmul(z_s, centroid.T) / temperature
        logits_v = torch.matmul(z_v, centroid.T) / temperature
        logits_t = torch.matmul(z_t, centroid.T) / temperature

        # 2. Multi-modal Cross Entropy
        # This enforces that syn[i] identifies centroid[i], vis[i] identifies centroid[i], etc.
        loss_s = F.cross_entropy(logits_s, labels)
        loss_v = F.cross_entropy(logits_v, labels)
        loss_t = F.cross_entropy(logits_t, labels)

        return (loss_s + loss_v + loss_t) / 3

    def train_2(self, epochs=50, batch_size=128):
        if os.path.exists(self.save_path):
            self.load(self.save_path)
            return self.model

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        loader = DataLoader(TensorDataset(self.syn_embs, self.vis_embs, self.tab_embs),
                            batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for syn, vis, tab in loader:
                syn, vis, tab = syn.to(self.device), vis.to(self.device), tab.to(self.device)

                # Get individual views and the shared centroid
                z_s, z_v, z_t, centroid = self.model(syn, vis, tab, return_centroid=True)

                # Calculate CoMM loss
                loss = self.comm_loss(z_s, z_v, z_t, centroid)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} | CoMM Loss: {total_loss/len(loader):.4f}")

        self.save(self.save_path)

    def train(self, epochs=500, batch_size=512, patience=5, min_delta=1e-4):
        if os.path.exists(self.save_path):
            self.load(self.save_path)
            return self.model

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        # --- Early Stopping Setup ---
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # Note: Ideally, use a separate validation loader here.
        # For this example, we will monitor the training loss trend.
        loader = DataLoader(TensorDataset(self.syn_embs, self.vis_embs, self.tab_embs),
                            batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for syn, vis, tab in loader:
                syn, vis, tab = syn.to(self.device), vis.to(self.device), tab.to(self.device)

                z_s, z_v, z_t, centroid = self.model(syn, vis, tab, return_centroid=True)
                loss = self.comm_loss(z_s, z_v, z_t, centroid)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} | CoMM Loss: {avg_loss:.4f}")

            # --- Early Stopping Logic ---
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    # Load the best weights before exiting
                    self.model.load_state_dict(best_model_wts)
                    break

        self.save(self.save_path)

    def transform(self, syn_embs=None, vis_embs=None, tab_embs=None, as_list: bool = False):
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
            ids = ['query_temp']

        self.model.eval()
        with torch.no_grad():
            syn = torch.from_numpy(syn_embs.astype('float32')).to(self.device)
            vis = torch.from_numpy(vis_embs.astype('float32')).to(self.device)
            tab = torch.from_numpy(tab_embs.astype('float32')).to(self.device)
            fused = self.model(syn, vis, tab).cpu().numpy()

        # RETURN STRUCTURED OUTPUT (MATCHES Fusion CLASS)
        if isinstance(ids, int):
            print({ids: fused})
            return {'id': ids, 'embedding': fused}
        else:
            result = {id_: fused[i] for i, id_ in enumerate(ids)}
            out = [{'id': k, 'embedding': v} for k, v in result.items()] if as_list else result
            print(out)
            return out

    def save(self, path: str = "fusion_model.pt"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dims': {
                'syn_dim': self.syn_embs.shape[1],
                'vis_dim': self.vis_embs.shape[1],
                'tab_dim': self.tab_embs.shape[1],
                'output_dim': self.model.output_dim
            },
            'item_ids': self.item_ids  # ← SAVE IDs FOR LATER USE
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)

        self.model.eval()
        return self.model

class TensorDataset(Dataset):
    def __init__(self, syn, vis, tab):
        self.syn, self.vis, self.tab = syn, vis, tab
    def __len__(self): return len(self.syn)
    def __getitem__(self, i):
        return self.syn[i], self.vis[i], self.tab[i]
