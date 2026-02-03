import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class SimpleFusion(nn.Module):
    """Minimal trainable fusion with InfoNCE loss"""
    def __init__(self, syn_dim: int, vis_dim: int, tab_dim: int, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(syn_dim + vis_dim + tab_dim, output_dim)
        self.output_dim = output_dim

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

class FusionTrainer:
    """Bare-minimum trainer using in-batch negatives (SimCLR style)"""
    def __init__(self, item_ids, syn_embs, vis_embs, tab_embs, output_dim=256, load_model=False):
        self.syn_embs = syn_embs.astype('float32')
        self.vis_embs = vis_embs.astype('float32')
        self.tab_embs = tab_embs.astype('float32')

        self.item_ids = item_ids
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_path = "my_fusion_v1.pt"

        self.model = SimpleFusion(384, 384, 384, 384)

        if load_model:
             self.model = self.load(self.save_path, self.device)
        else:
            self.model = SimpleFusion(
                syn_embs.shape[1],
                vis_embs.shape[1],
                tab_embs.shape[1],
                output_dim
            ).to(self.device)

    def train(self, epochs=10, batch_size=64, lr=3e-4, temperature=0.07):
        if not os.path.exists(self.save_path):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            dataset = TensorDataset(self.syn_embs, self.vis_embs, self.tab_embs)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                total_loss = 0
                for syn, vis, tab in loader:
                    # Move to device
                    syn, vis, tab = syn.to(self.device), vis.to(self.device), tab.to(self.device)

                    # Create two augmented views via modality dropout (minimal augmentation)
                    mask1 = torch.rand_like(syn) > 0.1  # 10% dropout
                    mask2 = torch.rand_like(syn) > 0.1

                    view1 = self.model(syn * mask1, vis * mask1, tab * mask1)
                    view2 = self.model(syn * mask2, vis * mask2, tab * mask2)

                    # InfoNCE loss: view1[i] ↔ view2[i] are positives
                    logits = (view1 @ view2.T) / temperature
                    labels = torch.arange(len(view1), device=self.device)
                    loss = F.cross_entropy(logits, labels)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            self.save(self.save_path)
        else:
            self.load(self.save_path)
        return self.model

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
