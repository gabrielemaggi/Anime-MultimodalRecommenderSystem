import torch
import torch.nn as nn
import torch.nn.functional as F


class Augumentations(nn.Module):
    def __init__(self, noise=0.05, mask_prob=0.1, modality_drop_prob=0.3):
        """
        Args:
            noise (float): Standard deviation of Gaussian noise added to embeddings.
            mask_prob (float): Probability of zeroing out individual features (Dropout).
            modality_drop_prob (float): Probability of zeroing out an ENTIRE modality.
        """
        super().__init__()
        self.noise_std = noise
        self.mask_prob = mask_prob
        self.modality_drop_prob = modality_drop_prob

    def forward(self, syn, vis, tab):
        """
        Input: Batch of embeddings for each modality.
        Output: Tuple of augmented embeddings (syn_aug, vis_aug, tab_aug)
        """
        # Apply Feature-level augmentations
        syn_aug = self._augment_single(syn)
        vis_aug = self._augment_single(vis)
        tab_aug = self._augment_single(tab)

        # Apply Modality-level augmentations (Collaboration forcing)
        # We ensure at least one modality is always kept (avoid zeroing all 3)
        syn_aug, vis_aug, tab_aug = self._random_modality_drop(
            syn_aug, vis_aug, tab_aug
        )

        return syn_aug, vis_aug, tab_aug

    def _augment_single(self, x):
        if not self.training:
            return x

        # 1. Add Gaussian Noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise

        # 2. Feature Dropout (Masking)
        # Create a binary mask (1 = keep, 0 = drop)
        mask = torch.bernoulli(torch.full_like(x, 1 - self.mask_prob))
        x = x * mask

        return x

    def _random_modality_drop(self, s, v, t):
        """
        Randomly drops full modalities.
        Logic: For each sample in batch, independently decide whether to kill syn, vis, or tab.
        """
        if not self.training:
            return s, v, t

        batch_size = s.size(0)

        # Generate random drop masks (Batch_Size, 1)
        # 1 = Keep, 0 = Drop
        # We use strict logic: never drop ALL modalities for a single sample.

        # Probabilities for keeping each modality
        p_keep = 1 - self.modality_drop_prob

        mask_s = torch.bernoulli(torch.full((batch_size, 1), p_keep, device=s.device))
        mask_v = torch.bernoulli(torch.full((batch_size, 1), p_keep, device=v.device))
        mask_t = torch.bernoulli(torch.full((batch_size, 1), p_keep, device=t.device))

        # Safety Check: If sum of masks is 0 (all dropped), force keep at least 'syn' (or random)
        # Sum masks across modalities
        total_mask = mask_s + mask_v + mask_t

        # Find indices where all are dropped (sum == 0)
        all_dropped = (total_mask == 0).squeeze()

        # For those rows, force mask_s to 1 (recover one modality)
        if all_dropped.any():
            mask_s[all_dropped] = 1.0

        return s * mask_s, v * mask_v, t * mask_t
