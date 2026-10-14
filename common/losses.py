"""Label-smoothed, class-weighted cross-entropy (Eq. 'loss') and mixup."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(counts: np.ndarray) -> torch.Tensor:
    """w_c = 1/sqrt(N_c), normalised to mean 1."""
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class SmoothWeightedCE(nn.Module):
    """L = -sum_c w_c [ (1-eps) y_c log p_c + eps/K log p_c ], averaged over the batch.

    Accepts hard labels (B,) or soft targets (B, K) (used with mixup).
    """

    def __init__(self, weights: torch.Tensor, eps: float = 0.1):
        super().__init__()
        self.register_buffer("w", weights)
        self.eps = eps

    def forward(self, logits, target):
        K = logits.shape[1]
        logp = F.log_softmax(logits.float(), dim=1)
        if target.dim() == 1:
            target = F.one_hot(target, K).float()
        t = (1 - self.eps) * target + self.eps / K
        return -(self.w[None, :] * t * logp).sum(1).mean()


def mixup(x, y, num_classes, alpha=0.2):
    """Returns mixed inputs and soft one-hot targets."""
    y1 = F.one_hot(y, num_classes).float()
    if alpha <= 0:
        return x, y1
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y1 + (1 - lam) * y1[idx]
