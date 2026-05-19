"""PyTorch loss functions used by trainable models."""

import torch


def masked_rmse_loss(pred, target, mask, eps=1e-8):
    """Compute RMSE over valid target points only."""
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid_count = mask.sum().clamp_min(1.0)
    mse = ((pred - target).pow(2) * mask).sum() / valid_count
    return torch.sqrt(mse + eps)
