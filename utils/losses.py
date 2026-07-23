"""PyTorch loss functions used by trainable models."""

import torch


def masked_rmse_loss(pred, target, mask, eps=1e-8):
    """Compute RMSE over valid target points only."""
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid_count = mask.sum().clamp_min(1.0)
    mse = ((pred - target).pow(2) * mask).sum() / valid_count
    return torch.sqrt(mse + eps)


def delta_vertical_smooth_loss(pred, T0, depth_tensor, mask=None, eps=1e-8):
    """Penalize abrupt layer-to-layer changes in delta = pred - T0.

    Shapes:
        pred/T0/mask: (B, D, H, W)
        depth_tensor: (D,)

    The depth spacing is used as a relative scale, while the average spacing keeps
    the loss magnitude close to a layer-to-layer delta difference.
    """
    if pred.shape[1] < 2:
        return pred.new_zeros(())

    delta = pred - T0
    dz = (depth_tensor[1:] - depth_tensor[:-1]).to(
        device=delta.device, dtype=delta.dtype
    )
    dz = dz.abs().clamp_min(eps)
    scale = (dz.mean() / dz).view(1, -1, 1, 1)
    diff = (delta[:, 1:] - delta[:, :-1]) * scale

    if mask is None:
        return diff.abs().mean()

    valid = (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
    valid = valid.to(device=diff.device, dtype=diff.dtype)
    valid_count = valid.sum().clamp_min(1.0)
    return (diff.abs() * valid).sum() / valid_count


def hvca_refiner_loss(pred, T0, target, mask, depth_tensor, lambda_delta_z=0.0):
    """Return HVCARefiner total loss and its tracked components.

    The checkpoint monitor still uses the RMSE component in train_hvca_refiner.py;
    the total loss is only used for backpropagation when lambda_delta_z > 0.
    """
    rmse_loss = masked_rmse_loss(pred, target, mask)
    delta_z_loss = delta_vertical_smooth_loss(pred, T0, depth_tensor, mask=mask)
    total_loss = rmse_loss + float(lambda_delta_z) * delta_z_loss
    return {
        "loss": total_loss,
        "rmse": rmse_loss,
        "delta_z": delta_z_loss,
    }
