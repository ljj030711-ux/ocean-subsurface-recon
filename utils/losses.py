"""PyTorch loss functions used by trainable models."""

import torch


def _validate_reconstruction_shapes(pred, target, mask):
    if pred.shape != target.shape:
        raise ValueError(
            "pred 与 target 形状必须完全一致，"
            f"实际 pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    if mask.shape != target.shape:
        raise ValueError(
            "mask 与 target 形状必须完全一致，"
            f"实际 mask={tuple(mask.shape)}, target={tuple(target.shape)}"
        )


def masked_rmse_loss(pred, target, mask, eps=1e-8):
    """Compute RMSE over valid target points only."""
    _validate_reconstruction_shapes(pred, target, mask)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid_count = mask.sum().clamp_min(1.0)
    mse = ((pred - target).pow(2) * mask).sum() / valid_count
    return torch.sqrt(mse + eps)


def masked_depth_mean_rmse_loss(pred, target, mask, eps=1e-8):
    """Average per-depth masked RMSE so every valid depth has equal weight.

    B0 optimizes the same masked RMSE independently at every depth.  B1 uses
    this aggregation to preserve that per-depth objective while training all
    output channels jointly.
    """
    _validate_reconstruction_shapes(pred, target, mask)
    if pred.ndim != 4:
        raise ValueError(
            "联合深度损失要求 (B,D,H,W)，"
            f"实际 pred={tuple(pred.shape)}"
        )

    mask = mask.to(device=pred.device, dtype=pred.dtype)
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid_count = mask.sum(dim=(0, 2, 3))
    squared_error = ((pred - target).pow(2) * mask).sum(dim=(0, 2, 3))
    valid_depths = valid_count > 0
    if not torch.any(valid_depths):
        return pred.sum() * 0.0
    depth_mse = squared_error[valid_depths] / valid_count[valid_depths]
    return torch.sqrt(depth_mse + eps).mean()


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
