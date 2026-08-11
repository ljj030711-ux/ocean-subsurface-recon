import torch
import torch.nn as nn
import pytest

from models.joint_du_unet import JointDuUnet
from utils.losses import masked_depth_mean_rmse_loss, masked_rmse_loss


@pytest.mark.parametrize("num_depths", [25, 28])
def test_joint_du_unet_outputs_every_depth(num_depths):
    model = JointDuUnet(num_depths=num_depths).eval()
    with torch.no_grad():
        pred = model(
            torch.randn(2, 1, 32, 32),
            torch.randn(2, 2, 16, 16),
        )
    assert pred.shape == (2, num_depths, 16, 16)


def test_joint_du_unet_has_no_attention_or_transformer():
    model = JointDuUnet(num_depths=25)
    forbidden = (
        nn.MultiheadAttention,
        nn.Transformer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
    )
    assert not any(isinstance(module, forbidden) for module in model.modules())


def test_equal_depth_loss_and_backward_are_finite():
    model = JointDuUnet(num_depths=3)
    pred = model(
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 2, 16, 16),
    )
    target = torch.randn_like(pred)
    mask = torch.ones_like(pred)
    mask[:, 1, :4, :4] = 0
    loss = masked_depth_mean_rmse_loss(pred, target, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_losses_reject_silent_channel_broadcasting():
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 25, 8, 8)
    mask = torch.ones_like(target)
    with pytest.raises(ValueError, match="形状必须完全一致"):
        masked_rmse_loss(pred, target, mask)
    with pytest.raises(ValueError, match="形状必须完全一致"):
        masked_depth_mean_rmse_loss(pred, target, mask)


def test_checkpoint_round_trip(tmp_path):
    source = JointDuUnet(num_depths=4).eval()
    checkpoint = tmp_path / "joint_du_unet.pth"
    torch.save(source.state_dict(), checkpoint)

    restored = JointDuUnet(num_depths=4).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    sst = torch.randn(1, 1, 32, 32)
    surface = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        expected = source(sst, surface)
        actual = restored(sst, surface)
    torch.testing.assert_close(actual, expected)
