import torch
import torch.nn.functional as F



def compute_eke(ssh, dx, dy, g=9.81, f=1e-4):
    """
    计算涡动动能 EKE（安全版本，维度严格对齐）

    Args:
        ssh: 海面高度异常 [B, C, H, W]
        dx: x 方向网格间距 (m)
        dy: y 方向网格间距 (m)

    Returns:
        eke: 涡动动能 [B, C, H, W]
    """
    # 中心差分
    dssh_dx = (ssh[:, :, :, 2:] - ssh[:, :, :, :-2]) / (2 * dx)   # [B,C,H,W-2]
    dssh_dy = (ssh[:, :, 2:, :] - ssh[:, :, :-2, :]) / (2 * dy)   # [B,C,H-2,W]

    # 对齐到公共内部区域
    dssh_dx = dssh_dx[:, :, 1:-1, :]   # [B,C,H-2,W-2]
    dssh_dy = dssh_dy[:, :, :, 1:-1]   # [B,C,H-2,W-2]

    # 地转流
    u = -g / f * dssh_dy
    v =  g / f * dssh_dx

    eke = 0.5 * (u ** 2 + v ** 2)

    # padding 回原始大小
    eke = F.pad(eke, (1, 1, 1, 1), mode="replicate")

    return eke

def compute_grad_ssh(ssh):
    """
    计算 SSH 梯度幅值（安全版本，维度严格对齐）
    
    Args:
        ssh: 海面高度异常 [B, C, H, W]
    
    Returns:
        grad: SSH 梯度幅值 [B, C, H, W]
    """
    # 中心差分
    dssh_dx = ssh[:, :, :, 2:] - ssh[:, :, :, :-2]   # [B,C,H,W-2]
    dssh_dy = ssh[:, :, 2:, :] - ssh[:, :, :-2, :]   # [B,C,H-2,W]

    # 对齐到公共内部区域
    dssh_dx = dssh_dx[:, :, 1:-1, :]                 # [B,C,H-2,W-2]
    dssh_dy = dssh_dy[:, :, :, 1:-1]                 # [B,C,H-2,W-2]

    # 梯度幅值
    grad = torch.sqrt(dssh_dx**2 + dssh_dy**2 + 1e-6)

    # padding 回原始大小
    grad = F.pad(grad, (1, 1, 1, 1), mode="replicate")

    return grad
