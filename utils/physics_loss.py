"""
物理约束损失函数

三个分量：
    1. 数据重建 MSE
    2. 静力学平衡约束：由预测温盐积分得到的 steric height 应与输入 SLA 一致
    3. 密度层结稳定性：海水密度应随深度递增（Brunt-Väisälä 频率 N² > 0）

海水线性状态方程：
    ρ = ρ₀ × (1 − α(T − T_ref) + β(S − S_ref))
    α ≈ 2×10⁻⁴ K⁻¹   热膨胀系数（温度升高 → 密度降低）
    β ≈ 7.6×10⁻⁴ psu⁻¹ 盐缩系数（盐度升高 → 密度增大）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RHO_0 = 1025.0
ALPHA = 2e-4
BETA = 7.6e-4
T_REF = 10.0
S_REF = 35.0

DEFAULT_DEPTH_LEVELS = [0, 10, 50, 100, 200, 300, 500, 700, 850, 1000]


def compute_density(temperature, salinity):
    """线性状态方程计算海水密度 (kg/m³)"""
    return RHO_0 * (1.0 - ALPHA * (temperature - T_REF)
                    + BETA * (salinity - S_REF))


class PhysicsLoss(nn.Module):
    """
    L = L_mse + λ_hydro · L_hydro + λ_strat · L_strat

    Args:
        depth_levels: 各层深度 (m)，用于计算层厚 Δz
        lambda_hydro: 静力学平衡惩罚权重
        lambda_strat: 密度稳定性惩罚权重
    """

    def __init__(self, depth_levels=None, lambda_hydro=0.01, lambda_strat=0.1):
        super().__init__()
        self.lambda_hydro = lambda_hydro
        self.lambda_strat = lambda_strat

        levels = depth_levels or DEFAULT_DEPTH_LEVELS
        dz = [float(levels[i + 1] - levels[i]) for i in range(len(levels) - 1)]
        self.register_buffer('dz', torch.tensor(dz, dtype=torch.float32))

    def forward(self, pred, target, sla=None):
        """
        Args:
            pred:   (B, D, H, W, 2)  [...,0]=T  [...,1]=S
            target: (B, D, H, W, 2)
            sla:    (B, H, W) 海面高度异常（可选）

        Returns:
            total_loss, {子项损失字典}
        """
        mse_loss = F.mse_loss(pred, target)

        pred_T = pred[..., 0]
        pred_S = pred[..., 1]
        rho = compute_density(pred_T, pred_S)

        # --- 静力学平衡：steric height ≈ SLA ---
        hydro_loss = torch.tensor(0.0, device=pred.device)
        if sla is not None and self.lambda_hydro > 0:
            delta_rho = rho - RHO_0
            mid_rho = (delta_rho[:, :-1] + delta_rho[:, 1:]) / 2.0
            steric = -(1.0 / RHO_0) * (mid_rho * self.dz.view(1, -1, 1, 1)).sum(dim=1)
            hydro_loss = F.mse_loss(steric, sla)

        # --- 密度层结稳定性：ρ(z_{k+1}) ≥ ρ(z_k) ---
        strat_loss = torch.tensor(0.0, device=pred.device)
        if self.lambda_strat > 0:
            drho = rho[:, :-1] - rho[:, 1:]
            strat_loss = F.relu(drho).mean()

        total = mse_loss + self.lambda_hydro * hydro_loss + self.lambda_strat * strat_loss
        return total, {
            'mse': mse_loss.item(),
            'hydrostatic': hydro_loss.item(),
            'stratification': strat_loss.item(),
            'total': total.item(),
        }
