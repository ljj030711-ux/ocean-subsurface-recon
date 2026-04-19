"""2DVar 变分反演方法。"""

import numpy as np
from scipy.optimize import minimize

from config import (
    VAR2D_BG_WEIGHT,
    VAR2D_GTOL,
    VAR2D_MAXITER,
    VAR2D_SLA_VAR,
    VAR2D_SSS_VAR,
)


def inversion_2dvar(sla_sss, t, c_depth=26):
    """
    2DVar 变分反演（单日单步）。

    Args:
        sla_sss: (T, 2, H, W) 海表 SLA/SSS
        t: 目标日期时间索引
        c_depth: 反演深度层数

    Returns:
        (1, c_depth, H, W) 反演结果
    """
    _, _, H, W = sla_sss.shape

    sss_surface_t = sla_sss[t, 1, :, :]
    sws_bg = np.tile(sss_surface_t[np.newaxis, :, :], (c_depth, 1, 1))

    sla_var = 0.01 ** 2
    sss_var = 0.1 ** 2
    bg_weight = 1e-3

    def observation_operator(x):
        x_r = x.reshape(c_depth, H, W)
        return np.array([np.mean(x_r) * 0.02, np.mean(x_r) * 0.05])

    def cost_function(x, obs, bg_flat, w):
        bg_term = w * np.sum((x - bg_flat) ** 2)
        h_x = observation_operator(x)
        obs_term = ((h_x[0] - obs[0]) ** 2) / sla_var + ((h_x[1] - obs[1]) ** 2) / sss_var
        return bg_term + obs_term

    obs_t = sla_sss[t].mean(axis=(1, 2))
    bg_flat = sws_bg.flatten()

    res = minimize(
        cost_function, bg_flat.copy(),
        args=(obs_t, bg_flat, bg_weight),
        method="L-BFGS-B",
        options={"maxiter": 50, "gtol": 1e-4, "disp": False},
    )

    print(f"2DVar 收敛={res.success}, J={res.fun:.6f}, iter={res.nit}")
    return res.x.reshape(1, c_depth, H, W).astype(np.float32)
