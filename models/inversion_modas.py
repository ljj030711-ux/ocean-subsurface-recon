"""MODAS 统计反演方法。"""

import numpy as np


def inversion_modas(sla_sss, sws_true, t, train_slice):
    """
    MODAS 统计反演（训练期逐像素逐层线性回归 + 单日预测）。

    Args:
        sla_sss: (T, 2, H, W) 海表 SLA/SSS
        sws_true: (T, D, H, W) 水下真值（训练集部分用于拟合）
        t: 目标日期时间索引
        train_slice: 训练集切片

    Returns:
        (1, D, H, W) 反演结果
    """
    _, _, H, W = sla_sss.shape
    _, C_depth, _, _ = sws_true.shape

    X_train = np.nan_to_num(sla_sss[train_slice], nan=0.0)
    Y_train = np.nan_to_num(sws_true[train_slice], nan=0.0)
    T_train = X_train.shape[0]

    ssh_train = X_train[:, 0, :, :]
    sss_train = X_train[:, 1, :, :]
    SSH_bar = np.mean(ssh_train, axis=0, dtype=np.float64).astype(np.float32)
    SSS_bar = np.mean(sss_train, axis=0, dtype=np.float64).astype(np.float32)
    dSSH_train = ssh_train - SSH_bar[np.newaxis, :, :]
    dSSS_train = sss_train - SSS_bar[np.newaxis, :, :]
    cross_mean = np.mean(dSSS_train * dSSH_train, axis=0, dtype=np.float64).astype(np.float32)

    n_feat = 4
    coeffs = np.zeros((C_depth, H, W, n_feat), dtype=np.float32)
    ridge = 1e-6
    eye4 = np.eye(n_feat, dtype=np.float64)

    print(f"MODAS 拟合中（{T_train} 天, {H}x{W} 网格, {C_depth} 层）...")
    for i in range(H):
        for j in range(W):
            d_sss = dSSS_train[:, i, j].astype(np.float64)
            d_ssh = dSSH_train[:, i, j].astype(np.float64)
            z_train = d_sss * d_ssh - float(cross_mean[i, j])
            X_aug = np.column_stack([np.ones(T_train, dtype=np.float64), d_sss, d_ssh, z_train])
            XtX = X_aug.T @ X_aug
            try:
                XtX_inv = np.linalg.inv(XtX + ridge * eye4)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)
            for k in range(C_depth):
                y_px = Y_train[:, k, i, j].astype(np.float64)
                coeffs[k, i, j, :] = (XtX_inv @ (X_aug.T @ y_px)).astype(np.float32)

    x_t = np.nan_to_num(sla_sss[t], nan=0.0)
    pred = np.zeros((C_depth, H, W), dtype=np.float32)
    for ii in range(H):
        for jj in range(W):
            d_sss = x_t[1, ii, jj] - SSS_bar[ii, jj]
            d_ssh = x_t[0, ii, jj] - SSH_bar[ii, jj]
            z = d_sss * d_ssh - cross_mean[ii, jj]
            x_vec = np.array([1.0, d_sss, d_ssh, z], dtype=np.float32)
            for k in range(C_depth):
                pred[k, ii, jj] = coeffs[k, ii, jj, :] @ x_vec

    print(f"MODAS 预测完成, 值范围: [{pred.min():.4f}, {pred.max():.4f}]")
    return pred[np.newaxis, ...].astype(np.float32)
