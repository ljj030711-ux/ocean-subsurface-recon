"""2dto3d 数据集定义。"""

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.io_2dto3d import clean_2dto3d, load_2dto3d_raw, validate_2dto3d_shapes


class DummyTwoDto3DDataset(Dataset):
    """
    2dto3d 模型的合成数据集：
      - surface_raw: (2, H, W)  -> [SLA, SSS]
      - target:      (D, H, W, 2) -> [..., 0]=temp, [..., 1]=salt
      - sla:         (H, W)（可选）
    """

    def __init__(self, num_samples=200, H=32, W=32, D=10, seed=42):
        rng = np.random.default_rng(seed)
        self.surface_raw = rng.normal(size=(num_samples, 2, H, W)).astype(np.float32)
        self.target = rng.normal(size=(num_samples, D, H, W, 2)).astype(np.float32)
        self.sla = self.surface_raw[:, 0]

    def __len__(self):
        return len(self.surface_raw)

    def __getitem__(self, idx):
        return {
            "surface_raw": torch.tensor(self.surface_raw[idx], dtype=torch.float32),
            "target": torch.tensor(self.target[idx], dtype=torch.float32),
            "sla": torch.tensor(self.sla[idx], dtype=torch.float32),
        }


class TwoDto3DDataset(Dataset):
    """
    从真实数据文件读取 2dto3d 数据：
      - 海表输入： (T,2,H,W)
      - 真值：     (T,D,H,W) 或 (T,D,H,W,2)

    若真值是 4D（只有一个变量），会自动扩展成最后一维 2 变量格式：
      target[..., 0] = temp(占位0)
      target[..., 1] = salt(原值)
    """

    def __init__(self, data_dir, normalize=False):
        self.data_dir = data_dir
        self.normalize = normalize

        self.surface_raw, self.target_data = load_2dto3d_raw(data_dir)
        validate_2dto3d_shapes(self.surface_raw, self.target_data)
        self.surface_raw, self.target_data = clean_2dto3d(
            self.surface_raw, self.target_data, normalize=normalize
        )

        if self.target_data.ndim == 4:
            t, d, h, w = self.target_data.shape
            temp = np.zeros((t, d, h, w), dtype=np.float32)
            salt = self.target_data.astype(np.float32)
            self.target_data = np.stack([temp, salt], axis=-1)
        elif self.target_data.ndim == 5 and self.target_data.shape[-1] == 2:
            pass
        else:
            raise ValueError(f"target_data 需要 (T,D,H,W) 或 (T,D,H,W,2)，实际：{self.target_data.shape}")

        self.T = int(self.surface_raw.shape[0])

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return {
            "surface_raw": torch.tensor(self.surface_raw[idx], dtype=torch.float32),
            "target": torch.tensor(self.target_data[idx], dtype=torch.float32),
            "sla": torch.tensor(self.surface_raw[idx, 0], dtype=torch.float32),
        }
