# EddyUNet 推理函数使用说明

本目录仅包含 `eddy_unet` 的函数化推理能力，不依赖训练入口。

## 文件说明

- `eddy_unet_infer.py`: 主函数实现
- `__init__.py`: 对外导出
- `run_eddy_unet_infer.py`: 最小调用示例

## 主函数

```python
infer_eddy_unet_26layers(
    select_day: str,
    data_dir: str,
    checkpoint_dir: str,
    use_physics_features: bool = False,
    target_level: int = 10,
    checkpoint_policy: str = "strict",
) -> dict
```

## 参数

- `select_day`: 日期，格式 `YYYY-MM-DD`。
- `data_dir`: 数据目录（例如 `./data/raw`）。
- `checkpoint_dir`: 26 层模型目录（例如 `./checkpoints/2dto2d/eddy_unet`）。
- `use_physics_features`: 是否使用物理特征（`EKE` 与 `gradSSH`）。
- `target_level`: 目标层索引，仅用于校验和元信息。
- `checkpoint_policy`:
  - `strict`（默认）：任意层 checkpoint 缺失直接报错。
  - `warn`：缺失层给出警告并使用随机初始化权重。

## 返回

- `pred`: `(1, 26, H, W)`，物理空间预测值（psu）。
- `target`: `(1, 26, H, W)`，物理空间真值（psu）。
- `summary`: `{mse, rmse, mae, r2}`。
- `meta`:
  - `metric_space = "physical"`
  - `metric_units`（`mse/psu^2`，`rmse/mae/psu`）
  - `pred_shape` / `target_shape`
  - `checkpoint_policy`
  - `missing_checkpoints`

## 运行示例

```bash
python inference/run_eddy_unet_infer.py
```

## 与现有口径一致性

该函数复用了现有 `EddyDataset`、`build_model`、`build_2dto2d_features`、`date_to_index` 和指标函数，
并保持与 `test.py --method eddy_unet` 相同的 26 层拼装与反标准化口径。
