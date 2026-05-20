# Eddy Inversion

海洋反演项目：由海表场（SST/SSH/SSS）重建水下多深度温度或盐度结构，统一支持变分、统计与深度学习方法。

## 建模范式

- `2dto2d`：每个变量、每个深度训练一个独立模型（当前：`Du_Unet`）。
- `2dto3d`：单模型一次输出所有深度（当前：`2dvar`、`modas`、`ocean_transformer`）。

输出目录统一为：

- `outputs/{paradigm}/{method}/...`
- `checkpoints/{paradigm}/{method}/...`
- Du_Unet 按变量额外分目录：`outputs/2dto2d/Du_Unet/{temperature|salinity}/...`
- Du_Unet 权重按变量额外分目录：`checkpoints/2dto2d/Du_Unet/{temperature|salinity}/...`

## 模型思路

### 1) Du_Unet（2dto2d）

- 思路：SST 高分辨率分支处理 `(B,1,160,160)`，SSH+SSS 低分辨率分支处理 `(B,2,64,64)`，融合后输出单层 `(B,1,64,64)`。
- 训练：一次选择 `temperature` 或 `salinity`，循环 5-300m 共 25 层；两类变量分别训练后共 50 个 checkpoint。
- 训练：按深度列表循环，目标为单层 `(B,1,H,W)`。
- 推理：逐深度加载模型并拼装为 `(1,25,64,64)`，再统一评估与绘图。
- 评估口径：`test.py` 中会将标准化输出反标准化回温度或盐度物理量后再评估。

### 2) 2dvar（2dto3d）

- 思路：变分反演，构造背景项与观测项代价函数，L-BFGS-B 迭代求解。
- 输出：`(1,D,H,W)`，默认 D=26（由 `--c-depth` 控制）。

### 3) modas（2dto3d）

- 思路：逐像素逐深度线性统计回归，在训练时段拟合系数，目标日预测整层场。
- 输出：`(1,D,H,W)`，D 由真值数据深度维决定（当前 26）。

### 4) ocean_transformer（2dto3d）

- 思路：CNN 提取海表空间特征 + 空间 Transformer + 深度 Transformer。
- 输出：`(B,D,H,W,2)`，最后一维是 `(temperature, salinity)`。
- 损失：`PhysicsLoss = MSE + 静力学平衡约束 + 层结稳定约束`。

## 项目整体架构

```mermaid
flowchart LR
  raw[raw_npy_data] --> ds[dataset_io_and_dataset]
  ds --> train[train.py]
  ds --> test[test.py]
  train --> ckpt[checkpoints_by_paradigm]
  ckpt --> test
  test --> eval[metrics_and_summary]
  eval --> map2d[plot_level_map]
  eval --> profile3d[plot_3d_metric_profile]
```



## 目录结构

```text
ocean-subsurface-recon/
├── config.py                     # 统一配置：范式、超参数、路径、常量
├── train.py                      # 训练入口（2dto2d/2dto3d）
├── test.py                       # 推理评估入口（2dto2d/2dto3d）
├── models/                       # 各模型/算法实现
├── datasets/                     # Dataset + 数据读入/前处理工具
│   ├── io_2dto2d.py
│   ├── io_2dto3d.py
│   ├── non_dl_preprocess.py
│   ├── date_utils.py
│   ├── dataset_2dto2d.py
│   └── dataset_2dto3d.py
├── utils/                        # physics / loss / metrics / viz
├── outputs/
│   ├── 2dto2d/
│   │   └── Du_Unet/
│   │       ├── temperature/
│   │       └── salinity/
│   └── 2dto3d/
└── checkpoints/
    ├── 2dto2d/
    │   └── Du_Unet/
    │       ├── temperature/
    │       └── salinity/
    └── 2dto3d/
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

## 训练 `train.py`

```bash
# 2dto2d: Du_Unet（一次训练一个变量的25层）
python train.py --method Du_Unet --target-var temperature \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw

# 可选：只训练指定深度索引，索引不是米数；9 表示 50m。
python train.py --method Du_Unet --target-var temperature \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw \
  --depth-indices 9

python train.py --method Du_Unet --target-var salinity \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw

# 2dto3d: ocean_transformer
python train.py --method ocean_transformer --data-dir ./data/raw
```

## 参数说明

| 参数 | 含义 |
| --- | --- |
| `--method` | 训练方法，当前支持 `Du_Unet` / `du_unet` 和 `ocean_transformer`。 |
| `--target-var` | Du_Unet 目标变量，`temperature` 或 `salinity`。 |
| `--start-date` | Du_Unet 数据起始日期，默认使用 `config.py` 中的 `DATA_START_DATE`。 |
| `--end-date` | Du_Unet 数据结束日期，不传时使用数据文件可覆盖的结束日期。 |
| `--epochs` | 覆盖默认训练轮数；不传时使用对应模型的配置值。 |
| `--batch-size` | 覆盖默认 batch size。 |
| `--lr` | 覆盖默认学习率。 |
| `--data-dir` | 原始数据目录。 |
| `--output-dir` | 训练历史和 loss 曲线输出根目录，默认 `./outputs`。 |
| `--checkpoint-dir` | checkpoint 输出根目录，默认 `./checkpoints`。 |
| `--dummy` | 仅用于 `ocean_transformer` 的合成数据调试。 |
| `--patience` | 早停 patience；不传时使用配置值。 |
| `--seed` | 随机种子。 |
| `--depth-indices` | Du_Unet 调试参数，指定训练哪些深度层索引，逗号分隔；不传时训练当前变量全部 25 层。 |

`--depth-indices` 是深度层索引，不是米数。Du_Unet 的索引从 5m 开始：

```text
0=5m, 1=10m, 2=15m, ..., 9=50m, 10=55m
```

### 推理`test.py`

```bash
# 2dto2d: Du_Unet（逐层加载checkpoint并拼装）
python test.py --method Du_Unet --target-var temperature --select-day 2023-03-12 --target-level 10 \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw --checkpoint-dir ./checkpoints/2dto2d/Du_Unet

python test.py --method Du_Unet --target-var salinity --select-day 2023-03-12 --target-level 10 \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw --checkpoint-dir ./checkpoints/2dto2d/Du_Unet

# 2dto3d: 2dvar
python test.py --method 2dvar --select-day 2023-06-15 --target-level 10 \
  --sla-sss-path ./data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy

# 2dto3d: modas
python test.py --method modas --select-day 2023-06-15 --target-level 10 \
  --sla-sss-path ./data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy \
  --sws-true-path ./data/raw/sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy
```

| 参数 | 含义 |
| --- | --- |
| `--method` | 推理/评估方法，支持 `Du_Unet` / `du_unet`、`ocean_transformer`、`2dvar`、`modas`。 |
| `--select-day` | 预测日期，格式 `YYYY-MM-DD`。 |
| `--target-level` | 可视化深度层索引，不是米数；Du_Unet 中 `10` 表示 55m。 |
| `--output-dir` | 推理产物输出根目录，默认 `./outputs`。 |
| `--checkpoint` | 单模型权重路径，主要用于 `ocean_transformer`。 |
| `--checkpoint-dir` | Du_Unet 多深度 checkpoint 目录；可传 `./checkpoints/2dto2d/Du_Unet`，代码会兼容查找 `{target_var}` 子目录。 |
| `--data-dir` | 深度学习方法的数据目录。 |
| `--target-var` | Du_Unet 预测目标变量，`temperature` 或 `salinity`。 |
| `--start-date` | Du_Unet 数据起始日期，应与训练 checkpoint 的数据范围一致。 |
| `--end-date` | Du_Unet 数据结束日期，应与训练 checkpoint 的数据范围一致。 |
| `--dummy` | 使用 2dto3d 合成数据调试。 |
| `--sla-sss-path` | 2DVar/MODAS 的 SLA/SSS 输入文件路径。 |
| `--sws-true-path` | MODAS 必需的真值文件路径；2DVar 可传入用于评估。 |
| `--c-depth` | 2DVar 输出深度层数，默认 26。 |

服务器数据目录如果不是 `./data/raw`，可把 `--data-dir` 替换为实际路径，例如 `/root/autodl-tmp/.autodl/data/raw`。Du_Unet 推理时的 `--start-date/--end-date` 应与对应 checkpoint 训练时使用的数据范围一致，否则标准化和反标准化统计量会不一致。

## 统一可视化产物

所有方法都会保存预测数组、网格指标和 summary。可视化产物按方法略有不同：

- Du_Unet：指定层预测-真值-绝对误差三联图 `map_panel_*`，以及 3D 误差剖面图 `plot_3d_metric_profile`。
- 2dto3d / Non-DL 方法：指定层 2D 图 `plot_level_map`，以及 3D 误差剖面图 `plot_3d_metric_profile`。

统一输出文件（位于 `outputs/{paradigm}/{method}/`；Du_Unet 位于 `outputs/2dto2d/Du_Unet/{target_var}/`）：

- `pred_{method}_{date}.npy`
- `grid_metrics_{method}_{date}.npz`
- `map_*_lvl{level}_{method}_{date}.png`
- `profile_{metric}_{method}_{date}.png`
- `summary_{method}_{date}.json`

Du_Unet 会在变量目录内保留变量名，例如：

- `pred_Du_Unet_temperature_20230615.npy`
- `grid_metrics_Du_Unet_salinity_20230615.npz`
- `map_panel_temperature_lvl10_Du_Unet_20230615.png`
- `summary_Du_Unet_temperature_20230615.json`

`summary` 中包含：

- `metric_units`：`mse/rmse/mae/r2/correlation` 对应单位说明
- `variables`：按变量保存 `mse/rmse/mae/r2/correlation`

所有模型的预测产物、图和指标均使用原始物理单位。深度学习模型会在训练/推理输入侧使用训练期拟合的月气候态距平和分层标准差归一化，并在评估前反归一化；MODAS 和 2DVar 保持原始单位流程。

Du_Unet 的训练 loss、summary 指标、二维误差图和三维误差剖面图只统计 `target_mask=1` 的有效标签区域。`grid_metrics_*.npz` 中无效区域写为 `NaN`，避免缺失标签区域被当作真实误差。

## 配置索引（config.py）

- 范式与方法：`PARADIGM_2DTO2D_METHODS`、`PARADIGM_2DTO3D_METHODS`
- 深度列表：`DEPTH_LEVELS_25M`（Du_Unet）、`DEPTH_LEVELS_26M`（旧 2dto3d/Non-DL）
- Du_Unet 训练：`DU_UNET_*`
- ocean_transformer 训练：`TWODTO3D_*`
- 推理可视化：`INFER_*`
