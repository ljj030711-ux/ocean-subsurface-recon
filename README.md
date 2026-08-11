# Ocean Subsurface Recon

本项目用海表场重建水下多深度温度或盐度结构，当前包含 `Du_Unet`、`Joint_Du_Unet`、`HVCARefiner`、`ocean_transformer`、`2dvar` 和 `modas`。

## 总览

### 建模范式

- `2dto2d`：使用二维卷积处理海表场。`Du_Unet` 为逐深度B0，`Joint_Du_Unet` 为一次输出全部深度的B1，`HVCARefiner` 为B0旁支的二阶段修正模型。
- `2dto3d`：单次输出多个深度。当前包括 `ocean_transformer`、`2dvar`、`modas`。

### 目录规则

- 训练/测试结果根目录：`outputs/{paradigm}/{method}/`
- checkpoint 根目录：`checkpoints/{paradigm}/{method}/`
- Du_Unet 输出：`outputs/2dto2d/Du_Unet/{temperature|salinity}/`
- Du_Unet 权重：`checkpoints/2dto2d/Du_Unet/{temperature|salinity}/`
- Joint_Du_Unet 输出：`outputs/2dto2d/Joint_Du_Unet/{temperature|salinity}/`
- Joint_Du_Unet 权重：`checkpoints/2dto2d/Joint_Du_Unet/{temperature|salinity}/`
- HVCARefiner 输出与权重：`outputs/2dto2d/HVCARefiner/`
- OceanTransformer 输出：`outputs/2dto3d/ocean_transformer/`
- 2DVar 输出：`outputs/2dto3d/2dvar/`
- MODAS 输出：`outputs/2dto3d/modas/`

### 运行日志和配置快照

Du_Unet、Joint_Du_Unet 和 HVCARefiner 的 train/test 入口会在运行前打印最终超参数，并保存 JSON 配置快照：

- `outputs/2dto2d/Du_Unet/Du_Unet_train_config.json`
- `outputs/2dto2d/Du_Unet/Du_Unet_test_config.json`
- `outputs/2dto2d/Joint_Du_Unet/{target_var}/Joint_Du_Unet_train_config.json`
- `outputs/2dto2d/Joint_Du_Unet/{target_var}/Joint_Du_Unet_test_config.json`
- HVCARefiner 训练配置：`<train --save-dir>/HVCARefiner_train_config.json`
- HVCARefiner 测试配置：`<test --save-dir>/HVCARefiner_test_config.json`

配置快照会记录命令行、时间、数据范围、batch size、学习率、深度、模型结构、模型参数量、checkpoint/cache/output 路径等信息。HVCARefiner 配置快照跟随本次运行的 `--save-dir` 保存，便于区分多组 refiner 实验。测试配置统一使用 `test` 命名；Du_Unet 单日测试会在 JSON 中记录 `single: true`，时间段测试记录 `single: false`。

### 安装

```bash
pip install -r requirements.txt
```

### 深度索引

Du_Unet 和 HVCARefiner 使用 `DEPTH_LEVELS_25M`，从 5m 开始：

```text
0=5m, 1=10m, 2=15m, ..., 9=50m, 10=55m, ..., 24=300m
```

`--depth-indices` 是索引，`--depths` 是实际米数。

## Du_Unet

Du_Unet 是第一阶段 `2dto2d` 模型：每个目标变量、每个深度单独训练一个二维模型，推理时逐层加载并拼接为完整 25 层剖面。

### train：逐变量逐深度训练

输入命令：

```bash
python train.py --method Du_Unet --target-var temperature \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw \
  --output-dir ./outputs \
  --checkpoint-dir ./checkpoints \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3 \
  --patience 10 \
  --seed 42
```

只训练指定深度索引用于调试：

```bash
python train.py --method Du_Unet --target-var salinity \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw \
  --depth-indices 9,10
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--method` | 固定为 `Du_Unet` 或 `du_unet`。 |
| `--target-var` | 训练目标变量，`temperature` 或 `salinity`。 |
| `--start-date` / `--end-date` | 数据日期范围；会影响数据切分和归一化统计。 |
| `--data-dir` | 原始 `.npy` 数据目录。 |
| `--output-dir` | 输出根目录，默认 `./outputs`。 |
| `--checkpoint-dir` | 权重根目录，默认 `./checkpoints`。 |
| `--epochs` | 训练轮数；不传时使用 `DU_UNET_EPOCHS`。 |
| `--batch-size` | batch size；不传时使用 `DU_UNET_BATCH_SIZE`。 |
| `--lr` | 学习率；不传时使用 `DU_UNET_LR`。 |
| `--patience` | early stopping patience；不传时使用 `DU_UNET_PATIENCE`。 |
| `--seed` | 随机种子。 |
| `--depth-indices` | 可选，逗号分隔深度索引；不传则训练当前变量全部 25 层。 |

输出文件：

| 路径 | 内容 |
| --- | --- |
| `outputs/2dto2d/Du_Unet/Du_Unet_train_config.json` | 本次 Du_Unet 训练配置快照。 |
| `outputs/2dto2d/Du_Unet/{target_var}/training_history_Du_Unet_{target_var}_depth{depth_m}m.npz` | 每个深度的 train/val loss 数组。 |
| `outputs/2dto2d/Du_Unet/{target_var}/loss_Du_Unet_{target_var}_depth{depth_m}m.png` | 每个深度的 loss 曲线。 |
| `checkpoints/2dto2d/Du_Unet/{target_var}/Du_Unet_{target_var}_depth{depth_m}m_best.pth` | 每个深度的最佳权重。 |

### test：单日预测和可视化

输入命令：

```bash
python test.py --method Du_Unet --target-var temperature \
  --select-day 2023-03-12 \
  --target-level 10 \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw \
  --checkpoint-dir ./checkpoints/2dto2d/Du_Unet \
  --output-dir ./outputs
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--method` | 固定为 `Du_Unet` 或 `du_unet`。 |
| `--target-var` | 测试变量，`temperature` 或 `salinity`。 |
| `--select-day` | 单日预测日期，格式 `YYYY-MM-DD`。 |
| `--target-level` | 二维可视化深度索引，不影响完整 25 层输出。 |
| `--start-date` / `--end-date` | 必须与训练 checkpoint 对应的数据范围保持一致。 |
| `--data-dir` | 原始数据目录。 |
| `--checkpoint-dir` | Du_Unet 多深度 checkpoint 目录；可传到 `Du_Unet` 根目录，代码会自动查找变量子目录。 |
| `--output-dir` | 输出根目录，默认 `./outputs`。 |

输出文件位于 `outputs/2dto2d/Du_Unet/{target_var}/`：

| 文件 | 内容 |
| --- | --- |
| `../Du_Unet_test_config.json` | 本次 Du_Unet 单日测试配置快照，位于 `outputs/2dto2d/Du_Unet/`，包含 `single: true`。 |
| `pred_Du_Unet_{target_var}_{YYYYMMDD}.npy` | 反标准化后的完整 25 层预测场。 |
| `grid_metrics_Du_Unet_{target_var}_{YYYYMMDD}.npz` | 网格 MAE/RMSE 等指标。 |
| `map_panel_{target_var}_lvl{level}_Du_Unet_{YYYYMMDD}.png` | 指定层预测、真值、误差三联图。 |
| `summary_Du_Unet_{target_var}_{YYYYMMDD}.json` | 单日汇总指标和单位说明。 |

### test/eval：时间段整体评估

输入命令：

```bash
python test.py --method Du_Unet --target-var temperature \
  --eval-start 2023-01-01 --eval-end 2023-12-31 \
  --eval-batch-size 32 \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --data-dir ./data/raw \
  --checkpoint-dir ./checkpoints/2dto2d/Du_Unet \
  --output-dir ./outputs
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--eval-start` / `--eval-end` | 闭区间评估日期；目前时间段模式仅支持 Du_Unet。 |
| `--eval-batch-size` | 时间段推理 batch size。 |
| 其他参数 | 与单日预测一致。 |

输出文件位于 `outputs/2dto2d/Du_Unet/{target_var}/`：

| 文件 | 内容 |
| --- | --- |
| `../Du_Unet_test_config.json` | 本次 Du_Unet 时间段测试配置快照，包含 `single: false`。 |
| `evaluation_metrics_Du_Unet_{target_var}_{YYYYMMDD}_{YYYYMMDD}.npz` | overall/by-depth/by-day 指标数组。 |
| `summary_Du_Unet_{target_var}_{YYYYMMDD}_{YYYYMMDD}.json` | 时间段汇总指标。 |
| `metrics_by_depth_Du_Unet_{target_var}_{YYYYMMDD}_{YYYYMMDD}.png` | 逐深度指标曲线。 |
| `metrics_by_day_Du_Unet_{target_var}_{YYYYMMDD}_{YYYYMMDD}.png` | 逐日指标曲线。 |

## B1：Joint_Du_Unet 联合输出基线

`Joint_Du_Unet` 保留B0完全相同的SST高分辨率分支、SSH/SSS低分辨率分支以及普通卷积U-Net骨干，只把最终输出从单层改为当前profile的全部深度层。模型不含注意力、Transformer、深度嵌入或垂向后处理。

- `legacy_small`：一次输出25层。
- `glorys12v1_all_size`：一次输出28层。
- 每个目标变量只训练一个模型。
- 训练损失为各深度masked RMSE的等权平均。
- B1训练禁止使用`--depth-indices`，保证始终联合学习全部深度。

### 上传代码到GitHub和服务器

本地B1工作目录及分支：

```text
/Users/lijunjie/Documents/python/ocean-subsurface-recon-b1
B1
```

本地提交并推送：

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon-b1
git status
git add -A
git status --short
git commit -m "feat: add B1 joint-depth U-Net baseline"
git push -u origin B1
```

如果不使用GitHub，可以直接把当前本地代码同步到AutoDL：

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon-b1
./scripts/upload_code_to_autodl.sh
```

默认上传到`/root/autodl-tmp/ocean-subsurface-recon-b1`。指定其他服务器目录：

```bash
./scripts/upload_code_to_autodl.sh /root/autodl-tmp/my-ocean-b1
```

该脚本会上传当前工作区，包括尚未提交的代码修改；它会排除`.git`、`data/`、`checkpoints/`、`outputs/`和Python缓存，不会删除服务器上已有文件。服务器地址、端口变化时可通过`SERVER_HOST`、`SERVER_PORT`和`SERVER_USER`覆盖默认值。

服务器首次下载到独立目录：

```bash
git clone --branch B1 --single-branch \
  git@github.com:ljj030711-ux/ocean-subsurface-recon.git \
  /root/ocean-subsurface-recon-b1
cd /root/ocean-subsurface-recon-b1
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

执行安装前应先激活服务器上用于本项目的Python/Conda环境。上述SSH地址要求服务器已经配置GitHub SSH key；否则可改用`https://github.com/ljj030711-ux/ocean-subsurface-recon.git`。

服务器已经下载过时更新：

```bash
cd /root/ocean-subsurface-recon-b1
git pull --ff-only origin B1
```

代码仓库不包含大型数据。`--data-dir`所指目录需要直接包含当前profile要求的`.npy`文件；不能指向存放原始NetCDF的`nc/`子目录。GLORYS12V1温度实验需要：

```text
SST_2002-01-01_2023-12-31_0_24_105_121.npy
SSH_2002-01-01_2023-12-31_0_24_105_121.npy
SSS_2002-01-01_2023-12-31_0_24_105_121.npy
T-FIELD_2002-01-01_2023-12-31_0_24_105_121_0-300.npy
```

### train：一次联合训练全部深度

下面以服务器上的GLORYS12V1 28层温度实验为例。请根据服务器实际位置替换`/root/autodl-fs/raw_all_size`和运行产物根目录。全分辨率训练建议从batch size 1开始，确认显存余量后再增加。

```bash
cd /root/ocean-subsurface-recon-b1
python -u train.py \
  --method joint_du_unet \
  --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir /root/autodl-fs/raw_all_size \
  --start-date 2002-01-01 \
  --end-date 2023-12-31 \
  --output-dir /root/autodl-tmp/ocean-subsurface-recon-b1-runs/outputs \
  --checkpoint-dir /root/autodl-tmp/ocean-subsurface-recon-b1-runs/checkpoints \
  --epochs 20 \
  --batch-size 1 \
  --lr 1e-3 \
  --patience 10 \
  --lazy-data \
  --normalization-chunk-size 8 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --persistent-workers \
  --seed 42
```

训练输出：

```text
/root/autodl-tmp/ocean-subsurface-recon-b1-runs/
├── checkpoints/2dto2d/Joint_Du_Unet/temperature/
│   └── Joint_Du_Unet_temperature_28depths_best.pth
└── outputs/2dto2d/Joint_Du_Unet/temperature/
    ├── Joint_Du_Unet_train_config.json
    ├── training_history_Joint_Du_Unet_temperature_28depths.npz
    └── loss_Joint_Du_Unet_temperature_28depths.png
```

首次正式运行会用2002–2021训练期拟合归一化统计量并保存在data目录的`normalization_stats/`中；之后训练和测试应复用同一统计文件，不要在正式测试时使用`--rebuild-normalization-stats`。

### test：单日预测

GLORYS12V1中约50m对应深度索引17（47.37m），索引18为55.76m：

```bash
cd /root/ocean-subsurface-recon-b1
python -u test.py \
  --method joint_du_unet \
  --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir /root/autodl-fs/raw_all_size \
  --start-date 2002-01-01 \
  --end-date 2023-12-31 \
  --checkpoint /root/autodl-tmp/ocean-subsurface-recon-b1-runs/checkpoints/2dto2d/Joint_Du_Unet/temperature/Joint_Du_Unet_temperature_28depths_best.pth \
  --output-dir /root/autodl-tmp/ocean-subsurface-recon-b1-runs/outputs \
  --select-day 2023-06-15 \
  --target-level 17 \
  --lazy-data \
  --normalization-chunk-size 8
```

### test/eval：2023年时间段评估

```bash
cd /root/ocean-subsurface-recon-b1
python -u test.py \
  --method joint_du_unet \
  --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir /root/autodl-fs/raw_all_size \
  --start-date 2002-01-01 \
  --end-date 2023-12-31 \
  --checkpoint /root/autodl-tmp/ocean-subsurface-recon-b1-runs/checkpoints/2dto2d/Joint_Du_Unet/temperature/Joint_Du_Unet_temperature_28depths_best.pth \
  --output-dir /root/autodl-tmp/ocean-subsurface-recon-b1-runs/outputs \
  --eval-start 2023-01-01 \
  --eval-end 2023-12-31 \
  --eval-batch-size 1 \
  --lazy-data \
  --normalization-chunk-size 8
```

时间段评估会输出overall、逐深度和逐日的MAE、RMSE、R²和相关系数，以及对应曲线和JSON汇总。测试时模型始终先输出全部28层；如传`--depth-indices`，只会在完整前向后选择部分深度计算指标。

## HVCARefiner

HVCARefiner 是 Du_Unet 之后的二阶段残差修正模型。它冻结 Du_Unet，先生成或读取 `T0` cache，再训练一个全深度 refiner。

### train：二阶段残差修正训练

输入命令：

```bash
python -u train_hvca_refiner.py \
  --target-var temperature \
  --data-dir ./data/raw \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --base-checkpoint-dir ./checkpoints/2dto2d/Du_Unet \
  --t0-cache-dir outputs/2dto2d/HVCARefiner/t0_cache_train \
  --val-cache-dir outputs/2dto2d/HVCARefiner/t0_cache_val \
  --save-dir outputs/2dto2d/HVCARefiner \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --dim 64 \
  --num-heads 4 \
  --num-layers 2 \
  --dropout 0.1 \
  --column-chunk-size 8192 \
  --num-workers 0 \
  --seed 42
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--target-var` | 修正变量，`temperature` 或 `salinity`，需与 Du_Unet checkpoint 对应。 |
| `--data-dir` | 原始数据目录。 |
| `--start-date` / `--end-date` | 数据日期范围。 |
| `--base-checkpoint-dir` | 第一阶段 Du_Unet checkpoint 根目录。 |
| `--t0-cache-dir` | 训练集 T0 cache 目录；不存在时会生成。 |
| `--val-cache-dir` | 验证集 T0 cache 目录；不存在时会生成。 |
| `--depths` | 可选，逗号分隔深度米数；不传使用全部 `DEPTH_LEVELS_25M`。 |
| `--epochs` | 训练轮数。 |
| `--batch-size` | cache dataset batch size。 |
| `--cache-shard-size` | 生成 T0 cache 时每个 shard 的样本数。 |
| `--lr` / `--weight-decay` | AdamW 优化器参数。 |
| `--dim` / `--num-heads` / `--num-layers` / `--dropout` | HVCARefiner 模型结构参数。 |
| `--column-chunk-size` | attention 分块列数，用于控制显存。 |
| `--num-workers` / `--prefetch-factor` / `--shard-cache-size` | cache DataLoader 读取参数。 |
| `--save-dir` | HVCA 输出目录，默认 `outputs/2dto2d/HVCARefiner`。 |
| `--device` | 可选，指定 `cuda`、`cpu` 等；不传自动选择。 |
| `--seed` | 随机种子。 |
| `--max-samples` | 可选，截断训练样本数用于调试。 |

输出文件：

| 路径 | 内容 |
| --- | --- |
| `<save-dir>/HVCARefiner_train_config.json` | 本次 HVCA 训练配置快照。 |
| `<save-dir>/best.pt` | 最佳 HVCARefiner checkpoint。 |
| `<save-dir>/training_history_HVCARefiner.npz` | train/val loss 数组。 |
| `<save-dir>/loss_HVCARefiner.png` | loss 曲线。 |
| `<t0-cache-dir>/` | 训练 T0 cache shard 和 metadata。 |
| `<val-cache-dir>/` | 验证 T0 cache shard 和 metadata。 |

### test/eval：时间段评估和空间解释图

输入命令：

```bash
python test_hvca_refiner.py \
  --target-var temperature \
  --data-dir ./data/raw \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --eval-start 2023-01-01 --eval-end 2023-12-31 \
  --checkpoint outputs/2dto2d/HVCARefiner/best.pt \
  --t0-cache-dir outputs/2dto2d/HVCARefiner/t0_cache_test \
  --base-checkpoint-dir ./checkpoints/2dto2d/Du_Unet \
  --save-dir outputs/2dto2d/HVCARefiner \
  --batch-size 4 \
  --column-chunk-size 8192 \
  --num-workers 0 \
  --spatial-panels \
  --target-levels 0,4,9,17,24
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--checkpoint` | 已训练的 HVCARefiner checkpoint，通常是 `best.pt`。 |
| `--t0-cache-dir` | 测试集 T0 cache 目录；不存在时会生成。 |
| `--base-checkpoint-dir` | 第一阶段 Du_Unet checkpoint 根目录。 |
| `--target-var` | 评估变量，需与 checkpoint 对应。 |
| `--eval-start` / `--eval-end` | 闭区间评估日期。 |
| `--depths` | 仅旧 checkpoint 缺少 `depth_values` 时使用。 |
| `--batch-size` | cache 推理 batch size。 |
| `--cache-shard-size` | 生成测试 T0 cache 时每个 shard 的样本数。 |
| `--dim` / `--num-heads` / `--num-layers` / `--dropout` | 仅旧 checkpoint 缺少 `model_config` 时使用。 |
| `--column-chunk-size` | 仅旧 checkpoint 缺少 `model_config` 时使用。 |
| `--thermocline-min` / `--thermocline-max` | 温跃层 RMSE 统计深度范围。 |
| `--spatial-panels` | 开启 T0/HVCA/truth/error/delta 二维解释面板输出。 |
| `--target-levels` | 空间解释面板使用的深度层索引。 |
| `--panel-days` | 可选，指定面板日期；不传则自动选择 best/median/worst。 |
| `--section-panels` | 开启 T0/HVCA/truth/error/delta 垂向剖面六子图。 |
| `--section-axis` | 剖面固定轴，`lat` 表示固定纬度画 depth-longitude，`lon` 表示固定经度画 depth-latitude。 |
| `--section-index` | 剖面固定轴索引，整数或 `middle`。 |
| `--section-width` | 固定轴方向平均格点宽度，默认 1 表示不做带平均。 |
| `--save-dir` | 输出目录，默认 `outputs/2dto2d/HVCARefiner`。 |
| `--device` / `--max-samples` | 设备指定和调试截断。 |

输出文件：

| 路径 | 内容 |
| --- | --- |
| `<save-dir>/HVCARefiner_test_config.json` | 本次 HVCA 测试配置快照，包含 `single: false`。 |
| `outputs/2dto2d/HVCARefiner/evaluation_metrics_HVCARefiner_{target_var}_{YYYY-MM-DD}_{YYYY-MM-DD}.npz` | baseline_T0 和 HVCA_refined 的 overall/by-depth/by-day 指标。 |
| `outputs/2dto2d/HVCARefiner/summary_HVCARefiner_{target_var}_{YYYY-MM-DD}_{YYYY-MM-DD}.json` | 汇总指标、温跃层指标、垂向梯度指标和面板记录。 |
| `outputs/2dto2d/HVCARefiner/metrics_by_depth_HVCARefiner_{target_var}_{YYYY-MM-DD}_{YYYY-MM-DD}.png` | baseline/refined 逐深度对比曲线。 |
| `outputs/2dto2d/HVCARefiner/metrics_by_day_HVCARefiner_{target_var}_{YYYY-MM-DD}_{YYYY-MM-DD}.png` | baseline/refined 逐日对比曲线。 |
| `outputs/2dto2d/HVCARefiner/map_hvca_panel_*` | 开启 `--spatial-panels` 时生成的二维解释面板。 |
| `outputs/2dto2d/HVCARefiner/section_hvca_panel_*` | 开启 `--section-panels` 时生成的垂向剖面解释面板。 |
| `outputs/2dto2d/HVCARefiner/t0_cache_test/` | 测试 T0 cache shard 和 metadata。 |

## OceanTransformer

OceanTransformer 是 `2dto3d` 深度学习模型，单模型直接输出多深度温盐场。

### train：端到端 2dto3d 训练

输入命令：

```bash
python train.py --method ocean_transformer \
  --data-dir ./data/raw \
  --output-dir ./outputs \
  --checkpoint-dir ./checkpoints \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-4 \
  --patience 10 \
  --seed 42
```

合成数据调试：

```bash
python train.py --method ocean_transformer --dummy
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--method` | 固定为 `ocean_transformer`。 |
| `--data-dir` | 原始数据目录。 |
| `--output-dir` | 输出根目录。 |
| `--checkpoint-dir` | checkpoint 根目录。 |
| `--epochs` / `--batch-size` / `--lr` / `--patience` | 覆盖 `TWODTO3D_*` 默认训练超参数。 |
| `--dummy` | 使用合成数据调试，不读真实数据。 |
| `--seed` | 随机种子。 |

输出文件：

| 路径 | 内容 |
| --- | --- |
| `outputs/2dto3d/ocean_transformer/training_history_ocean_transformer.npz` | train/val loss 数组。 |
| `outputs/2dto3d/ocean_transformer/loss_ocean_transformer.png` | loss 曲线。 |
| `checkpoints/2dto3d/ocean_transformer/ocean_transformer_best.pth` | 最佳 checkpoint。 |

### test：单日预测和可视化

输入命令：

```bash
python test.py --method ocean_transformer \
  --select-day 2023-06-15 \
  --target-level 5 \
  --data-dir ./data/raw \
  --checkpoint ./checkpoints/2dto3d/ocean_transformer/ocean_transformer_best.pth \
  --output-dir ./outputs
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--method` | 固定为 `ocean_transformer`。 |
| `--select-day` | 单日预测日期。 |
| `--target-level` | 可视化深度层索引。 |
| `--data-dir` | 原始数据目录。 |
| `--checkpoint` | 模型权重路径；不传或不存在时会使用随机初始化模型。 |
| `--dummy` | 使用合成测试数据。 |
| `--output-dir` | 输出根目录。 |

输出文件位于 `outputs/2dto3d/ocean_transformer/`：

| 文件 | 内容 |
| --- | --- |
| `pred_ocean_transformer_{YYYYMMDD}.npy` | 预测温盐场，形状为 `(1,D,H,W,2)`。 |
| `map_temp_lvl{level}_ocean_transformer_{YYYYMMDD}.png` | 指定层温度预测图。 |
| `map_salt_lvl{level}_ocean_transformer_{YYYYMMDD}.png` | 指定层盐度预测图。 |
| `grid_metrics_ocean_transformer_{YYYYMMDD}.npz` | 网格 MAE/RMSE。 |
| `map_mae_lvl{level}_ocean_transformer_{YYYYMMDD}.png` | 指定层 MAE 图。 |
| `summary_ocean_transformer_{YYYYMMDD}.json` | 温度、盐度汇总指标。 |

## 2DVar

2DVar 是非深度学习变分反演方法，没有独立训练脚本，只通过 `test.py` 做单日反演。

### test：单日变分反演

输入命令：

```bash
python test.py --method 2dvar \
  --select-day 2023-06-15 \
  --target-level 10 \
  --sla-sss-path ./data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy \
  --sws-true-path ./data/raw/sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy \
  --c-depth 26 \
  --output-dir ./outputs
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--method` | 固定为 `2dvar`。 |
| `--select-day` | 单日反演日期。 |
| `--target-level` | 可视化深度层索引。 |
| `--sla-sss-path` | SLA/SSS 输入文件路径。 |
| `--sws-true-path` | 可选真值文件；提供后会计算指标和误差图。 |
| `--c-depth` | 2DVar 输出深度层数，默认 26。 |
| `--output-dir` | 输出根目录。 |

输出文件位于 `outputs/2dto3d/2dvar/`：

| 文件 | 内容 |
| --- | --- |
| `pred_2dvar_{YYYYMMDD}.npy` | 单日反演结果。 |
| `map_temperature_lvl{level}_2dvar_{YYYYMMDD}.png` | 指定层预测图。 |
| `summary_2dvar_{YYYYMMDD}.json` | 单日汇总；未提供真值时指标为空。 |
| `grid_metrics_2dvar_{YYYYMMDD}.npz` | 提供真值时生成的网格指标。 |
| `map_mae_lvl{level}_2dvar_{YYYYMMDD}.png` | 提供真值时生成的指定层 MAE 图。 |

## MODAS

MODAS 是统计回归反演方法，没有独立训练脚本；`test.py` 会在运行时用固定训练切片拟合回归系数，然后预测目标日。

### test：单日统计反演

输入命令：

```bash
python test.py --method modas \
  --select-day 2023-06-15 \
  --target-level 10 \
  --sla-sss-path ./data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy \
  --sws-true-path ./data/raw/sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy \
  --output-dir ./outputs
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--method` | 固定为 `modas`。 |
| `--select-day` | 单日预测日期。 |
| `--target-level` | 可视化深度层索引。 |
| `--sla-sss-path` | SLA/SSS 输入文件路径。 |
| `--sws-true-path` | 必需，既用于拟合统计关系，也用于评估。 |
| `--output-dir` | 输出根目录。 |

输出文件位于 `outputs/2dto3d/modas/`：

| 文件 | 内容 |
| --- | --- |
| `pred_modas_{YYYYMMDD}.npy` | 单日预测结果。 |
| `map_temperature_lvl{level}_modas_{YYYYMMDD}.png` | 指定层预测图。 |
| `grid_metrics_modas_{YYYYMMDD}.npz` | 网格指标。 |
| `map_mae_lvl{level}_modas_{YYYYMMDD}.png` | 指定层 MAE 图。 |
| `summary_modas_{YYYYMMDD}.json` | 单日汇总指标。 |

## 指标口径

- 深度学习模型会在评估前反标准化回原始物理单位。
- Du_Unet、Joint_Du_Unet 和 HVCARefiner 的指标会排除 `target_mask=0` 的无效区域。
- 时间段评估包含 overall、by-depth、by-day 三类统计。
- overall RMSE 是所有有效点平方误差总体均值开方，不是每日 RMSE 的平均。

## 常用配置位置

- 路径和日期：`config.py` 中的 `DATA_DIR`、`OUTPUTS_ROOT`、`CHECKPOINTS_ROOT`、`DATA_START_DATE`、`TEST_START_DATE`。
- Du_Unet 超参数：`DU_UNET_*`。
- Joint_Du_Unet 沿用 `DU_UNET_*` 优化设置，并动态读取当前profile的深度数。
- OceanTransformer 超参数：`TWODTO3D_*`。
- 深度列表：`DEPTH_LEVELS_25M`、`DEPTH_LEVELS_26M`。
