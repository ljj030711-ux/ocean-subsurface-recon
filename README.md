# Ocean Subsurface Recon

本项目用海表场重建水下多深度温度或盐度结构，当前包含 `Du_Unet`、`Joint_Du_Unet`、`Joint_DUViT`、`B3_DepthFormer`、`HVCARefiner`、`ocean_transformer`、`2dvar` 和 `modas`。

## 总览

### 建模范式

- `2dto2d`：`Du_Unet`为逐深度B0，`Joint_Du_Unet`为联合输出B1，`Joint_DUViT`为在两个瓶颈加入空间ViT的B2，`B3_DepthFormer`为BOS启动的深度自回归Transformer，`HVCARefiner`为B0旁支的二阶段修正模型。
- `2dto3d`：单次输出多个深度。当前包括 `ocean_transformer`、`2dvar`、`modas`。

### 目录规则

- 训练/测试结果根目录：`outputs/{paradigm}/{method}/`
- checkpoint 根目录：`checkpoints/{paradigm}/{method}/`
- Du_Unet 输出：`outputs/2dto2d/Du_Unet/{temperature|salinity}/`
- Du_Unet 权重：`checkpoints/2dto2d/Du_Unet/{temperature|salinity}/`
- Joint_Du_Unet 输出：`outputs/2dto2d/Joint_Du_Unet/{temperature|salinity}/`
- Joint_Du_Unet 权重：`checkpoints/2dto2d/Joint_Du_Unet/{temperature|salinity}/`
- Joint_DUViT 输出：`outputs/2dto2d/Joint_DUViT/{temperature|salinity}/`
- Joint_DUViT 权重：`checkpoints/2dto2d/Joint_DUViT/{temperature|salinity}/`
- B3_DepthFormer 输出：`outputs/2dto2d/B3_DepthFormer/{temperature|salinity}/`
- B3_DepthFormer 权重：`checkpoints/2dto2d/B3_DepthFormer/{temperature|salinity}/`
- HVCARefiner 输出与权重：`outputs/2dto2d/HVCARefiner/`
- OceanTransformer 输出：`outputs/2dto3d/ocean_transformer/`
- 2DVar 输出：`outputs/2dto3d/2dvar/`
- MODAS 输出：`outputs/2dto3d/modas/`

### 运行日志和配置快照

Du_Unet、Joint_Du_Unet、Joint_DUViT、B3_DepthFormer和HVCARefiner的train/test入口会在运行前打印最终超参数，并保存JSON配置快照：

- `outputs/2dto2d/Du_Unet/Du_Unet_train_config.json`
- `outputs/2dto2d/Du_Unet/Du_Unet_test_config.json`
- `outputs/2dto2d/Joint_Du_Unet/{target_var}/Joint_Du_Unet_train_config_{depth_tag}.json`
- `outputs/2dto2d/Joint_Du_Unet/{target_var}/Joint_Du_Unet_test_config_{depth_tag}.json`
- `outputs/2dto2d/Joint_DUViT/{target_var}/Joint_DUViT_train_config_{depth_tag}.json`
- `outputs/2dto2d/Joint_DUViT/{target_var}/Joint_DUViT_test_config_{depth_tag}.json`
- `outputs/2dto2d/B3_DepthFormer/{target_var}/B3_DepthFormer_train_config_{depth_tag}.json`
- `outputs/2dto2d/B3_DepthFormer/{target_var}/B3_DepthFormer_test_config_{depth_tag}.json`
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

## B3：DepthFormer 深度自回归模型

B3 将 SST、SSH、SSS 编码为共享海表记忆，以可学习 BOS token 启动，按真实深度从浅到深生成目标温度层。模型结构、指定六层实验、CPU smoke test、训练/评估指令和完整超参数见 [B3 实现说明](docs/B3_DEPTHFORMER.md)。

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
| `--eval-start` / `--eval-end` | 闭区间评估日期；支持Du_Unet及联合深度模型。 |
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

`Joint_Du_Unet` 保留B0相同的SST高分辨率分支、SSH/SSS低分辨率分支和普通卷积U-Net骨干，只把逐层独立训练改成一个模型联合输出所选深度。模型不含注意力、Transformer、深度嵌入、垂向正则或HVCA。

- `--depth-indices`是原始profile层号，输出通道数自动等于传入层数，不需要手工配置。
- 不传`--depth-indices`时，自动联合训练当前profile的全部深度。
- checkpoint保存原始层号及其输出通道顺序；测试会据此校验和映射。
- 各输出层使用等权masked RMSE，以免有效海洋格点较少的深层被自动降低权重。
- 与B0公平对照的主实验使用同一组`0,4,9,17,24,27`，而不是28层训练后只测试6层。

### 从本地打包、备份并部署到服务器

不经过GitHub，运行仓库自带的完整部署脚本：

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon-b1
./scripts/deploy_upload.sh
```

脚本会把当前B1代码打成一个新的`.tar.gz`压缩包，上传到服务器，并先解压到临时目录校验。校验通过后，服务器现有的`/root/ocean-subsurface-recon-b1`会改名为带时间戳的备份目录，新代码再整体切换为正式目录。因此服务器代码与本地压缩包完全一致，本地已删除的旧文件不会残留，也不需要手工清理`tests/`。

默认服务器参数与原`deploy_upload.sh`一致，可按实际AutoDL实例覆盖：

```bash
SERVER_HOST=connect.bjb2.seetacloud.com \
SERVER_PORT=38916 \
SERVER_USER=root \
REMOTE_PARENT_DIR=/root \
./scripts/deploy_upload.sh
```

脚本不会打包`.git`、数据、checkpoint、outputs、logs、cache或密钥文件，所以训练数据和实验结果不会被代码部署覆盖。旧代码备份和本次上传的压缩包都会保留在服务器；解压或校验失败时，当前正式代码目录不会被替换。

### 与B0相同6层的联合训练和全年测试

进入同一个tmux shell后先设置路径。数据目录必须直接包含profile要求的四个`.npy`文件，不能指向NetCDF目录。

```bash
cd /root/ocean-subsurface-recon-b1

DATA=/root/autodl-tmp/raw_all_size/npy
STATS=/root/autodl-fs/raw_all_size/normalization_stats/glorys12v1_temperature_2002_2021.npz
B1_ROOT=/root/autodl-tmp/Joint_Du_Unet_all_size
B1_CKPT="$B1_ROOT/checkpoints"
B1_OUT="$B1_ROOT/outputs"
B1_LOG="$B1_ROOT/logs"
DEPTH_IDX=0,4,9,17,24,27
MODEL="$B1_CKPT/2dto2d/Joint_Du_Unet/temperature/Joint_Du_Unet_temperature_idx0-4-9-17-24-27_6depths_best.pth"

mkdir -p "$B1_CKPT" "$B1_OUT" "$B1_LOG"
set -o pipefail
```

训练一个联合6层模型：

```bash
python -u train.py \
  --method joint_du_unet --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir "$DATA" --normalization-stats-path "$STATS" \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --depth-indices "$DEPTH_IDX" \
  --epochs 50 --patience 10 --batch-size 32 --lr 0.001 \
  --num-workers 4 --prefetch-factor 2 --pin-memory --persistent-workers \
  --checkpoint-dir "$B1_CKPT" --output-dir "$B1_OUT" --seed 42 \
  2>&1 | tee -a "$B1_LOG/joint_du_unet_6depths_train.log"
```

训练完成后，权重应位于`$MODEL`。若显存不足，只降低`--batch-size`并在实验记录中注明；其余条件保持与B0一致。

对同一6层执行2023年全年测试：

```bash
python -u test.py \
  --method joint_du_unet --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir "$DATA" --normalization-stats-path "$STATS" \
  --checkpoint "$MODEL" \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --eval-start 2023-01-01 --eval-end 2023-12-31 \
  --depth-indices "$DEPTH_IDX" --eval-batch-size 8 \
  --spatial-panels --target-levels 0,1,2,3,4,5 \
  --output-dir "$B1_OUT" \
  2>&1 | tee -a "$B1_LOG/joint_du_unet_6depths_test.log"
```

`--target-levels` 使用联合输出内部的局部位置，因此六层模型传
`0,1,2,3,4,5`；它们依次对应原始 GLORYS 层号 `0,4,9,17,24,27`。
水平图为 `Joint prediction / Truth / Absolute error` 三联图。上面的主实验命令没有传入
任何 `section` 参数，因此只调用水平可视化，不生成垂向剖面；垂向模块仍然保留。
不传 `--panel-days` 时，程序会用六层等权的 `structure_score=(SSIM+ACC)/2`，
自动选择最接近 P10、P50、P90 的困难、典型和较好代表日期。如需人工覆盖，再显式传入
`--panel-days YYYY-MM-DD,...`。

如后续需要垂向分析，可在同一条全年测试命令中按需追加以下参数之一；当前水平主实验不追加：

```bash
# 单一剖面：固定中间纬度，画 depth-longitude
--section-panels --section-axis lat --section-index middle --section-width 1

# 多剖面：两个方向均在 10/25/50/75/90% 位置绘制
--multi-section-panels --section-percentiles 10,25,50,75,90 --section-width 1
```

`--section-panels` 只绘制 `--section-axis` 指定的一个方向；
`--multi-section-panels` 才会同时绘制固定纬度的 depth-longitude 和固定经度的
depth-latitude。两种模式均复用 P10/P50/P90 代表日期，输出
`Prediction / Truth / Absolute error` 垂向三联图。

Joint时间段评估保留原有pooled `overall/by_depth/by_day` 指标，同时新增
`day×depth` 的 MAE、RMSE、R²、correlation、SSIM 和 ACC。SSIM与ACC都使用
2002–2021年训练期月气候态距平：ACC表示冷暖异常的空间相关；SSIM表示局部水平结构相似度。
论文主要报告建议使用逐深度MAE/RMSE与SSIM/ACC；pooled R²和correlation仅作辅助，
因为混合多层后可能被垂向温差抬高。

主要产物路径：

```text
/root/autodl-tmp/Joint_Du_Unet_all_size/
├── checkpoints/2dto2d/Joint_Du_Unet/temperature/
│   └── Joint_Du_Unet_temperature_idx0-4-9-17-24-27_6depths_best.pth
├── outputs/2dto2d/Joint_Du_Unet/temperature/
│   ├── Joint_Du_Unet_train_config_idx0-4-9-17-24-27_6depths.json
│   ├── Joint_Du_Unet_test_config_idx0-4-9-17-24-27_6depths.json
│   ├── training_history_Joint_Du_Unet_temperature_idx0-4-9-17-24-27_6depths.npz
│   ├── loss_Joint_Du_Unet_temperature_idx0-4-9-17-24-27_6depths.png
│   ├── evaluation_metrics_Joint_Du_Unet_temperature_20230101_20231231.npz
│   ├── summary_Joint_Du_Unet_temperature_20230101_20231231.json
│   ├── metrics_by_depth_Joint_Du_Unet_temperature_20230101_20231231.png
│   ├── metrics_by_day_Joint_Du_Unet_temperature_20230101_20231231.png
│   ├── horizontal_structure_by_depth_Joint_Du_Unet_temperature_20230101_20231231.png
│   ├── horizontal_structure_by_day_Joint_Du_Unet_temperature_20230101_20231231.png
│   └── visualizations/idx0-4-9-17-24-27_6depths/
│       ├── horizontal/
│       │   └── map_joint_panel_temperature_gidx{original_depth_index}_lvl{local_level}_{date}.png
│       └── vertical/                       # 仅传section参数时生成
│           └── section_joint_panel_temperature_{section_tag}_{date}.png
└── logs/
    ├── joint_du_unet_6depths_train.log
    └── joint_du_unet_6depths_test.log
```

B1不训练HVCA。HVCA只保留为B0的两阶段旁支对照；把它接到B1后面会同时改变“联合输出”和“二阶段修正”两个变量，无法单独回答B1是否有效。

## B2：Joint_DUViT 双瓶颈空间ViT

`Joint_DUViT`从B1独立派生，不修改`Joint_Du_Unet`。模型保持相同的SST、SSH、SSS输入、双分支卷积骨干、普通跳连、融合头和动态联合深度输出，只新增两个空间ViT瓶颈：

- SST瓶颈：256通道、Patch 1、Embedding 256、8 heads、2层、MLP ratio 4。
- SSH/SSS瓶颈：64通道、Patch 2、Embedding 128、4 heads、2层、MLP ratio 4。
- 两个ViT均采用可插值的可学习二维位置编码、Pre-LayerNorm、GELU和dropout 0.1。
- 全局注意力Token上限分别为256和512，控制全区域数据的显存开销；CNN残差保留原始局部特征。
- 当前版本`skip_attention=false`、`vertical_transformer=false`，没有开启跳连Attention Gate，也没有HVCA。
- 六层输出通道数仍由`--depth-indices 0,4,9,17,24,27`自动得到，无需手工设置为6。
- 六层B2共5,802,950个可训练参数；B1六层为2,943,110个。

### 从本地打包、备份并部署到服务器

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon-b2
./scripts/deploy_upload.sh
```

脚本会上传压缩包、在服务器临时目录解压校验，并把旧的
`/root/ocean-subsurface-recon-b2`改名为带时间戳的备份后整体替换。服务器地址变化时：

```bash
SERVER_HOST=connect.bjb2.seetacloud.com \
SERVER_PORT=38916 \
SERVER_USER=root \
REMOTE_PARENT_DIR=/root \
./scripts/deploy_upload.sh
```

### 同六层训练

在同一个tmux shell设置路径：

```bash
cd /root/ocean-subsurface-recon-b2

DATA=/root/autodl-tmp/raw_all_size/npy
STATS=/root/autodl-fs/raw_all_size/normalization_stats/glorys12v1_temperature_2002_2021.npz
B2_ROOT=/root/autodl-tmp/Joint_DUViT_all_size
B2_CKPT="$B2_ROOT/checkpoints"
B2_OUT="$B2_ROOT/outputs"
B2_LOG="$B2_ROOT/logs"
DEPTH_IDX=0,4,9,17,24,27
MODEL="$B2_CKPT/2dto2d/Joint_DUViT/temperature/Joint_DUViT_temperature_idx0-4-9-17-24-27_6depths_best.pth"

mkdir -p "$B2_CKPT" "$B2_OUT" "$B2_LOG"
set -o pipefail
```

先用batch size 4训练；确认显存余量后可以提高，但同一组比较需记录实际batch size：

```bash
python -u train.py \
  --method joint_duvit --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir "$DATA" --normalization-stats-path "$STATS" \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --depth-indices "$DEPTH_IDX" \
  --epochs 50 --patience 10 --batch-size 4 --lr 0.001 \
  --num-workers 4 --prefetch-factor 2 --pin-memory --persistent-workers \
  --checkpoint-dir "$B2_CKPT" --output-dir "$B2_OUT" --seed 42 \
  2>&1 | tee -a "$B2_LOG/joint_duvit_6depths_train.log"
```

### 2023年全年测试和水平图

```bash
python -u test.py \
  --method joint_duvit --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir "$DATA" --normalization-stats-path "$STATS" \
  --checkpoint "$MODEL" \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --eval-start 2023-01-01 --eval-end 2023-12-31 \
  --depth-indices "$DEPTH_IDX" --eval-batch-size 4 \
  --spatial-panels --target-levels 0,1,2,3,4,5 \
  --output-dir "$B2_OUT" \
  2>&1 | tee -a "$B2_LOG/joint_duvit_6depths_test.log"
```

这条测试命令只生成水平面板，不传`section`参数，因此不调用垂向可视化。主要路径：

```text
/root/autodl-tmp/Joint_DUViT_all_size/
├── checkpoints/2dto2d/Joint_DUViT/temperature/
│   └── Joint_DUViT_temperature_idx0-4-9-17-24-27_6depths_best.pth
├── outputs/2dto2d/Joint_DUViT/temperature/
│   ├── Joint_DUViT_train_config_idx0-4-9-17-24-27_6depths.json
│   ├── Joint_DUViT_test_config_idx0-4-9-17-24-27_6depths.json
│   ├── training_history_Joint_DUViT_temperature_idx0-4-9-17-24-27_6depths.npz
│   ├── loss_Joint_DUViT_temperature_idx0-4-9-17-24-27_6depths.png
│   ├── evaluation_metrics_Joint_DUViT_temperature_20230101_20231231.npz
│   ├── summary_Joint_DUViT_temperature_20230101_20231231.json
│   ├── metrics_by_depth_Joint_DUViT_temperature_20230101_20231231.png
│   ├── metrics_by_day_Joint_DUViT_temperature_20230101_20231231.png
│   ├── horizontal_structure_by_depth_Joint_DUViT_temperature_20230101_20231231.png
│   ├── horizontal_structure_by_day_Joint_DUViT_temperature_20230101_20231231.png
│   └── visualizations/idx0-4-9-17-24-27_6depths/horizontal/
└── logs/
    ├── joint_duvit_6depths_train.log
    └── joint_duvit_6depths_test.log
```

B2与B1的主对比保持相同输入、六层、数据划分、归一化、损失、学习率、训练轮数和随机种子，重点比较逐深度RMSE以及年度SSIM、ACC。水平案例使用同一批日期，不能让两个模型各自挑选不同日期后直接比较。

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
- Du_Unet、Joint_Du_Unet、Joint_DUViT和HVCARefiner的指标会排除`target_mask=0`的无效区域。
- 时间段评估包含 overall、by-depth、by-day 三类统计。
- overall RMSE 是所有有效点平方误差总体均值开方，不是每日 RMSE 的平均。
- Joint_Du_Unet和Joint_DUViT额外保存`by_day_depth_*`，用于避免把时间、深度和空间混合后掩盖局部误差。
- Joint_Du_Unet和Joint_DUViT的SSIM、ACC按每天、每层单独计算，并等权汇总为年度水平结构指标。

## 常用配置位置

- 路径和日期：`config.py` 中的 `DATA_DIR`、`OUTPUTS_ROOT`、`CHECKPOINTS_ROOT`、`DATA_START_DATE`、`TEST_START_DATE`。
- Du_Unet 超参数：`DU_UNET_*`。
- Joint_Du_Unet 沿用 `DU_UNET_*` 优化设置，并动态读取当前profile的深度数。
- Joint_DUViT沿用`DU_UNET_*`优化设置；空间ViT结构参数位于`JOINT_DUVIT_*`。
- OceanTransformer 超参数：`TWODTO3D_*`。
- 深度列表：`DEPTH_LEVELS_25M`、`DEPTH_LEVELS_26M`。
