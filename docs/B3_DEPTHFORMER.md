# B3 DepthFormer 实现说明

> GL-Geoformer 官方源码与 B3 各模块的逐项对应关系见
> [B3 DepthFormer 与 GL-Geoformer 官方代码对应报告](B3_GLGeoFormer_CODE_MAPPING.md)。

B3 首先对 SST、SSH 和 SSS 进行一次编码，然后以可学习的 BOS token 为起点，
从浅到深生成指定的温度层。

## 代码结构

~~~text
models/b3_depthformer/
├── embeddings.py   # Patch Embedding、二维位置编码和真实深度编码
├── decoder.py      # 因果深度注意力、交叉注意力和 Patch Fold
├── model.py        # 海表编码器、Teacher Forcing 和自回归生成
└── __init__.py
tests/test_b3_depthformer.py
scripts/smoke_b3.py
~~~

`train.py` 和 `test.py` 只负责组织公共数据、checkpoint 与输出流程，因此归一化、
指标计算和可视化均与 B2 保持一致。

## 模型结构与官方代码复用关系

1. SST、SSH 和 SSS 分别使用独立的卷积 Patch Embedding。
2. 三个特征图会对齐到统一输出 patch 网格，因此允许源数据分辨率不同。
3. 融合后的 token 加入可学习二维位置编码，再进入海表 Transformer Encoder。
4. Decoder 的第一个 query 为 BOS + 空间位置编码 + 真实深度编码。
5. 后续 query 为 PatchEmbed（前一个温度层）+ 空间位置编码 + 当前深度编码。
6. 因果深度注意力禁止浅层位置读取尚未生成的更深层。
7. 海表交叉注意力读取共享的 SST/SSH/SSS memory。
8. Linear patch 投影和 Fold 恢复各深度温度图，小型卷积残差用于减轻 patch 边界伪影。

B3 在机制层面复用了 GL-Geoformer 的卷积 Patch Embedding、因果 mask、
Encoder-Decoder 交叉注意力、滚动生成和 Patch Fold，但没有复制其时间预测数据流程。
B3 将时间轴替换为真实深度轴，并将最后一个历史输入帧替换为可学习 BOS token。

## 当前六深度实验

GLORYS 原始深度索引：

~~~text
0,4,9,17,24,27
~~~

对应物理深度：

~~~text
0.494025, 5.078224, 13.467140, 47.373690, 155.850700, 266.040300 m
~~~

深度索引必须按照从浅到深的顺序传入。模型本身支持动态深度数量；后续不传
`--depth-indices` 时，会使用当前数据配置中的全部深度。

## 仅使用 CPU 的 Smoke Test

不需要真实数据或预训练权重：

~~~bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/smoke_b3.py
~~~

测试覆盖不同源分辨率、前向与反向传播、BOS 因果性、自回归生成、Scheduled
Sampling、checkpoint 重载和 B2 兼容路径。

## 六深度训练命令

~~~bash
cd /root/ocean-subsurface-recon-b3

DATA=/root/autodl-tmp/raw_all_size/npy
STATS=/root/autodl-fs/raw_all_size/normalization_stats/glorys12v1_temperature_2002_2021.npz
B3_ROOT=/root/autodl-tmp/B3_DepthFormer_all_size
B3_CKPT="$B3_ROOT/checkpoints"
B3_OUT="$B3_ROOT/outputs"
B3_LOG="$B3_ROOT/logs"
DEPTH_IDX=0,4,9,17,24,27
MODEL="$B3_CKPT/2dto2d/B3_DepthFormer/temperature/B3_DepthFormer_temperature_idx0-4-9-17-24-27_6depths_best.pth"

mkdir -p "$B3_CKPT" "$B3_OUT" "$B3_LOG"
set -o pipefail

python -u train.py \
  --method b3_depthformer --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir "$DATA" --normalization-stats-path "$STATS" \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --depth-indices "$DEPTH_IDX" \
  --epochs 50 --patience 10 --batch-size 16 --lr 0.0001 \
  --num-workers 4 --prefetch-factor 2 --pin-memory --persistent-workers \
  --b3-d-model 128 --b3-nhead 8 \
  --b3-encoder-layers 4 --b3-decoder-layers 3 \
  --b3-dim-feedforward 512 --b3-dropout 0.1 \
  --b3-sst-patch-size 16 --b3-surface-patch-size 4 \
  --b3-output-patch-size 8 --b3-refinement-channels 32 \
  --b3-base-grid-height 16 --b3-base-grid-width 16 \
  --b3-teacher-forcing-start 1.0 \
  --b3-teacher-forcing-end 0.5 \
  --b3-teacher-forcing-decay-epochs 30 \
  --b3-depth-attention causal \
  --checkpoint-dir "$B3_CKPT" --output-dir "$B3_OUT" --seed 42 \
  2>&1 | tee -a "$B3_LOG/b3_depthformer_6depths_train.log"
~~~

上次最终成功的 B3 实验使用 batch size 16。为保持后续实验可比，B3 默认值固定为
16；如因显存不足必须降低，应将该运行视为不同训练配置并单独记录，不能与 batch
size 16 的消融结果直接混用。

## 2023 全年评估

~~~bash
python -u test.py \
  --method b3_depthformer --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir "$DATA" --normalization-stats-path "$STATS" \
  --checkpoint "$MODEL" \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --eval-start 2023-01-01 --eval-end 2023-12-31 \
  --depth-indices "$DEPTH_IDX" --eval-batch-size 1 \
  --spatial-panels --target-levels 0,1,2,3,4,5 \
  --panel-days 2023-07-21,2023-08-12,2023-05-26 \
  --output-dir "$B3_OUT" \
  2>&1 | tee -a "$B3_LOG/b3_depthformer_6depths_test.log"
~~~

上面的日期是当前实验指定的固定案例，生成2023-07-21、2023-08-12和2023-05-26三个日期、六个深度的水平图，并在同一水平图目录保存一个压缩的 `case_fields_*.npz`。该文件包含预测、真值、有效掩膜、训练月气候态、距平、误差、经纬度和逐日逐深度指标，供B1/B2/B3使用统一色标重新绘图。如需同时生成多个位置的垂直剖面图，可在命令中追加：

~~~bash
  --multi-section-panels \
  --section-percentiles 10,25,50,75,90 \
  --section-width 1
~~~

若只生成一个垂直剖面，可改用 `--section-panels`，并通过 `--section-axis`、
`--section-index` 和 `--section-width` 指定位置。`--section-panels` 与
`--multi-section-panels` 不能同时使用。

训练和测试命令通过 `2>&1 | tee -a ...` 同时输出到终端并追加保存到日志文件。
程序还会独立保存配置 JSON、训练历史 NPZ、loss 曲线及评估结果，因此即使不使用
`tee`，结构化结果仍会写入输出目录。

## 因果深度注意力消融

消融比较原始 B3 的 `causal` 与等参数量的 `independent`。后者保留相同的
MultiheadAttention 层和全部权重，但用对角 mask 限制每个深度 token 只能关注自身。
海表 Encoder、海表交叉注意力、自回归前一层输入、重建头和张量尺寸均不变。因此该实验
隔离的是深度自注意力内部的跨深度信息交换，不是 HVCA 消融，也没有移除 B3 的自回归路径。

在服务器 B3 仓库运行：

~~~bash
cd /root/ocean-subsurface-recon-b3

python -u scripts/run_depth_fusion_ablation.py \
  --data-dir /root/autodl-tmp/raw_all_size/npy \
  --normalization-stats-path /root/autodl-fs/raw_all_size/normalization_stats/glorys12v1_temperature_2002_2021.npz \
  --experiment-root /root/autodl-tmp/B3_DepthFormer_depth_attention_ablation_bs16
~~~

追加 `--dry-run` 可只打印 causal/independent 两组的四条训练和测试命令。运行器锁定
batch size 16、seed 42、2002-01-01 至 2023-12-31 数据范围、2023 全年测试、六个相同深度以及
2023-07-21、2023-08-12、2023-05-26 三个二维图日期。运行中断后可使用同一命令续跑：
已有 best checkpoint 时跳过对应训练，已有完整 summary 时跳过对应推理；共同配置不一致时
拒绝混用结果。两组完成后会核验参数量并生成 `comparison.csv`。
旧的 `B3_DepthFormer_depth_attention_ablation` 目录包含 batch size 2 的产物，保留用于
追溯，但不得作为 batch size 16 实验的续跑目录。

当前 CUDA/PyTorch 环境中的 bicubic 位置编码反向算子
`upsample_bicubic2d_backward_out_cuda` 没有确定性实现，因此训练采用
`torch.use_deterministic_algorithms(True, warn_only=True)`：随机种子、模型初始化、
DataLoader 顺序、cuDNN benchmark 等复现条件仍受控，但相同 seed 重跑不保证权重和指标
逐位一致。该策略会写入每组训练配置和顶层 `experiment.json`。解释消融结果时应比较效应
大小；若两组差异很小，建议补充多个随机种子并报告均值与标准差。

## 输入与优化超参数

| 参数 | 取值 |
| --- | --- |
| 输入 | 按月气候态归一化的 SST、SSH、SSS |
| 目标 | 按月气候态归一化的温度场 |
| 深度索引 | 0,4,9,17,24,27 |
| 训练 / 验证 / 测试 | 2002-2021 / 2022 / 2023 |
| 损失函数 | 各深度等权的 masked RMSE |
| 优化器 | AdamW |
| 学习率 / 权重衰减 | 1e-4 / 1e-4 |
| Epoch / Early Stopping patience | 50 / 10 |
| Batch size | 16 |
| 梯度裁剪 | 1.0 |
| 学习率调度 | CosineAnnealing，最低学习率 5e-6 |
| Teacher Forcing | 前 30 个 epoch 从 1.0 线性下降至 0.5 |
| 验证方式 | 完全自回归 |
| 随机种子 | 42 |

## 模型超参数

| 参数 | 取值 |
| --- | --- |
| `d_model` / 注意力头数 | 128 / 8 |
| 海表 Encoder 层数 | 4 |
| 深度 Decoder 层数 | 3 |
| FFN 维度 / Dropout | 512 / 0.1 |
| SST patch size | 16 |
| SSH/SSS patch size | 4 |
| 输出 patch size | 8 |
| 局部修正通道数 | 32 |
| 位置编码基础网格 | 16 x 16，运行时插值 |
| 深度编码 | 对归一化 `log(1+depth)` 使用可学习 MLP |

## 输出约定

~~~text
B3_DepthFormer_all_size/
├── checkpoints/2dto2d/B3_DepthFormer/temperature/
│   └── B3_DepthFormer_temperature_idx0-4-9-17-24-27_6depths_best.pth
└── outputs/2dto2d/B3_DepthFormer/temperature/
    ├── B3_DepthFormer_train_config_idx0-4-9-17-24-27_6depths.json
    ├── B3_DepthFormer_test_config_idx0-4-9-17-24-27_6depths.json
    ├── training_history_B3_DepthFormer_temperature_idx0-4-9-17-24-27_6depths.npz
    ├── loss_B3_DepthFormer_temperature_idx0-4-9-17-24-27_6depths.png
    ├── evaluation_metrics_B3_DepthFormer_temperature_20230101_20231231.npz
    ├── summary_B3_DepthFormer_temperature_20230101_20231231.json
    ├── metrics_by_depth_B3_DepthFormer_temperature_20230101_20231231.png
    ├── metrics_by_day_B3_DepthFormer_temperature_20230101_20231231.png
    ├── horizontal_structure_by_depth_B3_DepthFormer_temperature_20230101_20231231.png
    ├── horizontal_structure_by_day_B3_DepthFormer_temperature_20230101_20231231.png
    └── visualizations/idx0-4-9-17-24-27_6depths/
        ├── horizontal/  # 水平三联图和单个case_fields压缩NPZ
        └── vertical/    # 可选的垂直剖面图
~~~
