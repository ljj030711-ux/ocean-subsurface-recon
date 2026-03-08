# Eddy Inversion - 涡旋反演项目

## 项目简介

这是一个**研究级的深度学习项目**，用于涡旋结构的水下温度反演。该项目采用标准的工程化架构，可直接支撑硕士论文的方法章节。

## 项目特点

✅ **完整的工程闭环**
- 数据读取 → 特征构造 → 模型 → 训练 → 测试
- 真实输出结果文件，支持后续分析

✅ **清晰的代码结构**
- 按功能模块划分（数据、模型、工具、训练）
- 易于维护、升级和扩展

✅ **标准化的数据格式**
- 统一 `.npy` 格式
- `[T, H, W]` 形状规范

✅ **物理约束集成**
- EKE (涡动动能) 计算
- SSH 梯度特征
- 光滑性正则化

## 目录结构

```
eddy_inversion/
├── data/
│   ├── raw/                  # 原始真实数据
│   └── demo/                 # 示例数据
│       ├── sst.npy          # 海表温度 [T, H, W]
│       ├── sss.npy          # 海表盐度 [T, H, W]
│       ├── ssh.npy          # 海面高度 [T, H, W]
│       └── subsurface.npy   # 水下目标（100m温度） [T, H, W]
│
├── datasets/
│   └── eddy_dataset.py       # PyTorch 数据集类
│
├── models/
│   └── eddy_cnn.py           # MVP 模型 (CNN)
│
├── utils/
│   ├── physics.py            # 物理量计算 (EKE, ∇SSH)
│   └── metrics.py            # 评估指标
│
├── train.py                  # 训练脚本
├── test.py                   # 推理 & 结果输出
├── config.py                 # 超参数配置
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

输出：
- `model.pth` - 训练好的模型权重

### 3. 测试与输出结果

```bash
python test.py
```

输出：
- `prediction.npy` - 反演结果 `[T, 1, H, W]`

## 数据格式规范

所有输入数据统一为 `.npy` 格式，形状规范：

| 变量 | 形状 | 说明 |
|------|------|------|
| sst | `[T, H, W]` | 海表温度 |
| sss | `[T, H, W]` | 海表盐度 |
| ssh | `[T, H, W]` | 海面高度 |
| target | `[T, H, W]` | 目标变量（如100m水温） |

其中：
- `T` - 时间步数
- `H` - 空间高度（纬度）
- `W` - 空间宽度（经度）

## 核心模块说明

### 物理工具 (`utils/physics.py`)

- **EKE 计算** - 涡动动能，反映涡旋强度
- **SSH梯度** - 地转流速度

### 数据集 (`datasets/eddy_dataset.py`)

- 自动加载所有输入变量
- 返回包含 sst, sss, ssh, target 的字典
- 支持动态特征工程

### 模型 (`models/eddy_cnn.py`)

- **输入**：5通道 `[sst, sss, ssh, eke, grad_ssh]`
- **输出**：1通道反演结果
- **架构**：4层 CNN (可升级到 UNet/SegFormer)

### 训练 (`train.py`)

- MSE 损失 + 光滑性正则化
- 自动保存最佳模型
- 实时损失监控

### 测试 (`test.py`)

- 加载训练好的模型
- 对所有样本进行推理
- 输出 `.npy` 结果文件

## 升级路线

### 第一阶段：替换骨干网络
```python
EddyAwareCNN → UNet / SegFormer
```

### 第二阶段：多深度反演
```python
target: [T, H, W] → [T, D, H, W]
```

### 第三阶段：物理约束增强
- 涡旋极性 mask
- 等密度面 loss
- 位势涡度 (PV) 约束

## 配置参数说明

编辑 `config.py` 调整：

```python
BATCH_SIZE = 4        # 批处理大小
EPOCHS = 20           # 训练轮数
LR = 1e-3             # 学习率
LAMBDA_SMOOTH = 0.1   # 光滑正则化系数
DEVICE = "cpu"        # 计算设备
```

## 数据准备

如需使用自己的数据，将其放入 `data/raw/`，需满足格式：

```python
sst.npy    # 形状 [T, H, W]
sss.npy    # 形状 [T, H, W]
ssh.npy    # 形状 [T, H, W]
target.npy # 形状 [T, H, W]
```

然后修改 `config.py` 中的 `DATA_DIR`。

## 输出解释

训练完成后：

```
model.pth        # PyTorch 模型文件
prediction.npy   # 推理结果，形状 [T, 1, H, W]
```

可使用 matplotlib 或 paraview 可视化：

```python
import numpy as np
import matplotlib.pyplot as plt

pred = np.load("prediction.npy")
plt.imshow(pred[0, 0])  # 第一个时间步的结果
plt.colorbar()
plt.show()
```

## 论文相关

该项目架构符合硕士论文方法章节的标准，包含：

- ✅ 数据处理与特征工程
- ✅ 物理约束设计
- ✅ 深度学习模型
- ✅ 训练与验证流程
- ✅ 结果评估指标

后续研究可基于本框架进行方法创新，而无需重构整个项目。

## 许可证

MIT License

## 联系方式

如有问题，欢迎讨论与反馈。

