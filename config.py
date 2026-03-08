# config.py
"""
全局配置文件 - 集中管理所有超参数
"""

# 数据配置
DATA_DIR = "./data/raw"
DATA_START_DATE = "2019-01-01"  # 第 0 个时间步对应日期（按天序列）
WINDOW_DAYS = 1                 # 当天输入（非滑窗）
PREDICT_HORIZON_DAYS = 0        # 当天预测当天
TRAIN_RATIO = 0.8               # 按时间顺序切分训练/验证

# 训练配置
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3
MODEL_NAME = "EddyUNet"  # 可选: EddyUNet / EddyResNet / EddyAwareCNN
USE_PHYSICS_FEATURES = False  # False: 仅 SSS/SSH(2通道), True: +EKE/+grad(4通道)

# 物理参数
DX = 10000.0   # 10 km
DY = 10000.0

# 正则化
LAMBDA_SMOOTH = 0.1

# 计算设备
DEVICE = "cpu"  # "cuda" 或 "cpu"

# 模型保存路径
MODEL_SAVE_PATH = "./model.pth"
PRED_SAVE_PATH = "./prediction.npy"

# 日志
VERBOSE = True
