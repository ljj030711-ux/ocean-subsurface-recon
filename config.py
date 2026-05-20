"""
全局配置文件 - 集中管理模型超参数与训练/推理常量
"""

# ==================== 全局路径与基础配置 ====================

DATA_DIR = "./data/raw"
OUTPUTS_ROOT = "./outputs"
CHECKPOINTS_ROOT = "./checkpoints"

DATA_START_DATE = "2002-01-01"
DATA_END_DATE = "2023-12-31"
TWODTO3D_DATA_START_DATE = "2019-01-01"
TWODTO3D_DATA_END_DATE = "2023-12-31"

TRAIN_END_DATE = "2021-12-31"
VAL_START_DATE = "2022-01-01"
VAL_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2023-12-31"
TOTAL_DAYS = 8035

WINDOW_DAYS = 1
PREDICT_HORIZON_DAYS = 0
TRAIN_RATIO = 0.8

VERBOSE = True
DEVICE = "cpu"
SEED = 42

# 可视化范围
LON_RANGE = (110, 118)
LAT_RANGE = (10, 18)
DEPTH_MAX = 300

# 物理参数
DX = 10000.0
DY = 10000.0

# ==================== 范式定义 ====================

PARADIGM_2DTO2D = "2dto2d"
PARADIGM_2DTO3D = "2dto3d"

PARADIGM_2DTO2D_METHODS = {"du_unet"}
PARADIGM_2DTO3D_METHODS = {"2dvar", "modas", "ocean_transformer"}

# 指定 26 层深度（旧参数， 单位 m）
DEPTH_LEVELS_26M = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    65, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300
]

# Du_Unet 使用 5-300m 共 25 层，每层、每个变量训练一个独立 2D->2D 模型。
DEPTH_LEVELS_25M = DEPTH_LEVELS_26M[1:]

# 结果保存路径拼接；target_var 用于 Du_Unet 的温度/盐度分目录。
def get_output_dir(paradigm, method, base_dir=OUTPUTS_ROOT, target_var=None):
    path = f"{base_dir}/{paradigm}/{method}"
    if target_var:
        path = f"{path}/{target_var}"
    return path


def get_checkpoint_dir(paradigm, method, base_dir=CHECKPOINTS_ROOT, target_var=None):
    path = f"{base_dir}/{paradigm}/{method}"
    if target_var:
        path = f"{path}/{target_var}"
    return path


# ==================== 数据文件管理 ====================

# Du_Unet 海表输入与温盐真值（默认都从 DATA_DIR 读取）
TWODTO2D_SST_FILENAME = "SST_2002-01-01_2023-12-31_10_18_110_118.npy"
TWODTO2D_SSH_FILENAME = "SSH_2002-01-01_2023-12-31_10_18_110_118.npy"
TWODTO2D_SSS_FILENAME = "SSS_2002-01-01_2023-12-31_10_18_110_118.npy"
TWODTO2D_TARGET_FILENAMES = {
    "temperature": "T-FIELD_2002-01-01_2023-12-31_10_18_110_118.npy",
    "salinity": "S-FIELD_2002-01-01_2023-12-31_10_18_110_118.npy",
}

# 2dto3d 旧流程暂时保持原始 2019-2023 数据文件。
TWODTO3D_SURFACE_FILENAME = "sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy"
TWODTO3D_TARGET_FILENAME = "sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy"

# ==================== Du_Unet（2dto2d） ====================

DU_UNET_EPOCHS = 20
DU_UNET_BATCH_SIZE = 32
DU_UNET_LR = 1e-3
DU_UNET_WEIGHT_DECAY = 1e-5
DU_UNET_PATIENCE = 10

DU_UNET_CKPT_NAME_TEMPLATE = "Du_Unet_{target_var}_depth{depth_m}m_best.pth"
DU_UNET_HISTORY_NAME_TEMPLATE = "training_history_Du_Unet_{target_var}_depth{depth_m}m.npz"
DU_UNET_LOSS_CURVE_TEMPLATE = "loss_Du_Unet_{target_var}_depth{depth_m}m.png"

# ==================== 2dvar（2dto3d） ====================

VAR2D_DEFAULT_C_DEPTH = len(DEPTH_LEVELS_26M)
VAR2D_SLA_VAR = 0.01 ** 2
VAR2D_SSS_VAR = 0.1 ** 2
VAR2D_BG_WEIGHT = 1e-3
VAR2D_MAXITER = 50
VAR2D_GTOL = 1e-4

# ==================== MODAS（2dto3d） ====================

MODAS_RIDGE = 1e-6

# ==================== OceanTransformer（2dto3d） ====================

TWODTO3D_IN_CHANNELS = 4
TWODTO3D_NUM_DEPTHS = 10
TWODTO3D_DEPTH_LEVELS = [0, 10, 50, 100, 200, 300, 500, 700, 850, 1000]
TWODTO3D_OUT_VARS = 2

TWODTO3D_D_MODEL = 128
TWODTO3D_NHEAD = 8
TWODTO3D_SPATIAL_LAYERS = 4
TWODTO3D_DEPTH_LAYERS = 2
TWODTO3D_DIM_FF = 512

TWODTO3D_LAMBDA_HYDRO = 0.01
TWODTO3D_LAMBDA_STRAT = 0.1

TWODTO3D_EPOCHS = 50
TWODTO3D_BATCH_SIZE = 4
TWODTO3D_LR = 1e-4
TWODTO3D_WEIGHT_DECAY = 1e-5
TWODTO3D_PATIENCE = 10

# ==================== 推理/可视化默认项 ====================

INFER_DEFAULT_TARGET_LEVEL = 10
INFER_PROFILE_METRIC = "rmse"
INFER_PROFILE_ZMAX = DEPTH_MAX
