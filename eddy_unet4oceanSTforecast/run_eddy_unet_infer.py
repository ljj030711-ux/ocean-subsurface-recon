"""EddyUNet 26 层推理最小调用示例。"""

import os
import sys
from pprint import pprint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # 作为包模块执行: python -m inference.run_eddy_unet_infer
    from inference import infer_eddy_unet_26layers
except ModuleNotFoundError:
    # 作为脚本执行: python inference/run_eddy_unet_infer.py
    from eddy_unet_infer import infer_eddy_unet_26layers


def main():
    result = infer_eddy_unet_26layers(
        select_day="2023-06-15",
        data_dir="./data/raw",
        checkpoint_dir="./checkpoints/2dto2d/eddy_unet",
        use_physics_features=False,
        target_level=10,
        checkpoint_policy="strict",
    )

    print("pred shape:", result["pred"].shape)
    print("target shape:", result["target"].shape)
    print("summary:")
    pprint(result["summary"])
    print("meta:")
    pprint(result["meta"])


if __name__ == "__main__":
    main()
