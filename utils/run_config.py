"""Utilities for printing and saving reproducible run configs."""

import json
import os
import sys
from datetime import datetime

import numpy as np

from config import PARADIGM_2DTO2D


def model_config_path(output_root, model_name, stage):
    return os.path.join(
        output_root, PARADIGM_2DTO2D, model_name, f"{model_name}_{stage}_config.json"
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def count_model_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def load_run_config(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_and_save_run_config(title, payload, config_path):
    config = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "config_path": config_path,
        **payload,
    }
    config = to_jsonable(config)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("=" * 60)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"运行配置已保存至: {config_path}")
    return config
