"""Run the paired B3 causal-depth-attention ablation from the published setup."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = Path("2dto2d/B3_DepthFormer/temperature")
DEPTH_TAG = "idx0-4-9-17-24-27_6depths"
MODES = ("causal", "independent")
TRAIN_BATCH_SIZE = 16


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/root/autodl-tmp/raw_all_size/npy")
    parser.add_argument(
        "--normalization-stats-path",
        default=(
            "/root/autodl-fs/raw_all_size/normalization_stats/"
            "glorys12v1_temperature_2002_2021.npz"
        ),
    )
    parser.add_argument(
        "--experiment-root",
        default="/root/autodl-tmp/B3_DepthFormer_depth_attention_ablation_bs16",
    )
    parser.add_argument("--start-date", default="2002-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--eval-start", default="2023-01-01")
    parser.add_argument("--eval-end", default="2023-12-31")
    parser.add_argument(
        "--panel-days",
        default="2023-07-21,2023-08-12,2023-05-26",
        help="固定为当前消融实验指定的三个二维图日期",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def arm_paths(root, mode):
    arm = root / mode
    return {
        "checkpoint_root": arm / "checkpoints",
        "output_root": arm / "outputs",
        "checkpoint": (
            arm
            / "checkpoints"
            / METHOD_DIR
            / f"B3_DepthFormer_temperature_{DEPTH_TAG}_best.pth"
        ),
        "summary": (
            arm
            / "outputs"
            / METHOD_DIR
            / "summary_B3_DepthFormer_temperature_20230101_20231231.json"
        ),
        "train_config_dir": arm / "outputs" / METHOD_DIR,
        "train_log": root / "logs" / f"b3_{mode}_train.log",
        "test_log": root / "logs" / f"b3_{mode}_test.log",
    }


def build_commands(args, root, mode):
    paths = arm_paths(root, mode)
    common = [
        "--method", "b3_depthformer",
        "--target-var", "temperature",
        "--dataset-profile", "glorys12v1_all_size",
        "--data-dir", args.data_dir,
        "--normalization-stats-path", args.normalization_stats_path,
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--depth-indices", "0,4,9,17,24,27",
    ]
    train = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "train.py"),
        *common,
        "--epochs", "50",
        "--patience", "10",
        "--batch-size", str(TRAIN_BATCH_SIZE),
        "--lr", "0.0001",
        "--num-workers", "4",
        "--prefetch-factor", "2",
        "--pin-memory",
        "--persistent-workers",
        "--b3-d-model", "128",
        "--b3-nhead", "8",
        "--b3-encoder-layers", "4",
        "--b3-decoder-layers", "3",
        "--b3-dim-feedforward", "512",
        "--b3-dropout", "0.1",
        "--b3-sst-patch-size", "16",
        "--b3-surface-patch-size", "4",
        "--b3-output-patch-size", "8",
        "--b3-refinement-channels", "32",
        "--b3-base-grid-height", "16",
        "--b3-base-grid-width", "16",
        "--b3-teacher-forcing-start", "1.0",
        "--b3-teacher-forcing-end", "0.5",
        "--b3-teacher-forcing-decay-epochs", "30",
        "--b3-depth-attention", mode,
        "--checkpoint-dir", str(paths["checkpoint_root"]),
        "--output-dir", str(paths["output_root"]),
        "--seed", str(args.seed),
    ]
    evaluate = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "test.py"),
        *common,
        "--checkpoint", str(paths["checkpoint"]),
        "--eval-start", args.eval_start,
        "--eval-end", args.eval_end,
        "--eval-batch-size", "1",
        "--spatial-panels",
        "--target-levels", "0,1,2,3,4,5",
        "--panel-days", args.panel_days,
        "--output-dir", str(paths["output_root"]),
        "--seed", str(args.seed),
    ]
    return train, evaluate, paths


def print_command(command, log_path):
    print(f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log_path))}")


def run_and_log(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def load_single_config(directory):
    matches = list(directory.glob(f"B3_DepthFormer_train_config_{DEPTH_TAG}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"无法唯一定位训练配置: {directory}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def write_comparison(root):
    rows = []
    parameter_counts = {}
    for mode in MODES:
        paths = arm_paths(root, mode)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        metrics = summary["variables"]["temperature"]["overall"]
        config = load_single_config(paths["train_config_dir"])
        stored_mode = config["model_config"].get("depth_attention", "causal")
        if stored_mode != mode:
            raise RuntimeError(
                f"{mode} 训练配置记录为 {stored_mode!r}，拒绝生成混合比较"
            )
        parameter_counts[mode] = config["parameter_count"]["total"]
        rows.append({"depth_attention": mode, **metrics})
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"两组参数量不一致: {parameter_counts}")
    comparison_path = root / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"消融汇总已保存: {comparison_path}")


def write_or_validate_experiment(root, experiment):
    """Create the run manifest or reject an incompatible resumed run."""
    path = root / "experiment.json"
    if not path.exists():
        path.write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    compared_keys = (
        "data_dir",
        "normalization_stats_path",
        "start_date",
        "end_date",
        "eval_start",
        "eval_end",
        "panel_days",
        "seed",
        "train_batch_size",
    )
    mismatches = [
        key
        for key in compared_keys
        if previous.get(key) != experiment.get(key)
    ]
    if mismatches:
        raise ValueError(
            "已有实验目录的共同配置不一致，不能安全续跑: "
            + ", ".join(mismatches)
        )
    # Refresh provenance fields added by newer runner versions.
    previous.update(
        {
            key: experiment[key]
            for key in (
                "determinism_policy",
                "known_nondeterministic_operation",
                "interpretation_note",
            )
        }
    )
    path.write_text(
        json.dumps(previous, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    fixed_controls = {
        "seed": (args.seed, 42),
        "start-date": (args.start_date, "2002-01-01"),
        "end-date": (args.end_date, "2023-12-31"),
        "eval-start": (args.eval_start, "2023-01-01"),
        "eval-end": (args.eval_end, "2023-12-31"),
    }
    changed = [
        f"{name}={actual!r}（应为 {expected!r}）"
        for name, (actual, expected) in fixed_controls.items()
        if actual != expected
    ]
    if changed:
        parser.error("为保证与上次B3实验可比，不允许改变：" + "；".join(changed))
    panel_days = [day.strip() for day in args.panel_days.split(",") if day.strip()]
    if panel_days != ["2023-07-21", "2023-08-12", "2023-05-26"]:
        parser.error(
            "为保证两组消融二维图完全可比，--panel-days 必须是 "
            "2023-07-21,2023-08-12,2023-05-26"
        )
    root = Path(args.experiment_root).expanduser().resolve()
    command_sets = [build_commands(args, root, mode) for mode in MODES]
    for train, evaluate, paths in command_sets:
        print_command(train, paths["train_log"])
        print_command(evaluate, paths["test_log"])
    if args.dry_run:
        return

    root.mkdir(parents=True, exist_ok=True)
    experiment = {
        **vars(args),
        "train_batch_size": TRAIN_BATCH_SIZE,
        "determinism_policy": "warn_only",
        "known_nondeterministic_operation": (
            "upsample_bicubic2d_backward_out_cuda"
        ),
        "interpretation_note": (
            "seed、初始化和数据顺序受控，但bicubic CUDA backward不保证逐位复现；"
            "消融结论应结合效应大小，必要时使用多随机种子。"
        ),
    }
    try:
        write_or_validate_experiment(root, experiment)
    except ValueError as exc:
        parser.error(str(exc))
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for train, evaluate, paths in command_sets:
        if paths["checkpoint"].exists():
            print(f"[resume] 已有checkpoint，跳过训练: {paths['checkpoint']}")
        else:
            run_and_log(train, paths["train_log"], env)
        if paths["summary"].exists():
            print(f"[resume] 已有完整评估summary，跳过推理: {paths['summary']}")
        else:
            run_and_log(evaluate, paths["test_log"], env)
    write_comparison(root)


if __name__ == "__main__":
    main()
