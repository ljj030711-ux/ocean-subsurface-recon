# B3 DepthFormer

B3 encodes SST, SSH and SSS once, then starts from a learned BOS token and
generates selected temperature layers from shallow to deep.

## Code structure

~~~text
models/b3_depthformer/
├── embeddings.py   # patch stems, 2-D position and real-depth embedding
├── decoder.py      # causal depth attention, cross-attention and patch fold
├── model.py        # surface encoder, teacher forcing and generation
└── __init__.py
tests/test_b3_depthformer.py
scripts/smoke_b3.py
~~~

train.py and test.py only orchestrate the shared data, checkpoint and output
pipeline. Normalization, metrics and visualization therefore stay aligned with
B2.

## Architecture and official-code reuse

1. SST, SSH and SSS use independent convolutional patch embeddings.
2. Feature maps are aligned to the output patch grid, so source resolutions may
   differ.
3. Fused tokens receive a learnable 2-D position map and enter the surface
   Transformer encoder.
4. The first decoder query is BOS + spatial position + real-depth embedding.
5. Later queries use PatchEmbed(previous layer) + spatial position +
   current-depth embedding.
6. Causal depth attention blocks access to deeper layers.
7. Surface cross-attention reads the shared SST/SSH/SSS memory.
8. Linear patch projection and fold reconstruct every temperature map; a small
   convolutional residual reduces patch-boundary artifacts.

The GL-Geoformer implementation is reused at the mechanism level:
convolutional patch embedding, causal masking, encoder-decoder cross-attention,
rolling generation and patch folding. Its time-forecasting data pipeline is not
copied. B3 replaces time with physical depth and replaces the last input frame
with a learned BOS token.

## Current six-depth experiment

Original GLORYS indices:

~~~text
0,4,9,17,24,27
~~~

Physical depths:

~~~text
0.494025, 5.078224, 13.467140, 47.373690, 155.850700, 266.040300 m
~~~

The indices must be ordered from shallow to deep. The model remains dynamic:
omitting --depth-indices later selects every depth in the active profile.

## CPU-only smoke test

No dataset or pretrained weight is required:

~~~bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/smoke_b3.py
~~~

The suite covers different source resolutions, forward/backward, causal BOS
behavior, autoregressive generation, scheduled sampling, checkpoint reload and
B2-compatible paths.

## Six-depth training command

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
  --epochs 50 --patience 10 --batch-size 2 --lr 0.0001 \
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
  --checkpoint-dir "$B3_CKPT" --output-dir "$B3_OUT" --seed 42 \
  2>&1 | tee -a "$B3_LOG/b3_depthformer_6depths_train.log"
~~~

If this configuration runs out of GPU memory, reduce only batch size from 2 to
1 for the first retry and record that change.

## 2023 full-year evaluation

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
  --output-dir "$B3_OUT" \
  2>&1 | tee -a "$B3_LOG/b3_depthformer_6depths_test.log"
~~~

## Input and optimization hyperparameters

| Parameter | Value |
| --- | --- |
| Inputs | monthly-climatology normalized SST, SSH, SSS |
| Target | monthly-climatology normalized temperature |
| Depth indices | 0,4,9,17,24,27 |
| Train / validation / test | 2002-2021 / 2022 / 2023 |
| Loss | equal-depth masked RMSE |
| Optimizer | AdamW |
| Learning rate / weight decay | 1e-4 / 1e-4 |
| Epochs / patience | 50 / 10 |
| Batch size | 2 |
| Gradient clipping | 1.0 |
| Scheduler | cosine, minimum learning rate 5e-6 |
| Teacher forcing | 1.0 to 0.5 over 30 epochs |
| Validation | fully autoregressive |
| Seed | 42 |

## Model hyperparameters

| Parameter | Value |
| --- | --- |
| d_model / heads | 128 / 8 |
| Surface encoder layers | 4 |
| Depth decoder layers | 3 |
| FFN dimension / dropout | 512 / 0.1 |
| SST patch size | 16 |
| SSH/SSS patch size | 4 |
| Output patch size | 8 |
| Local refinement channels | 32 |
| Position base grid | 16 x 16 with runtime interpolation |
| Depth encoding | learned MLP of normalized log(1 + depth) |

## Output contract

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
    └── visualizations/idx0-4-9-17-24-27_6depths/horizontal/
~~~
