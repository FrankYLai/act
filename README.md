# Exporting ACT with LEAPP

This is a focused fork of [Tony Zhao's original ACT
repository](https://github.com/tonyzhaozh/act). It demonstrates how to export the
complete numerical ACT policy with
[LEAPP](https://github.com/nvidia-isaac/leapp): image and joint-state
preprocessing, PyTorch inference, and action unnormalization.

The LEAPP boundaries are inserted directly into the original rollout in
`imitate_episodes.py`. There is no replacement inference pipeline. The export
produces one unnormalized action chunk; simulator stepping and playing that
chunk back at the control rate remain application responsibilities.

## Export contract

Inputs:

- `pixels_top`: raw top-camera RGB image, `uint8 [480, 640, 3]`
- `joint_position`: robot joint positions, `float64 [14]`

Output:

- `action_chunk`: 100 unnormalized joint-position targets,
  `float32 [1, 100, 14]`

One LEAPP node, `act_policy`, captures all three processing stages in a single
ONNX model (`act_policy.onnx`):

1. Preprocessing: image layout conversion and scaling plus joint normalization
2. ACT inference: ImageNet normalization, ResNet-18, transformer, and action
   head
3. Postprocessing: conversion to NumPy and action unnormalization

## Reproduce the export

The environment is managed with [uv](https://docs.astral.sh/uv/) and installs
the CUDA-enabled PyTorch build declared in `pyproject.toml`.

```bash
git clone https://github.com/FrankYLai/act.git
cd act
uv sync
```

### 1. Prepare the checkpoint

The original ACT repository does not publish a trained checkpoint. This script
downloads the pinned, Apache-2.0-licensed
[`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/original_act_repo)
checkpoint and converts it back to the original ACT layout.

```bash
uv run python leapp_export/prepare_checkpoint.py
```

The downloaded and converted weights are written under
`leapp_export/artifacts/`, which is ignored by Git and is not redistributed by
this repository.

### 2. Export ACT

```bash
MUJOCO_GL=egl uv run python leapp_export/export_act.py
```

The export is written to:

```text
leapp_export/artifacts/leapp_onnx/act_pipeline/act_pipeline.yaml
```

`export_act.py` only starts LEAPP, runs the existing ACT evaluation path once,
and compiles and validates the recorded graph. It disables cuDNN TF32 during
capture to compare against ONNX Runtime at FP32 precision. The policy
operations themselves remain in the original source.

### 3. Validate in simulation

```bash
MUJOCO_GL=egl uv run python -m leapp_export.compare_in_sim
```

This runs the original PyTorch policy and the LEAPP export from the same
simulator seed using the same rollout and action-chunk schedule. Both should
reach the environment's maximum reward. A side-by-side comparison video is
generated locally under `leapp_export/artifacts/`; it is not part of the
repository.

## Implementation notes

Two small additions support the export boundary:

- The raw dm_control image is copied before `torch.from_numpy()` because the
  original view has negative strides. This changes storage layout, not values.
- The original action-unnormalization expression is applied to the complete
  predicted chunk so the exported output is immediately usable by the rollout.

Temporal aggregation is disabled in this example. LEAPP exports the numerical
policy, while the existing ACT rollout owns action selection, chunk playback,
simulator interaction, and rendering.

## License

The original ACT code is distributed under the repository's [MIT
License](LICENSE). The DETR-derived code under `detr/` retains its separate
[Apache License 2.0](detr/LICENSE). The checkpoint is downloaded separately
under the license declared on its Hugging Face model page.
