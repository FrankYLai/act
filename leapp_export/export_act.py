import sys
from pathlib import Path
from unittest.mock import patch

import leapp
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detr"))

from imitate_episodes import eval_bc


CHECKPOINT = Path(__file__).resolve().parent / "artifacts/checkpoint"
EXPORT_DIR = Path("leapp_export/artifacts/leapp_onnx")

POLICY_CONFIG = {
    "lr": 1e-5,
    "num_queries": 100,
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 1,
    "nheads": 8,
    "camera_names": ["top"],
}

EVAL_CONFIG = {
    "ckpt_dir": str(CHECKPOINT),
    "state_dim": 14,
    "real_robot": False,
    "policy_class": "ACT",
    "onscreen_render": False,
    "policy_config": POLICY_CONFIG,
    "camera_names": ["top"],
    "episode_len": 400,
    "task_name": "sim_transfer_cube_human",
    "temporal_agg": False,
    "num_rollouts": 1,
    "eval_seed": 0,
}

ACT_ARGV = [
    "act-leapp-export",
    "--ckpt_dir",
    str(CHECKPOINT),
    "--policy_class",
    "ACT",
    "--task_name",
    "sim_transfer_cube_human",
    "--seed",
    "0",
    "--num_epochs",
    "0",
]


def main():
    # Match ONNX Runtime's FP32 precision during numerical validation.
    torch.backends.cudnn.allow_tf32 = False
    leapp.start(name="act_pipeline", save_path=str(EXPORT_DIR))
    with patch.object(sys, "argv", ACT_ARGV):
        eval_bc(EVAL_CONFIG, "policy_best.ckpt", save_episode=False)
    leapp.stop()
    leapp.compile_graph(visualize=True, validate=True)


if __name__ == "__main__":
    main()
