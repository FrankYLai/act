import sys
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
from leapp import InferenceManager

from leapp_export.export_act import ACT_ARGV, CHECKPOINT, EVAL_CONFIG
from imitate_episodes import eval_bc


GRAPH = Path("leapp_export/artifacts/leapp_onnx/act_pipeline/act_pipeline.yaml")
VIDEO = Path("leapp_export/artifacts/original_vs_leapp.mp4")


def main():
    original_config = {**EVAL_CONFIG, "video_prefix": "original"}
    with patch.object(sys, "argv", ACT_ARGV):
        original_success, _ = eval_bc(
            original_config,
            "policy_best.ckpt",
            save_episode=True,
        )

    manager = InferenceManager(str(GRAPH))
    exported_config = {**EVAL_CONFIG, "video_prefix": "leapp"}
    exported_success, _ = eval_bc(
        exported_config,
        "policy_best.ckpt",
        save_episode=True,
        inference_manager=manager,
    )

    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    original_video = imageio.get_reader(CHECKPOINT / "original0.mp4")
    exported_video = imageio.get_reader(CHECKPOINT / "leapp0.mp4")
    comparison_video = imageio.get_writer(VIDEO, fps=50)
    for original_frame, exported_frame in zip(original_video, exported_video):
        comparison_video.append_data(
            np.concatenate(
                [original_frame[:, :640], exported_frame[:, :640]],
                axis=1,
            )
        )
    original_video.close()
    exported_video.close()
    comparison_video.close()
    print(
        f"original_success={bool(original_success)} "
        f"leapp_success={bool(exported_success)} video={VIDEO}"
    )


if __name__ == "__main__":
    main()
