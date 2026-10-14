from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/source_checkpoint"
OUTPUT = ROOT / "artifacts/checkpoint"
REPO_ID = "lerobot/act_aloha_sim_transfer_cube_human"
REVISION = "57b06c959f625b9098b4a70f3d937ea47d594568"


def original_model_key(key: str) -> str:
    replacements = (
        ("model.backbone.", "model.backbones.0.0.body."),
        ("model.encoder.layers.", "model.transformer.encoder.layers."),
        ("model.decoder.layers.", "model.transformer.decoder.layers."),
        ("model.decoder.norm.", "model.transformer.decoder.norm."),
        ("model.decoder_pos_embed.", "model.query_embed."),
        ("model.encoder_img_feat_input_proj.", "model.input_proj."),
        ("model.encoder_robot_state_input_proj.", "model.input_proj_robot_state."),
        ("model.vae_encoder.layers.", "model.encoder.layers."),
        ("model.vae_encoder_cls_embed.", "model.cls_embed."),
        ("model.vae_encoder_action_input_proj.", "model.encoder_action_proj."),
        ("model.vae_encoder_robot_state_input_proj.", "model.encoder_joint_proj."),
        ("model.vae_encoder_latent_output_proj.", "model.latent_proj."),
        ("model.encoder_latent_input_proj.", "model.latent_out_proj."),
        (
            "model.encoder_robot_and_latent_pos_embed.",
            "model.additional_pos_embed.",
        ),
    )
    if key == "model.vae_encoder_pos_enc":
        return "model.pos_table"
    if key.startswith("model.action_head."):
        return key
    for source, target in replacements:
        if key.startswith(source):
            return target + key.removeprefix(source)
    raise KeyError(f"No original ACT mapping for {key}")


def main() -> None:
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=SOURCE,
        allow_patterns=["README.md", "convert_weights.py", "model.safetensors"],
    )
    source = load_file(SOURCE / "model.safetensors", device="cpu")
    state_dict = {
        original_model_key(key): value
        for key, value in source.items()
        if key.startswith("model.")
    }

    # These original outputs are unused during inference and were intentionally
    # removed by LeRobot's published forward conversion.
    state_dict["model.is_pad_head.weight"] = torch.zeros(1, 512)
    state_dict["model.is_pad_head.bias"] = torch.zeros(1)

    stats = {
        "qpos_mean": source[
            "normalize_inputs.buffer_observation_state.mean"
        ].numpy(),
        "qpos_std": source[
            "normalize_inputs.buffer_observation_state.std"
        ].numpy()
        + np.float32(1e-8),
        "action_mean": source["unnormalize_outputs.buffer_action.mean"].numpy(),
        "action_std": source["unnormalize_outputs.buffer_action.std"].numpy(),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, OUTPUT / "policy_best.ckpt")
    with (OUTPUT / "dataset_stats.pkl").open("wb") as stats_file:
        pickle.dump(stats, stats_file)
    (OUTPUT / "source.json").write_text(
        json.dumps({"repo_id": REPO_ID, "revision": REVISION}, indent=2) + "\n"
    )
    print(f"Prepared original ACT checkpoint under {OUTPUT}")


if __name__ == "__main__":
    main()
