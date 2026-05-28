# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import loguru
import torch
import argparse
import einops
import imageio
import json
import mimetypes
import numpy as np
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, VideoClip

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons import auto_offload_model
from hyvideo.commons.parallel_states import initialize_parallel_state, get_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local, CameraState

MY_WORLDPLAY_ROOT = Path(__file__).resolve().parents[2]
if str(MY_WORLDPLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(MY_WORLDPLAY_ROOT))

parallel_dims = initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

mapping = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def one_hot_to_one_dimension(one_hot):
    y = torch.tensor([mapping[tuple(row.tolist())] for row in one_hot])
    return y


def parse_pose_string(pose_string):
    """
    Parse pose string to motions list.
    Format: "w-3, right-0.5, d-4"
    - w: forward movement
    - s: backward movement
    - a: left movement
    - d: right movement
    - up: pitch up rotation
    - down: pitch down rotation
    - left: yaw left rotation
    - right: yaw right rotation
    - number after dash: duration in latents

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        list of dict: motions for generate_camera_trajectory_local
    """
    # Movement amount per frame
    forward_speed = 0.08  # units per frame
    yaw_speed = np.deg2rad(3)  # radians per frame
    pitch_speed = np.deg2rad(3)  # radians per frame

    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pose command: {cmd}. Expected format: 'action-duration'"
            )

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        num_frames = int(duration)

        # Parse action and create motion dict
        if action == "w":
            # Forward
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == "s":
            # Backward
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == "a":
            # Left
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == "d":
            # Right
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == "up":
            # Pitch up
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == "down":
            # Pitch down
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == "left":
            # Yaw left
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == "right":
            # Yaw right
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(
                f"Unknown action: {action}. Supported actions: w, s, a, d, up, down, left, right"
            )

    return motions


def pose_string_to_json(pose_string):
    """
    Convert pose string to pose JSON format.

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        dict: pose JSON with extrinsic and intrinsic parameters
    """
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)

    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    pose_json = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

    return pose_json


def incremental_poses_to_input(incremental_poses, prev_c2w=None, intrinsic=None):
    """
    Convert incremental poses to input tensors for interactive mode.

    Args:
        incremental_poses: list of 4x4 c2w matrices (np.ndarray)
        prev_c2w: 4x4 c2w matrix from previous round's last frame (np.ndarray)
                  If None, uses first frame of incremental_poses as reference
        intrinsic: 3x3 intrinsic matrix (np.ndarray)
                   If None, uses default intrinsic

    Returns:
        tuple: (w2c_list, intrinsic_list, action_one_label)
    """
    if intrinsic is None:
        intrinsic = np.array([
            [969.6969696969696, 0.0, 960.0],
            [0.0, 969.6969696969696, 540.0],
            [0.0, 0.0, 1.0],
        ])

    latent_num = len(incremental_poses)

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        c2w = incremental_poses[i]
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)

        intrinsic_copy = intrinsic.copy()
        intrinsic_copy[0, 0] /= intrinsic_copy[0, 2] * 2
        intrinsic_copy[1, 1] /= intrinsic_copy[1, 2] * 2
        intrinsic_copy[0, 2] = 0.5
        intrinsic_copy[1, 2] = 0.5
        intrinsic_list.append(intrinsic_copy)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)

    relative_c2w = np.zeros_like(c2ws)
    if prev_c2w is None:
        relative_c2w[0, ...] = c2ws[0, ...]
        C_inv = np.linalg.inv(c2ws[:-1])
        relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    else:
        prev_w2c = np.linalg.inv(prev_c2w)
        relative_c2w[0, ...] = prev_w2c @ c2ws[0]
        C_inv = np.linalg.inv(c2ws[:-1])
        relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]

    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    action_start_idx = 0 if prev_c2w is not None else 1
    for i in range(action_start_idx, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / torch.pi)
        else:
            trans_angles_deg = torch.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)

        if move_norms > move_norm_valid:
            if trans_angles_deg[2] < 60:
                trans_one_hot[i, 0] = 1
            elif trans_angles_deg[2] > 120:
                trans_one_hot[i, 1] = 1

            if trans_angles_deg[0] < 60:
                trans_one_hot[i, 2] = 1
            elif trans_angles_deg[0] > 120:
                trans_one_hot[i, 3] = 1

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1

    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.as_tensor(w2c_list), torch.as_tensor(intrinsic_list), action_one_label


def pose_to_input(pose_data, latent_num, tps=False):
    """
    Convert pose data to input tensors.

    Args:
        pose_data: str or dict
            - If str ending with '.json': path to JSON file
            - If str: pose string (e.g., "w-3, right-0.5, d-4")
            - If dict: pose JSON data
        latent_num: int, number of latents
        tps: bool, third person mode

    Returns:
        tuple: (w2c_list, intrinsic_list, action_one_label)
    """
    # Handle different input types
    if isinstance(pose_data, str):
        if pose_data.endswith(".json"):
            # Load from JSON file
            pose_json = json.load(open(pose_data, "r"))
        else:
            # Parse pose string
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(
            f"Invalid pose_data type: {type(pose_data)}. Expected str or dict."
        )

    pose_keys = list(pose_json.keys())
    latent_num_from_pose = len(pose_keys)
    assert latent_num_from_pose == latent_num, (
        f"pose corresponds to {latent_num_from_pose * 4 - 3} frames, num_frames "
        f"must be set to {latent_num_from_pose * 4 - 3} to ensure alignment."
    )

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[t_key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]  # direction vector
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:  # threshold for movement
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / torch.pi)  # convert to degrees
        else:
            trans_angles_deg = torch.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)

        # Determine movement and rotation actions
        if move_norms > move_norm_valid:  # threshold for movement
            if (not tps) or (
                tps == True
                and abs(rot_angles_deg[1]) < 5e-2
                and abs(rot_angles_deg[0]) < 5e-2
            ):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1  # right
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1  # left

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1  # up
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1  # down
    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.as_tensor(w2c_list), torch.as_tensor(intrinsic_list), action_one_label


def load_prompt_schedule_json(schedule_path, fallback_prompt):
    """
    Load a non-interactive prompt schedule.

    Supported JSON format:
    {
      "segments": [
        {"pose": "w-46", "prompt": "first prompt"},
        {"pose": "left-41", "prompt": "second prompt"}
      ]
    }

    Pose durations are latent steps, matching the existing pose string syntax.
    Prompt changes are applied at AR chunk boundaries.
    """
    with open(schedule_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            f"Invalid prompt schedule JSON: {schedule_path}. "
            "Expected a non-empty list or an object with a non-empty 'segments' list."
        )

    pose_parts = []
    prompt_schedule = []
    cumulative_motion_latents = 0
    current_prompt = fallback_prompt

    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Invalid segment #{idx}: expected an object.")
        pose = str(segment.get("pose", "")).strip()
        if not pose:
            raise ValueError(f"Invalid segment #{idx}: missing non-empty 'pose'.")

        prompt = segment.get("prompt", current_prompt)
        if prompt is None:
            prompt = current_prompt
        prompt = str(prompt)

        start_latent = cumulative_motion_latents if idx == 0 else cumulative_motion_latents + 1
        prompt_schedule.append({"start_latent": int(start_latent), "prompt": prompt})

        pose_parts.append(pose)
        cumulative_motion_latents += len(parse_pose_string(pose))
        current_prompt = prompt

    pose = ",".join(pose_parts)
    latent_num = cumulative_motion_latents + 1
    video_length = latent_num * 4 - 3
    return pose, video_length, prompt_schedule


def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(path, vid, fps=24)


def decode_latents_to_video(pipe, latents):
    latents = latents.to(device=pipe.execution_device, dtype=pipe.target_dtype)
    if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        latents = latents / pipe.vae.config.scaling_factor

    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda",
            dtype=pipe.vae_dtype,
            enabled=pipe.vae_autocast_enabled,
        ),
        auto_offload_model(pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading),
    ):
        video = pipe.vae.decode(latents, return_dict=False)[0]
    return (video / 2 + 0.5).clamp(0, 1).cpu().float()


def save_interactive_video(pipe, history_latents, accumulated_frames, path):
    if history_latents is not None:
        try:
            save_video(decode_latents_to_video(pipe, history_latents), path)
            return
        except Exception as exc:
            print(f"Failed to decode accumulated latents, falling back to chunk frames: {exc}")
    combined = torch.cat(accumulated_frames, dim=2)
    save_video(combined, path)


def save_action_prompt_memory(memory, path, base_prompt):
    payload = {
        "base_prompt": base_prompt,
        "entries": memory,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def default_prompt_cache_path(output_path):
    return os.path.join(output_path, "prompt_cache.json")


def read_prompt_cache_prompt(cache_path, chunk_id):
    try:
        from worldagent.vlm_environment_prompt_agent import read_prompt_from_cache

        return read_prompt_from_cache(Path(cache_path), chunk_id)
    except Exception as exc:
        print(f"Failed to read prompt cache for chunk {chunk_id}: {exc}")
        return None


def run_vlm_prompt_cache_task(args, video_path, action, source_chunk_id, used_prompt):
    try:
        from argparse import Namespace
        from worldagent.vlm_environment_prompt_agent import run_agent_to_prompt_cache

        cache_path = args.vlm_prompt_cache_path or default_prompt_cache_path(args.output_path or "./outputs")
        log_json = os.path.join(os.path.dirname(cache_path), "vlm_prompt_outputs.json")
        vlm_args = Namespace(
            video=video_path,
            action=action,
            previous_prompt=used_prompt,
            user_goal=args.prompt,
            base_url=args.vlm_base_url,
            model=args.vlm_model,
            api_key=args.vlm_api_key,
            max_frames=args.vlm_max_frames,
            max_image_size=args.vlm_max_image_size,
            temperature=args.vlm_temperature,
            max_tokens=args.vlm_max_tokens,
            timeout=args.vlm_timeout,
            frame_dir=args.vlm_frame_dir,
            log_json=log_json,
            log_jsonl=None,
            prompt_cache=cache_path,
            target_chunk_offset=args.vlm_prompt_target_offset,
            target_chunk_id=source_chunk_id + args.vlm_prompt_target_offset,
            chunk_id=source_chunk_id,
            cache_source="vlm_web",
            chunks_dir=None,
            max_chunks=None,
        )
        result = run_agent_to_prompt_cache(
            vlm_args,
            chunk_id=source_chunk_id,
            target_chunk_id=source_chunk_id + args.vlm_prompt_target_offset,
        )
        print(
            "VLM prompt cache updated: "
            f"chunk {source_chunk_id} -> {source_chunk_id + args.vlm_prompt_target_offset}"
        )
        return result
    except Exception as exc:
        print(f"VLM prompt cache task failed for chunk {source_chunk_id}: {exc}")
        import traceback
        traceback.print_exc()
        return None


WEB_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HY-WorldPlay Interactive</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #191d24;
      --panel-2: #202631;
      --line: #343c49;
      --text: #f4f7fb;
      --muted: #aeb8c7;
      --accent: #2fb7a6;
      --accent-2: #f2b84b;
      --danger: #ef6262;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      min-height: 100vh;
      padding: 18px;
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    aside {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .status strong { color: var(--text); }
    .prompt-box {
      display: grid;
      gap: 8px;
    }
    .prompt-box label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #11151b;
      color: var(--text);
      padding: 10px;
      font: inherit;
      line-height: 1.35;
    }
    textarea:focus {
      outline: 1px solid var(--accent);
      border-color: var(--accent);
    }
    .kbd {
      display: grid;
      grid-template-columns: repeat(3, minmax(58px, 1fr));
      gap: 8px;
      max-width: 260px;
    }
    .spacer { visibility: hidden; }
    button {
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:active, button.active { background: var(--accent); color: #06110f; }
    button:disabled { opacity: 0.55; cursor: wait; }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .actions button { min-height: 42px; font-size: 14px; }
    .actions .stop { border-color: color-mix(in srgb, var(--danger), var(--line)); }
    .player {
      display: grid;
      grid-template-rows: minmax(260px, 1fr) auto;
      overflow: hidden;
    }
    .stage {
      display: grid;
      place-items: center;
      padding: 16px;
      background: #080a0d;
      min-height: 0;
    }
    video {
      width: min(100%, 1120px);
      max-height: calc(100vh - 170px);
      aspect-ratio: 16 / 9;
      background: #000;
      border-radius: 6px;
    }
    .timeline {
      border-top: 1px solid var(--line);
      padding: 12px 14px;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      min-height: 68px;
    }
    .chip {
      white-space: nowrap;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
      background: var(--panel-2);
      font-size: 13px;
    }
    .chip.current { color: #101216; background: var(--accent-2); border-color: var(--accent-2); }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      video { max-height: 52vh; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>HY-WorldPlay</h1>
      <div class="status">
        <div>Status: <strong id="state">loading</strong></div>
        <div>Last: <strong id="last">none</strong></div>
        <div>Prompt cache: <strong id="cache">disabled</strong></div>
        <div>Total frames: <strong id="frames">0</strong></div>
      </div>
      <div class="prompt-box">
        <label for="prompt">Prompt for next chunk</label>
        <textarea id="prompt"></textarea>
      </div>
      <div class="kbd" aria-label="Movement controls">
        <div class="spacer"></div>
        <button data-action="w">W</button>
        <div class="spacer"></div>
        <button data-action="a">A</button>
        <button data-action="s">S</button>
        <button data-action="d">D</button>
        <div class="spacer"></div>
        <button data-action="up">↑</button>
        <div class="spacer"></div>
        <button data-action="left">←</button>
        <button data-action="down">↓</button>
        <button data-action="right">→</button>
      </div>
      <div class="actions">
        <button id="save">Save</button>
        <button id="quit" class="stop">Quit</button>
      </div>
    </aside>
    <section class="player">
      <div class="stage">
        <video id="video" controls autoplay muted playsinline></video>
      </div>
      <div id="timeline" class="timeline"></div>
    </section>
  </main>
  <script>
    const keyMap = {
      KeyW: "w", KeyA: "a", KeyS: "s", KeyD: "d",
      ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right"
    };
    const stateEl = document.getElementById("state");
    const lastEl = document.getElementById("last");
    const cacheEl = document.getElementById("cache");
    const framesEl = document.getElementById("frames");
    const video = document.getElementById("video");
    const timeline = document.getElementById("timeline");
    const promptInput = document.getElementById("prompt");
    const buttons = [...document.querySelectorAll("button[data-action]")];
    let chunks = [];
    let playingIndex = -1;
    let promptInitialized = false;

    async function post(path, body = {}) {
      const res = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    async function sendAction(action) {
      if (buttons.some(btn => btn.disabled)) return;
      setButtons(true);
      try {
        await post("/api/action", {action, prompt: promptInput.value});
        await refresh();
      } catch (err) {
        stateEl.textContent = err.message;
        setButtons(false);
      }
    }

    function setButtons(disabled) {
      buttons.forEach(btn => btn.disabled = disabled);
    }

    function renderTimeline() {
      timeline.innerHTML = "";
      chunks.forEach((chunk, index) => {
        const el = document.createElement("button");
        el.className = "chip" + (index === playingIndex ? " current" : "");
        const prompt = chunk.prompt ? ` · ${chunk.prompt.slice(0, 28)}` : "";
        el.textContent = `${chunk.round}: ${chunk.command}${prompt}`;
        el.onclick = () => playChunk(index);
        timeline.appendChild(el);
      });
    }

    function playChunk(index) {
      if (!chunks[index]) return;
      playingIndex = index;
      video.src = chunks[index].url + "?t=" + Date.now();
      video.play().catch(() => {});
      renderTimeline();
    }

    video.addEventListener("ended", () => {
      if (playingIndex + 1 < chunks.length) playChunk(playingIndex + 1);
    });

    async function refresh() {
      const status = await fetch("/api/status").then(r => r.json());
      if (!promptInitialized) {
        promptInput.value = status.base_prompt || "";
        promptInitialized = true;
      }
      stateEl.textContent = status.busy ? "rendering" : status.message;
      lastEl.textContent = status.last_command || "none";
      cacheEl.textContent = status.prompt_cache_status || "disabled";
      framesEl.textContent = status.total_frames;
      if (status.cached_next_prompt && document.activeElement !== promptInput) {
        promptInput.value = status.cached_next_prompt;
      }
      setButtons(status.busy || status.queue_size > 0);
      if (status.chunks.length !== chunks.length) {
        chunks = status.chunks;
        if (playingIndex === -1 || playingIndex < chunks.length - 1) {
          playChunk(chunks.length - 1);
        } else {
          renderTimeline();
        }
      } else {
        renderTimeline();
      }
    }

    document.addEventListener("keydown", event => {
      if (event.target === promptInput) return;
      if (event.repeat) return;
      const action = keyMap[event.code];
      if (!action) return;
      event.preventDefault();
      document.querySelector(`[data-action="${action}"]`)?.classList.add("active");
      sendAction(action);
    });
    document.addEventListener("keyup", event => {
      const action = keyMap[event.code];
      if (action) document.querySelector(`[data-action="${action}"]`)?.classList.remove("active");
    });
    buttons.forEach(btn => btn.addEventListener("click", () => sendAction(btn.dataset.action)));
    document.getElementById("save").onclick = () => post("/api/save").then(refresh);
    document.getElementById("quit").onclick = () => post("/api/quit").then(refresh);
    setInterval(refresh, 1500);
    refresh();
  </script>
</body>
</html>
"""


def start_web_controller(args, output_path, latent_num):
    command_queue = queue.Queue()
    chunks_dir = os.path.join(output_path, "web_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    prompt_cache_path = args.vlm_prompt_cache_path or default_prompt_cache_path(output_path)
    state = {
        "busy": False,
        "message": "ready",
        "last_command": None,
        "last_prompt": None,
        "last_prompt_source": "manual",
        "prompt_cache_enabled": bool(args.vlm_prompt_cache),
        "prompt_cache_path": prompt_cache_path,
        "prompt_cache_status": "disabled" if not args.vlm_prompt_cache else "ready",
        "total_frames": 0,
        "chunks": [],
        "base_prompt": args.prompt,
    }
    lock = threading.Lock()
    action_aliases = {
        "w": "w",
        "a": "a",
        "s": "s",
        "d": "d",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    }

    def json_response(handler, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def read_json(handler):
        content_length = int(handler.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = handler.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def command_duration_for_next_chunk(next_chunk_id):
        if next_chunk_id == 1 and not args.initial_pose:
            return latent_num - 1
        return latent_num

    def make_status():
        with lock:
            next_chunk_id = len(state["chunks"]) + 1
            cached_next_prompt = None
            if args.vlm_prompt_cache:
                cached_next_prompt = read_prompt_cache_prompt(prompt_cache_path, next_chunk_id)
            return {
                **state,
                "queue_size": command_queue.qsize(),
                "chunk_latents": latent_num,
                "next_command_duration": command_duration_for_next_chunk(next_chunk_id),
                "next_chunk_id": next_chunk_id,
                "cached_next_prompt": cached_next_prompt,
            }

    class WebHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = WEB_INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/status":
                json_response(self, make_status())
                return
            if parsed.path.startswith("/chunks/"):
                name = os.path.basename(parsed.path)
                path = os.path.join(chunks_dir, name)
                if not os.path.isfile(path):
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(os.path.getsize(path)))
                self.end_headers()
                with open(path, "rb") as f:
                    while True:
                        block = f.read(1024 * 1024)
                        if not block:
                            break
                        self.wfile.write(block)
                return
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/action":
                    payload = read_json(self)
                    command = payload.get("command")
                    action = payload.get("action")
                    if not command:
                        action = action_aliases.get(str(action).strip())
                        if not action:
                            json_response(self, {"error": "unknown action"}, status=400)
                            return
                    next_chunk_id = 1
                    with lock:
                        next_chunk_id = len(state["chunks"]) + 1
                    if not command:
                        command = f"{action}-{command_duration_for_next_chunk(next_chunk_id)}"
                    prompt_source = "manual"
                    prompt = None
                    if args.vlm_prompt_cache:
                        prompt = read_prompt_cache_prompt(prompt_cache_path, next_chunk_id)
                        if prompt:
                            prompt_source = "prompt_cache"
                    if not prompt:
                        prompt = str(payload.get("prompt") or args.prompt).strip()
                    command_queue.put(
                        {
                            "type": "action",
                            "command": command,
                            "prompt": prompt,
                            "prompt_source": prompt_source,
                            "target_chunk_id": next_chunk_id,
                        }
                    )
                    with lock:
                        state["message"] = "queued"
                        state["last_prompt"] = prompt
                        state["last_prompt_source"] = prompt_source
                    json_response(self, {"ok": True, "command": command})
                    return
                if parsed.path == "/api/save":
                    command_queue.put({"type": "save", "command": "save"})
                    with lock:
                        state["message"] = "save queued"
                    json_response(self, {"ok": True})
                    return
                if parsed.path == "/api/quit":
                    command_queue.put({"type": "quit", "command": "quit"})
                    with lock:
                        state["message"] = "quit queued"
                    json_response(self, {"ok": True})
                    return
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=500)
                return
            self.send_error(404)

    server = ThreadingHTTPServer((args.web_host, args.web_port), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Web UI running at http://{args.web_host}:{args.web_port}")

    def mark_busy(command, prompt=None):
        with lock:
            state["busy"] = True
            state["message"] = "rendering"
            state["last_command"] = command
            if prompt is not None:
                state["last_prompt"] = prompt

    def mark_ready(message="ready"):
        with lock:
            state["busy"] = False
            state["message"] = message

    def mark_prompt_cache(status):
        with lock:
            state["prompt_cache_status"] = status

    def add_chunk(round_count, command, prompt, frames):
        filename = f"chunk_{round_count:04d}.mp4"
        path = os.path.join(chunks_dir, filename)
        save_video(frames, path)
        with lock:
            state["chunks"].append(
                {
                    "round": round_count,
                    "command": command,
                    "prompt": prompt,
                    "url": f"/chunks/{filename}",
                    "frames": int(frames.shape[2]),
                    "created_at": time.time(),
                }
            )
            state["total_frames"] += int(frames.shape[2])
            state["busy"] = False
            state["message"] = "ready"
        return path

    return {
        "queue": command_queue,
        "server": server,
        "mark_busy": mark_busy,
        "mark_ready": mark_ready,
        "mark_prompt_cache": mark_prompt_cache,
        "add_chunk": add_chunk,
        "prompt_cache_path": prompt_cache_path,
    }


def rank0_log(message, level):
    if int(os.environ.get("RANK", "0")) == 0:
        loguru.logger.log(level, message)


def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def camera_center_normalization(w2c):
    c2w = np.linalg.inv(w2c)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    return np.linalg.inv(c2w_aligned)


def parse_pose_string_to_actions(pose_string, fps=24):
    """
    Parse pose string to frame-level action timeline.

    Format: pose string uses latent counts, where:
    - 1 latent = 4 frames
    - Special rule: first frame of entire video is extra (frame 0)
    - Example: "w-4,d-4" means:
      - w-4: forward for frames 0-16 (17 frames total: 1 extra + 4*4)
      - d-4: right for frames 17-32 (16 frames total: 4*4)

    Args:
        pose_string: str, comma-separated pose commands (e.g., "w-4,d-4")
        fps: int, frames per second for video (default: 24)

    Returns:
        list of dict with frame_idx and actions for each frame
    """
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    # Build frame-level actions list
    frame_actions = []
    is_first_command = True

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pose command: {cmd}. Expected format: 'action-duration'"
            )

        action = parts[0].strip()
        try:
            num_latents = int(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        # Convert latents to frames
        # First command gets 1 extra frame (the special frame 0)
        # Formula: first command = 1 + num_latents * 4, others = num_latents * 4
        if is_first_command:
            num_frames = 1 + num_latents * 4
            is_first_command = False
        else:
            num_frames = num_latents * 4

        # Map action to action values
        action_values = {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}

        if action == "w":
            action_values["forward"] = 1
        elif action == "s":
            action_values["forward"] = -1
        elif action == "a":
            action_values["left"] = 1
        elif action == "d":
            action_values["left"] = -1
        elif action == "up":
            action_values["pitch"] = 1
        elif action == "down":
            action_values["pitch"] = -1
        elif action == "left":
            action_values["yaw"] = -1
        elif action == "right":
            action_values["yaw"] = 1
        else:
            raise ValueError(f"Unknown action: {action}")

        # Add frame-level actions
        for _ in range(num_frames):
            frame_actions.append(action_values.copy())

    # Return frame-level timeline (each entry represents one frame)
    return frame_actions


def draw_rounded_rectangle(draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = xy
    diameter = radius * 2

    # Draw four corners (circles)
    draw.ellipse(
        [x1, y1, x1 + diameter, y1 + diameter], fill=fill, outline=outline, width=width
    )
    draw.ellipse(
        [x2 - diameter, y1, x2, y1 + diameter], fill=fill, outline=outline, width=width
    )
    draw.ellipse(
        [x1, y2 - diameter, x1 + diameter, y2], fill=fill, outline=outline, width=width
    )
    draw.ellipse(
        [x2 - diameter, y2 - diameter, x2, y2], fill=fill, outline=outline, width=width
    )

    # Draw two rectangles to fill the middle
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)

    # Draw border if outline is specified
    if outline:
        # Top and bottom lines
        draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
        draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
        # Left and right lines
        draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
        draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)


def create_wasd_keyboard(actions, key_size=70, key_spacing=6, corner_radius=14):
    """Create WASD keyboard overlay."""
    keyboard_width = 3 * key_size + 2 * key_spacing
    keyboard_height = 2 * key_size + key_spacing

    img = Image.new("RGBA", (keyboard_width, keyboard_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_normal = (0, 0, 0, 128)
    bg_active = (30, 120, 255, 220)
    text_color = (255, 255, 255, 255)
    font_size = 28

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except (IOError, OSError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", font_size
            )
        except (IOError, OSError):
            font = ImageFont.load_default()

    def draw_key(x, y, label, is_active):
        bg_color = bg_active if is_active else bg_normal
        draw_rounded_rectangle(
            draw,
            [x, y, x + key_size, y + key_size],
            radius=corner_radius,
            fill=bg_color,
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (key_size - text_width) // 2
        text_y = y + (key_size - text_height) // 2
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    forward_val = actions.get("forward", 0)
    left_val = actions.get("left", 0)

    w_active = forward_val > 0
    s_active = forward_val < 0
    a_active = left_val > 0
    d_active = left_val < 0

    wasd_keys = [
        ("W", 1, 0, w_active),
        ("A", 0, 1, a_active),
        ("S", 1, 1, s_active),
        ("D", 2, 1, d_active),
    ]

    for label, col, row, is_active in wasd_keys:
        x = col * (key_size + key_spacing)
        y = row * (key_size + key_spacing)
        draw_key(x, y, label, is_active)

    return img


def create_arrow_keyboard(actions, key_size=70, key_spacing=6, corner_radius=14):
    """Create arrow keys keyboard overlay with triangle symbols."""
    keyboard_width = 3 * key_size + 2 * key_spacing
    keyboard_height = 2 * key_size + key_spacing

    img = Image.new("RGBA", (keyboard_width, keyboard_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_normal = (0, 0, 0, 128)
    bg_active = (30, 120, 255, 220)
    text_color = (255, 255, 255, 255)

    def draw_key_with_triangle(x, y, direction, is_active):
        bg_color = bg_active if is_active else bg_normal
        draw_rounded_rectangle(
            draw,
            [x, y, x + key_size, y + key_size],
            radius=corner_radius,
            fill=bg_color,
        )
        cx = x + key_size // 2
        cy = y + key_size // 2
        size = key_size // 8
        if direction == "up":
            points = [
                (cx, cy - size),
                (cx - size, cy + size // 2),
                (cx + size, cy + size // 2),
            ]
        elif direction == "down":
            points = [
                (cx, cy + size),
                (cx - size, cy - size // 2),
                (cx + size, cy - size // 2),
            ]
        elif direction == "left":
            points = [
                (cx - size, cy),
                (cx + size // 2, cy - size),
                (cx + size // 2, cy + size),
            ]
        elif direction == "right":
            points = [
                (cx + size, cy),
                (cx - size // 2, cy - size),
                (cx - size // 2, cy + size),
            ]
        draw.polygon(points, fill=text_color)

    yaw_val = actions.get("yaw", 0)
    pitch_val = actions.get("pitch", 0)

    up_active = pitch_val > 0
    down_active = pitch_val < 0
    left_active = yaw_val < 0
    right_active = yaw_val > 0

    arrow_keys = [
        ("up", 1, 0, up_active),
        ("left", 0, 1, left_active),
        ("down", 1, 1, down_active),
        ("right", 2, 1, right_active),
    ]

    for direction, col, row, is_active in arrow_keys:
        kx = col * (key_size + key_spacing)
        ky = row * (key_size + key_spacing)
        draw_key_with_triangle(kx, ky, direction, is_active)

    return img


def blend_overlay(base_frame, overlay, position):
    """Blend an RGBA overlay onto a RGB frame."""
    x, y = position
    overlay_array = np.array(overlay)

    oh, ow = overlay_array.shape[:2]
    bh, bw = base_frame.shape[:2]

    x = max(0, min(x, bw - 1))
    y = max(0, min(y, bh - 1))

    if x + ow > bw:
        ow = bw - x
        overlay_array = overlay_array[:, :ow]
    if y + oh > bh:
        oh = bh - y
        overlay_array = overlay_array[:oh, :]

    if ow <= 0 or oh <= 0:
        return base_frame

    overlay_rgb = overlay_array[:, :, :3].astype(np.float32)
    overlay_alpha = overlay_array[:, :, 3:4].astype(np.float32) / 255.0

    base_region = base_frame[y : y + oh, x : x + ow].astype(np.float32)

    blended = (overlay_rgb * overlay_alpha + base_region * (1 - overlay_alpha)).astype(
        np.uint8
    )

    result = base_frame.copy()
    result[y : y + oh, x : x + ow] = blended

    return result


def add_keyboard_overlay_to_video(video_path, output_path, actions_timeline):
    """Add keyboard overlay to an existing video.

    Args:
        video_path: path to input video
        output_path: path to output video
        actions_timeline: list of action dicts, one per frame
    """
    try:
        video = VideoFileClip(video_path)
        width, height = video.size
        fps = video.fps
        duration = video.duration

        key_size = 70
        key_spacing = 6
        corner_radius = 14
        margin = 40

        keyboard_width = 3 * key_size + 2 * key_spacing
        keyboard_height = 2 * key_size + key_spacing

        wasd_pos = (margin, height - margin - keyboard_height)
        arrow_pos = (width - margin - keyboard_width, height - margin - keyboard_height)

        wasd_cache = {}
        arrow_cache = {}

        def get_actions_at_time(t):
            # Convert time to frame index
            frame_idx = int(t * fps)
            if 0 <= frame_idx < len(actions_timeline):
                return actions_timeline[frame_idx]
            return {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}

        def make_frame(t):
            frame = video.get_frame(t)
            actions = get_actions_at_time(t)

            def sign(x):
                if x > 0:
                    return 1
                elif x < 0:
                    return -1
                return 0

            wasd_key = (sign(actions.get("forward", 0)), sign(actions.get("left", 0)))
            arrow_key = (sign(actions.get("yaw", 0)), sign(actions.get("pitch", 0)))

            if wasd_key not in wasd_cache:
                wasd_cache[wasd_key] = create_wasd_keyboard(
                    actions, key_size, key_spacing, corner_radius
                )
            wasd_img = wasd_cache[wasd_key]

            if arrow_key not in arrow_cache:
                arrow_cache[arrow_key] = create_arrow_keyboard(
                    actions, key_size, key_spacing, corner_radius
                )
            arrow_img = arrow_cache[arrow_key]

            frame = blend_overlay(frame, wasd_img, wasd_pos)
            frame = blend_overlay(frame, arrow_img, arrow_pos)

            return frame

        output_clip = VideoClip(make_frame, duration=duration)
        output_clip = output_clip.set_fps(fps)

        if video.audio is not None:
            output_clip = output_clip.set_audio(video.audio)

        output_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            logger=None,
        )

        video.close()
        output_clip.close()

        return True

    except Exception as e:
        import traceback

        print(f"Error adding keyboard overlay: {e}")
        traceback.print_exc()
        return False


def generate_video(args):
    prompt_schedule = None
    if args.prompt_schedule_json:
        scheduled_pose, scheduled_video_length, prompt_schedule = load_prompt_schedule_json(
            args.prompt_schedule_json, args.prompt
        )
        args.pose = scheduled_pose
        args.video_length = scheduled_video_length
        if prompt_schedule:
            args.prompt = prompt_schedule[0]["prompt"]
        rank0_log(
            f"Loaded prompt schedule from {args.prompt_schedule_json}: "
            f"pose='{args.pose}', video_length={args.video_length}, "
            f"prompt_changes={len(prompt_schedule)}",
            "INFO",
        )

    assert (
        (args.video_length - 1) // 4 + 1
    ) % 4 == 0, "number of latents must be divisible by 4"
    initialize_infer_state(args)

    task = "i2v" if args.image_path else "t2v"

    enable_sr = args.sr

    # Build transformer_version based on flags
    transformer_version = f"{args.resolution}_{task}"
    assert transformer_version == "480p_i2v"

    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )

    extra_kwargs = {}
    if task == "i2v":
        extra_kwargs["reference_image"] = args.image_path

    enable_rewrite = args.rewrite
    if not args.rewrite:
        rank0_log(
            "Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.",
            "WARNING",
        )

    viewmats, Ks, action = pose_to_input(args.pose, (args.video_length - 1) // 4 + 1)

    if task == "i2v":
        extra_kwargs["reference_image"] = args.image_path

    out = pipe(
        enable_sr=enable_sr,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=args.video_length,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=enable_rewrite,
        return_pre_sr_video=args.save_pre_sr_video,
        viewmats=viewmats.unsqueeze(0),
        Ks=Ks.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=args.few_step,
        chunk_latent_frames=4 if args.model_type == "ar" else 16,
        model_type=args.model_type,
        user_height=args.height,
        user_width=args.width,
        transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
        prompt_schedule=prompt_schedule,
        **extra_kwargs,
    )

    # save video
    if int(os.environ.get("RANK", "0")) == 0:
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

        save_video_path = os.path.join(output_path, "gen.mp4")
        save_video_sr_path = os.path.join(output_path, "gen_sr.mp4")

        # Determine which video to process for UI overlay
        video_to_process = None
        final_video_path = None

        if enable_sr and hasattr(out, "sr_videos"):
            save_video(out.sr_videos, save_video_sr_path)
            print(f"Saved SR video to: {save_video_sr_path}")
            video_to_process = save_video_sr_path
            final_video_path = save_video_sr_path

            if args.save_pre_sr_video:
                save_video(out.videos, save_video_path)
                print(f"Saved original video (before SR) to: {save_video_path}")
        else:
            save_video(out.videos, save_video_path)
            print(f"Saved video to: {save_video_path}")
            video_to_process = save_video_path
            final_video_path = save_video_path

        # Add keyboard overlay if --with-ui is enabled and pose is a string
        if (
            args.with_ui
            and isinstance(args.pose, str)
            and not args.pose.endswith(".json")
        ):
            print(f"Adding keyboard overlay to video...")
            try:
                actions_timeline = parse_pose_string_to_actions(args.pose)

                # Create temporary output path for video with UI
                video_with_ui_path = os.path.join(output_path, "gen_with_ui_temp.mp4")

                if add_keyboard_overlay_to_video(
                    video_to_process, video_with_ui_path, actions_timeline
                ):
                    # Replace original video with UI version
                    os.replace(video_with_ui_path, final_video_path)
                    print(f"Successfully added keyboard overlay to: {final_video_path}")
                else:
                    print(f"Failed to add keyboard overlay, keeping original video")
                    if os.path.exists(video_with_ui_path):
                        os.remove(video_with_ui_path)
            except Exception as e:
                print(f"Error processing keyboard overlay: {e}")
                import traceback

                traceback.print_exc()


def generate_video_interactive(args):
    assert args.model_type == "ar", "Interactive mode requires model_type='ar'"

    interactive_video_length = args.interactive_video_length
    assert (interactive_video_length - 1) % 4 == 0, (
        f"interactive_video_length must be of form 4*n + 1, got {interactive_video_length}"
    )
    latent_num = (interactive_video_length - 1) // 4 + 1
    assert latent_num % 4 == 0, (
        "number of latents must be divisible by 4; "
        f"interactive_video_length={interactive_video_length} gives {latent_num} latents. "
        "Use values like 13, 29, 45, 61, ..."
    )

    initialize_infer_state(args)

    task = "i2v" if args.image_path else "t2v"

    enable_sr = False

    transformer_version = f"{args.resolution}_{task}"
    assert transformer_version == "480p_i2v"

    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )

    extra_kwargs = {}
    if task == "i2v":
        extra_kwargs["reference_image"] = args.image_path

    enable_rewrite = args.rewrite
    if not args.rewrite:
        rank0_log(
            "Warning: Prompt rewriting is disabled in interactive mode.",
            "WARNING",
        )

    camera_state = CameraState()

    if args.initial_pose:
        rank0_log(
            "initial_pose advances the camera before any latent history exists. "
            "For i2v interactive generation this can mismatch the reference image "
            "condition with the first generated camera pose.",
            "WARNING",
        )
        initial_motions = parse_pose_string(args.initial_pose)
        generate_camera_trajectory_local(initial_motions, camera_state)

    output_path = args.output_path or "./outputs"
    os.makedirs(output_path, exist_ok=True)
    accumulated_frames = []
    history_latents = None
    history_cond_latents = None
    history_viewmats = None
    history_Ks = None
    history_action = None
    action_prompt_memory = []
    round_count = 0
    is_rank0 = int(os.environ.get("RANK", "0")) == 0
    web_controller = None
    if args.web and is_rank0:
        web_controller = start_web_controller(args, output_path, latent_num)

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("Warning: OpenCV not found, display window will be disabled")

    display = None
    if is_rank0 and args.display_window and not args.web and has_cv2:
        from hyvideo.display import get_display, close_display
        display = get_display(create=True)

    print("\n" + "=" * 60)
    if args.web:
        print(f"Interactive Web Mode - open http://{args.web_host}:{args.web_port}")
    else:
        first_duration = latent_num if args.initial_pose else latent_num - 1
        print(
            "Interactive Mode - Enter pose commands "
            f"(first e.g. 'w-{first_duration}', then e.g. 'w-{latent_num}')"
        )
    print("Commands: w(forward), s(backward), a(left), d(right)")
    print("          up, down (pitch), left, right (yaw)")
    print("Type 'quit' or 'q' to exit, 'save' to save video")
    print("=" * 60 + "\n")

    web_idle_poll_seconds = 30

    while True:
        try:
            if is_rank0:
                if args.web:
                    try:
                        command_item = web_controller["queue"].get(
                            timeout=web_idle_poll_seconds
                        )
                    except queue.Empty:
                        command_item = {
                            "type": "noop",
                            "command": "",
                            "prompt": args.prompt,
                        }
                else:
                    print(f"\n--- Round {round_count + 1} ---")
                    command_item = {
                        "type": "action",
                        "command": input("Enter pose command: ").strip(),
                        "prompt": args.prompt,
                    }
            else:
                command_item = {}

            if torch.distributed.is_initialized():
                obj_list = [command_item]
                if get_parallel_state().sp_enabled:
                    group_src_rank = torch.distributed.get_global_rank(
                        get_parallel_state().sp_group, 0
                    )
                    torch.distributed.broadcast_object_list(
                        obj_list, src=group_src_rank, group=get_parallel_state().sp_group
                    )
                else:
                    torch.distributed.broadcast_object_list(
                        obj_list, src=0
                    )
                command_item = obj_list[0]

            if isinstance(command_item, str):
                command_item = {
                    "type": "action",
                    "command": command_item,
                    "prompt": args.prompt,
                }

            pose_input = str(command_item.get("command", "")).strip()
            command_type = command_item.get("type", "action")
            chunk_prompt = str(command_item.get("prompt") or args.prompt).strip()
            prompt_source = str(command_item.get("prompt_source") or "manual")
            if not chunk_prompt:
                chunk_prompt = args.prompt

            if command_type == "noop" or not pose_input:
                continue

            if is_rank0:
                print(f"\n--- Round {round_count + 1}: {pose_input} ---")

            if command_type == "quit" or pose_input in ["quit", "q", "exit"]:
                print("Exiting interactive mode...")
                break

            if command_type == "save" or pose_input in ["save"]:
                if accumulated_frames and is_rank0:
                    save_path = os.path.join(output_path, f"interactive_gen_{round_count}.mp4")
                    save_interactive_video(pipe, history_latents, accumulated_frames, save_path)
                    memory_path = os.path.join(output_path, f"interactive_memory_{round_count}.json")
                    save_action_prompt_memory(action_prompt_memory, memory_path, args.prompt)
                    print(f"Saved video to: {save_path}")
                    print(f"Saved action/prompt memory to: {memory_path}")
                    if web_controller is not None:
                        web_controller["mark_ready"](f"saved {os.path.basename(save_path)}")
                continue

            if web_controller is not None:
                web_controller["mark_busy"](pose_input, chunk_prompt)

            motions = parse_pose_string(pose_input)

            prev_c2w = camera_state.last_c2w.copy() if camera_state.last_c2w is not None else None

            next_camera_state = CameraState()
            next_camera_state.restore(camera_state.save())
            poses = generate_camera_trajectory_local(motions, next_camera_state)

            if prev_c2w is not None:
                incremental_poses = poses[1:]
            else:
                incremental_poses = poses

            if len(incremental_poses) == 0:
                print("No new poses to generate")
                continue
            if len(incremental_poses) != latent_num:
                expected_duration = latent_num if prev_c2w is not None else latent_num - 1
                raise ValueError(
                    f"Interactive command produced {len(incremental_poses)} latent poses, "
                    f"but interactive_video_length={interactive_video_length} requires {latent_num}. "
                    f"Use commands whose total duration is {expected_duration}, "
                    f"e.g. 'w-{expected_duration}'."
                )
            camera_state = next_camera_state

            w2c_list, Ks, action = incremental_poses_to_input(
                incremental_poses, prev_c2w=prev_c2w, intrinsic=np.array(camera_state.intrinsic)
            )

            out = pipe(
                enable_sr=False,
                prompt=chunk_prompt,
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                sr_num_inference_steps=None,
                video_length=interactive_video_length,
                negative_prompt=args.negative_prompt,
                seed=args.seed + round_count if args.seed else None,
                output_type="pt",
                prompt_rewrite=False,
                return_pre_sr_video=False,
                viewmats=w2c_list.unsqueeze(0),
                Ks=Ks.unsqueeze(0),
                action=action.unsqueeze(0),
                few_step=args.few_step,
                chunk_latent_frames=latent_num,
                model_type=args.model_type,
                user_height=args.height,
                user_width=args.width,
                history_latents=history_latents,
                history_cond_latents=history_cond_latents,
                history_viewmats=history_viewmats,
                history_Ks=history_Ks,
                history_action=history_action,
                start_latent_idx=history_latents.shape[2] if history_latents is not None else 0,
                transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
                **extra_kwargs,
            )

            frames = out.videos
            if history_latents is None:
                history_latents = out.latents
                history_cond_latents = out.cond_latents
                history_viewmats = out.viewmats
                history_Ks = out.Ks
                history_action = out.action
            else:
                history_latents = torch.cat([history_latents, out.latents], dim=2)
                history_cond_latents = torch.cat(
                    [history_cond_latents, out.cond_latents], dim=2
                )
                history_viewmats = torch.cat([history_viewmats, out.viewmats], dim=1)
                history_Ks = torch.cat([history_Ks, out.Ks], dim=1)
                history_action = torch.cat([history_action, out.action], dim=1)

            actions_dict = {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}
            for cmd in pose_input.split(","):
                cmd = cmd.strip()
                if not cmd:
                    continue
                parts = cmd.split("-")
                if len(parts) == 2:
                    action_name = parts[0].strip()
                    duration = int(parts[1].strip())
                    if action_name == "w":
                        actions_dict["forward"] = 1
                    elif action_name == "s":
                        actions_dict["forward"] = -1
                    elif action_name == "a":
                        actions_dict["left"] = 1
                    elif action_name == "d":
                        actions_dict["left"] = -1
                    elif action_name == "left":
                        actions_dict["yaw"] = -1
                    elif action_name == "right":
                        actions_dict["yaw"] = 1
                    elif action_name == "up":
                        actions_dict["pitch"] = 1
                    elif action_name == "down":
                        actions_dict["pitch"] = -1

            accumulated_frames.append(frames)
            if web_controller is not None and is_rank0:
                chunk_id = round_count + 1
                chunk_path = web_controller["add_chunk"](chunk_id, pose_input, chunk_prompt, frames)
                if args.vlm_prompt_cache:
                    web_controller["mark_prompt_cache"](f"analyzing chunk {chunk_id}")

                    def _vlm_task():
                        result = run_vlm_prompt_cache_task(
                            args=args,
                            video_path=chunk_path,
                            action=pose_input,
                            source_chunk_id=chunk_id,
                            used_prompt=chunk_prompt,
                        )
                        if result is None:
                            web_controller["mark_prompt_cache"](f"failed chunk {chunk_id}")
                        else:
                            target = chunk_id + args.vlm_prompt_target_offset
                            web_controller["mark_prompt_cache"](f"ready prompt {target}")

                    threading.Thread(target=_vlm_task, daemon=True).start()

            if display is not None:
                for i in range(frames.shape[2]):
                    frame = frames[0, :, i]
                    display.display_frame(frame.unsqueeze(0), actions_dict)

            round_count += 1
            action_prompt_memory.append(
                {
                    "round": round_count,
                    "command": pose_input,
                    "prompt": chunk_prompt,
                    "prompt_source": prompt_source,
                    "frames": int(frames.shape[2]),
                    "latent_frames": int(out.latents.shape[2]) if out.latents is not None else None,
                    "seed": args.seed + (round_count - 1) if args.seed else None,
                }
            )

            print(f"Generated {frames.shape[2]} frames. Total frames: {sum(f.shape[2] for f in accumulated_frames)}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            if web_controller is not None:
                web_controller["mark_ready"](f"error: {e}")
            import traceback
            traceback.print_exc()

    if display is not None:
        close_display()

    if accumulated_frames and is_rank0:
        output_path = args.output_path or "./outputs"
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, "interactive_final.mp4")
        save_interactive_video(pipe, history_latents, accumulated_frames, save_path)
        memory_path = os.path.join(output_path, "interactive_memory_final.json")
        save_action_prompt_memory(action_prompt_memory, memory_path, args.prompt)
        print(f"Saved final video to: {save_path}")
        print(f"Saved action/prompt memory to: {memory_path}")
    if web_controller is not None:
        web_controller["server"].shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Generate video using HunyuanWorld-1.5"
    )

    parser.add_argument(
        "--pose",
        type=str,
        default="./assets/pose/test_forward_32_latents.json",
        help="Path to pose JSON file or pose string (e.g., 'w-3, right-0.5, d-4')",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for video generation"
    )
    parser.add_argument(
        "--prompt_schedule_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file with non-interactive pose/prompt segments. "
            "When set, it overrides --pose and --video_length."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for video generation (default: empty string)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        required=True,
        choices=["480p", "720p"],
        help="Video resolution (480p or 720p)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--action_ckpt", type=str, required=True, help="Path to pretrained action model"
    )
    parser.add_argument(
        "--aspect_ratio", type=str, default="16:9", help="Aspect ratio (default: 16:9)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=127,
        help="Number of frames to generate (default: 127)",
    )
    parser.add_argument(
        "--sr",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable super resolution (default: true). "
        "Use --sr or --sr true/1 to enable, --sr false/0 to disable",
    )
    parser.add_argument(
        "--save_pre_sr_video",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Save original video before super resolution (default: false). "
        "Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, "
        "--save_pre_sr_video false/0 to disable",
    )
    parser.add_argument(
        "--rewrite",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable prompt rewriting (default: true). "
        "Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable",
    )
    parser.add_argument(
        "--offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable offloading (default: true). "
        "Use --offloading or --offloading true/1 to enable, "
        "--offloading false/0 to disable",
    )
    parser.add_argument(
        "--group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Enable group offloading (default: None, automatically enabled if offloading is enabled). "
        "Use --group_offloading or --group_offloading true/1 to enable, "
        "--group_offloading false/0 to disable",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Data type for transformer (default: bf16). "
        "bf16: faster, lower memory; fp32: better quality, slower, higher memory",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to reference image for i2v (if provided, uses i2v mode)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output file path for generated video (if not provided, saves to ./outputs/output.mp4)",
    )
    parser.add_argument(
        "--enable_torch_compile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable torch compile for transformer (default: false). "
        "Use --enable_torch_compile or --enable_torch_compile true/1 to enable, "
        "--enable_torch_compile false/0 to disable",
    )
    parser.add_argument(
        "--few_step",
        type=str_to_bool,
        nargs="?",
        const=False,
        default=False,
        help="Enable super resolution (default: true). "
        "Use --few_step or --few_step true/1 to enable, --few_step false/0 to disable",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["bi", "ar"],
        help="inference bidirectional or autoregressive model. ",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="height for generation (recommended to set as 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="width for generation (recommended to set as 832)",
    )
    parser.add_argument(
        "--with-ui",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Add keyboard overlay to generated video (default: false). "
        "Only works with pose string input, not JSON files. "
        "Use --with-ui or --with-ui true/1 to enable, --with-ui false/0 to disable",
    )

    parser.add_argument(
        "--use_sageattn",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable sageattn (default: false). "
        "Use --use_sageattn or --use_sageattn true/1 to enable, "
        "--use_sageattn false/0 to disable",
    )
    parser.add_argument(
        "--sage_blocks_range",
        type=str,
        default="0-53",
        help="Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)",
    )
    parser.add_argument(
        "--use_vae_parallel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable vae parallel (default: false). "
        "Use --use_vae_parallel or --use_vae_parallel true/1 to enable, "
        "--use_vae_parallel false/0 to disable",
    )
    # fp8 gemm related
    parser.add_argument(
        "--use_fp8_gemm",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable fp8 gemm for transformer (default: false). "
        "Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, "
        "--use_fp8_gemm false/0 to disable",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="fp8-per-block",
        help="Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-block)",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        default="double_blocks",
        help="Include patterns for fp8 gemm (default: double_blocks)",
    )
    parser.add_argument(
        "--transformer_resident_ar_rollout",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Keep transformer on GPU for entire AR rollout instead of per-chunk offloading (default: false). "
        "Reduces inference time without increasing peak VRAM. Only affects AR model_type with offloading enabled. "
        "Use --transformer_resident_ar_rollout or --transformer_resident_ar_rollout true to enable.",
    )
    parser.add_argument(
        "--interactive",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable interactive mode for real-time video generation (default: false). "
        "Use --interactive or --interactive true to enable.",
    )
    parser.add_argument(
        "--interactive_video_length",
        type=int,
        default=13,
        help="Number of frames to generate per interaction round (default: 13, which is 4 latents). "
        "Must produce a latent count divisible by 4: latents=(frames - 1)//4 + 1, "
        "so valid values include 13, 29, 45, 61, 77, ...",
    )
    parser.add_argument(
        "--initial_pose",
        type=str,
        default=None,
        help="Initial pose string for interactive mode (e.g., 'w-4'). "
        "If not provided, camera starts at origin.",
    )
    parser.add_argument(
        "--display_window",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable real-time video display window in interactive mode (default: true).",
    )
    parser.add_argument(
        "--web",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable browser-based interactive control for interactive mode (default: false).",
    )
    parser.add_argument(
        "--web_host",
        type=str,
        default="0.0.0.0",
        help="Host for the interactive web UI (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--web_port",
        type=int,
        default=7860,
        help="Port for the interactive web UI (default: 7860).",
    )
    parser.add_argument(
        "--vlm_prompt_cache",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable VLM prompt cache in interactive web mode.",
    )
    parser.add_argument(
        "--vlm_prompt_cache_path",
        type=str,
        default=None,
        help="Path to prompt_cache.json. Defaults to <output_path>/prompt_cache.json.",
    )
    parser.add_argument(
        "--vlm_prompt_target_offset",
        type=int,
        default=1,
        help="Target chunk offset for VLM prompt cache. 1 means chunk n writes prompt n+1.",
    )
    parser.add_argument(
        "--vlm_base_url",
        type=str,
        default=os.getenv("VLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible VLM base URL.",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default=os.getenv(
            "VLM_MODEL",
            "/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B",
        ),
        help="VLM model name/path served by vLLM.",
    )
    parser.add_argument(
        "--vlm_api_key",
        type=str,
        default=os.getenv("VLM_API_KEY", "None"),
        help="VLM API key for OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--vlm_max_frames",
        type=int,
        default=12,
        help="Max frames sampled from each generated chunk for VLM.",
    )
    parser.add_argument(
        "--vlm_max_image_size",
        type=int,
        default=1024,
        help="Max image dimension sent to VLM.",
    )
    parser.add_argument(
        "--vlm_temperature",
        type=float,
        default=0.1,
        help="VLM sampling temperature.",
    )
    parser.add_argument(
        "--vlm_max_tokens",
        type=int,
        default=1024,
        help="VLM max output tokens.",
    )
    parser.add_argument(
        "--vlm_timeout",
        type=int,
        default=600,
        help="VLM request timeout in seconds.",
    )
    parser.add_argument(
        "--vlm_frame_dir",
        type=str,
        default="/tmp/worldagent_vlm_frames",
        help="Temporary directory for VLM frame extraction.",
    )

    args = parser.parse_args()

    assert args.image_path is not None

    if args.interactive:
        generate_video_interactive(args)
    else:
        generate_video(args)


if __name__ == "__main__":
    main()
