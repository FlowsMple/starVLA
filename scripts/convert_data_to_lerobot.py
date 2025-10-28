"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
import sys
sys.path.append('.')
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    features = {
        "state.left_eef_position": {
            "dtype": "float32",
            "shape": (3,)
        },
        "state.left_eef_quaternion": {
            "dtype": "float32",
            "shape": (4,)
        },
        "state.left_gripper": {
            "dtype": "float32",
            "shape": (1,)
        },
        "state.left_joints": {
            "dtype": "float32",
            "shape": (7,)
        },
        "state.right_eef_position": {
            "dtype": "float32",
            "shape": (3,)
        },
        "state.right_eef_quaternion": {
            "dtype": "float32",
            "shape": (
                4,
            )
        },
        "state.right_gripper": {
            "dtype": "float32",
            "shape": (1,)
        },
        "state.right_joints": {
            "dtype": "float32",
            "shape": (7,)
        },
        "action.left_eef_position": {
            "dtype": "float32",
            "shape": (3,)
        },
        "action.left_eef_quaternion": {
            "dtype": "float32",
            "shape": (4,)
        },
        "action.left_delta_eef_position": {
            "dtype": "float32",
            "shape": (3,)
        },
        "action.left_delta_eef_quaternion": {
            "dtype": "float32",
            "shape": (4,)
        },
        "action.left_gripper": {
            "dtype": "float32",
            "shape": (1,)
        },
        "action.left_joints": {
            "dtype": "float32",
            "shape": (7,)
        },
        "action.right_eef_position": {
            "dtype": "float32",
            "shape": (
                3,
            )
        },
        "action.right_eef_quaternion": {
            "dtype": "float32",
            "shape": (
                4,
            )
        },
        "action.right_delta_eef_position": {
            "dtype": "float32",
            "shape": (
                3,
            )
        },
        "action.right_delta_eef_quaternion": {
            "dtype": "float32",
            "shape": (
                4,
            )
        },
        "action.right_gripper": {
            "dtype": "float32",
            "shape": (
                1,
            )
        },
        "action.right_joints": {
            "dtype": "float32",
            "shape": (7,)
        },
    }

    for cam in cameras:
        features[f"video.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640,3),
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False
            }
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
]:
    with h5py.File(ep_path, "r") as ep:
        state_joints = torch.from_numpy(ep["/observations/state_qpos"][:])
        state_eef = torch.from_numpy(ep["/observations/state_eef"][:])
        action_eef = torch.from_numpy(ep["/action_eef"][:])
        action_joints = torch.from_numpy(ep["/action_qpos"][:])
        action_delta_eef = torch.from_numpy(ep["/action_delta_eef"][:])
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state_joints,state_eef, action_eef,action_joints,action_delta_eef


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state_joints,state_eef, action_eef,action_joints,action_delta_eef = load_raw_episode_data(ep_path)
        num_frames = state_joints.shape[0]
        # add prompt
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)
        for i in range(num_frames):
            frame = {
                "state.left_eef_position": state_eef[i][:3],
                "state.left_eef_quaternion": state_eef[i][3:7],
                "state.left_gripper": state_eef[i][7:8],

                "state.right_eef_position": state_eef[i][8:11],
                "state.right_eef_quaternion": state_eef[i][11:15],
                "state.right_gripper": state_eef[i][15:16],

                "state.left_joints": state_joints[i][:7],
                "state.right_joints": state_joints[i][7:14],

                "action.left_eef_position": action_eef[i][:3],
                "action.left_eef_quaternion": action_eef[i][3:7],
                "action.left_gripper": action_eef[i][7:8],

                "action.right_eef_position": action_eef[i][8:11],
                "action.right_eef_quaternion": action_eef[i][11:15],
                "action.right_gripper": action_eef[i][15:16],

                "action.left_joints": action_joints[i][:7],
                "action.right_joints": action_joints[i][7:14],

                "action.left_delta_eef_position": action_delta_eef[i][:3],
                "action.left_delta_eef_quaternion": action_delta_eef[i][3:7],

                "action.right_delta_eef_position": action_delta_eef[i][7:10],
                "action.right_delta_eef_quaternion": action_delta_eef[i][10:14],
                "task": instruction,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"video.{camera}"] = img_array[i]

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def port_franka(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_franka-panda" if is_mobile else "franka-panda",
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_franka)
