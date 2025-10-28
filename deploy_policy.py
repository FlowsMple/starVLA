# Copyright 2025 InternVLA-M1. All rights reserved.
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [add fake sample and predict_action to match with starVLA].
"""
InternVLA M1 framework:
Vision-Language-Action diffusion model integrating:
  - Qwen2.5 vision-language backbone
  - Layer-wise QFormer aggregation
  - DINO multi-view visual encoder
  - DiT diffusion head for future action sequence prediction
Primary goal: predict continuous future actions conditioned on multi-view images + instruction.
"""

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info


from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.QwenPI import Qwen_PI

def encode_obs(observation):  # Post-Process Observation
    obs_0 = Image.fromarray(observation['observation']['head_camera']['rgb'])
    obs_1 = Image.fromarray(observation['observation']['left_camera']['rgb'])
    obs_2 = Image.fromarray(observation['observation']['right_camera']['rgb'])
    endpose = observation['endpose']
    state = np.concatenate([endpose['left_endpose'],[endpose['left_gripper']],endpose['right_endpose'],[endpose['right_gripper']]],axis=0)
    state = state.reshape(1,-1)
    return [obs_0,obs_1,obs_2],state


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    config_yaml = f"policy/starVLA/playground/Checkpoints/{usr_args['ckpt_setting']}/config.yaml"
    cfg = OmegaConf.load(config_yaml)
    model = Qwen_PI(cfg)   
    model_path = f"/mnt/data/zijian/starVLA/RoboTwin/policy/starVLA/playground/Checkpoints/{usr_args['ckpt_setting']}/checkpoints/steps_25000_pytorch_model.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs,state = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    predict_output = model.predict_action(batch_images=[obs], instructions=[instruction],state = [state])
    actions = predict_output['normalized_actions'][0]
    for action in actions:  # Execute each step of the action
        # TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass

