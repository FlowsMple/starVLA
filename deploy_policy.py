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

@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config
        llm_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size

        DiTConfig = {"num_layers": llm_layers, "input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32}
        self.config.framework.action_model.hidden_size = 2048 #check what this for?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = llm_hidden_size

        self.config.framework.action_model.DiTConfig = DiTConfig
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

def encode_obs(observation):  # Post-Process Observation
    obs_0 = Image.fromarray(observation['observation']['head_camera']['rgb'])
    obs_1 = Image.fromarray(observation['observation']['left_camera']['rgb'])
    obs_2 = Image.fromarray(observation['observation']['right_camera']['rgb'])
    endpose = observation['endpose']
    state = np.concatenate([endpose['left_endpose'],[endpose['left_gripper']],endpose['right_endpose'],[endpose['right_gripper']]],axis=0)
    state = state.reshape(1,-1)
    return [obs_0,obs_1,obs_2],state


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    config_yaml = f"policy/starVLA/playground/{usr_args['ckpt_setting']}/config.yaml"
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
    import pdb
    pdb.set_trace()
    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        # TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass

