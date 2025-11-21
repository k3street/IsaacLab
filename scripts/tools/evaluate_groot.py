# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to evaluate a 'Groot' policy (or any foundation model) in Isaac Lab.

This script loads the OpenArm bimanual environment and runs a policy loop.
It is designed to be a pipeline stage:
1. Load Pre-trained Model
2. Run Inference in Sim
3. Report Success Rate

Usage:
    ./isaaclab.sh -p ../IsaacLab/scripts/tools/evaluate_groot.py --task Isaac-Grasp-Cube-OpenArm-Bi-Play-v0 --num_demos 10
"""

import argparse
import sys
import os
import torch
import numpy as np
import time
import json
import tempfile
from huggingface_hub import hf_hub_download

# Add IsaacLab source to path
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source"))
sys.path.append(os.path.join(source_dir, "isaaclab"))
sys.path.append(os.path.join(source_dir, "isaaclab_tasks"))
sys.path.append(os.path.join(source_dir, "isaaclab_mimic"))
sys.path.append(os.path.join(source_dir, "isaaclab_assets"))
sys.path.append(os.path.join(source_dir, "isaaclab_rl"))

# Add OpenArm source to path
openarm_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../openarm_isaac_lab/source/openarm"))
sys.path.append(openarm_source_dir)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate Groot Policy in Isaac Lab.")
parser.add_argument("--task", type=str, default="Isaac-Grasp-Cube-OpenArm-Bi-Play-v0", help="Name of the task.")
parser.add_argument("--num_demos", type=int, default=5, help="Number of episodes to evaluate.")
parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to Groot model weights or HF ID.")
# parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras for vision-based policies.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import openarm  # noqa: F401
from isaaclab.utils import configclass

# LeRobot Imports
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import get_device_from_parameters

# -----------------------------------------------------------------------------
# GROOT POLICY WRAPPER
# -----------------------------------------------------------------------------
class GrootPolicyWrapper:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"[Groot] Loading model from {model_path}...")
        
        # 1. Load Config
        try:
            self.cfg = PreTrainedConfig.from_pretrained(model_path)
        except Exception as e:
            print(f"[Groot] Standard config load failed: {e}.")
            print(f"[Groot] Assuming {model_path} is a raw GR00T model checkpoint. Creating default GrootConfig.")
            from lerobot.policies.groot.configuration_groot import GrootConfig
            self.cfg = GrootConfig(base_model_path=model_path)
            
            # Manually populate features to satisfy validation
            self.cfg.input_features = {
                "observation.images.primary": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(self.cfg.max_state_dim,))
            }
            self.cfg.output_features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.cfg.max_action_dim,))
            }

            # Force disable flash attention in backbone config
            self.cfg.backbone_cfg = {
                'eagle_path': 'NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops',
                'load_bf16': False,
                'project_to_dim': None,
                'reproject_vision': False,
                'select_layer': 12,
                'tune_llm': False,
                'tune_visual': True,
                'use_flash_attention': False,
            }

        self.cfg.pretrained_path = model_path
        self.cfg.device = device
        
        # 2. Create Policy
        # We pass a dummy ds_meta to bypass the check in make_policy
        # The config already has the features, so we don't need real metadata
        class DummyDatasetMetadata:
            def __init__(self):
                self.features = {}
        
        self.policy = make_policy(self.cfg, ds_meta=DummyDatasetMetadata())
        self.policy.eval()
        
        # 3. Create Processors
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.cfg, 
            pretrained_path=model_path
        )
        
        print("[Groot] Model loaded successfully.")
        self.print_keys_once = True

    def reset(self):
        self.policy.reset()

    def get_action(self, obs, env=None, instruction="grasp the cube"):
        """
        Args:
            obs (dict): Observation dictionary from Isaac Lab.
            env (ManagerBasedRLEnv): The environment instance (optional, for direct state access).
            instruction (str): Text instruction for the task.
        Returns:
            torch.Tensor: Action tensor
        """
        # 1. Inspect expected keys on first run
        if self.print_keys_once:
            print(f"[Groot] Expected Input Features: {self.cfg.input_features.keys()}")
            self.print_keys_once = False

        # 2. Construct LeRobot Observation Dictionary
        lerobot_obs = {}
        
        # Map Joint State
        # Prefer reading directly from env to get all joints (including grippers)
        if env is not None:
            # Get all joint positions: (num_envs, num_joints)
            # OpenArm has 16 controlled joints (7 left + 7 right + 2 grippers) usually
            # But the articulation might have more (fingers).
            # We need to know the indices.
            # For now, let's assume the robot is defined such that we can just take the relevant ones.
            # Or we can rely on the 'policy' obs if we trust it.
            
            # Let's try to use the 'policy' obs first as it's safer for sim-to-real (which won't have 'env')
            # But if we are missing grippers, we might need to augment.
            pass

        if "policy" in obs:
            # Extract first 14 elements as joint positions (left + right arms)
            joint_pos_arms = obs["policy"][:, :14]
            
            # If we need gripper state and it's not in obs, we might need to fake it or get it from env
            # For now, let's assume Groot might be fine with just arms, or we append 0s for grippers
            # if the model expects 16 dims.
            # We'll check the expected shape dynamically if possible, but for now just pass arms.
            lerobot_obs["observation.state"] = joint_pos_arms
        
        # Map Images
        if "vision" in obs:
            # Map overhead camera
            if "overhead_rgb" in obs["vision"]:
                # LeRobot expects (B, C, H, W) usually, Isaac Lab gives (B, H, W, C)
                img = obs["vision"]["overhead_rgb"]
                if img.shape[-1] == 3: # HWC -> CHW
                    img = img.permute(0, 3, 1, 2)
                lerobot_obs["observation.images.top"] = img
                # Also try 'primary' as a common key
                lerobot_obs["observation.images.primary"] = img

            # Map wrist cameras
            if "left_wrist_rgb" in obs["vision"]:
                img = obs["vision"]["left_wrist_rgb"]
                if img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2)
                lerobot_obs["observation.images.left_wrist"] = img
            
            if "right_wrist_rgb" in obs["vision"]:
                img = obs["vision"]["right_wrist_rgb"]
                if img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2)
                lerobot_obs["observation.images.right_wrist"] = img

        # Map Text Instruction
        lerobot_obs["text"] = [instruction] * obs["policy"].shape[0]

        # 3. Filter to only keys expected by the model
        input_keys = set(self.cfg.input_features.keys())
        filtered_obs = {k: v for k, v in lerobot_obs.items() if k in input_keys}
        
        # 4. Run Inference
        # Move to device
        filtered_obs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in filtered_obs.items()}
        
        # Preprocess
        try:
            filtered_obs = self.preprocessor(filtered_obs)
        except Exception as e:
            print(f"[Groot] Preprocessing failed: {e}")
            pass

        with torch.inference_mode():
            action = self.policy.select_action(filtered_obs)
        
        # Postprocess
        action = self.postprocessor(action)
        
        return action

# -----------------------------------------------------------------------------
# MAIN EVALUATION LOOP
# -----------------------------------------------------------------------------
def main():
    # Parse env config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    
    # Ensure cameras are enabled
    if args_cli.enable_cameras:
        # Force enable cameras if not already
        pass

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Load Policy
    policy = GrootPolicyWrapper(args_cli.model_path, device=args_cli.device)
    
    # Reset
    obs, _ = env.reset()
    policy.reset()
    
    print(f"[INFO] Starting evaluation for {args_cli.num_demos} episodes...")
    
    for episode_i in range(args_cli.num_demos):
        # Reset environment
        obs, _ = env.reset()
        policy.reset()
        
        step_count = 0
        while True:
            # Get Action
            action = policy.get_action(obs, env=env)
            
            # Apply Action
            # If action is 14 dims (joints only), we might need to append gripper
            if action.shape[-1] == 14:
                # Append gripper actions (open/close)
                # For now, keep them open (positive) or closed (0)
                gripper_action = torch.ones((action.shape[0], 2), device=env.device) * 0.04
                action = torch.cat([action, gripper_action], dim=-1)
            
            # Step
            obs, rew, terminated, truncated, info = env.step(action)
            
            step_count += 1
            
            if terminated or truncated:
                print(f"Episode {episode_i} finished after {step_count} steps.")
                break
    
    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
