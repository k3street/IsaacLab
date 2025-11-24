
import os
import sys
import re
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil
import time

# Configuration
CONFIG_FILE = "/home/kimate/open_arm_10Things/openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/joint_pos_env_cfg.py"
EVAL_SCRIPT = "/home/kimate/open_arm_10Things/IsaacLab/scripts/tools/evaluate_groot.py"
OUTPUT_DIR = "/home/kimate/open_arm_10Things/IsaacLab/output_images"
CALIB_DIR = "/home/kimate/open_arm_10Things/IsaacLab/calibration_images"

# Test Parameters
# We want to test different pitch angles (downward tilt)
# Base rotation is 90 deg around Z (to align camera X with gripper Y)
# Then we pitch down around local X
PITCH_ANGLES = [30, 45, 60, 75, 90] 
Z_OFFSETS = [-0.25] # Keep the one we just set, or try others if needed

def calculate_quaternion(pitch_deg):
    # Base: 90 deg rotation around Z
    # q_base = [0, 0, sin(45), cos(45)] = [0, 0, 0.7071, 0.7071] (x,y,z,w)
    # But Isaac Lab uses (w, x, y, z) in config? 
    # Wait, the user's config had `rot=(0.7071, 0, 0, 0.7071)` with convention="ros".
    # Let's stick to the scipy logic we used in the terminal which worked.
    # Terminal: q_base = [0.0, 0.0, 0.7071068, 0.7071068] (x,y,z,w)
    
    r_base = R.from_quat([0.0, 0.0, 0.7071068, 0.7071068])
    r_pitch = R.from_euler('x', pitch_deg, degrees=True)
    r_new = r_base * r_pitch
    q_new = r_new.as_quat() # x, y, z, w
    
    # Isaac Lab config expects (w, x, y, z) usually
    # BUT the user's file has `rot=(w, x, y, z)` format based on previous edits?
    # Let's check the file content again.
    # The file has `rot=(0.6830, 0.1830, 0.1830, 0.6830)`
    # My terminal output for 30 deg was: `0.6830, 0.1830, 0.1830, 0.6830` (w, x, y, z)
    # So we need to return (w, x, y, z)
    
    return (q_new[3], q_new[0], q_new[1], q_new[2])

def update_config(pitch_deg, z_offset):
    quat = calculate_quaternion(pitch_deg)
    quat_str = f"({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})"
    pos_str = f"(0.10, 0.0, {z_offset})"
    
    print(f"Updating config to Pitch={pitch_deg}, Z={z_offset} -> Rot={quat_str}")
    
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    
    # Pattern:
    # (left_wrist_camera.*?offset=CameraCfg\.OffsetCfg\(\s*pos=)\(.*?\)(,\s*rot=)\(.*?\)(,)
    # Group 1: prefix up to `pos=`
    # Match: `(...)` (the old pos tuple)
    # Group 2: `, rot=`
    # Match: `(...)` (the old rot tuple)
    # Group 3: `,` (trailing comma)
    
    # Replacement: \1{pos_str}\2{quat_str}\3
    
    pattern_left = r'(left_wrist_camera.*?offset=CameraCfg\.OffsetCfg\(\s*pos=)\(.*?\)(,\s*rot=)\(.*?\)(,)'
    replacement_left = f'\\1{pos_str}\\2{quat_str}\\3'
    
    content = re.sub(pattern_left, replacement_left, content, flags=re.DOTALL)
    
    pattern_right = r'(right_wrist_camera.*?offset=CameraCfg\.OffsetCfg\(\s*pos=)\(.*?\)(,\s*rot=)\(.*?\)(,)'
    replacement_right = f'\\1{pos_str}\\2{quat_str}\\3'
    
    content = re.sub(pattern_right, replacement_right, content, flags=re.DOTALL)
    
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)

def run_evaluation():
    # Construct the command with the environment setup as specified by the user
    setup_cmd = (
        "ln -sfn /home/kimate/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64 _isaac_sim && "
        "source ../openarm_isaac_sim/scripts/setup_isaac_ros_env.sh && "
        "export VIRTUAL_ENV= && "
        "export PYTHONPATH=/home/kimate/open_arm_10Things/IsaacLab/source/isaaclab:/home/kimate/open_arm_10Things/IsaacLab/source/isaaclab_tasks:/home/kimate/open_arm_10Things/IsaacLab/source/isaaclab_mimic:/home/kimate/open_arm_10Things/openarm_isaac_lab/source/openarm:$PYTHONPATH && "
        "export PATH=/home/kimate/open_arm_10Things/IsaacLab/_isaac_sim/kit/python/bin:$PATH && "
        "./isaaclab.sh -p scripts/tools/evaluate_groot.py "
        "--task Isaac-Grasp-Cube-OpenArm-Bi-Play-v0 "
        "--num_demos 1 "
        "--enable_cameras "
        "--max_steps 1"
    )
    
    print("Running simulation...")
    # Run with shell=True to allow sourcing and exporting
    subprocess.run(setup_cmd, shell=True, check=True, cwd="/home/kimate/open_arm_10Things/IsaacLab", executable="/bin/bash")

def save_results(pitch, z_offset):
    if not os.path.exists(CALIB_DIR):
        os.makedirs(CALIB_DIR)
        
    # Copy the generated images
    # We expect output_images/ep0_step0_left_wrist_rgb.png and right_wrist_rgb.png
    
    src_left = os.path.join(OUTPUT_DIR, "ep0_step0_left_wrist_rgb.png")
    src_right = os.path.join(OUTPUT_DIR, "ep0_step0_right_wrist_rgb.png")
    
    dst_left = os.path.join(CALIB_DIR, f"pitch_{pitch}_z_{z_offset}_left.png")
    dst_right = os.path.join(CALIB_DIR, f"pitch_{pitch}_z_{z_offset}_right.png")
    
    if os.path.exists(src_left):
        shutil.copy(src_left, dst_left)
        print(f"Saved {dst_left}")
    
    if os.path.exists(src_right):
        shutil.copy(src_right, dst_right)
        print(f"Saved {dst_right}")

def main():
    print("Starting Camera Calibration...")
    
    for pitch in PITCH_ANGLES:
        for z in Z_OFFSETS:
            try:
                update_config(pitch, z)
                run_evaluation()
                save_results(pitch, z)
            except Exception as e:
                print(f"Error processing Pitch={pitch}, Z={z}: {e}")
                
    print(f"\nCalibration complete. Images saved to {CALIB_DIR}")
    print("Please inspect the images and choose the best pitch angle.")

if __name__ == "__main__":
    main()
