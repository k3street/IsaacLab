import sys
import os

# Add paths
sys.path.append("/home/kimate/open_arm_10Things/IsaacLab/source/isaaclab")
sys.path.append("/home/kimate/open_arm_10Things/IsaacLab/source/isaaclab_tasks")
sys.path.append("/home/kimate/open_arm_10Things/IsaacLab/source/isaaclab_mimic")
sys.path.append("/home/kimate/open_arm_10Things/openarm_isaac_lab/source/openarm")

try:
    from isaaclab.devices.gamepad.se3_gamepad import Se3Gamepad
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
