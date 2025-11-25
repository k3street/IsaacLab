import sys
import os
import inspect

# Add IsaacLab source to path
# Assuming this script is in IsaacLab/
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "source"))
sys.path.append(os.path.join(source_dir, "isaaclab"))

# We might need to launch the app to load extensions, but let's try importing first.
# Some modules might depend on 'omni' which is only available when the app is running.
# But 'omni.isaac.lab.devices' might be importable if it's just python code.

try:
    from isaaclab.devices import Se3Gamepad
    print("Imported from isaaclab.devices")
except ImportError:
    try:
        from omni.isaac.lab.devices import Se3Gamepad
        print("Imported from omni.isaac.lab.devices")
    except ImportError as e:
        print(f"Failed to import: {e}")
        sys.exit(1)

print("Se3Gamepad docstring:")
print(Se3Gamepad.__doc__)
print("\nSe3Gamepad init signature:")
print(inspect.signature(Se3Gamepad.__init__))
