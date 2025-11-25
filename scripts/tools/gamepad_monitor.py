#!/usr/bin/env python3
"""Lightweight visualizer for Se3Gamepad inputs.

Run with:
    ./isaaclab.sh -p scripts/tools/gamepad_monitor.py --print_events
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

# Add IsaacLab sources to sys.path (mirrors record_demos.py setup)
SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(SOURCE_ROOT, "source/isaaclab"))
sys.path.append(os.path.join(SOURCE_ROOT, "source/isaaclab_tasks"))
sys.path.append(os.path.join(SOURCE_ROOT, "source/isaaclab_mimic"))
sys.path.append(os.path.join(SOURCE_ROOT, "source/isaaclab_assets"))
sys.path.append(os.path.join(SOURCE_ROOT, "source/isaaclab_rl"))

OPENARM_SOURCE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../openarm_isaac_lab/source/openarm")
)
sys.path.append(OPENARM_SOURCE_ROOT)

from isaaclab.app import AppLauncher

# These globals are populated after launching the simulator because importing
# `isaaclab.devices` pulls in OpenXR and other Omniverse modules that require
# a running SimulationApp first.
Se3Gamepad = None  # type: ignore[name-defined]
Se3GamepadCfg = None  # type: ignore[name-defined]
ui = None  # type: ignore[name-defined]


@dataclass
class MonitorCfg:
    """User-adjustable parameters for the visualizer."""

    pos_sensitivity: float
    rot_sensitivity: float
    dead_zone: float
    update_hz: float
    print_threshold: float
    print_events: bool


class LoopRate:
    """Very small helper to cap the UI refresh rate."""

    def __init__(self, hz: float) -> None:
        self._period = 1.0 / max(hz, 1e-3)
        self._last = time.perf_counter()

    def sleep(self) -> None:
        target = self._last + self._period
        while True:
            now = time.perf_counter()
            if now >= target:
                self._last = now
                return
            time.sleep(min(target - now, 0.001))


class GamepadVisualizer:
    """Creates an omni.ui panel that mirrors the current gamepad command."""

    AXIS_LABELS: Sequence[str] = (
        "Δx (m)",
        "Δy (m)",
        "Δz (m)",
        "Δroll (rad)",
        "Δpitch (rad)",
        "Δyaw (rad)",
        "Gripper",
    )

    def __init__(self, cfg: MonitorCfg) -> None:
        self._cfg = cfg
        self._device = Se3Gamepad(
            Se3GamepadCfg(
                pos_sensitivity=cfg.pos_sensitivity,
                rot_sensitivity=cfg.rot_sensitivity,
                dead_zone=cfg.dead_zone,
                sim_device="cpu",
            )
        )
        self._axis_models = [ui.SimpleFloatModel(0.0) for _ in self.AXIS_LABELS]
        self._value_labels = []
        self._status_label = None
        self._source_label = None
        self._build_window()

    def _describe_backend(self) -> str:
        if getattr(self._device, "_gamepad", None) is not None:
            name = self._device._input.get_gamepad_name(self._device._gamepad)
            return f"Carb device: {name}"
        if getattr(self._device, "_evdev_device", None) is not None:
            return f"evdev device: {self._device._evdev_device.name}"
        return "No controller detected"

    def _build_window(self) -> None:
        window = ui.Window(
            "Gamepad Monitor",
            width=420,
            height=360,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
        )
        with window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("SE(3) command preview", height=20, style={"font_size": 18})
                self._source_label = ui.Label(self._describe_backend())
                self._status_label = ui.Label("Waiting for samples…")
                ui.Spacer(height=6)
                for idx, label in enumerate(self.AXIS_LABELS):
                    with ui.HStack(spacing=8, height=24):
                        ui.Label(label, width=120)
                        slider = ui.FloatSlider(
                            min=-1.0,
                            max=1.0,
                            step=0.001,
                            model=self._axis_models[idx],
                        )
                        slider.enabled = False
                        value_label = ui.Label("0.000", width=80, alignment=ui.Alignment.RIGHT_CENTER)
                        self._value_labels.append(value_label)
                ui.Spacer(height=4)
                with ui.HStack(spacing=4):
                    ui.Label("Shortcuts: press ESC or close the window to exit.")

    def poll(self) -> None:
        command = self._device.advance()
        if not isinstance(command, torch.Tensor):
            return
        arr = command.detach().cpu().numpy()
        timestamp = time.strftime("%H:%M:%S")
        if self._status_label is not None:
            self._status_label.text = f"Last sample: {timestamp}"
        for idx, value in enumerate(arr[: len(self.AXIS_LABELS)]):
            clipped = float(np.clip(value, -1.0, 1.0))
            self._axis_models[idx].value = clipped
            if idx < len(self._value_labels):
                self._value_labels[idx].text = f"{value: .3f}"
        if self._cfg.print_events and np.any(np.abs(arr) >= self._cfg.print_threshold):
            print(f"[{timestamp}] cmd = {np.array2string(arr, precision=3)}", flush=True)

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self._device.reset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw SE3 gamepad inputs.")
    parser.add_argument("--headless", action="store_true", help="Run without a visible window (prints only).")
    parser.add_argument("--update_hz", type=float, default=30.0, help="UI refresh rate.")
    parser.add_argument("--pos_sensitivity", type=float, default=1.0, help="Position gain passed to Se3Gamepad.")
    parser.add_argument("--rot_sensitivity", type=float, default=1.6, help="Rotation gain passed to Se3Gamepad.")
    parser.add_argument("--dead_zone", type=float, default=0.01, help="Dead zone passed to Se3Gamepad.")
    parser.add_argument("--print_events", action="store_true", help="Log command vectors to stdout when active.")
    parser.add_argument(
        "--print_threshold",
        type=float,
        default=0.05,
        help="Absolute magnitude that triggers stdout logging when --print_events is used.",
    )
    parser.add_argument(
        "--force_evdev",
        action="store_true",
        help="Skip the Carb listener and read the controller via /dev/input (matches record_demos fallback).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.force_evdev:
        os.environ["ISAACLAB_FORCE_EVDEV"] = "1"

    # Ensure we launch with the same Python runtime as Isaac Sim, even if a user
    # activated a different virtual environment in their shell.
    isaac_python = os.path.join(os.path.dirname(__file__), "../../_isaac_sim/python.sh")
    if not os.environ.get("PYTHON_EXECUTABLE"):
        os.environ["PYTHON_EXECUTABLE"] = isaac_python

    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    # Import Omniverse-dependent modules now that the app is running.
    global Se3Gamepad, Se3GamepadCfg, ui  # noqa: PLW0603
    from isaaclab.devices.gamepad.se3_gamepad import Se3Gamepad as _Se3Gamepad, Se3GamepadCfg as _Se3GamepadCfg
    import omni.ui as _ui

    Se3Gamepad = _Se3Gamepad
    Se3GamepadCfg = _Se3GamepadCfg
    ui = _ui

    cfg = MonitorCfg(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        dead_zone=args.dead_zone,
        update_hz=args.update_hz,
        print_threshold=args.print_threshold,
        print_events=args.print_events,
    )
    visualizer = GamepadVisualizer(cfg)
    rate = LoopRate(args.update_hz)

    try:
        while simulation_app.is_running():
            simulation_app.update()
            visualizer.poll()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
