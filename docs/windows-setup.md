# Windows Setup Guide

This guide covers setting up the LeKiwi Cat Player project on Windows.

## Prerequisites

- Windows 10/11 (64-bit)
- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda
- [Git for Windows](https://git-scm.com/download/win)
- (Optional) [CMake](https://cmake.org/download/) — only if you need to build apriltag-pybind11 from source
- (Optional) Orbbec camera + [Orbbec Viewer / USB driver](https://www.orbbec.com/developers/orbbec-sdk/)

## Hardware Requirements

The full pipeline requires an **Orbbec RGB-D camera** (e.g., Femto Bolt, Gemini 2) connected via USB. Without the camera, you can still work on planning/detection code using saved frames or test images.

The robot bridge (`scripts/robot_bridge.py`) requires network access to the LeKiwi robot at `192.168.100.1`.

## Installation

Open **Anaconda Prompt** (or any terminal with conda available):

```powershell
# Clone the repo
git clone <repo-url> lekiwi_catplayer
cd lekiwi_catplayer

# Create conda environment
conda create -n lekiwi python=3.11 -y
conda activate lekiwi

# Install Python dependencies
pip install -r requirements.txt
```

### Orbbec Camera SDK (pyorbbecsdk)

The `pyorbbecsdk` package in requirements.txt provides pre-built Windows amd64 wheels. If `pip install` fails:

1. Check https://github.com/orbbec/pyorbbecsdk/releases for offline `.whl` files
2. Or try the v2 package: `pip install pyorbbecsdk2`
3. Install the [Orbbec USB driver](https://www.orbbec.com/developers/orbbec-sdk/) — Windows needs this for camera access

### AprilTag Detector (pupil-apriltags)

Pre-built Windows wheels are available on PyPI — `pip install pupil-apriltags` should work directly.

You do **not** need to build the `third_party/apriltag-pybind11` submodule on Windows. That submodule is a Linux-specific alternative; the `pupil-apriltags` pip package replaces it.

## Running the Pipeline

### With a camera connected

```powershell
conda activate lekiwi

# Basic detection + visualization (opens browser at http://localhost:8080)
python -m perception.viz_detection --target cat

# With AprilTag localization (base-frame transform)
python -m perception.viz_detection --localize --target cat --distance 0.2
```

For the full pipeline with robot bridge, use two separate terminal windows:

```powershell
# Terminal 1: perception + visualization
python -m perception.viz_detection --localize --target cat

# Terminal 2: robot bridge (requires robot on network)
python scripts/robot_bridge.py
```

Note: The bash script `scripts/run_pipeline.sh` does not work on Windows. Use the manual two-terminal approach above, or run it via Git Bash.

### Without a camera (development only)

Without an Orbbec camera, you cannot run `viz_detection.py` directly. For development and testing of non-camera modules:

```powershell
# Test YOLO detection on a saved image
python -m perception.test_detection --image path\to\test_image.jpg

# Verify imports work (planning/grid code)
python -c "from perception.viz_detection import generate_grid_map, astar_grid; print('OK')"
```

## Compute T_base_tag (one-time calibration)

This script computes the static camera-to-base transform from a URDF file:

```powershell
pip install yourdfpy
python scripts/compute_T_base_tag.py
```

This saves `common/T_base_tag.npy`. You only need to run this once (or when the robot URDF changes). The `.npy` file can also be copied from a Linux machine.

Note: The URDF path in `scripts/compute_T_base_tag.py` is hardcoded to a Linux path. Update it to your local URDF location before running.

## Robot Host (SSH to LeKiwi)

The robot host script (`scripts/start_robot_host.sh`) uses `sshpass` which is Linux-specific. On Windows, connect to the robot manually:

```powershell
# Use Windows OpenSSH (type password when prompted)
ssh catplayer@192.168.100.1

# Then on the robot:
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/lerobot
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi --robot.cameras="{}" --host.connection_time_s=600
```

Or use PuTTY / Windows Terminal with saved credentials.

## Known Issues on Windows

| Issue | Workaround |
|-------|-----------|
| `pyorbbecsdk` import fails | Install Orbbec USB driver; try `pyorbbecsdk2` or download wheel from GitHub releases |
| `cv2` import crash | Ensure only one of `opencv-python` / `opencv-python-headless` is installed: `pip uninstall opencv-python-headless` |
| ZMQ "address already in use" | Another instance is running on the same port. Kill it via Task Manager or change ports in `common/config.py` |
| `T_base_tag.npy` not found | Run `python scripts/compute_T_base_tag.py` or copy the file from a configured machine |
| YOLO model download slow/fails | Download `yolo11x-seg.pt` manually from Ultralytics and place in repo root |
| `bash` scripts don't run | Use the PowerShell equivalents shown above, or install Git Bash |

## Using WSL as an Alternative

If you encounter persistent platform issues, Windows Subsystem for Linux (WSL2) provides a full Linux environment:

```powershell
# Install WSL (one-time, from admin PowerShell)
wsl --install -d Ubuntu-22.04

# Then inside WSL, follow the Linux setup:
bash setup_env.sh
```

USB camera passthrough to WSL2 requires [usbipd-win](https://github.com/dorssel/usbipd-win):

```powershell
# From Windows (admin PowerShell)
usbipd list
usbipd bind --busid <BUS-ID>
usbipd attach --wsl --busid <BUS-ID>
```

## Summary: What Works on Windows

| Component | Status |
|-----------|--------|
| YOLO detection (ultralytics) | Works |
| AprilTag localization (pupil-apriltags) | Works |
| A* path planning / grid map | Works |
| Viser 3D visualization | Works |
| Orbbec camera (pyorbbecsdk) | Works with driver installed |
| Robot bridge (ZMQ to LeKiwi) | Works (need network access to robot) |
| `setup_env.sh` | Use manual steps above instead |
| `scripts/run_pipeline.sh` | Use two terminals instead |
| `scripts/start_robot_host.sh` | SSH manually instead |
