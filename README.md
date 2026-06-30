# LeKiwi Cat Player (逗猫机器人)

Cat-teasing robot using the LeKiwi mobile manipulator. Detects a cat via YOLO-seg + AprilTag localization, plans a navigation path via A* on a 2D occupancy grid, and drives the robot to a standoff position.

## Architecture

Single-process perception pipeline (`perception/viz_detection.py`) + separate robot bridge (`scripts/robot_bridge.py`):

```
Orbbec Camera → YOLO Detection → AprilTag Localization → Grid Map → A* Path
                                                                       ↓
Browser (viser 3D viewer) ← viz_detection ← path plan ← robot_bridge → LeKiwi
```

- **viz_detection.py** — Camera capture, YOLO-seg object detection, AprilTag-based base-frame localization, 2D occupancy grid, A* path planning, viser 3D visualization (http://localhost:8080)
- **robot_bridge.py** — Subscribes to planned path over ZMQ, translates to velocity commands, sends to LeKiwi robot over the network

## Quick Start

```bash
conda activate lekiwi
pip install -r requirements.txt

# Full pipeline (two terminals)
python -m perception.viz_detection --localize --target cat --distance 0.3
python scripts/robot_bridge.py

# Or use the launcher script (Linux/Mac)
bash scripts/run_pipeline.sh --localize --target cat --distance 0.3
```

Open http://localhost:8080 for the real-time 3D visualization with grid map and path overlay.

## Setup Guides

- [Mac Quick Start](docs/mac-quickstart.md)
- [Windows Setup](docs/windows-setup.md)

## Project Structure

```
common/              Config, message types, ZMQ helpers
perception/          Camera, detection, localization, main pipeline
scripts/             Robot bridge, pipeline launcher, robot host SSH
docs/                Platform setup guides
```

## Dependencies

- Python 3.11, pyzmq, numpy, opencv-python
- ultralytics (YOLO-seg), pupil-apriltags, pyorbbecsdk (Orbbec camera)
- viser (3D web visualization)

## Hardware

- Orbbec RGB-D camera (e.g., Femto Bolt, Gemini 2)
- LeKiwi mobile manipulator (network-connected at 192.168.100.1)
- AprilTag (tagStandard41h12, 35.8mm) mounted on the robot

## Network Setup (Mac)

Connect your Mac to the robot via USB Ethernet adapter, then set a static IP:

```bash
# Set static IP on the USB Ethernet interface
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.100.2 255.255.255.0

# Verify connectivity
ping 192.168.100.1

# Revert to DHCP when done
sudo networksetup -setdhcp "USB 10/100/1000 LAN"
```

If your adapter has a different name, run `networksetup -listallhardwareports` to find it.
