# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cat-teasing robot using the LeKiwi mobile manipulator. Detects a cat via YOLO-seg + AprilTag localization, plans a navigation path via A* on a 2D occupancy grid, and drives the robot to a standoff position.

## Commands

```bash
# Environment setup (one-time)
bash setup_env.sh

# Run the pipeline (two terminals)
conda activate lekiwi
python -m perception.viz_detection --localize --target cat --distance 0.3
python scripts/robot_bridge.py

# Or use the launcher script
bash scripts/run_pipeline.sh --localize --target cat --distance 0.3

# Visualization only (no robot control)
python -m perception.viz_detection --localize --target cat --no-zmq

# Compute camera-to-base transform (one-time calibration)
python scripts/compute_T_base_tag.py
```

Open http://localhost:8080 for the viser 3D viewer.

There is no test suite, linter, or CI configured.

## Architecture

Two-process system: a single perception pipeline and a separate robot bridge.

### Data flow

```
Orbbec Camera → YOLO-seg Detection → AprilTag Localization → Grid Map → A* Path
                                                                           ↓
Browser (viser @ :8080) ← viz_detection.py ─── ZMQ PUB ───→ robot_bridge.py → LeKiwi
```

### Process responsibilities

- **perception/viz_detection.py** — Main pipeline. Camera capture, YOLO-seg object detection, AprilTag-based base-frame transform, 2D occupancy grid generation, A* path planning, viser 3D visualization. Publishes detection + path on ZMQ PUB (port 5560).
- **scripts/robot_bridge.py** — Subscribes to path via ZMQ, computes velocity commands, sends to LeKiwi robot over network (192.168.100.1:5555). Supports `--arm-only`, `--base-only`, or combined mode.

### Key modules

- `perception/camera.py` — Orbbec RGB-D camera wrapper (pyorbbecsdk)
- `perception/detection.py` — YOLO-seg model loading, inference, 3D point extraction, AABB fitting
- `perception/localization.py` — AprilTag detection, pose estimation, frame transforms
- `common/config.py` — All tunable parameters (detection, grid map, distances, ZMQ ports)
- `common/types.py` — `MsgType` enum (ZMQ message topics)
- `common/zmq_message.py` — `ZmqMessage` dataclass with `create/to_bytes`, `make_publisher/publish` helpers

### ZMQ topology

- `viz_detection.py` PUB binds `tcp://127.0.0.1:5560` (topic: `cat_bbox_3d`)
- `viz_detection.py` PUB binds `tcp://127.0.0.1:5570` (topic: `bridge_control` — GUI enable/e-stop)
- `robot_bridge.py` SUB connects to both, PUSH to robot at `tcp://192.168.100.1:5555`

## Dependencies

- Python 3.11 via conda (`lekiwi` env)
- pyzmq, numpy, opencv-python, ultralytics, viser, pupil-apriltags, pyorbbecsdk
