# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cat-teasing robot using the LeKiwi mobile manipulator. Detects a cat via GroundingDINO + AprilTag localization, plans a partially-visible play position, navigates via A* path planning, and waves a wand toy. Currently runs in simulation mode with a fake camera node.

## Commands

```bash
# Environment setup (one-time)
bash setup_env.sh

# Run the full demo (all nodes + visualization)
conda activate lekiwi && bash run_demo.sh

# Run a single node
conda activate lekiwi && python -m nodes.<node_name>
# e.g. python -m nodes.state_node

# Visualization
# Open http://localhost:8080 after run_demo.sh
```

There is no test suite, linter, or CI configured.

## Architecture

Decoupled node-based system. Each node is a standalone process communicating over ZMQ PUB/SUB (one port per publisher). Nodes are run as Python modules from the repo root.

### Data flow

```
fake_camera_node ──→ state_node ──→ play_planner_node ──→ base_planner_node
       ↑                                                → arm_planner_node
       │                                                → controller_node
       └────────────────────────────────────────────────────┘ (closed loop)
                                                        → viz_node (WebSocket → browser)
```

### Node responsibilities

- **fake_camera_node** — Simulates perception: publishes robot pose, cat bounding box, and occupancy grid at 5Hz. Integrates velocity commands from controller to simulate motion.
- **state_node** — Central state machine. Subscribes to perception + planner outputs, publishes `WorldState` that all other nodes consume.
- **play_planner_node** — Selects a "peek" position around the cat (annulus sampling scored by partial visibility, obstacle proximity, travel cost).
- **base_planner_node** — A* path planning on inflated 2D occupancy grid (50×50 cells, 5cm resolution).
- **arm_planner_node** — Selects predefined arm poses (HOME/PLAY/RETRACT) based on state.
- **controller_node** — Proportional velocity controller with waypoint tracking and safety stop.
- **viz_node** — aiohttp server bridging ZMQ → WebSocket to an HTML5 Canvas frontend.

### State machine

`INIT → WAIT_FOR_ROBOT → WAIT_FOR_CAT → SELECT_PLAY_TARGET → NAVIGATE_TO_TARGET → ARM_READY → PLAY → (cycle back to SELECT_PLAY_TARGET)`

Emergency: any state → `CAT_TOO_CLOSE → STOP` if distance < 0.8m.

### ZMQ topology

All addresses defined in `common/config.py`. Each publisher binds a unique port (5560–5565). Messages are JSON-serialized `ZmqMessage` with topic-based multipart routing (topic = message type string).

### Key shared types

- `common/types.py` — `MsgType` enum (message topics)
- `common/zmq_message.py` — `ZmqMessage` dataclass with `create/to_bytes/from_bytes`, plus `make_publisher/make_subscriber/publish/receive` helpers
- `common/config.py` — All tunable parameters (distances, map size, control gains, arm poses, ports)

## Dependencies

- Python 3.11 via conda (`lekiwi` env)
- pyzmq, numpy, aiohttp
- Third-party submodule: `third_party/apriltag-pybind11` (C++ with pybind11, requires CMake build)
