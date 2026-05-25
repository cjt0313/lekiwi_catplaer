# LeKiwi Cat Player (逗猫机器人)

Cat-teasing robot using the LeKiwi mobile manipulator. The system detects a cat, plans a partially-visible play position, navigates via A* path planning, and waves a wand toy.

## Architecture

Modular node-based system communicating over ZMQ PUB/SUB:

```
fake_camera_node → state_node → play_planner_node → base_planner_node
                                                   → arm_planner_node
                                                   → controller_node
                                                   → viz_node (browser)
```

**Sense–Plan–Act loop:**
- **Perception**: GroundingDINO detection, AprilTag localization, 2D occupancy grid from RGB-D
- **Planning**: Partial-visibility heuristic goal selection, A* path planning with collision inflation, predefined arm pose library
- **Control**: Proportional velocity tracking, joint position commands, distance-based safety stop

## State Machine

`INIT → WAIT_FOR_ROBOT → WAIT_FOR_CAT → SELECT_PLAY_TARGET → NAVIGATE_TO_TARGET → ARM_READY → PLAY → (cycle)`

Emergency transition: `CAT_TOO_CLOSE → STOP` (safety distance < 0.8m)

## Key Algorithms

- **Play position selection**: Annulus sampling around cat (radii 0.8/1.0/1.2m, 32 angles), scored by partial visibility (Bresenham ray cast), obstacle clearance, and travel cost
- **Path planning**: A* on 50×50 occupancy grid (2.5m × 2.5m, 5cm resolution) with 20cm collision radius inflation
- **Control**: Proportional controller (KP_linear=0.5, KP_angular=1.0) with waypoint tracking

## Quick Start

```bash
conda activate lekiwi
pip install -r requirements.txt
bash run_demo.sh
```

Open http://localhost:8080 for real-time visualization (2D map, robot/cat positions, planned path, state panel).

## Project Structure

```
common/          Config, message types, ZMQ helpers
nodes/           All processing nodes (fake_camera, state, planners, controller, viz)
viz/             Browser-based visualization (HTML5 Canvas + WebSocket)
third_party/     AprilTag pybind11 bindings
```

## Dependencies

- Python 3.10+, pyzmq, numpy, aiohttp
- Optional: python-pptx (slide generation)
