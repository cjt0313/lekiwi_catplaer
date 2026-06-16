#!/bin/bash
# Launch all nodes for the fake-data demo.
# Usage: conda activate lekiwi && bash run_demo.sh

set -e
trap 'kill $(jobs -p) 2>/dev/null; wait' EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== LeKiwi Cat Player Demo ==="
echo "Starting all nodes..."

python -m nodes.fake_camera_node &
sleep 0.5

python -m nodes.state_node &
python -m nodes.play_planner_node &
python -m nodes.base_planner_node &
python -m nodes.arm_planner_node &
python -m nodes.controller_node &
python -m nodes.viz_node &

echo "All nodes running. Visualization at http://localhost:8080"
echo "Press Ctrl+C to stop."
wait
