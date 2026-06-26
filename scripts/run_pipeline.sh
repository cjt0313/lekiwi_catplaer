#!/bin/bash
# Start the full pipeline: viz_detection (perception + planning) + robot_bridge.
# Usage: bash scripts/run_pipeline.sh [--arm-only|--base-only]
#
# Requires:
#   - Robot host running (bash scripts/start_robot_host.sh)
#   - conda env: lekiwi (for bridge), gspy312 (for viz)

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

BRIDGE_ARGS="${@}"

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $VIZ_PID $BRIDGE_PID 2>/dev/null
    wait $VIZ_PID $BRIDGE_PID 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# Start viz_detection (perception + path planning)
echo "[1/2] Starting viz_detection..."
python -m perception.viz_detection --localize --distance 0.2 &
VIZ_PID=$!
sleep 2

# Start robot bridge
echo "[2/2] Starting robot_bridge ${BRIDGE_ARGS}..."
python scripts/robot_bridge.py $BRIDGE_ARGS &
BRIDGE_PID=$!

echo ""
echo "Pipeline running. Open http://localhost:8080 for visualization."
echo "Use the Robot ON/OFF and E-STOP buttons in the GUI."
echo "Press Ctrl+C to stop everything."
echo ""

wait
