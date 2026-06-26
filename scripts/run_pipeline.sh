#!/bin/bash
# Start the full pipeline: viz_detection (perception + planning) + robot_bridge.
#
# Usage:
#   bash scripts/run_pipeline.sh [VIZ_OPTIONS] [-- BRIDGE_OPTIONS]
#
# Examples:
#   bash scripts/run_pipeline.sh --localize --target cat --distance 0.2
#   bash scripts/run_pipeline.sh --localize --target cup -- --arm-only
#   bash scripts/run_pipeline.sh --target cat --fps 5 -- --base-only
#
# Viz options (before --):
#   --target CLASS    Detection target (default: cat)
#   --localize        Enable AprilTag localization (base-frame transform)
#   --distance M      Standoff distance in meters (default: 0.2)
#   --fps N           Viz update rate (default: 3)
#   --port PORT       Viser port (default: 8080)
#
# Bridge options (after --):
#   --arm-only        Only arm flirting
#   --base-only       Only base path following
#
# Requires:
#   - Robot host running (bash scripts/start_robot_host.sh)

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Split args on "--": before goes to viz, after goes to bridge
VIZ_ARGS=""
BRIDGE_ARGS=""
seen_separator=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        seen_separator=true
        continue
    fi
    if $seen_separator; then
        BRIDGE_ARGS="$BRIDGE_ARGS $arg"
    else
        VIZ_ARGS="$VIZ_ARGS $arg"
    fi
done

# Default viz args if none given
if [ -z "$VIZ_ARGS" ]; then
    VIZ_ARGS="--localize --distance 0.2"
fi

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $VIZ_PID $BRIDGE_PID 2>/dev/null
    wait $VIZ_PID $BRIDGE_PID 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# Start viz_detection (perception + path planning)
echo "[1/2] Starting viz_detection ${VIZ_ARGS}..."
python -m perception.viz_detection $VIZ_ARGS &
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
