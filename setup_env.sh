#!/bin/bash
# One-shot setup: create conda env, install deps, build apriltag.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up lekiwi environment ==="

# Create conda env
conda create -n lekiwi python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate lekiwi

# Install Python deps
pip install -r "$SCRIPT_DIR/requirements.txt"

# Build apriltag-pybind11
echo "Building apriltag-pybind11..."
cd "$SCRIPT_DIR/third_party/apriltag-pybind11"
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j"$(nproc)"

# Copy .so to site-packages
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
cp apriltag.cpython-*.so "$SITE_PACKAGES/"

echo "=== Setup complete ==="
echo "Activate with: conda activate lekiwi"
echo "Run demo with: cd $SCRIPT_DIR && bash run_demo.sh"
