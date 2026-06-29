#!/bin/bash
# One-shot setup: create conda env and install deps.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up lekiwi environment ==="

# Create conda env
conda create -n lekiwi python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate lekiwi

# Install Python deps
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "=== Setup complete ==="
echo "Activate with: conda activate lekiwi"
echo "Run pipeline with: cd $SCRIPT_DIR && bash scripts/run_pipeline.sh --localize --target cat"
