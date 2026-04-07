#!/bin/bash
set -e

VENV_DIR="${1:-$HOME/jurica_venv}"

echo "Creating venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing core packages ..."
pip install --upgrade pip
pip install numpy matplotlib tqdm pulp shapely
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "Installing swig 4.3.0 (required for VisiLibity build) ..."
pip install swig==4.3.0

echo "Installing VisiLibity ..."
pip install VisiLibity --no-build-isolation

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
