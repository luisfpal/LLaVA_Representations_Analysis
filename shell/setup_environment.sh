#!/bin/bash

# Go to project root (assumed one level up from script)
cd "$(dirname "$0")/.."
# dirname "$0" is the directory of this script

# Install uv and clean up the installer script
curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh
sh install_uv.sh
rm install_uv.sh

# Install dependencies
uv sync
# uv pip install flash-attn==2.5.7 --no-build-isolation

# Activate the environment
source .venv/bin/activate
# As with conda this step might be needed every time we want to use the environment
# Particularly because uv manages the environment locally from the project root
