#!/bin/bash

# Go to project root
cd ../

# Install uv
pip install uv

# Install dependencies
uv sync
uv pip install flash-attn==2.5.7 --no-build-isolation
