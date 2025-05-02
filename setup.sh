#!/bin/bash

# Create necessary directories
mkdir -p ~/.cache/torch/hub/checkpoints

# Download required model files
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

# Install dependencies
pip install -r requirements.txt 