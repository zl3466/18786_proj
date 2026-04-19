#!/bin/bash

# LlamaFactory Training Script for Qwen 2.5 7B on SQL Skeleton Dataset
# Usage: bash run_llamafactory.sh

# Activate environment if using conda
# conda activate text-to-sql

# Install llamafactory if not already installed
# pip install llamafactory

# Run training
llamafactory-cli train train/qwen_skeleton_llamafactory.yaml

echo "Training completed! Check ./qwen-skeleton-qlora-adapter for outputs"










