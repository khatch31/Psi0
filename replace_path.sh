#!/bin/bash

# Script to replace old path with new path in all files

OLD_PATH="/home/ubuntu/Desktop/world_models_project/psi0_workspace"
NEW_PATH="/home/songlin/Projects/psi0_kyle/psi0_workspace"
TARGET_DIR="../training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254/"

echo "Replacing occurrences of:"
echo "  OLD: $OLD_PATH"
echo "  NEW: $NEW_PATH"
echo "In directory: $TARGET_DIR"
echo ""

# Find all files and replace the path (using # as delimiter to avoid escaping slashes)
find "$TARGET_DIR" -type f -exec sed -i "s#${OLD_PATH}#${NEW_PATH}#g" {} +

echo "✓ Replacement complete!"
