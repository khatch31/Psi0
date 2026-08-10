#!/usr/bin/env bash
#
# Zip each 2026-06-19* item into its own .zip file for upload.
#
# Usage:
#   zip_for_upload.sh [TARGET_DIR] [PATTERN]
#
#   TARGET_DIR  Directory to scan (default: the saved_inference dir).
#   PATTERN     Glob prefix to match (default: 2026-06-19*).
#
set -euo pipefail

TARGET_DIR="${1:-$HOME/Desktop/world_models_project/psi0_workspace/saved_inference}"
# PATTERN="${2:-2026-06-*}"
# PATTERN="${2:-2026*}"

PATTERN="*"

cd "$TARGET_DIR"

shopt -s nullglob
matches=( $PATTERN )

if [ ${#matches[@]} -eq 0 ]; then
  echo "No items matching '$PATTERN' in $TARGET_DIR"
  exit 0
fi

for item in "${matches[@]}"; do
  # Skip the zip files themselves so reruns don't re-zip output.
  case "$item" in
    *.zip) continue ;;
  esac

  zipname="${item%/}.zip"

  if [ -e "$zipname" ]; then
    echo "Skipping '$item' -> '$zipname' already exists"
    continue
  fi

  echo "Zipping '$item' -> '$zipname'"
  zip -r "$zipname" "$item"
done

echo "Done."


# ./make_inference_videos.sh

# python3 -u add_frame_numbers.py \
# ../saved_inference/deployment_videos
