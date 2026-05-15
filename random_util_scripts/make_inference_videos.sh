#!/bin/bash
# Create MP4 videos from saved inference images.
#
# Usage:
#   ./make_inference_videos.sh [session_dir]
#
# If session_dir is omitted, falls back to $PSI_HOME/saved_inference/<latest>.
# Outputs:
#   <session_dir>/deployment_video_30hz.mp4
#   <session_dir>/policy_video_10hz.mp4

# ./random_util_scripts/make_inference_videos.sh $PSI_HOME/saved_inference/2026-05-15_15-57-17

set -e

# ── resolve session dir ──────────────────────────────────────────────────────
if [ -n "$1" ]; then
    SESSION_DIR="$1"
else
    INFER_ROOT="${PSI_HOME:-$HOME}/saved_inference"
    SESSION_DIR=$(ls -td "$INFER_ROOT"/*/ 2>/dev/null | head -1)
    if [ -z "$SESSION_DIR" ]; then
        echo "ERROR: could not find a session directory under $INFER_ROOT"
        echo "Usage: $0 <session_dir>"
        exit 1
    fi
    SESSION_DIR="${SESSION_DIR%/}"  # strip trailing slash
fi

echo "Session: $SESSION_DIR"

DEPLOY_DIR="$SESSION_DIR/deployment_time_inference/images"
POLICY_DIR="$SESSION_DIR/policy_time_inference/images"

# ── deployment video: 30 Hz ──────────────────────────────────────────────────
if [ -d "$DEPLOY_DIR" ]; then
    echo ""
    echo "==> Building deployment video (30 Hz)..."
    TMP=$(mktemp /tmp/deploy_list_XXXX.txt)

    # Filenames: img_{N}_batch0_img0.png — sort numerically by N (field 2)
    ls "$DEPLOY_DIR" \
      | grep -E '^img_[0-9]+_batch0_img0\.png$' \
      | awk -F'_' '{printf "%d %s\n", $2, $0}' \
      | sort -n \
      | awk -v dir="$DEPLOY_DIR" '{print "file \047" dir "/" $2 "\047"; print "duration 0.033333"}' \
      > "$TMP"

    N=$(wc -l < "$TMP")
    echo "    frames: $N"

    ffmpeg -y -r 30 -f concat -safe 0 -i "$TMP" \
      -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
      "$SESSION_DIR/deployment_video_30hz.mp4"

    rm "$TMP"
    echo "    saved: $SESSION_DIR/deployment_video_30hz.mp4"
else
    echo "SKIP: $DEPLOY_DIR not found"
fi

# ── policy video: 10 Hz ──────────────────────────────────────────────────────
if [ -d "$POLICY_DIR" ]; then
    echo ""
    echo "==> Building policy video (10 Hz)..."
    TMP=$(mktemp /tmp/policy_list_XXXX.txt)

    # Filenames:
    #   img_{s}_{inference_counter}_batch0_img0.png  (NF=5, field 3 = counter)
    #   img_initial_{inference_counter}_batch0_img0.png  (NF=5, field 3 = counter)
    # Sort by field 3 ($(NF-2)) numerically.
    ls "$POLICY_DIR" \
      | grep -E '^img_.*_batch0_img0\.png$' \
      | awk -F'_' '{printf "%d %s\n", $(NF-2), $0}' \
      | sort -n \
      | awk -v dir="$POLICY_DIR" '{print "file \047" dir "/" $2 "\047"; print "duration 0.1"}' \
      > "$TMP"

    N=$(wc -l < "$TMP")
    echo "    frames: $N"

    ffmpeg -y -r 10 -f concat -safe 0 -i "$TMP" \
      -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
      "$SESSION_DIR/policy_video_10hz.mp4"

    rm "$TMP"
    echo "    saved: $SESSION_DIR/policy_video_10hz.mp4"
else
    echo "SKIP: $POLICY_DIR not found"
fi

echo ""
echo "Done."
