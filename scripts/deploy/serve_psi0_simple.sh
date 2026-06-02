#!/bin/bash

set -e

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Serving on GPU $CUDA_VISIBLE_DEVICES"

# Accept RUN_DIR and CKPT_STEP as command line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RUN_DIR CKPT_STEP"
    exit 1
fi

RUN_DIR=$1
CKPT_STEP=$2

uv run --active --group psi --group serve serve_psi0 \
    --host 0.0.0.0 \
    --port 22085 \
    --policy=psi0 \
    --run-dir=$RUN_DIR \
    --ckpt-step=$CKPT_STEP \
    --action-exec-horizon=24 \
    --rtc

    # --port 22085 \

# bash scripts/deploy/serve_psi0_simple.sh \
# "$PSI_HOME/training_output/finetune/G1WholebodyTabletopGraspMP-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605262113" \
# 40000