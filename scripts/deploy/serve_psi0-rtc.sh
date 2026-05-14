#!/bin/bash

source .venv-psi/bin/activate
which python

export CUDA_VISIBLE_DEVICES=0
echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"

CHECKPOINT_DIR=/home/songlin/Projects/psi0_kyle/psi0_workspace/training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254
CHECKPOINT_STEP=40000

# python src/psi/deploy/psi_serve_rtc-trainingtimertc.py \
python src/psi/deploy/psi_serve_rtc-trainingtimertc_zeros.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}

# bash scripts/deploy/serve_psi0-rtc.sh