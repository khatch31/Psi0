#!/bin/bash

source .venv-psi/bin/activate
which python

export CUDA_VISIBLE_DEVICES=0
echo "Serving on GPU(s): $CUDA_VISIBLE_DEVICES"

echo "PSI_HOME: $PSI_HOME"

# Control-loop rate. Must match the rate the checkpoints were trained at.
# 1/30 for 30Hz checkpoints, 1/10 for 10Hz checkpoints.
# CTRL_PERIOD_SEC=$(python -c "print(1./30)")   # 30Hz
CTRL_PERIOD_SEC=$(python -c "print(1./10)")     # 10Hz
echo "CTRL_PERIOD_SEC: $CTRL_PERIOD_SEC"

# Checkpoints offered in the Ctrl+C menu, as RUN_DIR:STEP.
# The FIRST entry is loaded onto the GPU at startup; the rest are parked in CPU RAM
# (~6-10GB each) and swapped in on demand.
#
# IMPORTANT: --ckpt takes SPACE-SEPARATED values on a SINGLE flag. Repeating the flag
# (--ckpt A --ckpt B) does NOT append -- only the last one survives and you would silently
# serve a single checkpoint. Add entries to the one --ckpt list below.
#
# NOTE: this server uses ONE global control rate for the whole session, so every checkpoint
# listed here must have been trained at the same rate. Do not mix a 10Hz and a 30Hz
# checkpoint in one launch -- one of them will be played back at the wrong speed.

CKPT_10HZ_ROOT=$PSI_HOME/training_output/10Hz/finetune
# CKPT_30HZ_ROOT=$PSI_HOME/training_output/30Hz/finetune

python src/psi/deploy/psi_serve_rtc-trainingtimertc_zeros_multickpt.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --ctrl_period_sec=${CTRL_PERIOD_SEC} \
    --ckpt \
        "${CKPT_10HZ_ROOT}/fold_towel.real.flow1000.cosine.lr1.0e-04.b128.gpus1.2607181739:38000"

# Add more checkpoints as further space-separated values of the SAME --ckpt flag, e.g.:
#     --ckpt \
#         "${CKPT_10HZ_ROOT}/fold_towel....:38000" \
#         "${CKPT_10HZ_ROOT}/<another_run_dir>:20000" \
#         "${CKPT_10HZ_ROOT}/<yet_another_run_dir>:30000"

# bash scripts/deploy/serve_psi0-rtc-multickpt.sh
# curl -X POST http://localhost:8014/reset
#
# Press Ctrl+C in the server terminal for the checkpoint menu.
# The control loop pauses while the menu is open and for the duration of a swap, so the
# robot holds its last commanded pose. A second Ctrl+C at the menu quits.
