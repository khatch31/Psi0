#!/bin/bash

source .venv-psi/bin/activate
which python



PORT=${1:-8014}
ACTION_EXEC_HORIZON=${2:-24}
CUDA_DEVICE=${3:-0}
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
echo "PORT: $PORT, ACTION_EXEC_HORIZON: $ACTION_EXEC_HORIZON, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Echo $nprocs GPUs"

echo "PSI_HOME: $PSI_HOME"

# # CHECKPOINT_DIR=/home/ubuntu/Desktop/world_models_project/psi0_workspace/training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254
# CHECKPOINT_DIR=$PSI_HOME/training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254
# # CHECKPOINT_DIR=$PSI_HOME/training_output/finetune/open_a_drawer_g1_2.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2606020423
# # CHECKPOINT_STEP=40000
# # CHECKPOINT_STEP=10000
# # CHECKPOINT_STEP=20000
# CHECKPOINT_STEP=30000

CHECKPOINT_DIR=$PSI_HOME/training_output/finetune/put_dumpling_into_plate_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2606171731
CHECKPOINT_STEP=40000

# CHECKPOINT_DIR=$PSI_HOME/training_output/finetune/push_duck_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2606181756
# CHECKPOINT_STEP=10000

# CHECKPOINT_DIR=$PSI_HOME/training_output/finetune/stack_two_boxes.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2606250328
# CHECKPOINT_STEP=30000

# CHECKPOINT_DIR=$PSI_HOME/training_output/finetune/place_a_cube_in_a_bag.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2606170034
# CHECKPOINT_STEP=20000

echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "CHECKPOINT_STEP: $CHECKPOINT_STEP"



python src/psi/deploy/psi0_serve_simple_zeros.py \
    --host 0.0.0.0 \
    --port ${PORT} \
    --action_exec_horizon ${ACTION_EXEC_HORIZON} \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}

# bash scripts/deploy/serve_psi0-simple-zeros.sh 8014 24 0
# bash scripts/deploy/serve_psi0-simple-zeros.sh 8015 24 3
# bash scripts/deploy/serve_psi0-simple-zeros.sh 8016 24 4
# bash scripts/deploy/serve_psi0-simple-zeros.sh 8017 24 7
# curl -X POST http://localhost:8014/reset