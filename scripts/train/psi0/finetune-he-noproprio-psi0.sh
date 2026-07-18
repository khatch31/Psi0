#!/bin/bash

export OMP_NUM_THREADS=32
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=0,3,4
BATCH_SIZE=42

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "BATCH_SIZE: $BATCH_SIZE"

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw pick-toys"
    exit 1
fi

export task="$1"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${2:-$default_exp}
master_port=${3:-29500}


echo "Task: $task"
echo "default_exp: $default_exp"
echo "Experiment name: $exp"
echo "master_port: $master_port"

args="
finetune_real_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=${BATCH_SIZE} \
--train.max_checkpoints_to_keep=1_000 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=2_000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=real_10Hz \
--data.train_repo_ids=$task \
--data.transform.repack.pad-action-dim=36 \
--data.transform.repack.pad-state-dim=36 \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=36 \
--data.transform.field.pad-state-dim=36 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=$PSI_HOME/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=36 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=36 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8 \
--train.output_dir=$PSI_HOME/training_output/10Hz \
--model.zero-states \
--model.zero-last-8-actions \
--train.save_ckpts_to_s3=1 \
--train.s3_save_uri="s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/world_models_project/psi0_workspace/training_output/10Hz"
"
### CLAUDE ### To upload each checkpoint to S3 (via `aws s3 sync`) and then delete the local
###            copy to save disk, append --train.save_ckpts_to_s3=1 and --train.s3_save_uri=s3://...
###            Both are required together. Default is OFF (save_ckpts_to_s3=0 in TrainConfig).
### END CLAUDE ###
### CLAUDE ### To disable proprioception conditioning, append --model.zero-states to the args above.
###            Default is proprioception ON (zero_states=False in Psi0ModelConfig).
### END CLAUDE ###

### CLAUDE ### To zero out the last 8 action dims (rpy + height + torso_vx + torso_vy + torso_vyaw + target_yaw)
###            during training and L1 eval, append --model.zero-last-8-actions to the args above.
###            Default is OFF (zero_last_8_actions=False in Psi0ModelConfig).
### END CLAUDE ###

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$master_port scripts/train.py \
    ${args}

# scripts/train/psi0/finetune-he-noproprio-psi0.sh place_a_cube_in_a_bag place_a_cube_in_a_bag 29500
# scripts/train/psi0/finetune-he-noproprio-psi0.sh put_dumpling_into_plate_g1 put_dumpling_into_plate_g1 29501
# scripts/train/psi0/finetune-he-noproprio-psi0.sh stack_two_boxes stack_two_boxes 29502
# scripts/train/psi0/finetune-he-noproprio-psi0.sh stack_two_cubes_g1 stack_two_cubes_g1 29503
# scripts/train/psi0/finetune-he-noproprio-psi0.sh open_a_drawer_g1 open_a_drawer_g1_second 29504
# scripts/train/psi0/finetune-he-noproprio-psi0.sh push_duck_g1 push_duck_g1 
# scripts/train/psi0/finetune-he-noproprio-psi0.sh erase_a_table_g1 erase_a_table_g1 29501
# scripts/train/psi0/finetune-he-noproprio-psi0.sh use_eraser_to_wipe_desk_g1 use_eraser_to_wipe_desk_g1 29500
# scripts/train/psi0/finetune-he-noproprio-psi0.sh fold_towel fold_towel 29501
# scripts/train/psi0/finetune-he-noproprio-psi0.sh insert_flower_into_vase insert_flower_into_vase 29502



# --data.root_dir=$PSI_HOME/data/real_10Hz \