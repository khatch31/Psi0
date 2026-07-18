


# suite=Articulated
# task=adjust_the_angle_of_a_phone_stand
# task=open_a_drawer_g1

# suite=Basic
# task=place_a_cube_in_a_bag
# task=put_dumpling_into_plate_g1
# task=stack_two_boxes
# task=stack_two_cubes_g1
# task=push_duck_g1

# suite=Tool_use
# task=erase_a_table_g1
# task=use_eraser_to_wipe_desk_g1

# suite=deformable
# task=fold_towel

# suite=Precision
# task=insert_flower_into_vase

echo "PSI_HOME: $PSI_HOME"
cd ..
pwd

# python3 -u scripts/data/raw_to_lerobot_he.py \
#   --data-root=$PSI_HOME/data/HE_RAW/$suite \
#   --work-dir=$PSI_HOME/data/real \
#   --repo-id=psi0-real-g1 \
#   --robot-type=g1 \
#   --task=$task

# task=put_dumpling_into_plate_g1
task=use_eraser_to_wipe_desk_g1

python3 -u scripts/data/raw_to_lerobot_he.py \
  --data-root=$PSI_HOME/data/HumanoidEveryday_raw \
  --work-dir=$PSI_HOME/data/real_10Hz \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task \
  --subsample 3

python3 -u scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real_10Hz \
  --task=$task

cp $PSI_HOME/data/real_10Hz/$task/meta/stats.json $PSI_HOME/data/real_10Hz/$task/meta/stats_psi0.json