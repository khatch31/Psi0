


# suite=Articulated
# task=adjust_the_angle_of_a_phone_stand
# task=open_a_drawer_g1

suite=Basic
# task=place_a_cube_in_a_bag
# task=put_dumpling_into_plate_g1
# task=stack_two_boxes
task=stack_two_cubes_g1

echo "PSI_HOME: $PSI_HOME"
cd ..
pwd

python3 -u scripts/data/raw_to_lerobot_he.py \
  --data-root=$PSI_HOME/data/HE_RAW/$suite \
  --work-dir=$PSI_HOME/data/real \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task

python3 -u scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real \
  --task=$task

cp $PSI_HOME/data/real/$task/meta/stats.json $PSI_HOME/data/real/$task/meta/stats_psi0.json