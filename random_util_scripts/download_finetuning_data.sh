export task=Pick_bottle_and_turn_and_pour_into_cup

hf download USC-PSI-Lab/psi-data \
  real/$task.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

# unzip $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task.zip -d $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task
unzip $PSI_HOME/data/real/$task.zip -d $PSI_HOME/data/real/$task

python scripts/data/patch_lerobot_meta.py $PSI_HOME/data/real/$task


