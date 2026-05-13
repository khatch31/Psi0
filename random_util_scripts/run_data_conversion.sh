


### CLAUDE ### Fix data-root (was one level too deep) and define task variable so iter_tasks gets a non-empty filter list
# task=adjust_the_angle_of_a_phone_stand
task=open_a_drawer_g1
### END CLAUDE ###

echo "PSI_HOME: $PSI_HOME"
cd ..
pwd

python3 -u scripts/data/raw_to_lerobot_he.py \
  --data-root=$PSI_HOME/data/HE_RAW/Articulated \
  --work-dir=$PSI_HOME/data/real \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task

python3 -u scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real \
  --task=$task

cp $PSI_HOME/data/real/$task/meta/stats.json $PSI_HOME/data/real/$task/meta/stats_psi0.json