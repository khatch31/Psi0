


SUITE=simple-eval

### CLAUDE ### Download all .zip files in the SUITE subfolder instead of a single task
# TASK_NAME=G1WholebodyTabletopGraspMP-v0
#
# hf download USC-PSI-Lab/psi-data \
#  $SUITE/$TASK_NAME.zip \
#  --local-dir=$PSI_HOME/data \
#  --repo-type=dataset
#
# cd ~/Desktop/world_models_project/psi0_workspace/data/$SUITE
# pwd
# unzip $TASK_NAME.zip
# rm $TASK_NAME.zip

hf download USC-PSI-Lab/psi-data \
 --include "$SUITE/*.zip" \
 --local-dir=$PSI_HOME/data \
 --repo-type=dataset

cd ~/Desktop/world_models_project/psi0_workspace/data/$SUITE
pwd

for zip_file in *.zip; do
    unzip "$zip_file"
    rm "$zip_file"
done
### END CLAUDE ###
