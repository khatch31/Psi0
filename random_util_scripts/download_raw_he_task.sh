


# TASK_NAME=fold_towel
# TASK_URL="https://www.dropbox.com/scl/fi/vwh85kp72jmxl691dmxvs/fold_towel.zip?rlkey=aicqzcudidqymksm32y5eptll&st=up122jap&dl=1" # Make sure you make it end in 1 not 0

cd /data/kylehatch/world_models_project/psi0_workspace/data/HumanoidEveryday_raw
pwd

# curl -L "${TASK_URL}" -o ${TASK_NAME}.zip
# unzip ${TASK_NAME}.zip

du -sh .
rm *.zip

du -sh .
echo "rm -r */*/depth"
rm -r */*/depth
du -sh .
echo "rm -r */*/lidar"
rm -r */*/lidar
du -sh .

# aws s3 sync . s3://tri-ml-sandbox-16011-us-west-2-datasets/kylehatch/world_models_project/psi0_workspace/data/HumanoidEveryday_raw