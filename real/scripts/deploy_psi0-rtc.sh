#!/bin/bash

PORT=8014
TASK="default/open_a_drawer_g1"
# TASK="the robot uses its right hand to grab the drawer kept on the middle of the desk and pull it out towards the robot."

cd "$(dirname "$0")/../teleop"

# python ../deploy/psi-inference_rtc.py \
python ../deploy/psi-inference_rtc_old.py \
    --port "$PORT" \
    --task "$TASK"

# bash ./real/scripts/deploy_psi0-rtc_old.sh
# bash ./real/scripts/deploy_psi0-rtc.sh