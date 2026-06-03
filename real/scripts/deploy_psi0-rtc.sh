#!/bin/bash

PORT=8014
# TASK="default/open_a_drawer_g1"
TASK="the robot uses its right hand to grab the drawer kept on the middle of the desk and pull it out towards the robot."

cd "$(dirname "$0")/../teleop"

python ../deploy/psi-inference_rtc.py \
    --port "$PORT" \
    --task "$TASK"

# bash ./real/scripts/deploy_psi0-rtc.sh