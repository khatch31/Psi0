#!/bin/bash

PORT=8014
# TASK="default/open_a_drawer_g1"
# TASK="the robot uses its right hand to grab the drawer kept on the middle of the desk and pull it out towards the robot."
# TASK="Use the left hand to pick up the pink round toy on the left and place it inside the orange plate on the desk."
# TASK="The robot uses its right hand to push the yellow duck toy to the center of the white desk."
TASK="The robot uses its right hand to push the yellow duck toy to the center of the white desk."
# TASK="the robot uses its right hand to pick up the grey box kept on the middle on the desk and place it on top of another box next to it in a right angle"
# TASK="the robot uses its right hand to pick up the cube kept on the middle of the desk and put it inside the bag next to it"

cd "$(dirname "$0")/../teleop"

# python ../deploy/psi-inference_rtc.py \
python ../deploy/psi-inference_rtc_old.py \
    --port "$PORT" \
    --task "$TASK"

# bash ./real/scripts/deploy_psi0-rtc_old.sh
# bash ./real/scripts/deploy_psi0-rtc.sh