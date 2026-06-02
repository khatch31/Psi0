

export dr=level-0

# # set entrypoint and agent to eval_decoupled_wbc.py and psi0_decoupled_wbc if the evaluating task ends with Teleop, which
# # means the task data is collected using teleoperation:
# export entry=eval_decoupled_wbc.py
# export agent=psi0_decoupled_wbc

# set entrypoint and agent to eval.py and psi0 if the evaluating task ends with MP, which means the task data is generated using CuRobo Motion planning:
export entry=eval.py
export agent=psi0

export OMNI_KIT_ACCEPT_EULA=yes

task=G1WholebodyTabletopGraspMP-v0

echo "dr: $dr"
echo "entry: $entry"
echo "agent: $agent"
echo "task: $task"

# cd third_party/SIMPLE
echo "pwd:" && pwd

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

echo "timestamp: $timestamp"

source $PSI_HOME/SIMPLE/.venv/bin/activate
which python3
python3 --version

export MUJOCO_GL=egl

echo "MUJOCO_GL: $MUJOCO_GL"

# python3 -u src/simple/cli/$entry \
# python3 -u third_party/SIMPLE/src/simple/cli/$entry \
python3 -u $PSI_HOME/SIMPLE/src/simple/cli/$entry \
	simple/$task \
	$agent \
	$dr \
	--host=localhost \
	--port=22085 \
	--sim-mode=mujoco_isaac \
    --headless \
	--data-format=lerobot \
	--data-dir=$PSI_HOME/data/evals/simple-eval-scratch/$task/$dr \
    --eval-dir=$PSI_HOME/SIMPLE_saved_inference/$timestamp

	# --data-dir=$PSI_HOME/data/evals/simple-eval/$task/$dr \

# bash random_util_scripts/deploy_psi_simple.sh