


EP_IDX=2
RUN_DIR="finetune/G1WholebodyTabletopGraspMP-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605262113"

python3 -u examples/simple/openloop_eval.py \
--num-inference-steps 10 \
--eps-idx $EP_IDX \
--zero_states \
--gpu 0 \
--run-dir "$PSI_HOME/training_output/$RUN_DIR" \
--output-dir "$PSI_HOME/open_loop_eval/$RUN_DIR" 

# bash examples/simple/run_open_loop_eval.sh