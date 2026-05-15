

source .env
source .venv-psi/bin/activate

echo "PSI_HOME: $PSI_HOME"

python3 -u scripts/deploy/dummy_client_psi0-rtc.py \
--run-dir $PSI_HOME/training_output/finetune/open_a_drawer_g1.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2605062254 \
--host localhost --port 8014

# bash scripts/deploy/launch_dummy_client.sh