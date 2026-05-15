

cd ..

pwd 



python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='USC-PSI-Lab/psi-model',
    allow_patterns='psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/**',
    local_dir='$PSI_HOME/cache/checkpoints'
)
# "

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='USC-PSI-Lab/psi-model',
    allow_patterns='psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/**',
    local_dir='$PSI_HOME/cache/checkpoints'
)
"