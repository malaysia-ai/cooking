#!/bin/bash

apt update
apt install unzip ffmpeg -y
apt update && apt install -y locales
locale-gen en_US.UTF-8
cd /workspace
pip3 install huggingface-hub wandb multiprocess

cmd1="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'dialects-processed-*.zip', local_dir = './', max_workers = 20)
\"
"

cmd2="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='malaysia-ai/STT-Whisper', repo_type='dataset', allow_patterns = 'speech-instructions-*.zip', local_dir = './', max_workers = 20)
\"
"

cmd3="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
wget https://huggingface.co/datasets/malaysia-ai/STT-Whisper/resolve/main/mallm-v2.zip
wget https://huggingface.co/datasets/malaysia-ai/STT-Whisper/resolve/main/tatabahasa-v2.zip
"

bash -c "$cmd1" &
pid1=$!

bash -c "$cmd2" &
pid2=$!

bash -c "$cmd3" &
pid3=$!

wait $pid1 $pid2 $pid3