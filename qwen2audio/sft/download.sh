#!/bin/bash

apt update
apt install unzip ffmpeg -y
apt update && apt install -y locales
locale-gen en_US.UTF-8
cd /workspace
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
tar -xf 7z2301-linux-x64.tar.xz
pip3 install huggingface-hub wandb multiprocess

cmd1="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
allow_patterns = ['mixtral-audio-instruction.zip', 'random-question-chunks.zip', 'sample-filter-gpt-omni-voiceassistant-400k-*.zip']
snapshot_download(repo_id='malaysia-ai/Speech-to-Speech', repo_type='dataset', 
                  allow_patterns = allow_patterns, local_dir = './')
\"
python3 -c \"
from huggingface_hub import snapshot_download
allow_patterns = ['mixtral-audio-instruction.zip', 'random-question-chunks.zip', 'sample-filter-gpt-omni-voiceassistant-400k-*.zip']
snapshot_download(repo_id='malaysia-ai/Speech-to-Speech', repo_type='dataset', 
                  allow_patterns = allow_patterns, local_dir = './')
\"
wget https://raw.githubusercontent.com/malaysia-ai/cooking/refs/heads/main/qwen2audio/sft/unzip.py
python3 unzip.py
"

cmd2="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='huseinzol05/malaysian-audio-qa-pretraining', repo_type='dataset', 
                  local_dir = './audio-qa-pretrained')
\"
"

bash -c "$cmd1" &
pid1=$!

bash -c "$cmd2" &
pid2=$!

wait $pid1 $pid2