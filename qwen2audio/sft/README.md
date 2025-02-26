# Qwen2Audio SFT

## requirements

1. Install libraries,

```bash
apt update
apt install ninja-build vim -y
pip3 install torch==2.6.0 torchaudio==2.6.0 deepspeed==0.15.4 mosaicml-streaming
pip3 install datasets evaluate peft librosa soundfile
pip3 install git+https://github.com/malaysia-ai/qwen2audio-multipack
pip3 install git+https://github.com/malaysia-ai/ml-cross-entropy-lora-lm-head
pip3 install git+https://github.com/malaysia-ai/accelerate-torch-compile-speechlm
```

Optional,

```bash
pip3 install notebook==6.5.6
apt install screen -y
screen -dmS jupyter_session bash -c "jupyter notebook --NotebookApp.token='' --no-browser --allow-root --notebook-dir='/workspace'"
```

## Dataset

1. Transcription instructions, https://huggingface.co/datasets/malaysia-ai/STT-Whisper
2. Speech instructions, https://huggingface.co/datasets/malaysia-ai/Speech-Instructions

### Download

```bash
bash download.sh
wget https://gist.githubusercontent.com/huseinzol05/2e26de4f3b29d99e993b349864ab6c10/raw/9b2251f3ff958770215d70c8d82d311f82791b78/unzip.py
python3 unzip.py
```

## Dataset preparation

1. We use multipacking 4k context length with proper multi-documents masking, [packing-text-instructions.ipynb](packing-text-instructions.ipynb).
2. We combined multipacking 4k context length with audio instruction dataset, [packing-audio-instructions.ipynb](packing-audio-instructions.ipynb).

### Download

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='huseinzol05/malaysian-audio-instructions', repo_type='dataset', local_dir = './malaysian-audio-instructions', max_workers = 20)
"
```

## SFT

### LoRA

1. Finetune,

```bash
cd /workspace
HF_HOME="/workspace/cache" huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
bash 128.sh
```