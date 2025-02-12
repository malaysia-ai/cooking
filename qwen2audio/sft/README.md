# Qwen2Audio SFT

## requirements

1. Install libraries,

```bash
apt update
apt install ninja-build vim -y
pip3 install torch==2.5.1 torchaudio==2.5.1 deepspeed==0.15.4 mosaicml-streaming
pip3 install datasets evaluate peft librosa soundfile
pip3 install git+https://github.com/malaysia-ai/qwen2audio-multipack
pip3 install git+https://github.com/malaysia-ai/ml-cross-entropy-lora-lm-head
```

Optional,

```bash
pip3 install notebook==6.5.6
apt install screen -y
screen -dmS jupyter_session bash -c "jupyter notebook --NotebookApp.token='' --no-browser --allow-root --notebook-dir='/workspace'"
```

## dataset preparation

1. We use multipacking 4k context length with proper multi-documents masking, [packing-text-instructions.ipynb](packing-text-instructions.ipynb).
2. We combined multipacking 4k context length with audio instruction dataset, [packing-audio-instructions.ipynb](packing-audio-instructions.ipynb).

## SFT

### LoRA

1. Finetune,

```bash
cd /workspace
HF_HOME="/workspace/cache" huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
bash 128.sh
```