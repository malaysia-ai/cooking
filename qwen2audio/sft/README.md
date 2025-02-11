# Qwen2Audio SFT

## requirements

1. Install libraries,

```bash
apt update
apt install ninja-build vim -y
pip3 install torch==2.5.1 torchaudio==2.5.1 deepspeed==0.15.4
pip3 install datasets evaluate peft librosa soundfile
pip3 install git+https://github.com/malaysia-ai/qwen2audio-multipack
pip3 install git+https://github.com/malaysia-ai/ml-cross-entropy-lora-lm-head
```

## dataset preparation

1. We use multipacking 4k context length with proper multi-documents masking.
2. We combined multipacking 4k context length with audio instruction dataset, [packing.ipynb](packing.ipynb).

## SFT

### LoRA

1. Finetune,

```bash
wget https://raw.githubusercontent.com/mesolitica/malaya/refs/heads/master/session/llama3/ds_config_zero3.json
bash 128_deepspeed.sh
```