# Parler TTS

## requirements

1. Prepare dataset,

**We use remote server**,

```bash
cd /workspace
apt update
apt install ffmpeg zip screen -y
git clone https://github.com/malaysia-ai/async-parler-tts
cd async-parler-tts
pip3 install -e .
pip3 install wandb multiprocess accelerate==1.1.1 datasets evaluate transformers==4.47.0
cd training
wget https://huggingface.co/datasets/huseinzol05/mesolitica-tts-combined/resolve/main/tmp_dataset_audio.zip
wget https://huggingface.co/datasets/huseinzol05/mesolitica-tts-combined/resolve/main/audio_code_tmp.zip
unzip tmp_dataset_audio.zip
unzip audio_code_tmp.zip
```

- `tmp_dataset_audio.zip` and `audio_code_tmp.zip` already generated in local machine, it will auto generate during first time running the script and it will take a long time to generate. Basically it is a prepared dataset that already convert the audio to speech tokens.

## dataset

Dataset at https://huggingface.co/datasets/mesolitica/TTS, notebook preparation at [prepare-parlertts.ipynb](prepare-parlertts.ipynb).

## Finetune

### Tiny V1

```bash
bash parler-tiny.sh
```

### Mini V1

```bash
bash parler-mini.sh
```
