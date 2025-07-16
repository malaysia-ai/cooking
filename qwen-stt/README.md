# Qwen STT

Continue finetuning on https://huggingface.co/datasets/malaysia-ai/Malaysian-STT dataset, support both whole and streaming inference.

## How to

1. Download the audio,

```bash
huggingface-cli download --repo-type dataset \
--include '*.zip' \
--local-dir './' \
--max-workers 20 \
malaysia-ai/Malaysian-STT

wget https://gist.githubusercontent.com/huseinzol05/2e26de4f3b29d99e993b349864ab6c10/raw/9b2251f3ff958770215d70c8d82d311f82791b78/unzip.py
python3 unzip.py
```

2. Convert audio to speech tokens,

```bash
python3 convert_glm4.py --path '*segment/*.mp3' --replication 20
python3 convert_glm4.py --path '*whole/*.mp3' --replication 20
```