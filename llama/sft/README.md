# Llama based SFT

## requirements

```bash
pip3 install -r requirements.txt
```

## dataset preparation

We use multipacking 3k context length with proper multi-documents masking, [packing.ipynb](packing.ipynb).

By using LoRA CCE loss, we can fit 256 LoRA on almost linear layers including embedding layer without need to use offloading.

## SFT

### LoRA

#### Llama 3.2 1B Instruct

1. Finetune,

```bash
bash 3.1-1b.sh
```

2. Merge back [merge-1b.ipynb](merge-1b.ipynb).

#### Llama 3.2 3B Instruct

1. Finetune,

```bash
bash 3.1-3b.sh
```