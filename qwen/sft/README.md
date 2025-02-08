# Qwen based SFT

## requirements

```bash
pip3 install -r requirements.txt
```

## dataset preparation

We use multipacking 3k context length with proper multi-documents masking, [packing.ipynb](packing.ipynb).

By using LoRA CCE loss, we can fit 256 LoRA on almost linear layers including embedding layer without need to use offloading.

## SFT

### LoRA

#### Qwen 2.5 1.5B Instruct

```bash
bash 2.5-1.5b.sh
```