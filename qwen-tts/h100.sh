
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install transformers==4.55.0 datasets accelerate
pip3 install mosaicml-streaming
pip3 install wandb
pip3 install liger-kernel==0.6.2
wget https://huggingface.co/datasets/mesolitica/Flash-Attention3-whl/resolve/main/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64-2.7.1-12.8.whl -O flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
pip3 install flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
pip3 install pip -U
pip3 install git+https://github.com/apple/ml-cross-entropy
