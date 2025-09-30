pip3 install transformers==4.55.0 datasets accelerate
pip3 install mosaicml-streaming
pip3 install torchaudio==2.7.0
pip3 install tf_keras==2.16.0 --no-deps
pip3 install wandb
pip3 install liger-kernel==0.6.2
wget https://huggingface.co/datasets/malaysia-ai/Flash-Attention3-wheel/resolve/main/flash_attn_3-3.0.0b1-cp39-abi3-linux_aarch64-2.7.0-12.8.whl -O flash_attn_3-3.0.0b1-cp39-abi3-linux_aarch64.whl
pip3 install flash_attn_3-3.0.0b1-cp39-abi3-linux_aarch64.whl
pip3 install pip -U
pip3 install git+https://github.com/apple/ml-cross-entropy
