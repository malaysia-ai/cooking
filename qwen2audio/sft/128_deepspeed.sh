WANDB_PROJECT="lora-embedding-128-qwen2audio-7b" \
HF_HOME="/workspace/cache" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 3 \
-m train \
--deepspeed ds_config_zero3.json \
--model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 2 \
--output_dir lora-embedding-128-qwen2audio-7b \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file audio-qa-pretrained-text \
--logging_steps 1 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--block_size 4096 \
--save_steps 100 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 128 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 10 \
--dataloader_prefetch_factor 4