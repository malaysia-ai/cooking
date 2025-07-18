WANDB_PROJECT="Qwen-Qwen2.5-1.5B-STT-10k" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
torchrun --nproc_per_node 1 \
-m qwen2_multipacking \
--model_name_or_path "Qwen/Qwen2.5-1.5B" \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 4 \
--output_dir Qwen-Qwen2.5-1.5B-STT-10k \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file packing-qwen2 \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 200 \
--block_size 10240 \
--save_steps 100 \
--save_total_limit 5 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5