WANDB_PROJECT="malaysian-Qwen2.5-1.5B-Instruct" \
CUDA_VISIBLE_DEVICES="0,1" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 2 \
-m train \
--model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
--per_device_train_batch_size 3 \
--gradient_accumulation_steps 4 \
--output_dir malaysian-Qwen2.5-1.5B-Instruct \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file packing-4k \
--logging_steps 1 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--block_size 24576 \
--save_steps 100 \
--save_total_limit 3 \
--gradient_checkpointing true \
--ddp_find_unused_parameters false \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--torch_compile