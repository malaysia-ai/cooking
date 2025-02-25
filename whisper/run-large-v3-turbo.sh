WANDB_PROJECT=malaysian-whisper-large-v3-turbo-v3 \
CUDA_VISIBLE_DEVICES="0,1" \
torchrun --nproc_per_node 2 \
-m whisper \
--model_name_or_path "openai/whisper-large-v3-turbo" \
--train_dataset_name "mosaic-stt" \
--eval_steps 1000 \
--save_steps 1000 \
--warmup_steps 100 \
--learning_rate 0.00005 \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 48 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 4 \
--output_dir "malaysian-whisper-large-v3-turbo-v3" \
--do_train \
--gradient_checkpointing \
--predict_with_generate \
--max_label_length 400 \
--bf16 \
--torch_compile