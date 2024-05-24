export num_gpus=8
export output_dir="outputs/pretrain_tuning"
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0 torchrun --master_port "$port" examples/pretrain_tuning.py \
    --model_type llama \
    --HiTaskType "CAUSAL_LM" \
    --deepspeed "dsconfig/zero0_config.json" \
    --model_name_or_path /mounts/work/lyk/hierFT/llama2-7b \
    --dataset_dir "data" \
    --data_cache_dir "data_cache_dir" \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --seed 12345 \
    --fp16 \
    --max_steps 1000 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir $output_dir/model \
    --overwrite_output_dir \
    --logging_first_step True \
    --lora_rank 8 \
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --hier_tuning \
    --group_element $1 \
    --optimizer_strategy $2