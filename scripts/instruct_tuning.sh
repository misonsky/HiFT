export num_gpus=2
export output_dir="outputs/instruct_tuning"
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES="0,2" torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/instruct_tuning.py \
    --model_type opt \
    --HiTaskType "CAUSAL_LM" \
    --optim "adamw_torch" \
    --deepspeed "dsconfig/zero0_config.json" \
    --model_name_or_path opt-7b  \
    --dataset_dir alpaca_data \
    --validation_split_percentage 0.01 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --seed 12345 \
    --fp16 \
    --tf32 true \
    --num_train_epochs 1 \
    --lr_scheduler_type "cosine" \
    --learning_rate 1e-5 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --preprocessing_num_workers 4 \
    --max_seq_length 512 \
    --output_dir $output_dir/model \
    --overwrite_output_dir \
    --logging_first_step True \
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end \
    --hier_tuning \
    --group_element $1 \
    --optimizer_strategy $2