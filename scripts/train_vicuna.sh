export num_gpus=2
export output_dir="outputs/output_vicuna"
port=$(shuf -i25000-30000 -n1)
#--fsdp "full_shard auto_wrap" \
CUDA_VISIBLE_DEVICES="0,2" torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/vicuna_train.py \
    --model_type llama \
    --HiTaskType "CAUSAL_LM" \
    --optim "lion_32bit" \
    --deepspeed "dsconfig/zero0_config.json" \
    --model_name_or_path /mounts/work/lyk/hierFT/llama2-7b \
    --data_path data/dummy_conversation.json \
    --eval_data_path data/sharegpt_clean.json \
    --output_dir $output_dir/model \
    --num_train_epochs 3 \
    --do_train \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --model_max_length 2800 \
    --lazy_preprocess True \
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end \
    --hier_tuning \
    --group_element $1 \
    --optimizer_strategy $2

