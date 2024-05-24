export num_gpus=1
export output_dir="outputs/squad_ia3"
port=$(shuf -i25000-30000 -n1)
# CUDA_VISIBLE_DEVICES=3 python run_glue.py \
CUDA_VISIBLE_DEVICES="7" torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/run_qa.py \
--model_name_or_path /mounts/work/lyk/hierFT/roberta-base \
--dataset_name squad \
--do_train \
--do_eval \
--optim "adamw_hf" \
--deepspeed "dsconfig/zero0_config.json" \
--peft_type "ia3" \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 3e-5 \
--num_train_epochs 100 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.02 \
--seed 0 \
--fp16 \
--weight_decay 0 \
--load_best_model_at_end
