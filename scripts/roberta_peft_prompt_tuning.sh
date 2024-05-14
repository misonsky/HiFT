export num_gpus=1
export output_dir="outputs/cola"
port=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES="6" torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/run_glue.py \
--model_name_or_path roberta-base \
--task_name cola \
--do_train \
--do_eval \
--do_predict \
--optim "adamw_hf" \
--peft_type "prompt_tuning" \
--max_seq_length 492 \
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
--weight_decay 0 \
--load_best_model_at_end
