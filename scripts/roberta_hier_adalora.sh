export num_gpus=1
export output_dir="outputs/cola"
CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=$num_gpus examples/run_glue.py \
--model_name_or_path roberta-base \
--task_name cola \
--do_train \
--do_eval \
--do_predict \
--optim "adamw_hf" \
--peft_type "adalora" \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 5e-4 \
--num_train_epochs 200 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--hier_tuning \
--seed 0 \
--weight_decay 0.4 \
--group_element $1 \
--optimizer_strategy $2 \
--load_best_model_at_end
