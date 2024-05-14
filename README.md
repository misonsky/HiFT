# HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy

This repo contains the source code of the Python package `HiFT` and several examples of how to integrate it with PyTorch models, such as those in Hugging Face. We only support PyTorch for now. See [our paper](https://arxiv.org/abs/2401.15207) for a detailed description of ·`HiFT`. `HiFT` supports FPFT of 7B models for 24G GPU memory devices under mixed precision without using any memory saving techniques and various optimizers including `AdamW`, `AdaGrad`, `SGD`, etc. 

**HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy** <br>
*Yongkang Liu, Yiqun Zhang, Qian Li, Tong Liu, Shi Feng, Daling Wang, Yifei Zhang, Hinrich Schütze* <br>
Paper: https://arxiv.org/abs/2401.15207 <br>

## News  

*26/1/2024*: Publish the first version of `HiFT` manuscript  
*25/2/2024*: Publish the second version of `HiFT` manuscript and source code  
*1/5/2024*:  Updated HiFT support for `LoRA`     
*10/5/2024*:  Adapt the optimizer provided by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)  
13/5/2024*:  Adapt `Adalora`,`LoRA`, `IA3`, `P_tuning`, `Prefix_tuning` , `Prompt_tuning` [peft](https://github.com/huggingface/peft) method.​  


## Repository Overview  

There are several directories in this repo:
* [hift/](hift) contains the source code for the package `hift`, which needs to be installed to run the examples we provide;
* [examples](examples) Contains `HiFT`-based `NER`, `QA`, `classification`, `text generation`,`instruction fine-tuning`, and `pre-training` example implementation. 
* [scripts](scripts) contains the script for running examples we provide.  
* [dsconfig](dsconfig) contains configuration files required for mixed precision.  
* [data](data) contains examples for instruction fine-tuning and pre-training. 

### Out-of-memory issues

Instruction fine-tuning 7B model on A6000 (48G), and the experimental results show that the maximum sequence length supported by HiFT is 2800.  Beyond this limit, `OOM` issues may occur.

| Model  | Seq Length | Max Batch Size |
| ------ | ---------- | -------------- |
| Alpaca | 512        | 8              |
| Vicuna | 2800       | 1              |



## Quickstart  

1. **Installing `hift` is simply**
 ```bash
 pip install hift
 ```

2. Import `hift` package  

```
### generation task  

from hift import HiFTSeq2SeqTrainer,GetCallBack,peft_function,Seq2SeqTrainer

### classification taks  

from hift import HiFTrainer,GetCallBack,PEFTrainer,peft_function


### QA task  

from hift import HiFTQuestionAnsweringTrainer,GetCallBack,QuestionAnsweringTrainer,peft_function
```

3. **Add `HiFT` configuration**
```
@dataclass
class HiFTArguments(ModelArguments):
    HiTaskType: str = field(
        default="SEQ_CLS",
        metadata={"help": ("HiTaskType should be consistent with PEFT TaskType" )},
    )
    peft_type: str = field(
        default=None,
        metadata={"help": ("peft_type should be in [lora,adalora,ia3,p_tuning,prefix_tuning,prompt_tuning]" )},
    )
    init_text:str = field(
        default="Predict if sentiment of this review is positive, negative or neutral",
        metadata={
            "help": (
                "the init prompt text for prompt tuning"
            )
        },
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": ("rank for lora or adalora" )},
    )
    peft_path : Optional[str] = field(default=None)
    virtual_tokens:int = field(
        default=20,
        metadata={"help": ("the number of virtual tokens for p_tuning, prefix_tuning and prefix_tuning" )},
    )
    group_element: int = field(
        default=1,
        metadata={"help": ("number element for each group parameters" )},
    )
    optimizer_strategy: str = field(
        default="down2up",
        metadata={"help": ("optimizer strategy of ['down2up','down2up','random']" )},
    )
    hier_tuning: bool = field(
        default=False,
        metadata={
            "help": (
                "hierarchical optimization for LLMS"
            )
        },
    )
    freeze_layers: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Index of the frozen layer"
            )
        },
    )
```



**HiTaskType** should be consistent with `PEFT` **TaskType**.  

   > **sequence classification**, **multiple choice tasks**: `TaskType.SEQ_CLS`    
   >
   > **question answering** task: `TaskType.QUESTION_ANS`  
   >
   > **sequence labeling** task: `TaskType.TOKEN_CLS`  
   >
   > **generation** task: `TaskType.CAUSAL_LM`   

**group_element**: the number of layers included in a block. Default value is `1`. 

**freeze_layers**: Layers you want to freeze during fine-tuning. You should provide the index of the corresponding layer. The **index** of the embedding layer is **0**, the index of the first layer is **1**,... 




4. **Using `HiFT` Trainer**  

`HiFT` inherits the trainer of huggingface, so you can directly use the trainer provided by hift to replace the original trainer.  

- **Classification Task**

```

if model_args.hier_tuning:#hier_tuning
        trainer = HiFTrainer(
            hiFThandler = GetCallBack(model_args.model_name_or_path),
            HiTaskType = model_args.HiTaskType,
            group_element = model_args.group_element,
            strategy = model_args.optimizer_strategy,
            hier_tuning= model_args.hier_tuning,
            peft_type = model_args.peft_type,
            freeze_layers = model_args.freeze_layers,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
  else:
        trainer = PEFTrainer(
            peft_type = model_args.peft_type,
            args=training_args,
            model=model,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
```

**QA Task**

```
if model_args.hier_tuning:
        trainer = HiFTQuestionAnsweringTrainer(
            hiFThandler = GetCallBack(model_args.model_name_or_path),
            HiTaskType = model_args.HiTaskType,
            group_element = model_args.group_element,
            strategy = model_args.optimizer_strategy,
            hier_tuning= model_args.hier_tuning,
            peft_type = model_args.peft_type,
            freeze_layers = model_args.freeze_layers,
            eval_examples=eval_examples if training_args.do_eval else None,
            post_process_function=post_processing_function,
            args=training_args,
            model=model,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics)
 else:
        trainer = QuestionAnsweringTrainer(
            peft_type = model_args.peft_type,
            eval_examples=eval_examples if training_args.do_eval else None,
            post_process_function=post_processing_function,
            args=training_args,
            model=model,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics)
```

- **Generation Task**  

  ```
  if model_args.hier_tuning:#hier_tuning
          trainer = HiFTSeq2SeqTrainer(
              hiFThandler = GetCallBack(model_args.model_name_or_path),
              HiTaskType = model_args.HiTaskType,
              group_element = model_args.group_element,
              strategy = model_args.optimizer_strategy,
              hier_tuning= model_args.hier_tuning,
              peft_type = model_args.peft_type,
              freeze_layers = model_args.freeze_layers,
              args=training_args,
              model=model,
              train_dataset=train_dataset if training_args.do_train else None,
              eval_dataset=eval_dataset if training_args.do_eval else None,
              compute_metrics=compute_metrics if training_args.predict_with_generate else None,
              tokenizer=tokenizer,
              data_collator=data_collator
          )
   else:
          trainer = Seq2SeqTrainer(
              peft_type = model_args.peft_type,
              args=training_args,
              model=model,
              train_dataset=train_dataset if training_args.do_train else None,
              eval_dataset=eval_dataset if training_args.do_eval else None,
              tokenizer=tokenizer,
              data_collator=data_collator,
              compute_metrics=compute_metrics if training_args.predict_with_generate else None,
          )
  ```

  

## Adapt Model to HiFT

`HiFT` supports any model. It is very easy to adapt to `HiFT`. 

> 1. Define the task types supported by your model in `TaskTInterface`.
> 2. Provides `regular expressions` for the `embedding layer` and different task `header layers`. The purpose of the regular expression is to uniquely identify the layer name of the corresponding layer.
> 3. Provide regular expressions except embedding layer and header layer in `others_pattern` interface. 

The simplest way is to provide the layer names for all layers in `others_pattern` interface, and the other interfaces return an empty list `[]`.  Below is the RoBerta's example.

```
class RobertaCallBack(HiFTCallBack):
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
        self.check_task_type(taskType,"RoBERTa",self.TaskTInterface)
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf'\.embedding\.']
        else:
            return [rf'\.embeddings\.']
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["classifier"]
    @property
    def token_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["classifier"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["qa_outputs"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']
        

```

### Instruction fine-tuning -- Vicuna

![vicuna](figure\vicuna.png)

```
### The parameters have not been fine-tuned, this is just a demo. Please adjust the parameters based on your data.

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
```

### Instruction fine-tuning -- Alpaca

![Alpaca](figure\alpaca.png)



```
### The parameters have not been fine-tuned, this is just a demo. Please adjust the parameters based on your data.

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
```



### Pre-Training  

![pretrain](figure\pretrain.png)



```
### This is just a demo. Please adjust the parameters based on your data.

export num_gpus=8
export output_dir="outputs/pretrain_tuning"
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0 torchrun --master_port "$port" examples/pretrain_tuning.py \
    --model_type llama \
    --HiTaskType "CAUSAL_LM" \
    --deepspeed "dsconfig/zero0_config.json" \
    --model_name_or_path llama2-7b \
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
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
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
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --hier_tuning \
    --group_element $1 \
    --optimizer_strategy $2
```

### PEFT-Tuning  

```

export num_gpus=8
export output_dir="outputs/e2e_opt"
port=$(shuf -i25000-30000 -n1)
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
CUDA_VISIBLE_DEVICES=7 torchrun --master_port "$port" examples/run_generation.py \
--model_name_or_path llama2-7b \
--model_type llama \
--HiTaskType "CAUSAL_LM" \
--peft_type "lora" \
--dataset_name e2e_nlg \
--do_train \
--do_eval \
--padding_side "left" \
--group_by_length \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 8 \
--save_strategy epoch \
--evaluation_strategy epoch \
--predict_with_generate \
--learning_rate 5e-5 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--max_eval_samples 2000 \
--model_max_length 512 \
--num_train_epochs 5 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.0  \
--num_beams 10 \
--seed 0 \
--fp16 \
--weight_decay 0.0 \
--load_best_model_at_end \
--weight_decay 0

```

### HIFT + PEFT

```

export num_gpus=8
export output_dir="outputs/e2e_opt"
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/run_generation.py \
--model_name_or_path /mounts/work/lyk/hierFT/llama2-7b \
--model_type llama \
--HiTaskType "CAUSAL_LM" \
--peft_type "lora" \
--dataset_name e2e_nlg \
--do_train \
--do_eval \
--deepspeed "dsconfig/zero0_config.json" \
--padding_side "left" \
--group_by_length \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--save_strategy epoch \
--evaluation_strategy epoch \
--predict_with_generate \
--learning_rate 5e-5 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--max_eval_samples 2000 \
--model_max_length 512 \
--num_train_epochs 5 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.0  \
--num_beams 10 \
--seed 0 \
--fp16 \
--weight_decay 0.0 \
--load_best_model_at_end \
--hier_tuning \
--weight_decay 0 \
--group_element $1 \
--optimizer_strategy $2

```




## Introduction  

![Algorithm](figure\alg.png)  

The detailed training process is shown in Algorithm. The first step is to determine the update strategy. Then freeze all layers. The layers to be updated, denoted by $E$, are selected from the queue $Q$ based on the parameter $m$. The selected layer $E$ is removed from head of the queue $Q$ and added to the tail of $Q$ to wait for the next update. Select the parameter $\theta_s$ that needs to be updated from $M$ based on $E$, set the parameter $\theta_s$ to a computable gradient state and set the update parameter group of optimizer $P$ to $\theta_s$. Before parameter updates, the states parameters  of optimizer $P$ related to $\theta_s$ could be moved to GPU devices. After the completion of weight updates, the corresponding gradients are clean up and optimizer states parameters are moved to CPU. When all layers have been updated once, adjust the learning rate once.    

`HiFT`  iteratively updates a subset of parameters at each training step, and it will modify the full parameter after multiple steps. This vastly reduces the GPU memory requirements for fine-tuninglarge language models enables efficient task-switching during deployment all without introducing inference latency. HiFT also outperforms several other adaptation methods including adapter, prefix-tuning, and fine-tuning.

`HiFT` is a model-independent and optimizer-independent full-parameter fine-tuning method that can be integrated with the PEFT method. 

*optimizers*: The latest version of `HiFT` is adapted to the `Adam`, `AdamW`, `SGD`, `Adafactor` and `Adagrad` optimizers.     

*Model*: The latest version of `HiFT` supports `BERT`, `RoBERTa`, `GPT-2`, `GPTNeo`,`GPT-NeoX`,`OPT` and `LLaMA-based` models.  


**Experiments** on **OPT-13B** (with 1000 examples). **ICL**: in-context learning; **LP**: linear probing; **FPFT**: full fine-tuning; Prefix: prefix-tuning. All experiments use prompts from MeZO.  

![OPT-13b](figure\opt13.png)



GPU memory usage of fine-tuning **LLaMA (7B)** on the **E2E** dataset.  **Total** represents the total memory used during fine-tuning. **Mixed** represents fine-tuning with **standard mixed precision** and **Mixed^Hi^** represents the mixed precision adapted to `HiFT`. **Para** represents the memory occupied by the model **parameters**; **Gra** represents the memory occupied by the gradient;  **Sta** represents the memory occupied by the **optimizer state**. **PGS** represents the sum of memory occupied by **parameters** , **gradients** and **optimizer state** .  

<img src="figure\llama.png" alt="llama-memory" style="zoom:150%;" />  

## Mixed Precision  

[Source Code](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/fp16)

```
class FP16_Optimizer(DeepSpeedOptimizer):
    def __init__(self,
       init_optimizer,
       deepspeed=None,
       static_loss_scale=1.0,
       dynamic_loss_scale=False,
       initial_dynamic_scale=2**32,
       dynamic_loss_args=None,
       verbose=True,
       mpu=None,
       clip_grad=0.0,
       fused_adam_legacy=False,
       has_moe_layers=False,
       timers=None):
                 
       ....
       self.fp16_groups = []
       self.fp16_groups_flat = []
       self.fp32_groups_flat = []
                 
       ...
                 
       for i, param_group in enumerate(self.optimizer.param_groups):
           ...
           self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
           ...
                            
```


The memory required to load **1B** parameters is **3.72GB** (10^9 $\times$ 4 /1024/1024/1024). Standard mixed precision stores both **single-precision** and **half-precision** model parameters. Assuming you are using standard mixed precision fine-tuning of the **7B** model, compared with **single-precision** fine-tuning, **mixed precision** requires an additional about **13G** GPU memory overhead to store half-precision model parameters. Only when the dynamic GPU memory reduction reaches 13GB does mixed precision demonstrate its advantages.  This requires using large batch size. 

We reimplement the mixed-precision algorithm to adapt to `HiFT`'s fine-tuning algorithm, which ensures that single-precision model parameters do not incur additional GPU memory overhead.  



## Citation
```BibTeX
@article{liu2024hift,
  title={HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy},
  author={Liu, Yongkang and Zhang, Yiqun and Li, Qian and Feng, Shi and Wang, Daling and Zhang, Yifei and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2401.15207},
  year={2024}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution.
