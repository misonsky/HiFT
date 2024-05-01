# HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy

This repo contains the source code of the Python package `HiFT` and several examples of how to integrate it with PyTorch models, such as those in Hugging Face. We only support PyTorch for now. See [our paper](xxxx) for a detailed description of ·`HiFT`. `HiFT` supports FPFT of **7B** models for **24G** GPU memory devices under mixed precision without using any memory saving techniques and various optimizers including `AdamW`, `AdaGrad`, `SGD`, etc. 

**HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy** <br>
*Yongkang Liu, Yiqun Zhang, Qian Li, Tong Liu, Shi Feng, Daling Wang, Yifei Zhang, Hinrich Schütze* <br>
Paper: https://arxiv.org/abs/2401.15207 <br>

## News  

*26/1/2024*: Publish the first version of `HiFT` manuscript  
*25/2/2024*: Publish the second version of `HiFT` manuscript and source code  
*1/5/2024*:  Updated HiFT support for `LoRA`     

## Repository Overview  

There are several directories in this repo:
* [hift/](hift) contains the source code for the package `hift`, which needs to be installed to run the examples we provide;
* [examples](examples) contains an example implementation of `HiFT` in **BERT**, **RoBERTa**, **GPT-2**, **GPT-Neo**,**GPT-NeoX**,**OPT** and **LLaMA** using our package.   
* [scripts](scripts) contains the script for running examples we provide.  
* [dsconfig](dsconfig) contains configuration files required for mixed precision.  


## Quickstart  

1. **Installing `hift` is simply**
 ```bash
 pip install hift
 # Alternatively
 # pip install git+https://github.com/misonsky/HiFT 
 ```

2. Import `hift` package  

```
from hift import HiFTrainer, HiFTSeq2SeqTrainer, GetCallBack
```

3. **Add `HiFT` configuration**
```
@dataclass
class HiFTArguments(ModelArguments):
    HiTaskType: str = field(
        default="SEQ_CLS",
        metadata={"help": ("HiTaskType should be consistent with PEFT TaskType" )},
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
    lora_tuning:bool = field(
        default=False,
        metadata={
            "help": (
                "whether using lora tuning"
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

***parameter introduction***

**HiTaskType** should be consistent with PEFT TaskType.  

   > **sequence classification**, **multiple choice tasks**: `TaskType.SEQ_CLS`    
   >
   > **question answering** task: `TaskType.QUESTION_ANS`  
   >
   > **sequence labeling** task: `TaskType.TOKEN_CLS`  
   >
   > **generation** task: `TaskType.CAUSAL_LM`   

**group_element**: the number of layers included in a block. Default value is ok. 

**lora_tuning**: HiFT fine-tuning in LoRA mode.

**freeze_layers**: Layers you want to freeze during fine-tuning. You should provide the index of the corresponding layer. The **index** of the embedding layer is **0**, the index of the first layer is **1**,... 


4. **Using `HiFT` Trainer**  

`HiFT` inherits the trainer of huggingface, so you can directly use the trainer provided by hift to replace the original trainer.  

```
if model_args.hier_tuning:#hier_tuning
     trainer = HiFTrainer(
            hiFThandler = GetCallBack(model_args.model_name_or_path),
            HiTaskType = model_args.HiTaskType,
            group_element = model_args.group_element,
            strategy = model_args.optimizer_strategy,
            hier_tuning= model_args.hier_tuning,
            lora_tuning = model_args.lora_tuning,
            freeze_layers = model_args.freeze_layers,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            args=training_args
        )
else:
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
```



## Register Your Model  

Theoretically `HiFT` supports any model. For a new model:

> 1. provide the task interface provided by your model in `TaskTInterface`.
> 2. For different task types, please provide irregular layer regular expressions for identifying them, such as functions `SequenceClassificationSpecialLayer`, `QuestionAnsweringSpecialLayer`, `CausalLMSpecialLayer`  
> 3. Match different task types with corresponding regular expressions, such as `GetSpecialLayer`  
> 4. provide a regular expression in `pattern_name` that identifies all layers.
> 5. The process of extracting identifiers for each layer is provided in group_model.****

```
class ModelCallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        super().__init__(freeze_layers,strategy)
        self.number_position = 5 if lora_tuning else 3
    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = [r"xxxx","xxxx","xxxx"]
       return special_layers
    @classmethod
    def QuestionAnsweringSpecialLayer(cls):
        special_layers = [r"xxxx","xxxx","xxxx"]
        return special_layers
    @classmethod
    def CausalLMSpecialLayer(cls):
        special_layers = [r"xxxx","xxxx","xxxx"]
        return special_layers
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For OPT the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        if taskType == TaskType.SEQ_CLS:
             return cls.SequenceClassificationSpecialLayer()
        if taskType == TaskType.TOKEN_CLS:
             return cls.TokenClassificationSpecialLayer()
        if taskType == TaskType.CAUSAL_LM:
            return cls.CausalLMSpecialLayer()
    def pattern_name(self,special_layers):
        patterns = [rf'\.\d+\.']
        patterns.extend([rf'{layer}' for layer in special_layers])
        pattern = '|'.join(patterns)
        return pattern
    def check_selection(self,elements,name_search):
        pattern_element = ["\."+element+"\." if element.isdigit() else element for element in elements]
        if len(name_search) >1:
            name_search = name_search[:1]
        assert len(name_search)==1
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in pattern_element]
        if sum(signal_value)<=0:
            return False
        else:
            return True
    def group_model(self,model,special_layers,num_position=2,lora_tuning=False):
        ....
        

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
