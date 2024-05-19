#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional,Tuple,List,Dict
from pathlib import Path
import datasets
import torch
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    GPT2Config,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    BloomForCausalLM,
    BloomTokenizerFast,
    BloomConfig,
    CTRLLMHeadModel,
    CTRLTokenizer,
    CTRLConfig,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTJForCausalLM,
    GPTJConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    OpenAIGPTConfig,
    OPTForCausalLM,
    OPTConfig,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    TransfoXLConfig,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLMConfig,
    XLNetLMHeadModel,
    XLNetTokenizer,
    XLNetConfig,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


from hift import HiFTSeq2SeqTrainer,GetCallBack,peft_function,Seq2SeqTrainer
from peft import PeftModel

from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer,GPT2Config),
    "geo":(AutoModelForCausalLM,AutoTokenizer,AutoConfig),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer,CTRLConfig),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,OpenAIGPTConfig),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer,XLNetConfig),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer,TransfoXLConfig),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer,XLMConfig),
    "gptj": (GPTJForCausalLM, AutoTokenizer,GPTJConfig),
    "bloom": (BloomForCausalLM, BloomTokenizerFast,BloomConfig),
    "llama": (LlamaForCausalLM, AutoTokenizer,AutoConfig),
    "opt": (OPTForCausalLM, GPT2Tokenizer,OPTConfig),
}

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    pretraining_tp: int = field(
        default=1,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type:str= field(
        default=None,
        metadata={"help": ("Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()) )},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=512)

@dataclass
class HiFTArguments(ModelArguments):
    HiTaskType: str = field(
        default="CAUSAL_LM",
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

logger = logging.getLogger(__name__)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_peft_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_peft_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_peft_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def main():

    parser = HfArgumentParser((HiFTArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    model_class, tokenizer_class,model_config = MODEL_CLASSES[model_args.model_type]
    
    config = model_config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if training_args.pretraining_tp >1:
        config.pretraining_tp = training_args.pretraining_tp

    tokenizer = tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    if model_args.peft_type:
        if model_args.peft_path is not None:
            logger.info("Peft from pre-trained model")
            model = PeftModel.from_pretrained(model, model_args.peft_path)
        else:
            model = peft_function(model,
                    config = config,
                    peft_type = model_args.peft_type,
                    task_type = model_args.HiTaskType,
                    rank=model_args.lora_rank,
                    virtual_tokens=model_args.virtual_tokens,
                    tokenizer_name_or_path=model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                    init_text=model_args.init_text if model_args.peft_type=="prompt_tuning" else None,
                    peft_config=None)
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset=None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]
            logger.info(f"Training files: {' '.join(files)}")
            train_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        if data_args.validation_file is not None:
            with training_args.main_process_first(desc="loading and tokenization"):
                files = [data_args.validation_file]
                logger.info(f"Evaluation files: {' '.join(files)}")
                eval_dataset = build_instruction_dataset(
                    data_path=files,
                    tokenizer=tokenizer,
                    max_seq_length=data_args.max_seq_length,
                    data_cache_dir = None,
                    preprocessing_num_workers = data_args.preprocessing_num_workers)
            logger.info(f"Num eval_samples  {len(eval_dataset)}")
            logger.info("eval example:")
            logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
        else:
            instruct_datasets = train_dataset.train_test_split(data_args.validation_split_percentage)
            eval_dataset = instruct_datasets["test"]
            train_dataset = instruct_datasets["train"]

            logger.info(f"Num train_samples  {len(train_dataset)}")
            logger.info("training example:")
            logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
            logger.info(f"Num eval_samples  {len(eval_dataset)}")
            logger.info("eval example:")
            logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    # Initialize our Trainer
    if model_args.hier_tuning:
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
            compute_metrics=None,
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
            compute_metrics=None,
        )
    if model_args.peft_type:
        trainer.add_callback(SavePeftModelCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] =len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()