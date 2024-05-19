# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import logging
from dataclasses import dataclass, field
import json
import sys
import os
import math
import pathlib
from typing import Dict, Optional, Sequence,Tuple,List

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

from hift import HiFTSeq2SeqTrainer,GetCallBack,peft_function,Seq2SeqTrainer
from peft import PeftModel
from build_dataset import DataCollatorForSupervisedDataset
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

replace_llama_attn_with_flash_attn()
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
    "llama": (AutoModelForCausalLM, AutoTokenizer,AutoConfig),
    "opt": (OPTForCausalLM, GPT2Tokenizer,OPTConfig),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
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
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    model_type:str= field(
        default=None,
        metadata={"help": ("Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()) )},
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


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
local_rank = None

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

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and hasattr(tokenizer,"legacy") and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and hasattr(tokenizer,"legacy") and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        key_name = "conversations"
        if "items" in raw_data[i]:
            key_name = "items"
        rank0_print("Formatting inputs...")
        sources = [example[key_name] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        key_name = "conversations"
        if "items" in self.raw_data[i]:
            key_name = "items"
        ret = preprocess([self.raw_data[i][key_name]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (HiFTArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model_class, tokenizer_class,model_config = MODEL_CLASSES[model_args.model_type]
    
    config = model_config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    
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

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # Start trainner
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
            train_dataset=data_module['train_dataset'] if training_args.do_train else None,
            eval_dataset=data_module['eval_dataset'] if training_args.do_eval else None,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            tokenizer=tokenizer,#data_collator=data_collator   
        )
    else:
        trainer = Seq2SeqTrainer(
            peft_type = model_args.peft_type,
            args=training_args,
            model=model,
            train_dataset=data_module['train_dataset'] if training_args.do_train else None,
            eval_dataset=data_module['eval_dataset'] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
    
    if model_args.peft_type:
        trainer.add_callback(SavePeftModelCallback)
    
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        # Save model
        model.config.use_cache = True
        trainer.save_state()
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)
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
    train()
