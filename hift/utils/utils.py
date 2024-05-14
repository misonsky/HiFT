#coding=utf-8
from peft import (
    PeftType,
    LoraConfig,
    TaskType,
    get_peft_model,
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig
)
from peft.tuners.adalora import RankAllocator
import torch

def _update_ipt(self, model):
    # Update the sensitivity and uncertainty for every weight
    for n, p in model.named_parameters():
        if "lora_" in n and self.adapter_name in n:
            if n not in self.ipt:
                self.ipt[n] = torch.zeros_like(p)
                self.exp_avg_ipt[n] = torch.zeros_like(p)
                self.exp_avg_unc[n] = torch.zeros_like(p)
            with torch.no_grad():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                self.ipt[n] = (p * p.grad).abs().detach()
                # Sensitivity smoothing
                self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                # Uncertainty quantification
                self.exp_avg_unc[n] = (
                    self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                )

def adalora_peft(model,task_type,rank=8,peft_config=None):
    """
    loss = model(**input).loss
    loss.backward()
    optimizer.step()
    model.base_model.update_and_allocate(i_step)
    optimizer.zero_grad()
    """
    RankAllocator.update_ipt = _update_ipt
    if not peft_config:
        peft_config = AdaLoraConfig(
            peft_type="ADALORA", 
            task_type=task_type, 
            r=rank, 
            lora_alpha=32, 
            lora_dropout=0.01)
    ada_model = get_peft_model(model, peft_config)
    return ada_model
def ia3_peft(model,task_type,peft_config=None):
    if not peft_config:
        peft_config = IA3Config(
            peft_type="IA3",
            task_type=task_type,
            feedforward_modules=["w0"])
    ia3_model = get_peft_model(model, peft_config)
    return ia3_model
def lora_peft(model,task_type,rank=8,peft_config=None):
    if not peft_config:
        peft_config = LoraConfig(
            task_type=task_type,
            r=rank,
            lora_alpha=32,
            lora_dropout=0.01)
    lora_model = get_peft_model(model, peft_config)
    return lora_model
def p_tuning(model,task_type,virtual_tokens=20,token_dim=768,hidden_size=768,att_heads = 12,num_layers=12,peft_config=None):
    if not peft_config:
        peft_config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=task_type,
            num_virtual_tokens=virtual_tokens,
            token_dim=token_dim,
            num_attention_heads=att_heads,
            num_layers=num_layers,
            encoder_hidden_size=hidden_size)
    p_model = get_peft_model(model, peft_config)
    return p_model

def prefix_tuning(model,task_type,virtual_tokens=20,token_dim=768,hidden_size=768,att_heads = 12,num_layers=12,peft_config=None):
    if not peft_config:
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type=task_type,
            num_virtual_tokens=virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=1,
            num_attention_heads=att_heads,
            num_layers=num_layers,
            encoder_hidden_size=hidden_size)
    prefix_model = get_peft_model(model, peft_config)
    return prefix_model
def prompt_tuning(model,task_type,virtual_tokens=20,token_dim=768,att_heads = 12,num_layers=12,tokenizer_name_or_path=None,prompt_tuning_init_text=None,peft_config=None):
    if not peft_config:
        peft_config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type=task_type,
            num_virtual_tokens=virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=1,
            num_attention_heads=att_heads,
            num_layers=num_layers,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
            tokenizer_name_or_path=tokenizer_name_or_path)
    prompt_model = get_peft_model(model, peft_config)
    return prompt_model
def get_model_config(config):
    hidden_size,att_heads,num_layers,token_dim = None,None,None,None
    if hasattr(config,"hidden_size"):
        hidden_size = config.hidden_size
    
    if hasattr(config,"n_embd"):
        token_dim = config.n_embd
    if hasattr(config,"word_embed_proj_dim"):
        token_dim = config.word_embed_proj_dim
    
    if hasattr(config,"num_attention_heads"):
        att_heads = config.num_attention_heads
    if hasattr(config,"num_heads"):
        att_heads = config.num_heads
    if hasattr(config,"n_head"):
        att_heads = config.n_head
    
    if hasattr(config,"num_hidden_layers"):
        num_layers  = config.num_hidden_layers
    if hasattr(config,"n_layer"):
        num_layers = config.n_layer
    if hasattr(config,"num_layers"):
        num_layers = config.num_layers
    
    if not token_dim:
        token_dim = hidden_size
    return hidden_size,att_heads,num_layers,token_dim
def peft_function(model,config,peft_type,task_type,rank=8,virtual_tokens=20,tokenizer_name_or_path=None,init_text=None,peft_config=None):
    hidden_size,att_heads,num_layers,token_dim =get_model_config(config)
    if "adalora" == peft_type.lower():
        return adalora_peft(model,task_type,rank=rank,peft_config=None)
    if "lora" == peft_type.lower():
        return lora_peft(model,task_type,rank=rank,peft_config=None)
    if "ia3" == peft_type.lower():
        return ia3_peft(model,task_type,peft_config=None)
    if "p_tuning" == peft_type.lower():
        return p_tuning(model,task_type,
                        virtual_tokens=virtual_tokens,
                        token_dim=token_dim,
                        hidden_size=hidden_size,
                        att_heads = att_heads,
                        num_layers= num_layers,
                        peft_config=None)
    if "prefix_tuning" == peft_type.lower():
        return prefix_tuning(model,task_type,
                        virtual_tokens=virtual_tokens,
                        token_dim=token_dim,
                        hidden_size=hidden_size,
                        att_heads = att_heads,
                        num_layers= num_layers,
                        peft_config=None)
    if "prompt_tuning" == peft_type.lower():
        return prompt_tuning(model,task_type,
                        virtual_tokens=virtual_tokens,
                        token_dim=token_dim,
                        att_heads = att_heads,
                        num_layers=num_layers,
                        tokenizer_name_or_path = tokenizer_name_or_path,
                        prompt_tuning_init_text=init_text,
                        peft_config=None)
    else:
        raise ValueError("unsupported {} peft fine-tuning mode".format(peft_type))