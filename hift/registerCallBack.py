import re
from transformers.utils import logging
from peft import TaskType

logger = logging.get_logger(__name__)
class HiFTCallBack(object):
    def __init__(self,freeze_layers,strategy) -> None:
        self.strategy = strategy
        self.freeze_layers = freeze_layers
    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = ["embeddings","classifier"]
       return special_layers
    @classmethod
    def TokenClassificationSpecialLayer(cls):
        special_layers = ["embeddings","classifier"]
        return special_layers
    @classmethod
    def QuestionAnsweringSpecialLayer(cls):
        special_layers = ["embeddings","qa_outputs"]
        return special_layers
    @classmethod
    def GetSpecialLayer(cls,taskType):
        if taskType == TaskType.SEQ_CLS:
            return cls.SequenceClassificationSpecialLayer()
        if taskType == TaskType.TOKEN_CLS:
            return cls.TokenClassificationSpecialLayer()
        if taskType == TaskType.QUESTION_ANS:
            return cls.QuestionAnsweringSpecialLayer()
    
    def nest_fun(self,layers,subname):
        signal_value = [1 if len(re.compile(layern).findall(subname))>0 else 0 for layern in layers]
        if sum(signal_value)<=0:
            return False
        return True
    def group_model(self,model,special_layers,num_position=2,lora_tuning=False):
        group_parameters = []
        non_lora_identifier = []
        for pname in special_layers:
            for name,_ in model.named_parameters():
                matches = re.compile(pname).findall(name)
                if len(matches)>0:
                    if pname not in group_parameters:
                        if not lora_tuning:
                            group_parameters.append(pname)
                else:
                    items = name.split(".")
                    flags = [1 if self.nest_fun(special_layers,item) else 0 for item in items]
                    if sum(flags)<=0:
                        layerNum = items[num_position]
                        assert layerNum.isdigit()
                        if layerNum not in group_parameters:
                            group_parameters.append(layerNum)
            
        if len(self.freeze_layers)>0:
            for index in self.freeze_layers:
                group_parameters[index]=-1
                group_parameters = [element for element in group_parameters if element != -1]
        if self.strategy=="up2down":
           group_parameters.reverse()
        elif self.strategy == "random":
            random.shuffle(group_parameters)
        elif self.strategy != "down2up":
            raise ValueError("providing proper strategy")
        print(group_parameters)
        return group_parameters

class RobertaCallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        super().__init__(freeze_layers,strategy)
        self.number_position = 5 if lora_tuning else 3
    def pattern_name(self,special_layers):
        patterns = [rf'\.\d+\.']
        patterns.extend([rf'{layer}' if "embeddings" not in layer else rf'\.{layer}\.' for layer in special_layers])
        pattern = '|'.join(patterns)
        return pattern
    def check_selection(self,elements,name_search):
        pattern_element = ["\."+element+"\." if element.isdigit() or "embeddings" in element else element for element in elements]
        assert len(name_search)==1
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in pattern_element]
        if sum(signal_value)<=0:
            return False
        else:
            return True
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For RoBERTa the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        return super().GetSpecialLayer(taskType)
class BERTCallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        
        super().__init__(freeze_layers,strategy)
        self.number_position = 5 if lora_tuning else 3
    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = ["embeddings","pooler","classifier"]
       return special_layers
    @classmethod
    def TokenClassificationSpecialLayer(cls):
        special_layers = ["embeddings","pooler","classifier"]
        return special_layers
    @classmethod
    def QuestionAnsweringSpecialLayer(cls):
        special_layers = ["embeddings","pooler","qa_outputs"]
        return special_layers
    def pattern_name(self,special_layers):
        patterns = [rf'\.\d+\.']
        patterns.extend([rf'{layer}' if "embeddings" not in layer else rf'\.{layer}\.' for layer in special_layers])
        pattern = '|'.join(patterns)
        return pattern
    def check_selection(self,elements,name_search):
        pattern_element = ["\."+element+"\." if element.isdigit() or "embeddings" in element else element for element in elements]
        assert len(name_search)==1
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in pattern_element]
        if sum(signal_value)<=0:
            return False
        else:
            return True
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For BERT the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        return super().GetSpecialLayer(taskType)
class GPT2CallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        super().__init__(freeze_layers,strategy)
        self.number_position = 4 if lora_tuning else 2

    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = [r"w[^ ]e","score"]
       return special_layers
    @classmethod
    def TokenClassificationSpecialLayer(cls):
        special_layers = [r"w[^ ]e","classifier"]
        return special_layers
    @classmethod
    def QuestionAnsweringSpecialLayer(cls):
        special_layers = [r"w[^ ]e","qa_outputs"]
        return special_layers
    @classmethod
    def LMHeadModelSpecialLayer(cls):
        special_layers = [r"w[^ ]e","ln_f"]
        return special_layers
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For GPT-2 the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        if taskType == TaskType.CAUSAL_LM:
            return cls.LMHeadModelSpecialLayer()
        return super().GetSpecialLayer(taskType)
    def pattern_name(self,special_layers):
        patterns = [rf'\.\d+\.']
        patterns.extend([rf'{layer}' for layer in special_layers])
        pattern = '|'.join(patterns)
        return pattern
    def check_selection(self,elements,name_search):
        pattern_element = ["\."+element+"\." if element.isdigit() else element for element in elements]
        assert len(name_search)==1
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in pattern_element]
        if sum(signal_value)<=0:
            return False
        else:
            return True

class GPTNeoXCallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        super().__init__(freeze_layers,strategy)
        self.number_position = 4 if lora_tuning else 2
        
    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = [r"w[^ ]e","score"]
       return special_layers
    @classmethod
    def TokenClassificationSpecialLayer(cls):
        special_layers = [r"w[^ ]e","classifier"]
        return special_layers
    @classmethod
    def QuestionAnsweringSpecialLayer(cls):
        special_layers = [r"w[^ ]e","qa_outputs"]
        return special_layers
    @classmethod
    def CausalLMSpecialLayer(cls):
        special_layers = [r"w[^ ]e","ln_f"]
        return special_layers
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For GPTNeoX the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        if taskType == TaskType.CAUSAL_LM:
            return cls.CausalLMSpecialLayer()
        return super().GetSpecialLayer(taskType)
    def pattern_name(self,special_layers):
        patterns = [rf'\.\d+\.']
        patterns.extend([rf'{layer}' for layer in special_layers])
        pattern = '|'.join(patterns)
        return pattern
    def check_selection(self,elements,name_search):
        pattern_element = ["\."+element+"\." if element.isdigit() else element for element in elements]
        assert len(name_search)==1
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in pattern_element]
        if sum(signal_value)<=0:
            return False
        else:
            return True

class OPTCallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        super().__init__(freeze_layers,strategy)
        self.number_position = 5 if lora_tuning else 3
    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = [r"w[^ ]e","score"]
       return special_layers
    @classmethod
    def QuestionAnsweringSpecialLayer(cls):
        special_layers = [r"w[^ ]e","qa_outputs"]
        return special_layers
    @classmethod
    def CausalLMSpecialLayer(cls):
        special_layers = [r"embed_[^ ]+","final_layer_norm"]
        return special_layers
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For OPT the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        if taskType == TaskType.CAUSAL_LM:
            return cls.CausalLMSpecialLayer()
        return super().GetSpecialLayer(taskType)
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

class LLaMaFamilyCallBack(HiFTCallBack):
    TaskTInterface = [TaskType.SEQ_CLS,TaskType.CAUSAL_LM]
    def __init__(self,freeze_layers,strategy,lora_tuning=False):
        super().__init__(freeze_layers,strategy)
        self.number_position = 4 if lora_tuning else 2
    @classmethod
    def SequenceClassificationSpecialLayer(cls):
       special_layers = ["embed_tokens","norm","score"]
       return special_layers
    @classmethod
    def CausalLMSpecialLayer(cls):
        special_layers = ["embed_tokens","norm","lm_head"]
        return special_layers
    @classmethod
    def GetSpecialLayer(cls,taskType):
        logger.warning("For LLaMaFamily the HiTaskType should be {}".format(" , ".join(cls.TaskTInterface)))
        assert taskType in cls.TaskTInterface
        if taskType == TaskType.CAUSAL_LM:
            return cls.CausalLMSpecialLayer()
        super().GetSpecialLayer(taskType)
    def pattern_name(self,special_layers):
        patterns = [rf'\.\d+\.']
        patterns.extend([rf'{layer}' if "norm" not in layer else rf'\.\d+\.|\.{layer}\.' for layer in special_layers])
        pattern = '|'.join(patterns)
        return pattern
    def check_selection(self,elements,name_search):
        pattern_element = ["\."+element+"\." if element.isdigit() or "norm" in element else element for element in elements]
        assert len(name_search)==1
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in pattern_element]
        if sum(signal_value)<=0:
            return False
        else:
            return True


MODDELS_HiFT_PROCESS={
    "roberta":RobertaCallBack,
    "bert":BERTCallBack,
    "gpt2":GPT2CallBack,
    "gptneox":GPTNeoXCallBack,
    "gptneo":GPTNeoXCallBack,
    "opt":OPTCallBack,
    "llamafamily":LLaMaFamilyCallBack,
}

def GetCallBack(model_name_path):
    if "roberta" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["roberta"]
    if "bert" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["bert"]
    if "gpt2" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["gpt2"]
    if "gptneox" in model_name_path.lower() or "gpt-neox" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["gptneox"]
    if "gptneo" in model_name_path.lower() or "gpt-neo" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["gptneo"]
    if "opt" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["opt"]
    if "llamafamily" in model_name_path.lower() or "llama" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["llamafamily"]
    