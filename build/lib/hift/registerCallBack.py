import re
from transformers.utils import logging
from peft import TaskType

logger = logging.get_logger(__name__)
class HiFTCallBack(object):
    def __init__(self,freeze_layers,strategy,taskType,peft_type) -> None:
        self.strategy = strategy
        self.freeze_layers = freeze_layers
        self.peft_type = peft_type
        self.taskType = taskType
        self.pattern_list= self.GetSpecialLayer()
    @property
    def emb_pattern(self):
        return list()
    @property
    def others_pattern(self):
        return list()
    @property
    def seq_cls_head(self):
        return list()
    @property
    def token_cls_head(self):
        return list()
    @property
    def qa_cls_head(self):
        return list()
    @property
    def causal_head(self):
        return list()
    @property
    def others_pattern(self):
        return list()
    def SequenceClassificationSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.seq_cls_head)
        return special_layers
    def TokenClassificationSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.token_cls_head)
        return special_layers
    def QuestionAnsweringSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.qa_cls_head)
        return special_layers
    def CausalLMSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.causal_head)
        return special_layers
    def check_selection(self,elements,name_search):
        if len(name_search)<=0:
            return False
        elements = elements = [element if '\\' in element else re.escape(element) for element in elements]
        # print("elements",elements)
        # print("name_search",name_search)
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in elements]
        if sum(signal_value)<=0:
            return False
        else:
            return True
    def check_task_type(self,taskType,model_name,TaskTInterface):
        logger.warning("For {} the HiTaskType should be {}".format(model_name," , ".join(TaskTInterface)))
        assert taskType in TaskTInterface
    def GetSpecialLayer(self):
        if self.taskType == TaskType.SEQ_CLS:
            return self.SequenceClassificationSpecialLayer()
        if self.taskType == TaskType.TOKEN_CLS:
            return self.TokenClassificationSpecialLayer()
        if self.taskType == TaskType.QUESTION_ANS:
            return self.QuestionAnsweringSpecialLayer()
        if self.taskType == TaskType.CAUSAL_LM:
            return self.CausalLMSpecialLayer()
        else:
            raise ValueError("......unsupported task type......")
    
    def group_model(self,model):
        group_parameters = []
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            for pattern in self.pattern_list:
                matches = re.compile(pattern).findall(name)
                if len(matches)>0:
                    if matches[0] not in group_parameters:
                        group_parameters.append(matches[0])
        if hasattr(self,"merge_param"):
            group_parameters = self.merge_param(group_parameters)
        if len(self.freeze_layers)>0:
            for index in self.freeze_layers:
                group_parameters[int(index)]=-1
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
    LayerNumbers = {"lora":5,"adalora":5,"ia3":5,"p_tuning":3,"prefix_tuning":5,"prompt_tuning":-1}
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
        # self.number_position = RobertaCallBack.LayerNumbers[peft_type] if peft_type else 3
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
class BERTCallBack(HiFTCallBack):
    LayerNumbers = {"lora":5,"adalora":5,"ia3":5,"p_tuning":3,"prefix_tuning":5,"prompt_tuning":-1}
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
        # self.number_position = BERTCallBack.LayerNumbers[peft_type] if peft_type else 3
        self.check_task_type(taskType,"BERTa",self.TaskTInterface)
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
            return ["pooler","classifier"]
    @property
    def token_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["pooler","classifier"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["pooler","qa_outputs"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']
class GPT2CallBack(HiFTCallBack):
    LayerNumbers = {"lora":4,"adalora":4,"ia3":4,"p_tuning":3,"prefix_tuning":4,"prompt_tuning":-1}
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
        # self.number_position = GPT2CallBack.LayerNumbers[peft_type] if peft_type else 2
        self.check_task_type(taskType,"GPT2",self.TaskTInterface)
    def merge_param(self,group_parameters):
        group_parameters = self.emb_pattern + [param for param in group_parameters if len(re.compile(self.emb_pattern[0]).findall(param))<=0]
        
        return group_parameters
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return [rf"\.w[^ ]e\."]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["score"]
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
    def causal_head(self):
        if self.peft_type:
            return [rf"\.ln_f\."]
        else:
            return [rf"\.ln_f\."]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']
            
class GPTNeoXCallBack(HiFTCallBack):
    LayerNumbers = {"lora":4,"adalora":4,"ia3":4,"p_tuning":3,"prefix_tuning":4,"prompt_tuning":-1}
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
        # self.number_position = GPTNeoXCallBack.LayerNumbers[peft_type] if peft_type else 2
        self.check_task_type(taskType,"GPTNeoX",self.TaskTInterface)
    def merge_param(self,group_parameters):
        group_parameters = self.emb_pattern + [param for param in group_parameters if len(re.compile(self.emb_pattern[0]).findall(param))<=0]
        
        return group_parameters
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return [rf"\.w[^ ]e\."]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["score"]
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
    def causal_head(self):
        if self.peft_type:
            return [rf"\.ln_f\."]
        else:
            return [rf"\.ln_f\."]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']

class OPTCallBack(HiFTCallBack):
    LayerNumbers = {"lora":5,"adalora":5,"ia3":5,"p_tuning":3,"prefix_tuning":5,"prompt_tuning":-1}
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
        # self.number_position = OPTCallBack.LayerNumbers[peft_type] if peft_type else 3
        self.check_task_type(taskType,"OPT",self.TaskTInterface)
    
    def merge_param(self,group_parameters):
        group_parameters = self.emb_pattern + [param for param in group_parameters if len(re.compile(self.emb_pattern[0]).findall(param))<=0]
        return group_parameters
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return [rf"\.embed_[^ ]+\."]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["score"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["qa_outputs"]
    @property
    def causal_head(self):
        if self.peft_type:
            return ["final_layer_norm"]
        else:
            return ["final_layer_norm"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']

class LLaMaFamilyCallBack(HiFTCallBack):
    LayerNumbers = {"lora":4,"adalora":4,"ia3":4,"p_tuning":3,"prefix_tuning":4,"prompt_tuning":-1}
    def __init__(self,freeze_layers,strategy,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.CAUSAL_LM]
        # self.number_position = OPTCallBack.LayerNumbers[peft_type] if peft_type else 2
        self.check_task_type(taskType,"LLaMA",self.TaskTInterface)
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return ["embed_tokens"]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["model.norm.weight","score"]
    @property
    def causal_head(self):
        if self.peft_type:
            return ["lm_head"]
        else:
            return ["model.norm.weight","lm_head"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']

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
    