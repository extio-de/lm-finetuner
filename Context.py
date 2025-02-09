import configparser
import torch

from peft import PeftModel
from sympy.printing import str

class Context:
    model: PeftModel
    
    accel: bool
    device: str
    purgeTargetDirectories: bool
    showChatTemplate: bool
    
    locBaseModel: str
    locDataset: str
    locWorkdir: str
    locAdapter: str
    locFull: str
    locValidation: str
    locGraderModel: str
    locCustomPromptTemplate: str
    
    train: bool
    storeAdapter: bool
    trEpochs: int
    trMaxSeqLength: int
    trPerDeviceTrainBatchSize: int
    trFindAutoBatchSize: bool
    trGradientAccSteps: int
    trGradientCheckpointing: bool
    trGroupByLength: bool
    trPacking: bool
    trOptim : str
    trSchedulerType: str
    
    qLora : bool
    loraR: int
    loraAlpha: int
    loraDropout: float
    loraBias: str
    loraTaskType: str
    loraLayers: list
    
    validate: bool
    vInplace: bool
    vAbortOnFail: bool
    vPasses: int
    vGenMaxTokens: int
    vExpected: int
    vQuantModel: bool
    vQuantGrader: bool
    vGraderOnCpu: bool
    
    mergeFull: bool
    
    def load(self, cfgFile):
        cfg = configparser.ConfigParser()
        if not cfg.read(cfgFile):
            raise Exception("Cannot read configuration file: " + cfgFile)
        
        self.accel = torch.cuda.is_available() and cfg.get("Operation", "device") != "cpu"
        self.device = (cfg.get("Operation", "device") if self.accel else "cpu") or "cpu"
        self.purgeTargetDirectories = cfg.get("Operation", "purgeTargetDirectories").lower() == "true"
        self.showChatTemplate = cfg.get("Operation", "showChatTemplate").lower() == "true"
        
        self.locBaseModel = cfg.get("Trainer", "locBaseModel")
        self.locDataset = cfg.get("Trainer", "locDataset")
        self.locWorkdir = cfg.get("Trainer", "locWorkdir")
        self.locAdapter = cfg.get("Trainer", "locAdapter")
        self.locFull = cfg.get("Merger", "locFull")
        self.locValidation = cfg.get("Validation", "locValidation")
        self.locGraderModel = cfg.get("Validation", "locGraderModel")
        self.locCustomPromptTemplate = cfg.get("Trainer", "locCustomPromptTemplate") or None
        
        self.train = cfg.get("Trainer", "train").lower() == "true"
        self.storeAdapter = cfg.get("Trainer", "storeAdapter").lower() == "true"
        self.trEpochs = int(cfg.get("Trainer", "trEpochs") or "0") or None
        self.trMaxSeqLength = int(cfg.get("Trainer", "trMaxSeqLength") or "0") or None
        if not self.trMaxSeqLength:
            raise Exception("trMaxSeqLength is not configured")
        self.trPerDeviceTrainBatchSize = int(cfg.get("Trainer", "trPerDeviceTrainBatchSize") or "0") or None
        self.trFindAutoBatchSize = cfg.get("Trainer", "trFindAutoBatchSize").lower() == "true" 
        self.trGradientAccSteps = int(cfg.get("Trainer", "trGradientAccSteps") or "0") or None
        self.trGradientCheckpointing = cfg.get("Trainer", "trGradientCheckpointing").lower() == "true" 
        self.trGroupByLength = cfg.get("Trainer", "trGroupByLength").lower() == "true"
        self.trPacking = cfg.get("Trainer", "trPacking").lower() == "true"
        self.trOptim = cfg.get("Trainer", "trOptim") or None
        self.trSchedulerType = cfg.get("Trainer", "trSchedulerType") or None
        
        self.qLora = cfg.get("Lora", "qLora").lower() == "true"
        self.loraR = int(cfg.get("Lora", "loraR") or "0") or 64
        self.loraAlpha = int(cfg.get("Lora", "loraAlpha") or "0") or 16
        self.loraDropout = float(cfg.get("Lora", "loraDropout") or "0") or 0.1
        self.loraBias = cfg.get("Lora", "loraBias") or "none"
        self.loraTaskType = cfg.get("Lora", "loraTaskType") or "CAUSAL_LM"
        self.loraLayers = (cfg.get("Lora", "loraLayers") or "q_proj,k_proj").split(",")
        
        self.validate = cfg.get("Validation", "validate").lower() == "true"
        self.vInplace = cfg.get("Validation", "vInplace").lower() == "true"
        self.vAbortOnFail = cfg.get("Validation", "vAbortOnFail").lower() == "true"
        self.vPasses = int(cfg.get("Validation", "vPasses") or "0") or 10
        self.vGenMaxTokens = int(cfg.get("Validation", "vGenMaxTokens") or "0") or 100
        self.vExpected = int(cfg.get("Validation", "vExpected") or "0") or 70
        self.vQuantModel = cfg.get("Validation", "vQuantModel").lower() == "true"
        self.vQuantGrader = cfg.get("Validation", "vQuantGrader").lower() == "true"
        self.vGraderOnCpu = cfg.get("Validation", "vGraderOnCpu").lower() == "true"
        
        self.mergeFull = cfg.get("Merger", "mergeFull").lower() == "true"
        
        print(self.__dict__)
