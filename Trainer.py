import gc
import torch

from Context import Context
from Dataset import Dataset
from ModelLoader import ModelLoader
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from Validator import Validator

class Trainer:
    def train(self, context: Context):
        if not context.train:
            return
        
        baseModel = ModelLoader().load(context.locBaseModel, False, True, context)
        sftTrainer = self.__createTrainer(baseModel, context)
        try:
            vInplace = context.validate and context.vInplace
            validator = None
            if vInplace:
                validator = Validator()
            
            try:
                cnt = 0
                continueTraining = True
                while continueTraining:
                    cnt += 1
                    print(f"Training run {cnt}")
                    
                    self.__train(sftTrainer)
                    
                    if vInplace:
                        continueTraining = not validator.validateInPlace(sftTrainer.tokenizer, sftTrainer.model, context)
                    else:
                        continueTraining = False
                
                print(f"Total training runs: {cnt} epochs {cnt * context.trEpochs}")
                
            finally:
                if vInplace:
                    print(f"Validation statistics:\n{validator.statistics}")
                    validator.unload(context)
            
            if context.storeAdapter:
                self.__storeAdapter(sftTrainer, context)
        
        finally:
            del sftTrainer
            del baseModel
            gc.collect()
            if context.accel:
                torch.cuda.empty_cache()
        
    def __createTrainer(self, baseModel, context: Context):
        tokenizer = AutoTokenizer.from_pretrained(
                context.locBaseModel,
                trust_remote_code = True)
        if tokenizer.pad_token == None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = Dataset().scan(context.locDataset, tokenizer, context)
        
        peftConfig = LoraConfig(
                r = context.loraR,
                lora_alpha = context.loraAlpha,
                lora_dropout = context.loraDropout,
                bias = context.loraBias,
                task_type = context.loraTaskType
                #target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
        )
        
        sftArgs = {}
        if context.accel:
            sftArgs["fp16"] = True
        else:
            sftArgs["bf16"] = True
        if context.trMaxSeqLength:
            sftArgs["max_seq_length"] = context.trMaxSeqLength
        if context.trPerDeviceTrainBatchSize:
            sftArgs["per_device_train_batch_size"] = context.trPerDeviceTrainBatchSize
        if context.trFindAutoBatchSize:
            sftArgs["auto_find_batch_size"] = context.trFindAutoBatchSize
        if context.trGradientAccSteps:
            sftArgs["gradient_accumulation_steps"] = context.trGradientAccSteps
        if context.trGradientCheckpointing:
            sftArgs["gradient_checkpointing"] = context.trGradientCheckpointing
        if context.trGroupByLength:
            sftArgs["group_by_length"] = context.trGroupByLength
        if context.trPacking:
            sftArgs["packing"] = context.trPacking
        sftConfig = SFTConfig(
                dataset_text_field="text",
                num_train_epochs = context.trEpochs,
                output_dir=context.locWorkdir,
                save_strategy="no",
                #neftune_noise_alpha=5,
                **sftArgs
        )
        
        sftTrainer = SFTTrainer(
                model = baseModel,
                train_dataset = dataset,
                tokenizer = tokenizer,
                peft_config = peftConfig,
                args = sftConfig
        )
        
        return sftTrainer
        
    def __train(self, sftTrainer):
        sftTrainer.train()
        print("Training finished")
        
    def __storeAdapter(self, sftTrainer, context: Context):
        print("Saving adapter...")
        
        sftTrainer.model.save_pretrained(context.locAdapter)
        sftTrainer.tokenizer.save_pretrained(context.locAdapter)