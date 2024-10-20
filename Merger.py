from Context import Context
from ModelLoader import ModelLoader
from transformers import AutoTokenizer
from peft import PeftModel

class Merger:
    def mergeAndStore(self, context: Context):
        if not context.mergeFull:
            return
        if not context.storeAdapter:
            raise Exception("Configuration problem: mergeFull requires storeAdapter")
        
        print("Reloading original model on CPU")
        baseModel = ModelLoader().load(context.locBaseModel, True, False, context)
        peftModel = PeftModel.from_pretrained(baseModel, context.locAdapter)
        
        print("Merging")
        mergedModel = peftModel.merge_and_unload()
        
        print("Storing model")
        mergedModel.save_pretrained(save_directory = context.locFull)
        
        print("Storing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
                context.locBaseModel,
                trust_remote_code = True)
        tokenizer.save_pretrained(context.locFull)
        