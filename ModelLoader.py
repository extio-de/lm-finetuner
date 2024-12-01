import torch

from Context import Context
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

class ModelLoader:
    def load(self, path: str, forceCpu: bool, q4OnGpu: bool, context: Context):
        q4 = context.qLora and q4OnGpu and context.accel and not forceCpu
        
        bbConfig = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True, # reduce precision loss
           bnb_4bit_compute_dtype=torch.float16
        )
        torchDataType = torch.float16 if context.accel and not forceCpu else torch.bfloat16
        
        print(f"Loading model {path} (gpu={context.accel and not forceCpu}, q4={q4}, dt={torchDataType})...")
        
        baseModel = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=bbConfig if q4 else None,
                torch_dtype=torchDataType,
                trust_remote_code=True)
        if not forceCpu:
            baseModel = baseModel.to(context.device)
        
        print(baseModel)
        
        return baseModel
