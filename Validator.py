import torch
import gc
import json

from Context import Context
from ModelLoader import ModelLoader
from Dataset import Dataset
from transformers import AutoTokenizer
from peft import PeftModel
from dataclasses import dataclass

class Validator:
    peftTokenizer = None
    peftModel = None
    extPeftTokenizer = None
    extPeftModel = None
    graderTokenizer = None
    graderModel = None
    statistics = []
    
    def validate(self, context: Context):
        if not context.validate or context.vInplace:
            return True
        if not context.storeAdapter:
            raise Exception("Configuration problem: validate requires storeAdapter")
                
        print("Validation")
        
        validations = None
        self.__loadPeftModel(context)
        try:
            validations = self.__askPeftModel(context)
        finally:
            self.unload(context)
        
        self.__loadGraderModel(context)
        try:
            return self.__grade(validations, context)
        finally:
            self.unload(context)
    
    def validateInPlace(self, peftTokenizer, peftModel, context: Context):
        if not context.validate or not context.vInplace:
            return True
        
        print("In-Place validation")
        
        self.extPeftTokenizer = peftTokenizer
        self.extPeftModel = peftModel
        validations = self.__askPeftModel(context)
        
        if self.graderModel == None:
            self.__loadGraderModel(context)
        return self.__grade(validations, context)
    
    def __loadPeftModel(self, context: Context):
        self.peftTokenizer = AutoTokenizer.from_pretrained(context.locBaseModel, trust_remote_code = True)
        if self.peftTokenizer.pad_token == None:
            self.peftTokenizer.pad_token = self.peftTokenizer.eos_token
        
        baseModel = ModelLoader().load(context.locBaseModel, False, context.vQuantModel, context)
        self.peftModel = PeftModel.from_pretrained(baseModel, context.locAdapter)
    
    def __askPeftModel(self, context: Context):
        tokenizer = self.extPeftTokenizer or self.peftTokenizer
        model = self.extPeftModel or self.peftModel
        
        result = Validations([], 0)
        
        dataset = Dataset().loadDataDir("json", context.locValidation, context)
        
        print("Asking " + str(dataset["validation"].num_rows * context.vPasses) + " questions...")
        
        noQuestion = 0
        result.total = 0
        for record in dataset["validation"]:
            answers = []
            for _ in range(0, context.vPasses):
                prompt = ""
                userPromptType = ""
                trimPrompt = True
                
                userPromptType = "chat"
                userPrompt = record[userPromptType]
                if userPrompt:
                    chat = [
                        {"role": "user", "content": f"{userPrompt}"}
                    ]
                    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt = True, tokenize = False)
                else:
                    userPromptType = "chatCompletion"
                    userPrompt = record[userPromptType]
                    if userPrompt:
                        chat = [
                            {"role": "user", "content": f"{userPrompt}"}
                        ]
                        prompt = tokenizer.apply_chat_template(chat, continue_final_message = True, tokenize = False)
                    else:
                        userPromptType = "completion"
                        userPrompt = prompt = record[userPromptType]
                        trimPrompt = False
                        if not userPrompt:
                            continue
                
                tokenized = tokenizer(prompt, return_tensors="pt", return_attention_mask = True, add_special_tokens = False).to(context.device)
                
                outputs = model.generate(**tokenized, max_new_tokens = context.vGenMaxTokens, pad_token_id = tokenizer.eos_token_id)
                
                output = str(tokenizer.decode(outputs[0]))
                if trimPrompt:
                    output = output.replace(prompt, "")
                if tokenizer.eos_token:
                    output = output.replace(tokenizer.eos_token, "")
                if tokenizer.pad_token:
                    output = output.replace(tokenizer.pad_token, "").rstrip()
                
                answers.append(output)
                
                noQuestion += 1
                print(noQuestion, end = " " if noQuestion%10>0 else "\n", flush = True)
                
            result.total += len(answers) * (len(record["oneOf"] or []) + len(record["string"] or []))
            result.validations.append(Validation(userPrompt, userPromptType, answers, record["oneOf"], record["string"]))
            
        print("")
        
        return result
    
    def __loadGraderModel(self, context: Context):
        self.graderTokenizer = AutoTokenizer.from_pretrained(context.locGraderModel, trust_remote_code = True)
        if self.graderTokenizer.pad_token == None:
            self.graderTokenizer.pad_token = self.graderTokenizer.eos_token
        self.graderModel = ModelLoader().load(context.locGraderModel, context.vGraderOnCpu, context.vQuantGrader, context)
    
    def __grade(self, validations, context: Context):
        print("Grading...")
        
        cntEvaluations = 0
        cntPassed = 0
        cntValidations = 0
        cntCompletion = 0
        
        for validation in validations.validations:
            cntCompletion += 1
            print(f"{validation.userPromptType} {cntCompletion}/{len(validations.validations)}: {validation.userPrompt}")
            
            for answer in validation.answers:
                print(f"    * {answer}")
                
                if len(validation.evalOneOfs) > 0:
                    cntEvaluations += 1
                    passed = False
                    for gradingQuestion in validation.evalOneOfs:
                        cntValidations += 1
                        print(f"        > ({cntValidations}/{validations.total}) {gradingQuestion}")
                        
                        chat = [
                            {"role": "system", "content": """You are a grader that evaluates the relevance of a given text to a user question.
Please provide a binary response 'true' or 'false' for the following text.
'true' means that the text provides a truthful answer to the question, while 'false' means that it does not.
Provide no preamble and a short explanation. Return the response in JSON format with the following field: "passed", "explanation\""""},
                            {"role": "user", "content": f"""The text is: '{answer}'
The question is: '{gradingQuestion}'"""}
                        ]
                        prompt = self.graderTokenizer.apply_chat_template(chat, add_generation_prompt = True, tokenize = False)
                        tokenized = self.graderTokenizer(prompt, return_tensors="pt", return_attention_mask = True, add_special_tokens = False)
                        if not context.vGraderOnCpu:
                            tokenized = tokenized.to(context.device)
                        
                        outputs = self.graderModel.generate(**tokenized, max_new_tokens = context.vGenMaxTokens, pad_token_id = self.graderTokenizer.eos_token_id)
                        
                        output = self.graderTokenizer.decode(outputs[0])
                        output = output.replace(prompt, "")
                        if self.graderTokenizer.eos_token:
                            output = output.replace(self.graderTokenizer.eos_token, "")
                        if self.graderTokenizer.pad_token:
                            output = output.replace(self.graderTokenizer.pad_token, "").rstrip()
                        
                        questionPassed = False
                        try:
                            processed = output[output.index("{") : output.index("}") + 1]
                            print(processed)
                            parsed = json.loads(processed)
                            questionPassed = "true" in str(parsed["passed"]).lower()
                            
                        except:
                            if "\"passed\"" in output.lower():
                                print(output)
                                print(f"            -> Warning: Json malformed but parseable")
                                questionPassed = any(field in output.lower() for field in ("\"passed\": true", "\"passed\":true", "\"passed\": \"true\"", "\"passed\":\"true\"")) 
                                
                            else:
                                print(f"            -> ERROR: Cannot parse: {output}")
                        
                        passed |= questionPassed
                        print("            -> " + ("PASSED" if questionPassed else "FAILED"))
                        
                    print("        => " + ("PASSED" if passed else "FAILED"))
                    if passed:
                        cntPassed += 1
                
                if len(validation.evalStrings) > 0:
                    cntEvaluations += 1
                    passed = False
                    for evalString in validation.evalStrings:
                        cntValidations += 1
                        print(f"        > ({cntValidations}/{validations.total}) Contains string {evalString}")
                        passed |= evalString.lower() in answer.lower()
                    print("        => " + ("PASSED" if passed else "FAILED"))
                    if passed:
                        cntPassed += 1
        
        passedPerc = int(float(cntPassed) / float(cntEvaluations) * 100.0)
        result = passedPerc >= context.vExpected
        self.statistics.append((result, passedPerc))
        
        print("")
        print("#############")
        print("Final result: " + str(cntPassed) + "/" + str(cntEvaluations) + " -> " + str(passedPerc) + "%")
        print("--> " + ("PASSED" if result else "FAILED"))
        
        return result
    
    def unload(self, context: Context):
        if self.peftTokenizer != None:
            del self.peftTokenizer
            self.peftTokenizer = None
        if self.peftModel != None:
            self.peftModel.unload()
            del self.peftModel
            self.peftModel = None
        if self.graderTokenizer != None:
            del self.graderTokenizer
            self.graderTokenizer = None
        if self.graderModel != None:
            del self.graderModel
            self.graderModel = None
        self.statistics = []
        
        gc.collect()
        
        if context.accel:
            torch.cuda.empty_cache()        
    
@dataclass
class Validations:
    validations: list
    total: int

@dataclass
class Validation:
    userPrompt: str
    userPromptType: str
    answers: list
    evalOneOfs: list
    evalStrings: list
