import pathlib

from Context import Context
from datasets import load_dataset, concatenate_datasets

class Dataset:
    def scan(self, location: str, tokenizer, context: Context):
        print(f"Scanning dataset directory {location}")
        
        directory = pathlib.Path(location)
        files = [f for f in directory.iterdir() if f.is_file()]
        
        datasets = []
        for file in files:
            print(file.name)
            
            if file.name.lower().endswith(".txt"):
                textDataset = self.loadFile("text", str(file), context)
                self.__loadTextDataset(textDataset, datasets, tokenizer)
                
            elif file.name.lower().endswith(".json") or file.name.lower().endswith(".jsonl"):
                qaDataset = self.loadFile("json", str(file), context)
                if "text" in qaDataset['train'].features:
                    self.__loadTextDataset(qaDataset, datasets, tokenizer)
                elif "history" in qaDataset['train'].features or "instruct" in qaDataset['train'].features or "completion" in qaDataset['train'].features or "question" in qaDataset['train'].features or "answer" in qaDataset['train'].features:
                    self.__loadQaDataset(qaDataset, datasets, tokenizer, context)
                else:
                    print("Cannot load dataset, json dataset neither contains 'text' nor 'question', 'answer'")
        
        if len(datasets) == 0:
            raise Exception("No datasets have been found. Check configuration")
        
        return concatenate_datasets(datasets)
    
    def loadFile(self, type_: str, path: str, context: Context, **kwargs):
        return self.__load(type_, context, data_files=path, **kwargs)

    def loadDataDir(self, type_: str, location: str, context: Context, **kwargs):
        return self.__load(type_, context, data_dir=location, **kwargs)
    
    def __load(self, type_: str, context: Context, **kwargs):
        print("Loading dataset...")
        
        dataset = load_dataset(type_, **kwargs)
        print(dataset)
        
        return dataset
    
    def __loadTextDataset(self, textDataset, datasets, tokenizer):
        def datasetTextEncoder(record):
            return tokenizer(record["text"])
        datasets.append(textDataset.map(datasetTextEncoder)["train"])

    def __loadQaDataset(self, qaDataset, datasets, tokenizer, context):
        customChatTemplate = None
        if context.locCustomPromptTemplate != None:
            with open(context.locCustomPromptTemplate, 'r') as file:
                customChatTemplate = file.read()
        
        def datasetChatEncoder(record):
            chat = []
            try:
                if record["history"]:
                    chat.append({"role": "assistant", "content": f"{record['history']}"})
            except KeyError:
                pass
            try:
                if record["question"]:
                    chat.append({"role": "user", "content": f"{record['question']}"})
            except KeyError:
                pass
            try:
                if record["instruct"]:
                    chat.append({"role": "user", "content": f"{record['instruct']}"})
            except KeyError:
                pass
            try:
                if record["answer"]:
                    chat.append({"role": "assistant", "content": f"{record['answer']}"})
            except KeyError:
                pass
            try:
                if record["completion"]:
                    chat.append({"role": "assistant", "content": f"{record['completion']}"})
            except KeyError:
                pass
            
            if context.showChatTemplate:
                print(tokenizer.apply_chat_template(chat, chat_template = customChatTemplate, add_generation_prompt = False, tokenize = False, return_dict = False))
            
            return tokenizer.apply_chat_template(chat, chat_template = customChatTemplate, add_generation_prompt = False, tokenize = True, return_dict = True)
        datasets.append(qaDataset.map(datasetChatEncoder, remove_columns=qaDataset.column_names["train"])["train"])
