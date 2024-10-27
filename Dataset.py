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
                datasets.append(textDataset["train"])
                
            elif file.name.lower().endswith(".json") or file.name.lower().endswith(".jsonl"):
                qaDataset = self.loadFile("json", str(file), context)
                if "text" in qaDataset['train'].features:
                    datasets.append(qaDataset["train"])
                elif "question" in qaDataset['train'].features and "answer" in qaDataset['train'].features:
                    def datasetChatEncoder(record):
                        chat = (
                            {"role": "user", "content": f"{record['question']}"},
                            {"role": "assistant", "content": f"{record['answer']}"},
                        )
                        return {"text": tokenizer.apply_chat_template(chat, continue_final_message = True, tokenize = False)}
                    datasets.append(qaDataset.map(datasetChatEncoder, remove_columns=("question", "answer"))["train"])
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
