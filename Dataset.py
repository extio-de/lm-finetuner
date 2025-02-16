import pathlib

from Context import Context
from datasets import load_dataset, concatenate_datasets

from datasets import disable_caching
disable_caching()

class Dataset:
    def scan(self, location: str, tokenizer, context: Context):
        print(f"Scanning dataset directory {location}")

        customChatTemplate = None
        if context.locCustomPromptTemplate != None:
            with open(context.locCustomPromptTemplate, 'r') as file:
                customChatTemplate = file.read()
        
        datasets = []
        directory = pathlib.Path(location)
        files = [f for f in directory.iterdir() if f.is_file()]
        for file in files:
            print(file.name)
            
            if file.name.lower().endswith(".txt"):
                textDataset = self.loadFile("text", str(file), context)
                self.__loadTextDataset(textDataset, datasets, tokenizer, context)
                
            elif file.name.lower().endswith(".json") or file.name.lower().endswith(".jsonl"):
                qaDataset = self.loadFile("json", str(file), context)
                if "text" in qaDataset['train'].features:
                    self.__loadTextDataset(qaDataset, datasets, tokenizer, context)
                elif "history" in qaDataset['train'].features or "instruct" in qaDataset['train'].features or "completion" in qaDataset['train'].features or "question" in qaDataset['train'].features or "answer" in qaDataset['train'].features:
                    self.__loadQaDataset(qaDataset, datasets, tokenizer, customChatTemplate, context)
                elif "conversation" in qaDataset['train'].features:
                    self.__loadConversationDataset(qaDataset, datasets, tokenizer, customChatTemplate, context)
                else:
                    print("Cannot load dataset, json dataset is malformed")
        
        if len(datasets) == 0:
            raise Exception("No datasets have been found. Check configuration")
        
        return concatenate_datasets(datasets).shuffle(seed = 4711)
    
    def loadFile(self, type_: str, path: str, context: Context, **kwargs):
        return self.__load(type_, context, data_files=path, **kwargs)

    def loadDataDir(self, type_: str, location: str, context: Context, **kwargs):
        return self.__load(type_, context, data_dir=location, **kwargs)
    
    def __load(self, type_: str, context: Context, **kwargs):
        print("Loading dataset...")
        
        dataset = load_dataset(type_, **kwargs)
        print(dataset)
        
        return dataset
    
    def __loadTextDataset(self, textDataset, datasets, tokenizer, context):
        def datasetTextEncoder(batch):
            inputIds = []
            attentionMask = []
            size = len(list(batch.values())[0])
            limit = context.trMaxSeqLength - 20
            for i in range(size):
                encoded = tokenizer(batch["text"][i])
                inputIds.extend([encoded['input_ids'][chunk:chunk + limit] for chunk in range(0, len(encoded['input_ids']), limit)])
                attentionMask.extend([encoded['attention_mask'][chunk:chunk + limit] for chunk in range(0, len(encoded['attention_mask']), limit)])
            
            return {"input_ids": inputIds, "attention_mask": attentionMask}
        
        datasets.append(textDataset.map(datasetTextEncoder, batched = True, batch_size = 250, num_proc = 4, remove_columns="text")["train"])
    
    def __loadQaDataset(self, qaDataset, datasets, tokenizer, customChatTemplate, context):
        MAPPING = {"history": "assistant",
                   "question": "user",
                   "instruct": "user",
                   "answer": "assistant",
                   "completion": "assistant"}
        
        def datasetChatEncoder(batch):
            inputIds = []
            attentionMask = []
            size = len(list(batch.values())[0])
            columns = [x for x in MAPPING.keys() if x in batch]
            
            for i in range(size):
                chat = []
                for column in columns:
                    chat.append({"role": MAPPING[column], "content": batch[column][i]})
                
                encoded = self.__format(chat, tokenizer, customChatTemplate, context)
                inputIds.append(encoded['input_ids'])
                attentionMask.append(encoded['attention_mask'])
            
            return {"input_ids": inputIds, "attention_mask": attentionMask}
        
        datasets.append(qaDataset.map(datasetChatEncoder, batched = True, batch_size = 250, num_proc = 4, remove_columns=qaDataset.column_names["train"])["train"])
    
    def __loadConversationDataset(self, qaDataset, datasets, tokenizer, customChatTemplate, context):
        def datasetConversationEncoder(batch):
            inputIds = []
            attentionMask = []
            for records in batch['conversation']:
                chat = []
                cnt = 0
                for record in records:
                    curCnt = len(tokenizer.tokenize(record['user'])) + len(tokenizer.tokenize(record['assistant'])) + 15
                    if cnt + curCnt >= context.trMaxSeqLength:
                        encoded = self.__format(chat, tokenizer, customChatTemplate, context)
                        inputIds.append(encoded['input_ids'])
                        attentionMask.append(encoded['attention_mask'])
                        chat = []
                        cnt = curCnt
                    else:
                        cnt += curCnt
                    chat.append({"role": "user", "content": f"{record['user']}"})
                    chat.append({"role": "assistant", "content": f"{record['assistant']}"})
                encoded = self.__format(chat, tokenizer, customChatTemplate, context)
                inputIds.append(encoded['input_ids'])
                attentionMask.append(encoded['attention_mask'])
            return {"input_ids": inputIds, "attention_mask": attentionMask}
        
        datasets.append(qaDataset.map(datasetConversationEncoder, batched = True, batch_size = 250, num_proc = 4, remove_columns=qaDataset.column_names["train"])["train"])
    
    def __format(self, chat, tokenizer, customChatTemplate, context):
        if context.showChatTemplate:
            print(tokenizer.apply_chat_template(chat, chat_template = customChatTemplate, add_generation_prompt = False, tokenize = False, return_dict = False))
        return tokenizer.apply_chat_template(chat, chat_template = customChatTemplate, add_generation_prompt = False, tokenize = True, return_dict = True)
