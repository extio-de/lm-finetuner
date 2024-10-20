# lm-finetuner

**Program Name:** Data Processing Pipeline

**Purpose:** The program is a data processing pipeline that trains a machine learning model, validates the results, and merges and stores the output. It is designed to be highly customizable and can be used for various machine learning tasks, including text classification, language modeling, and question answering.

**Key Features:**

1. **Configuration Management:** The program uses a configuration loader to load settings from a configuration file, which controls various aspects of the pipeline, including training settings, validation settings, and data loading settings.
2. **Data Loading:** The program has a data loader class that can load datasets from various sources, including text files, JSON files, and JSONL files, and can process the data using a tokenizer.
3. **Model Training:** The program uses a trainer class that trains a model using the SFT (Supervised Fine-tuning) algorithm and can save the trained model and its tokenizer to a specified location.
4. **Validation:** The program has a validator class that performs validation and grading of user responses using two models: a PEFT model for generating answers and a grader model for evaluating the correctness of the generated answers.
5. **Model Merging:** The program has a merger class that can merge a pre-trained model with an adapter and store the result.
6. **Model Loading:** The program has a model loader class that can load a pre-trained causal language model from a given path, with support for loading on a GPU and/or with 4-bit quantization.

**Overall:** The program is designed to be a flexible and customizable data processing pipeline that can be used for various machine learning tasks, from training and validating models to loading and merging pre-trained models.

*["requests":8,"requestDuration":"PT23.926013939S","inTokens":6994,"outTokens":1277,"tps":55.52173913043478]*
