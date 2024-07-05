# ReBase: Training Task Experts through Retrieval Based Distillation

**Re**trieval **Base**d Distillation (**ReBase**) is a method that first retrieves data from rich online sources and then transforms them into domain-specific data. This method greatly enhances data diversity. Moreover, **ReBase** generates Chain-of-Thought reasoning and distills the reasoning capacity of LLMs. 
 
## Quick Start
### Token Prepare

Please prepare the Anthropic REST API key and the Huggingface token (with write access). Copy these into the `config.py` file located in both the `data_preparation` and `finetune` folders.

### Environment Setup
```
conda create -n rebase python=3.10
conda activate rebase
pip install -r requirements.txt
```
- More details about unsloth install could refer to: https://github.com/unslothai/unsloth

### Run Scripts
You can modify and run run_scripts.sh to pass all the procedures needed. However, it is recommended to read through the following steps to understand more details about what we are doing here. 

### Stage 1: Data Preparation

```
cd data_preparation
```

Step 1: download datasets and and merge them into a corpus.
```
python merge_datasets.py 
```
- This step takes 1 or 2 days to finish, you could also find a subset of `dataset_index.json` to process data.

Step 2: get corpus embedding
```
python embed_corpus.py
```

Step 3: retrieve from corpus for a specific task
```
python retrieve_data.py --task_name squad
```
- available task_name including: 'mconala', 'mnli', 'squad', 'bbh' (This task includes following subtasks:['date_understanding', 'logical_deduction_five_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'object_counting', 'word_sorting', 'hyperbaton', 'sports_understanding', 'boolean_expressions', 'tracking_shuffled_objects_seven_objects', 'ruin_names', 'tracking_shuffled_objects_three_objects', 'causal_judgement', 'reasoning_about_colored_objects', 'logical_deduction_seven_objects', 'temporal_sequences', 'salient_translation_error_detection', 'tracking_shuffled_objects_five_objects', 'geometric_shapes', 'disambiguation_qa', 'dyck_languages', 'navigate', 'formal_fallacies', 'web_of_lies', 'snarks', 'penguins_in_a_table', 'logical_deduction_three_objects'])

Step 4: transform retrieved data to training data
```
python dataset_transformer.py --task_name squad
```

### Stage 2: Finetune Model

```
cd finetune
```

Step 5: finetune model using transformed data

```
python finetune.py --model_name unsloth/llama-3-8b-bnb-4bit --data_path ../data_preparation/tasks/squad/transformed_data_score_use_full_row_dataset.csv --finetuned_model_name Username/modelname 
```

- `model_name`: Name of the pre-trained model.
- `data_path`: Path to the training data CSV file.
- `finetuned_model_name`: Specify the name of the fine-tuned model. Replace "Username" with your Huggingface username and create a model name of your choice.
