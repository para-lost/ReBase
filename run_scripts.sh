#!/bin/bash

TASK="bbh"
# SUBTASK is only for Big Bench Hard (bbh) task.
SUBTASK="boolean_expressions"
USERNAME=""
MODELNAME="unsloth/llama-3-8b-bnb-4bit "
CUSTOMIZED_MODELNAME=""
echo "TASK is set to: $TASK"
echo "USERNAME is set to: $USERNAME"
echo "MODELNAME is set to: $MODELNAME"
echo "CUSTOMIZED_MODELNAME is set to: $CUSTOMIZED_MODELNAME"

cd data_preparation
python merge_datasets.py 
python embed_corpus.py
python retrieve_data.py --task_name $TASK
python dataset_transformer.py --task_name $TASK
cd ..

cd finetune
if [ "$TASK" == "bbh" ]; then
    python finetune.py --model_name $MODELNAME --data_path ../data_preparation/tasks/$TASK/$SUBTASK/transformed_data_score_use_full_row_dataset.csv --finetuned_model_name $USERNAME/$CUSTOMIZED_MODELNAME 
else
    python finetune.py --model_name $MODELNAME --data_path ../data_preparation/tasks/$TASK/transformed_data_score_use_full_row_dataset.csv --finetuned_model_name $USERNAME/$CUSTOMIZED_MODELNAME 
fi
