from unsloth import FastLanguageModel
import torch
import csv
from datasets import Dataset
from prompt import squad_prompt_template
from trl import SFTTrainer
from transformers import TrainingArguments
import argparse
from config import HF_TOKEN

def formatting_prompts_func(examples):
    texts = []
    num_examples = len(examples['question'])
    for index in range(num_examples): 
        
        input = examples['question'][index]
        output = examples['answer'][index]
        text = prompt_template.format(input=str(input), response=str(output)) + EOS_TOKEN
        texts.append(text)
        
    return { "text" : texts, }


def load_dataset_from_csv(file_path):
    """Load the dataset from a CSV file back into a Python dictionary."""    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        dataset_rows = {'question': [], 'answer': []}
        for row in reader:
            # Each row is a dictionary with keys matching the CSV column headers ('input' and 'output' )
            dataset_rows['question'].append(row['input'])
            dataset_rows['answer'].append(row['output'])
    return Dataset.from_dict(dataset_rows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and fine-tune a language model.")
    parser.add_argument("--model_name", type=str, help="Name of the pre-trained model.")
    parser.add_argument("--data_path", type=str, help="Path to the training data CSV file.")
    parser.add_argument("--finetuned_model_name", type=str, help="Name of the fine-tuned model.")

    args = parser.parse_args()
    max_seq_length = 2048 
    dtype = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = True,
        token = HF_TOKEN
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    prompt_template = squad_prompt_template

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    dataset = load_dataset_from_csv(args.data_path)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # Train model
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 20,
            max_steps = -1,
            num_train_epochs=1,
            learning_rate = 3e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "unsloth_outputs_8_shots",
            report_to="none", 
        ),
    )
    trainer_stats = trainer.train()

    # Save model to huggingface hub
    model.push_to_hub(args.finetuned_model_name, token = HF_TOKEN)
    tokenizer.push_to_hub(args.finetuned_model_name, token = HF_TOKEN)