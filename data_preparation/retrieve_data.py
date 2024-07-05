from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch
import pickle
from pathlib import Path
import os
import argparse


def get_model(device="cuda"):
    """Start with sentence bert, could go with more complicated designs"""
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    model = model.to(device)
    return model


def batch_encode(model, texts, batch_size=128):
    all_embeddings = []
    for start_index in range(0, len(texts), batch_size):
        batch_texts = texts[start_index:start_index+batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_tensor=True)
        all_embeddings.extend(embeddings)
    return all_embeddings


def load_full_embedding():
    #Load sentences & embeddings from disc
    embeddings_file_path = 'embeddings_dataset_final.pkl'  # File to store embeddings_1
    with open(embeddings_file_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['ids']
        stored_embeddings = stored_data['embeddings']
    stored_values_file = 'stored_values.pkl'
    with open(stored_values_file, 'rb') as f:
        stored_values = pickle.load(f)
    return stored_sentences, stored_embeddings, stored_values


def load_dataset_embedding():
    #Load sentences & embeddings from disc
    embeddings_file_path = 'embeddings_dataset_final.pkl'  # File to store embeddings_1
    with open(embeddings_file_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['ids']
        stored_embeddings = stored_data['embeddings']
    
    stored_values_file = 'stored_dict.pkl'
    with open(stored_values_file, 'rb') as f:
        stored_values = pickle.load(f)
    return stored_sentences,stored_embeddings


stored_sentences, stored_embeddings, stored_values = load_full_embedding()
stored_sentences2,stored_dataset_embeddings = load_dataset_embedding()


def score_data_points(input_prompt_list, output_prompt_list, instruction, model, use_sample=False, batch_size=64):
    scores = {}
    prompt_embedding_list = []
    dataset_embedding = model.encode([instruction], convert_to_tensor=True)
    for input_prompt, output_prompt in zip(input_prompt_list, output_prompt_list):
        prompt_embedding = model.encode([input_prompt, output_prompt], convert_to_tensor=True)
        prompt_embedding_list.append(prompt_embedding)

    stored_embeddings_tensor = torch.tensor(stored_embeddings).to(prompt_embedding.device) 
    input_scores_list = []
    output_scores_list = []
    # Compute cosine similarities
    for prompt_embedding in prompt_embedding_list:
        input_scores = util.pytorch_cos_sim(prompt_embedding[0], stored_embeddings_tensor)
        output_scores = util.pytorch_cos_sim(prompt_embedding[1], stored_embeddings_tensor)
        input_scores_list.append(input_scores)
        output_scores_list.append(output_scores)
    input_scores = torch.stack(input_scores_list).mean(dim=0)
    output_scores = torch.stack(output_scores_list).mean(dim=0)
    del stored_embeddings_tensor
    stored_dataset_embeddings_tensor = torch.tensor(stored_dataset_embeddings).to(prompt_embedding.device)
    dataset_scores = util.pytorch_cos_sim(dataset_embedding[0], stored_dataset_embeddings_tensor)
    
    # Iterate through each stored sentence to populate the scores dictionary
    for idx, sentence_id in enumerate(stored_sentences):
        key = sentence_id  # Assuming sentence_id can be used directly as key
        value = stored_values[idx]
        if key not in scores:
            scores[key] = {'input': [], 'output': [], 'dataset': []}

        # Append the cosine similarity scores and corresponding values
        # Note: The actual 'value' associated with each embedding needs to be retrieved if necessary
        dataset_score_float = float(dataset_scores[0][idx])
        input_score_float = float(input_scores[0][idx])
        output_score_float = float(output_scores[0][idx])

        scores[key]['input'].append((input_score_float, key, value))
        scores[key]['output'].append((output_score_float, key, value))
        scores[key]['dataset'].append((dataset_score_float, key, value))

    return scores
                

def select_top_data_points_average_score(scores, max_points=3000, banned_name=None):
    final_scores_with_values = []
    for key, score_lists in scores.items():
        # Find the data point with the maximum input score and its corresponding value
        max_input = max(score_lists['input'], key=lambda x: x[0])
        max_input_score, max_input_id, max_input_value = max_input

        # Find the data point with the maximum output score and its corresponding value
        max_output = max(score_lists['output'], key=lambda x: x[0])
        max_output_score, max_output_id, max_output_value = max_output

        # Find the data point with the maximum output score and its corresponding value
        max_dataset = max(score_lists['dataset'], key=lambda x: x[0])
        max_dataset_score, max_dataset_id, max_dataset_value = max_dataset
        
        # Calculate the final score
        final_score = (max_input_score + max_output_score + max_dataset_score) / 3
        final_scores_with_values.append((final_score, key, max_input_value, max_output_value))

    # Sort by final score and select top data points
    final_scores_with_values.sort(key=lambda x: x[0], reverse=True)
    tot = 0
    top_data_points = []
    for item in final_scores_with_values:
        if banned_name in item[1]:
            continue
        if tot == max_points:
            break
        tot += 1
        top_data_points.append(item)
    return top_data_points


def select_top_data_points_average_rank(scores, max_points=3000, use_method='average'):
    final_scores_with_values = []
    for key, score_lists in scores.items():
        # Find the data point with the maximum input score and its corresponding value
        max_input = max(score_lists['input'], key=lambda x: x[0])
        max_input_score, max_input_id, max_input_value = max_input

        # Find the data point with the maximum output score and its corresponding value
        max_output = max(score_lists['output'], key=lambda x: x[0])
        max_output_score, max_output_id, max_output_value = max_output

        # Calculate the final score
        final_scores_with_values.append((max_input_score, max_output_score, key,  max_input_value, max_output_value))

    # Sort by final score and select top data points
    if use_method == 'average':
        final_scores_with_values = sort_by_average_rank(final_scores_with_values)
    elif use_method == 'max':
        final_scores_with_values = sort_by_max_rank(final_scores_with_values)
    top_data_points = final_scores_with_values[:max_points]
    return top_data_points


def calculate_ranks(items, score_index):
    # Sort items based on a specific score and calculate ranks
    sorted_items = sorted(items, key=lambda x: x[score_index], reverse=True)
    ranks = {item: rank for rank, item in enumerate(sorted_items, start=1)}
    return ranks
    

def sort_by_average_rank(items):
    # Calculate ranks based on input and output scores
    input_ranks = calculate_ranks(items, 0)  # 0 is the index for input_score
    output_ranks = calculate_ranks(items, 1)  # 1 is the index for output_score
    new_item_list = []
    # Calculate average rank for each item
    for item in items:
        input_rank = input_ranks[item]
        output_rank = output_ranks[item]
        avg_rank = (input_rank + output_rank) / 2
        new_item_list.append((item, avg_rank))
        # item.append(avg_rank)  # Append average rank to each item

    # Sort items based on average rank
    sorted_by_avg_rank = sorted(new_item_list, key=lambda x: x[-1])  # Sorting by the last element, which is avg_rank

    return sorted_by_avg_rank


def sort_by_max_rank(items):
    # Calculate ranks based on input and output scores
    input_ranks = calculate_ranks(items, 0)  # 0 is the index for input_score
    output_ranks = calculate_ranks(items, 1)  # 1 is the index for output_score
    new_item_list = []
    # Calculate average rank for each item
    for item in items:
        input_rank = input_ranks[item]
        output_rank = output_ranks[item]
        max_rank = max(input_rank, output_rank)
        new_item_list.append((item, max_rank))
        # item.append(avg_rank)  # Append average rank to each item

    # Sort items based on average rank
    sorted_by_avg_rank = sorted(new_item_list, key=lambda x: x[-1])  # Sorting by the last element, which is avg_rank

    return sorted_by_avg_rank


def retrieve_data(jsonl_file_path, input_prompt, output_prompt, instruction, model, task_name, compare_type="score"):
    scored_data = score_data_points(input_prompt, output_prompt, instruction, model)
    if compare_type == "score":
        top_data_points = select_top_data_points_average_score(scored_data, banned_name=task_name)
    else:
        top_data_points = select_top_data_points_average_rank(scored_data)
    return top_data_points


def write_data_to_json(file_path, data):
    print('get into write data to json')
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    

def get_dataset(input_example, output_example, instruction, task_name):
    model = get_model()
    jsonl_file_path = "datasets_merged_flattened_final.jsonl"
    retrieved_data = retrieve_data(jsonl_file_path, input_example, output_example, instruction, model, task_name)
    return retrieved_data


def retrieve(task_name):
    if task_name=='mconala':
        instruction = """
                Given a Japanese instruction, generate the according python code.
                """
        input_prompt = "スペースで区切られた入力`stdin`を変数に格納して表示する"
        output_prompt = "for line in stdin: a = line.rstrip().split(' ') print(a)"
        input_prompt2 = "HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する"
        output_prompt2 = "f = open('test.html', 'w') f.write(text.encode('utf-8'))"
        input_prompt3 = "tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る"
        output_prompt3 = "search = api.search(q=word)"
        input_prompt_list = [input_prompt, input_prompt2, input_prompt3]
        output_prompt_list = [output_prompt, output_prompt2, output_prompt3]
        
    elif task_name == 'mnli':
        instruction = """
            Given a premise sentence and a hypothesis sentence, predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).
        """
        input_prompt = "Premise: She smiled back. Hypothesis: She was so happy she couldn't stop smiling."
        output_prompt = "Neutral"   
        input_prompt2 = "Premise: And to show just how fast Japan's new rulers were catching on, two punitive expeditions were launched against Korea and China in the grand manner of 19th-century gunboat diplomacy. Hypothesis: Japan's new rulers were catching on quickly."
        output_prompt2 = "Entailment"
        input_prompt3 = "Premise: Fun for adults and children. Hypothesis: Fun for only children."
        output_prompt3 = "Contradiction"
        input_prompt_list = [input_prompt, input_prompt2, input_prompt3]
        output_prompt_list = [output_prompt, output_prompt2, output_prompt3]

    elif task_name == 'squad':
        instruction = """
        Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.
        """
        input_prompt = """Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."""
        output_prompt = """Santa Clara"""
        input_prompt2 = """Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."""
        output_prompt2 = """Vistula River"""
        input_prompt3 = """Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."""
        output_prompt3 = """Europe"""
        input_prompt_list = [input_prompt, input_prompt2, input_prompt3]
        output_prompt_list = [output_prompt, output_prompt2, output_prompt3]

    retrieved_data = get_dataset(input_prompt_list, output_prompt_list, instruction, task_name)
    output_file_path = "./tasks/"+task_name+"/retrieved_data_rank_score_dataset.json"
    write_data_to_json(output_file_path, retrieved_data)

def write_data_to_json(file_path, data):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

                
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

def get_original_rows_from_file(task_name, use_sample=False, start=0, tot_num=3000, neglect_dataset=False):
    
    stored_values_file = 'stored_dict.pkl'
    with open(stored_values_file, 'rb') as f:
        stored_values = pickle.load(f)

    json_file_path = "./tasks/" + task_name + "/retrieved_data_rank_score_dataset.json"
    data_list = load_json_file(json_file_path)
    row_list = []
    cur_num = 0
    for num, item in enumerate(data_list):
        if cur_num==tot_num:
            break
        location = item[1]
        if neglect_dataset:
            if "'dataset_id': 'gsm8k'" in location:
                print("found original gsm8k dataset!")
                continue
        try:
            # Retrieve the row
            row = stored_values[location]
            row_list.append(row)
        except:
            print("cannot find" + location)
        cur_num += 1

    output_file_path = "./tasks/" + task_name + "/selected_dataset.json"
    write_data_to_json(output_file_path, row_list)
    
    return row_list
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve data for a specific task.") 
    parser.add_argument("--task_name", type=str, help="Name of the task.")  
    args = parser.parse_args()
    TASK_NAME_LIST = [args.task_name]
    for task_name in TASK_NAME_LIST:
        if "bbh" not in task_name:
            retrieve(task_name)
            get_original_rows_from_file(task_name)
        else:
            with open('./bbh/instructions.json', 'r') as file:
                data_instructions = json.load(file)
            with open('./bbh/template.json', 'r') as file:
                data_examples = json.load(file)
            for task_name in data_examples.keys():
                instruction = data_instructions[task_name]
                example_list = data_examples[task_name]
                input_prompt_list = [example[0] for example in example_list]
                output_prompt_list = [example[1].split("answer is ")[1] for example in example_list]
                path = Path("./tasks/bbh/"+task_name)
                path.mkdir(parents=True, exist_ok=True)
                output_file_path = "./tasks/bbh/"+task_name+"/retrieved_data_rank_score_dataset.json"
                jsonl_file_path = "datasets_merged_flattened_final.jsonl"
                retrieved_data = get_dataset(input_prompt_list, output_prompt_list, instruction, task_name)
                write_data_to_json(output_file_path, retrieved_data)
                get_original_rows_from_file("bbh/"+task_name)
                
            
    